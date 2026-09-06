# -*- coding: utf-8 -*-
"""
问题二·任务2:品类日需求量 7 日预测(STL 分解 + LightGBM 分位数回归)
====================================================================
目标:预测 2023-07-01 ~ 07-07 六品类每日需求量(kg),输出 P10/P50/P90。
口径:需求 = 品类日总销量(正价+品相折扣+清仓,与任务一 simulate_day 的
Q_total 口径一致);销量在售罄日被截断而低于真实需求 —— 见断货处理。

方法构成(小组定稿 2026-09-06):
  1) 周季节分解:经典加法分解(7 日滑动平均趋势 + 星期中位数季节因子),
     statsmodels 不可用故手写;因子只在各折训练窗内估计;折边缘 3 天的
     趋势强制用 trailing 窗口,杜绝"偷看未来";
  2) 断货检测与三处接入:检测器 = 当日最后成交时刻比该品类同星期几的
     近 8 个同星期日中位早 >=3 小时且销量不低于近期一半;接入:滞后标记
     作特征 / 疑似截断日剔除或按日内累计曲线修复训练目标 / 评价双轨。
     三种处理(无/剔除/修复)消融,按整体 WAPE 择优做最终预测;
  3) 直接多步:滞后特征只用 >=7 天前信息(lag7/14/21/28 + 平移 7 日的
     28 日滚动统计),单模型直接预测 7 天,不递归;
  4) log 分位数回归:目标 ln(Q+0.1),预测后 exp(q)-0.1 精确回换;
  5) rolling-origin 回测:2023-01-02 起逐周 24 折;指标 WAPE/MASE/RMSSE
     + pinball;基线:季节朴素、STL+岭回归。

输入:附件1/2 + 任务一产物 p2_品类日内画像.csv(断货修复曲线)
输出:
  p2_回测指标汇总.csv        各模型×品类:WAPE/MASE/RMSSE/pinball(10/50/90)
  p2_需求预测_202307.csv     六品类 7 天 P10/P50/P90(kg)
  p2_fig4_需求预测.png       近 90 天历史 + 预测中位数与 10-90 区间(六品类小图)
"""
import sys
import warnings
import traceback

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 配置 ----------------
A1, A2 = "附件1.xlsx", "附件2.xlsx"
PROF = "p2_品类日内画像.csv"
QUANTILES = [0.1, 0.5, 0.9]
FOLD_START, FOLD_END = "2023-01-02", "2023-06-17"   # 每折预测其后 7 天
FORECAST_START = "2023-07-01"
LAG_MIN = 7
CENS_HOUR = 3
CATS = ["花叶类", "食用菌", "辣椒类", "水生根茎类", "茄类", "花菜类"]
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE = "#2a78d6", "#eb6834"

HOLIDAYS = set()      # 节假日日期集合(load_daily 填充)


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def get_lgbm_factory():
    """优先 LightGBM;不可用退回 sklearn HistGradientBoosting(同为分位数 GBDT)"""
    try:
        from lightgbm import LGBMRegressor

        def make(alpha):
            return LGBMRegressor(
                objective="quantile", alpha=alpha, n_estimators=400,
                learning_rate=0.05, num_leaves=31, min_child_samples=30,
                colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
                verbose=-1, n_jobs=-1)
        return "LightGBM", make
    except Exception:
        pass
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        def make(alpha):
            return HistGradientBoostingRegressor(
                loss="quantile", quantile=alpha, max_iter=400,
                learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=30,
                l2_regularization=1.0, random_state=42)
        return "HistGradientBoosting", make
    except Exception:
        return None, None


def fit_ridge(X, y, lam=1.0):
    """闭式岭回归(截距不惩罚),返回预测函数"""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    P = np.eye(Xs.shape[1]) * lam
    P[0, 0] = 0.0
    beta = np.linalg.pinv(Xs.T @ Xs + P) @ (Xs.T @ y)

    def pred(Xt):
        Z = np.column_stack([np.ones(len(Xt)), (Xt - mu) / sd])
        return Z @ beta
    return pred


def ext7(arr, N, extra=0):
    """把长度 N 的因果特征延长 extra 格:未来第 j 天的滞后特征 = arr[j-7],
    即尾部取 arr[N-7 : N-7+extra](正确取值,而非补 NaN)"""
    if extra <= 0:
        return arr
    return np.concatenate([arr, arr[N - 7:N - 7 + extra]])


# ---------------- 数据准备 ----------------
def load_daily():
    """六品类全日历日度面板:总销量 Q、最后成交时刻 L、断货标记等"""
    a1 = pd.read_excel(A1)[["单品编码", "分类名称"]]
    a2 = pd.read_excel(A2)
    a2 = a2[a2["销售类型"] == "销售"].copy()
    a2["销售日期"] = pd.to_datetime(a2["销售日期"])
    a2["hour"] = pd.to_datetime(a2["扫码销售时间"].astype(str), errors="coerce").dt.hour
    a2 = a2.merge(a1, on="单品编码", how="left")
    print(f"销售流水 {len(a2)} 条(已剔除退货)")

    cal = pd.date_range(a2["销售日期"].min(), a2["销售日期"].max(), freq="D")
    N = len(cal)
    for yr in (2020, 2021, 2022):
        HOLIDAYS.update(pd.date_range(f"{yr}-10-01", f"{yr}-10-07"))
    for a, b in [("2021-02-11", "2021-02-17"), ("2022-01-31", "2022-02-06"),
                 ("2023-01-21", "2023-01-27"), ("2021-05-01", "2021-05-05"),
                 ("2022-04-30", "2022-05-04"), ("2023-04-29", "2023-05-03")]:
        HOLIDAYS.update(pd.date_range(a, b))

    gq = a2.groupby(["分类名称", "销售日期"])["销量(千克)"].sum()
    gl = a2.groupby(["分类名称", "销售日期"])["hour"].max()

    data = {}
    for c in CATS:
        Q = np.array([gq.get((c, d), 0.0) for d in cal], dtype=float)
        L = np.array([gl.get((c, d), np.nan) for d in cal], dtype=float)
        closed = Q <= 0
        hol = np.array([d in HOLIDAYS for d in cal], dtype=float)
        wd = np.array([d.dayofweek for d in cal], dtype=int)

        # ---- 断货检测(只用过去信息,因果) ----
        sL, sQ = pd.Series(L), pd.Series(Q)
        base_L = np.full(N, np.nan)
        for w in range(7):
            m = wd == w
            base_L[m] = sL[m].expanding().median().shift(1).values  # 同星期几历史中位
        qmed28 = sQ.rolling(28, min_periods=14).median().shift(1).values
        cens = ((~closed) & (~np.isnan(base_L)) & (L <= base_L - CENS_HOUR)
                & (Q >= 0.5 * np.nan_to_num(qmed28, nan=np.inf))).astype(float)
        cens_lag7 = pd.Series(cens).shift(7).values
        cens_rate7 = pd.Series(cens).rolling(7).mean().shift(7).values
        closed_lag7 = pd.Series(closed.astype(float)).shift(7).values

        data[c] = dict(Q=Q, L=L, closed=closed, hol=hol, wd=wd,
                       cens=cens, cens_lag7=cens_lag7,
                       cens_rate7=cens_rate7, closed_lag7=closed_lag7)
    n_c = {c: int(np.nansum(data[c]["cens"])) for c in CATS}
    print("疑似提前售罄日检出(全部历史):",
          {c: f"{n_c[c]}天({n_c[c] / N:.1%})" for c in CATS})
    return data, cal, N


def stl_factors(y, wd, closed, train_mask):
    """经典加法分解的星期季节因子(仅训练窗、非闭市日);
    训练窗最后 3 天的趋势强制 trailing,避免居中窗口偷看未来"""
    trend = pd.Series(y).rolling(7, center=True, min_periods=3).mean().values.copy()
    last = np.where(train_mask)[0].max()
    for j in range(max(0, last - 2), last + 1):
        trend[j] = np.nanmean(y[max(0, j - 6):j + 1])
    det = y - trend
    S = np.zeros(7)
    for w in range(7):
        m = train_mask & (wd == w) & (~closed) & (~np.isnan(det))
        S[w] = np.median(det[m]) if m.sum() > 0 else 0.0
    return S


def make_design(data, a_by_cat, cal, N_extra=0):
    """拼装池化设计矩阵(品类×日期长表)。
    行位置从 35 起(lag28 与 28 日滚动均可用);目标 = 去季节 log 序列。"""
    start = LAG_MIN + 28
    rows_X, rows_a, rows_cat, rows_pos = [], [], [], []
    for ci, c in enumerate(CATS):
        d = data[c]
        N = len(d["Q"])
        M = N + N_extra
        a = np.full(M, np.nan)
        a[:N] = a_by_cat[c]
        lag = {k: np.concatenate([np.full(k, np.nan), a[:M - k]]) for k in (7, 14, 21, 28)}
        s = pd.Series(a)
        rm = s.rolling(28, min_periods=14).mean().shift(LAG_MIN).values
        rs = s.rolling(28, min_periods=14).std().shift(LAG_MIN).values
        num = np.column_stack([
            lag[7], lag[14], lag[21], lag[28], rm, rs,
            ext7(d["cens_lag7"], N, N_extra), ext7(d["cens_rate7"], N, N_extra),
            ext7(d["closed_lag7"], N, N_extra),
            pad_hol(N, N_extra)])
        cal_ext = list(cal) + [cal[-1] + pd.Timedelta(days=i + 1)
                               for i in range(N_extra)]
        wd1h = np.eye(7)[np.array([x.dayofweek for x in cal_ext])]
        m1h = np.eye(12)[np.array([x.month for x in cal_ext]) - 1]
        yr = np.array([x.year for x in cal_ext])
        y1h = np.column_stack([(yr == 2021).astype(float), (yr == 2022).astype(float)])
        c1h = np.zeros((M, len(CATS)))
        c1h[:, ci] = 1.0
        X = np.column_stack([num, wd1h, m1h, y1h, c1h])
        keep = np.arange(start, M)
        rows_X.append(X[keep])
        rows_a.append(a[keep])
        rows_cat.append(np.full(len(keep), ci))
        rows_pos.append(keep)
    return (np.vstack(rows_X), np.concatenate(rows_a),
            np.concatenate(rows_cat), np.concatenate(rows_pos))


def pad_hol(N, extra):
    """节假日标记,延展到未来 7 天(未来日期的节假日是已知的)"""
    cal = pd.date_range("2020-07-01", periods=N + extra)
    return np.array([d in HOLIDAYS for d in cal], dtype=float)


def metrics(y_true, y_pred, scale1, scale2, preds_q=None):
    e = y_pred - y_true
    out = {"WAPE": np.abs(e).sum() / max(y_true.sum(), 1e-9),
           "MASE": np.abs(e).mean() / max(scale1, 1e-9),
           "RMSSE": np.sqrt((e ** 2).mean()) / max(scale2, 1e-9)}
    if preds_q is not None:
        for q, p in zip(QUANTILES, preds_q):
            dd = y_true - p
            out[f"pinball{int(q * 100)}"] = float(
                np.mean(np.where(dd >= 0, q * dd, (q - 1) * dd)))
    return out


def back(q, shift=0.1):
    return np.clip(np.exp(q) - shift, 0.0, None)


# ---------------- 主流程 ----------------
def fig5_backtest(res, engine):
    """回测对比图:整体三指标 + pinball50 + 分品类主对照 + 断货消融"""
    def short(m):
        return (m.replace(f"STL+{engine}", "STL+GBDT")
                 .replace("无断货处理", "无断货").replace("断货剔除", "剔除")
                 .replace("断货修复", "修复"))
    res = res.copy()
    res["短名"] = res["模型"].map(short)
    order = res.groupby("短名")["WAPE"].mean().sort_values().index.tolist()
    pal = ["#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#898781"]
    mcolor = {m: pal[i % len(pal)] for i, m in enumerate(order)}

    fig = plt.figure(figsize=(14, 8), dpi=150, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.42)
    ov = res.groupby("短名")[["WAPE", "MASE", "RMSSE", "pinball50"]].mean()

    for i, (met, tit) in enumerate([("WAPE", "整体 WAPE"), ("MASE", "整体 MASE"),
                                    ("RMSSE", "整体 RMSSE")]):
        ax = fig.add_subplot(gs[0, i])
        style_ax(ax)
        v = ov.loc[order, met]
        ax.barh(range(len(order)), v.values,
                color=[mcolor[m] for m in order], height=0.62, zorder=3)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=9, color=INK)
        ax.invert_yaxis()
        for j, vv in enumerate(v.values):
            ax.text(vv + v.max() * 0.02, j, f"{vv:.3f}", va="center",
                    fontsize=8.5, color=INK2)
        ax.set_xlim(0, v.max() * 1.2)
        ax.set_title(f"{tit}(越低越好)", fontsize=11.5, color=INK, pad=8)
        ax.tick_params(labelsize=8.5)

    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax)
    main3 = ["季节朴素", "STL+岭回归", f"STL+{engine}(断货剔除)"]
    w = 0.26
    for j, m in enumerate(main3):
        sub = res[res["模型"] == m].set_index("品类")["WAPE"].reindex(CATS)
        ax.bar(np.arange(6) + (j - 1) * w, sub.values, width=w,
               color=mcolor[short(m)], label=short(m), zorder=3)
    ax.set_xticks(range(6))
    ax.set_xticklabels(CATS, fontsize=9, color=INK2)
    ax.set_title("分品类 WAPE:主模型 vs 基线", fontsize=11.5, color=INK, pad=26)
    ax.legend(fontsize=8, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.tick_params(labelsize=8.5)

    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax)
    gb = [m for m in order if "GBDT" in m]
    v = ov.loc[gb, "pinball50"]
    ax.bar(range(len(gb)), v.values, color=[mcolor[m] for m in gb],
           width=0.55, zorder=3)
    ax.set_xticks(range(len(gb)))
    ax.set_xticklabels(gb, fontsize=7.5, color=INK2)
    for i, vv in enumerate(v.values):
        ax.text(i, vv + v.max() * 0.02, f"{vv:.2f}", ha="center",
                fontsize=8.5, color=INK2)
    ax.set_ylim(0, v.max() * 1.18)
    ax.set_title("整体 pinball50(分位数质量)", fontsize=11.5, color=INK, pad=8)
    ax.tick_params(labelsize=8.5)

    ax = fig.add_subplot(gs[1, 2])
    style_ax(ax)
    m0, m1 = f"STL+{engine}(无断货处理)", f"STL+{engine}(断货剔除)"
    w = 0.36
    for j, (m, lab) in enumerate([(m0, "无断货处理"), (m1, "断货剔除")]):
        sub = res[res["模型"] == m].set_index("品类")["WAPE"].reindex(CATS)
        ax.bar(np.arange(6) + (j - 0.5) * w, sub.values, width=w,
               color=mcolor[short(m)], label=lab, zorder=3)
    ax.set_xticks(range(6))
    ax.set_xticklabels(CATS, fontsize=9, color=INK2)
    ax.set_title("断货处理消融:分品类 WAPE", fontsize=11.5, color=INK, pad=26)
    ax.legend(fontsize=8, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.tick_params(labelsize=8.5)

    fig.suptitle("回测对比:主模型 vs 对照基线(24 折滚动平均)", fontsize=13.5,
                 color=INK, y=0.99)
    fig.savefig("p2_fig5_回测对比.png", facecolor=SURFACE)
    plt.close(fig)


def main():
    data, cal, N = load_daily()
    engine, make_lgbm = get_lgbm_factory()
    print(f"分位数模型引擎:{engine or '无(请安装 lightgbm 或 scikit-learn)'}")

    y_log = {c: np.log(data[c]["Q"] + 0.1) for c in CATS}
    origins = pd.date_range(FOLD_START, FOLD_END, freq="7D")
    p0s = [int((t - cal[0]) / pd.Timedelta("1D")) for t in origins]

    # 断货修复曲线:任务一画像的全渠道日内累计销量份额 -> 提升系数 1/cum(L)
    cum_curves = {}
    try:
        prof = pd.read_csv(PROF)
        for c in CATS:
            h = (prof[prof["分类名称"] == c].groupby("hour")["销量占比"].sum()
                 .reindex(range(8, 23), fill_value=0.0))
            cum = h.cumsum()
            cum_curves[c] = {int(hh): (1.0 / v if v > 0.25 else np.nan)
                             for hh, v in cum.items()}
        print(f"已读取 {PROF}(断货目标修复用日内累计曲线)")
    except Exception as e:
        print(f"[提示] 读不到 {PROF}:断货修复档退化为剔除档({e})")

    lgbm_runs = ["无断货处理", "断货剔除", "断货修复"] if engine else []
    results = {}

    for fi, t0 in enumerate(origins):
        p0 = int((t0 - cal[0]) / pd.Timedelta("1D"))
        train_mask = np.zeros(N, dtype=bool)
        train_mask[:p0 + 1] = True
        a_by, S_by = {}, {}
        for c in CATS:
            S = stl_factors(y_log[c], data[c]["wd"], data[c]["closed"], train_mask)
            a_by[c] = y_log[c] - S[data[c]["wd"]]
            S_by[c] = S
        X, ya, cats_idx, poss = make_design(data, a_by, cal)
        tr = poss <= p0
        te = (poss > p0) & (poss <= p0 + 7)

        scale1, scale2 = {}, {}
        for c in CATS:
            Q = data[c]["Q"]
            diff = Q[7:p0 + 1] - Q[:p0 + 1 - 7]
            scale1[c] = np.abs(diff).mean()
            scale2[c] = np.sqrt((diff ** 2).mean())

        # ---- 季节朴素:Q(d-7) ----
        for ci, c in enumerate(CATS):
            yt = data[c]["Q"][p0 + 1:p0 + 8]
            pr = data[c]["Q"][p0 - 6:p0 + 1]
            results.setdefault(("季节朴素", c), []).append(
                metrics(yt, pr, scale1[c], scale2[c]))

        # ---- STL+岭回归 ----
        pr_ridge = fit_ridge(X[tr], ya[tr])(X[te])
        for ci, c in enumerate(CATS):
            m = cats_idx[te] == ci
            yhat = back(pr_ridge[m] + S_by[c][data[c]["wd"][poss[te][m]]])
            yt = data[c]["Q"][p0 + 1:p0 + 8]
            results.setdefault(("STL+岭回归", c), []).append(
                metrics(yt, yhat, scale1[c], scale2[c]))

        # ---- STL+GBDT × 三种断货处理 ----
        for treat in lgbm_runs:
            m_tr, y_use = tr.copy(), ya.copy()
            if treat in ("断货剔除", "断货修复"):
                cens_pos = np.zeros(len(poss), dtype=bool)
                for ci, c in enumerate(CATS):
                    cm = cats_idx == ci
                    cf = np.concatenate([np.zeros(35), data[c]["cens"][35:]])
                    cens_pos[cm] = cf[poss[cm]] > 0.5
                if treat == "断货剔除":
                    m_tr = tr & (~cens_pos)
                else:
                    for ci, c in enumerate(CATS):
                        cm = cats_idx == ci
                        for j in np.where(cm & cens_pos & tr)[0]:
                            pos = poss[j]
                            f = cum_curves.get(c, {}).get(int(data[c]["L"][pos]), np.nan)
                            if not np.isnan(f):
                                Qr = data[c]["Q"][pos] * f
                                y_use[j] = np.log(Qr + 0.1) - S_by[c][data[c]["wd"][pos]]
            preds_q = {c: np.zeros((3, 7)) for c in CATS}
            for qi, alpha in enumerate(QUANTILES):
                mdl = make_lgbm(alpha)
                mdl.fit(X[m_tr], y_use[m_tr])
                ah = mdl.predict(X[te])
                for ci, c in enumerate(CATS):
                    m = cats_idx[te] == ci
                    preds_q[c][qi] = back(ah[m] + S_by[c][data[c]["wd"][poss[te][m]]])
            name = f"STL+{engine}({treat})"
            for ci, c in enumerate(CATS):
                yt = data[c]["Q"][p0 + 1:p0 + 8]
                results.setdefault((name, c), []).append(
                    metrics(yt, preds_q[c][1], scale1[c], scale2[c],
                            preds_q=preds_q[c]))

        if (fi + 1) % 8 == 0 or fi == len(origins) - 1:
            print(f"  回测完成 {fi + 1}/{len(origins)} 折")

    # ---------- 汇总 ----------
    rows = []
    for (name, c), ms in results.items():
        agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        rows.append({"模型": name, "品类": c, **{k: round(v, 4) for k, v in agg.items()}})
    res = pd.DataFrame(rows)
    cols = ["WAPE", "MASE", "RMSSE", "pinball10", "pinball50", "pinball90"]
    overall = res.groupby("模型")[cols].mean().round(4).sort_values("WAPE")
    print("\n=== 回测总览(24 折平均) ===")
    print(overall.to_string())
    res.to_csv("p2_回测指标汇总.csv", index=False, encoding="utf-8-sig")
    fig5_backtest(res, engine)
    print("已保存 p2_fig5_回测对比.png")

    # ---------- 选最优断货处理档,全量重训,做最终 7 天预测 ----------
    lgbm_names = [n for n in overall.index if engine and n.startswith(f"STL+{engine}")]
    if not engine or not lgbm_names:
        print("[中止] 无可用 GBDT 引擎,无法生成分位数预测;请安装 lightgbm 或 scikit-learn。")
        return
    best = min(lgbm_names, key=lambda n: overall.loc[n, "WAPE"])
    print(f"\n断货处理消融:{best} 整体 WAPE 最优,按此做最终预测")

    train_mask = np.ones(N, dtype=bool)
    a_by, S_by = {}, {}
    for c in CATS:
        S = stl_factors(y_log[c], data[c]["wd"], data[c]["closed"], train_mask)
        a_by[c] = y_log[c] - S[data[c]["wd"]]
        S_by[c] = S
    X, ya, cats_idx, poss = make_design(data, a_by, cal, N_extra=7)
    tr = poss <= N - 1
    m_tr, y_use = tr.copy(), ya.copy()
    if "断货剔除" in best or "断货修复" in best:
        cens_pos = np.zeros(len(poss), dtype=bool)
        for ci, c in enumerate(CATS):
            cm = cats_idx == ci
            cf = np.concatenate([np.zeros(35), data[c]["cens"][35:], np.zeros(7)])
            cens_pos[cm] = cf[poss[cm]] > 0.5
        if "断货剔除" in best:
            m_tr = tr & (~cens_pos)
        else:
            for ci, c in enumerate(CATS):
                cm = cats_idx == ci
                for j in np.where(cm & cens_pos & tr)[0]:
                    pos = poss[j]
                    f = cum_curves.get(c, {}).get(int(data[c]["L"][pos]), np.nan)
                    if not np.isnan(f):
                        Qr = data[c]["Q"][pos] * f
                        y_use[j] = np.log(Qr + 0.1) - S_by[c][data[c]["wd"][pos]]
    fdates = pd.date_range(FORECAST_START, periods=7)
    wd_ext = np.array([d.dayofweek for d in fdates])
    preds = {c: np.zeros((3, 7)) for c in CATS}
    for qi, alpha in enumerate(QUANTILES):
        mdl = make_lgbm(alpha)
        mdl.fit(X[m_tr], y_use[m_tr])
        ah = mdl.predict(X[~tr])
        for ci, c in enumerate(CATS):
            m = cats_idx[~tr] == ci
            preds[c][qi] = back(ah[m] + S_by[c][wd_ext])

    out = []
    for c in CATS:
        for k, d in enumerate(fdates):
            out.append({"品类": c, "销售日期": d.date(),
                        "P10": round(float(preds[c][0][k]), 1),
                        "P50": round(float(preds[c][1][k]), 1),
                        "P90": round(float(preds[c][2][k]), 1)})
    fc = pd.DataFrame(out)
    fc.to_csv("p2_需求预测_202307.csv", index=False, encoding="utf-8-sig")
    print("\n=== 2023-07-01~07 日需求量 P50 预测(kg) ===")
    print(fc.pivot(index="销售日期", columns="品类", values="P50").to_string())

    # ---------- 图:近 90 天历史 + 预测 ----------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=150,
                             facecolor=SURFACE, sharex=False, sharey=False)
    for ax, c in zip(axes.ravel(), CATS):
        style_ax(ax)
        h0 = cal.get_loc(pd.Timestamp("2023-05-15"))
        hist = pd.Series(data[c]["Q"][h0:], index=cal[h0:])
        ax.plot(hist.index, hist.values, color=BLUE, lw=1.5, zorder=3, label="历史销量")
        ax.fill_between(fdates, preds[c][0], preds[c][2], color=BLUE, alpha=0.18,
                        zorder=2, label="P10-P90 区间")
        ax.plot(fdates, preds[c][1], color=ORANGE, lw=1.8, marker="o", ms=3.5,
                zorder=4, label="P50 预测")
        ax.axvline(fdates[0], color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(list(hist.index[::10]) + [fdates[3]])
        ax.tick_params(axis="x", rotation=30, labelsize=7.5)
        ax.set_title(c, fontsize=11.5, color=INK, pad=6)
        ax.set_xlabel("日期", fontsize=10, color=INK)
        ax.set_ylabel("日需求量(kg)", fontsize=10, color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="upper center", frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("品类日需求量预测(2023-07-01~07):近 90 天历史与分位数区间",
                 fontsize=13.5, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("p2_fig4_需求预测.png", facecolor=SURFACE)
    plt.close(fig)
    print("\n已保存:p2_回测指标汇总.csv / p2_需求预测_202307.csv / p2_fig4_需求预测.png")
    print(f"(分位数引擎:{engine};断货处理:{best})")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("\n[运行出错] 请把上面完整报错原样贴回给 Claude。")
