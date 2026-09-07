# -*- coding: utf-8 -*-
"""
问题二·任务2a:弹性关系三层构建 —— 删失修正 + 2SLS主/DML对照 (v3)
====================================================================
口径(docs/adr/0003, 2026-09-06 用户确认):
  层1 删失修正:销量=min(需求,库存),高需求日被封顶把|E|压向0。
      库存代理 cap_d = max(S_d, 品类日销量居中15日滚动中位);另用
      断货时点信号(Jain-Rudi-Wang 2015:晚市销量份额异常低→当日提前
      售罄)做删失判定交叉验证。主方法=EM(变上限Tobit, 对数空间)修复
      被封顶日的真实需求;对照=直接剔除删失日。
  层2 识别:lnQ* = a − b·P + 年月FE + 星期FE,P 用品类加权批发价对数
      作工具变量 2SLS(主);对照=DML-PLIV(Chernozhukov 2018 部分线性
      IV,5 折交叉拟合,HistGradientBoosting 拟合多余函数)。
      观测弹性定位为真实弹性的下界(MSI 24-139)。
  层3 形式:半对数(最优价 P*=1/b 天然内点);对照=Laspeyres 固定权重
      品类均价 vs 销量加权均价(构成效应偏差量化)。

输入:附件1/2/3
输出:p2o2_弹性参数.csv / p2o2_fig_需求曲线.png(供 p2_定价优化.py 消费)
"""
import sys, os, warnings, traceback
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

plt.rcParams["font.sans-serif"] = ["Noto Serif CJK SC", "Noto Sans CJK SC",
                                   "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A1, A2, A3 = (os.path.join(_ROOT, f) for f in ("附件1.xlsx", "附件2.xlsx", "附件3.xlsx"))
CATS = ["花叶类", "食用菌", "辣椒类", "水生根茎类", "茄类", "花菜类"]
HOUR_LO, HOUR_HI = 9, 22
EVENING_HOUR = 19          # 断货时点信号:18时后(≥19时)销量份额
ROLL_MED_WIN = 15          # 库存代理:居中15日滚动中位
EM_MAXITER, EM_TOL = 200, 1e-8
HAC_LAG = 21                # Newey-West 滞后(日度数据,覆盖月内自相关)
N_FOLDS = 5                # DML 交叉拟合折数
SEED = 20260906

SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


# ---------------- 面板构建(SKU 级一次读取) ----------------
def load_sku_panel():
    """SKU×日期 级别:销量、正价均单价;并聚出品类日度面板。"""
    a1 = pd.read_excel(A1)[["单品编码", "分类名称"]]
    a2 = pd.read_excel(A2)
    a2 = a2[a2["销售类型"] == "销售"].copy()
    a2["销售日期"] = pd.to_datetime(a2["销售日期"])
    a2["hour"] = pd.to_datetime(a2["扫码销售时间"].astype(str), errors="coerce").dt.hour
    a2 = a2.merge(a1, on="单品编码", how="left")
    a2["is_disc"] = a2["是否打折销售"].astype(str).str.strip() == "是"
    a2 = a2[a2["分类名称"].isin(CATS)]
    a2["rev_"] = a2["销量(千克)"] * a2["销售单价(元/千克)"]

    # SKU×日 均单价(全部成交;用于 Laspeyres)与销量
    sku = a2.groupby(["分类名称", "单品编码", "销售日期"]).agg(
        q=("销量(千克)", "sum"), rev=("rev_", "sum")).reset_index()
    sku["p"] = sku["rev"] / sku["q"]

    # 正价 SKU×日 均价(Laspeyres 用"挂牌价"口径更干净)
    nd = a2[~a2["is_disc"]].groupby(["分类名称", "单品编码", "销售日期"]).agg(
        qn=("销量(千克)", "sum"), revn=("rev_", "sum")).reset_index()
    nd["pn"] = nd["revn"] / nd["qn"]

    # 品类×日:总销量、正价加权均价、逐小时销量、批发价
    daily = a2.groupby(["分类名称", "销售日期"]).agg(
        S=("销量(千克)", "sum")).reset_index()
    ndc = a2[~a2["is_disc"]].copy()
    listp = (ndc.groupby(["分类名称", "销售日期"])
                .agg(rp=("rev_", "sum"), q=("销量(千克)", "sum"))
                .assign(p_list_obs=lambda d: d["rp"] / d["q"])["p_list_obs"]
                .reset_index())
    ev = (a2[a2["hour"] >= EVENING_HOUR].groupby(["分类名称", "销售日期"])[
              "销量(千克)"].sum().rename("S_ev").reset_index())
    a3 = pd.read_excel(A3)
    a3["日期"] = pd.to_datetime(a3["日期"])
    ws = a2.groupby(["分类名称", "销售日期", "单品编码"])["销量(千克)"].sum().reset_index()
    ws = ws.merge(a3, left_on=["销售日期", "单品编码"], right_on=["日期", "单品编码"],
                  how="left").dropna(subset=["批发价格(元/千克)"])
    ws["wp"] = ws["销量(千克)"] * ws["批发价格(元/千克)"]
    wavg = (ws.groupby(["分类名称", "销售日期"])
              .agg(wp=("wp", "sum"), q=("销量(千克)", "sum"))
              .assign(c_whl=lambda d: d["wp"] / d["q"])["c_whl"]
              .reset_index())
    panel = (daily.merge(listp, on=["分类名称", "销售日期"])
                  .merge(wavg, on=["分类名称", "销售日期"])
                  .merge(ev, on=["分类名称", "销售日期"], how="left"))
    panel["S_ev"] = panel["S_ev"].fillna(0.0)
    panel["ev_share"] = panel["S_ev"] / panel["S"]
    panel["kappa_obs"] = panel["p_list_obs"] / panel["c_whl"] - 1.0
    return panel, sku, nd


# ---------------- 层1:删失检测 + EM 修复 ----------------
def detect_censoring(p):
    """删失判定(Jain-Rudi-Wang 2015 时点信号):当日≥19时销量份额低于该
    品类分布的 q10 → 判定当日提前售罄(需求撞上库存),记为删失日。
    附带报告 q5 阈值下的占比作稳健性。"""
    p = p.sort_values("销售日期").copy()
    q10 = p["ev_share"].quantile(0.10)
    p["cens_ev"] = p["ev_share"] < q10
    p["censored"] = p["cens_ev"]
    return p, float(q10)


def em_correct(p_cat):
    """变上限 Tobit 的 EM(对数空间):删失日 sales=库存,真实需求 ≥ 观测,
    lnQ 在截断点 ln S_d 之下侧截断,用截尾均值 E[lnD|lnD>lnS] 补。
    X=截距+年月FE+星期FE(不含 P——P 内生,补全后由 2SLS 重估其系数)。
    返回补全后的 lnQ*、截断点、删失标记与 sigma。"""
    d = p_cat.copy()
    X = fe_matrix(d["销售日期"]).to_numpy(float)
    lncut = np.log(d["S"].to_numpy(float))
    y = np.log(d["S"].to_numpy(float))
    cens = d["censored"].to_numpy(bool)
    beta = np.linalg.lstsq(X[~cens], y[~cens], rcond=None)[0]
    sig = max((y[~cens] - X[~cens] @ beta).std(), 0.05)
    yf = y.copy()
    for _ in range(EM_MAXITER):
        z = (lncut[cens] - X[cens] @ beta) / sig
        lam = norm.pdf(z) / np.maximum(1 - norm.cdf(z), 1e-12)
        yf[cens] = X[cens] @ beta + sig * lam
        nb = np.linalg.lstsq(X, yf, rcond=None)[0]
        r = yf - X @ nb
        nsig = np.sqrt((r @ r) / len(r))
        if np.abs(nb - beta).max() < EM_TOL and abs(nsig - sig) < EM_TOL:
            beta, sig = nb, nsig
            break
        beta, sig = nb, nsig
    return yf, lncut, cens, float(sig)


# ---------------- FE 与 2SLS ----------------
def fe_matrix(dates):
    d = pd.DatetimeIndex(dates)
    X = pd.get_dummies(d.to_period("M").astype(str), drop_first=True, dtype=float)
    X = pd.concat([X, pd.get_dummies(d.dayofweek, prefix="wd",
                                     drop_first=True, dtype=float)], axis=1)
    X.insert(0, "const", 1.0)
    return X


def winsor(s, lo=0.005, hi=0.995):
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)


def semilog_2sls(pk):
    """lnQ = a − b·P + FE;P 工具变量=ln(批发价)。返回 b,t(b),F1,n。
    (半对数系数 b>0:价格升 b·dP 使 lnQ 降 b·dP)"""
    Z = fe_matrix(pk["销售日期"]).copy()
    Z["logc"] = np.log(pk["c_whl"].to_numpy())      # 位置赋值,防索引错位
    X = Z.drop(columns=["logc"]).copy()
    X["P"] = pk["p_list_obs"].to_numpy()
    y = np.log(pk["Q_adj"].to_numpy())
    Zm, Xm = Z.to_numpy(float), X.to_numpy(float)
    ok = np.isfinite(Zm).all(1) & np.isfinite(Xm).all(1) & np.isfinite(y)
    Zm, Xm, yv = Zm[ok], Xm[ok], y[ok]
    b1, *_ = np.linalg.lstsq(Zm, Xm[:, -1], rcond=None)
    r1 = Xm[:, -1] - Zm @ b1
    s1 = np.sqrt(r1.var() * len(r1) / max(len(r1) - Zm.shape[1], 1))
    t_logc = b1[-1] / (s1 * np.sqrt(np.linalg.pinv(Zm.T @ Zm)[-1, -1]))
    PZ = Zm @ np.linalg.pinv(Zm.T @ Zm) @ Zm.T
    XtPZ_X = Xm.T @ PZ @ Xm
    # 一阶段拟合过强时 Schur 补病态,lstsq 比 solve 稳健
    bvec = np.linalg.lstsq(XtPZ_X, Xm.T @ PZ @ yv, rcond=None)[0]
    cond_x = np.linalg.cond(XtPZ_X)
    resid = yv - Xm @ bvec
    sig2 = resid @ resid / max(len(yv) - Xm.shape[1], 1)
    # Newey-West HAC 标准误:日度销量序列自相关,iid-SE 会低估σ、夸大数据
    # 对先验的权重。Bartlett 核,滞后 L 天。
    G = XtPZ_X
    XP = PZ @ Xm                          # n×k,第一阶段拟合值
    su = XP * resid[:, None]              # 得分 x̂_t·u_t
    n, k = su.shape
    L = min(HAC_LAG, n // 4)
    Om = (su.T @ su).copy()
    for h in range(1, L + 1):
        w = 1.0 - h / (L + 1)
        Gh = su[h:].T @ su[:-h]
        Om += w * (Gh + Gh.T)
    V = sig2 * np.linalg.pinv(G) @ Om @ np.linalg.pinv(G)
    return float(-bvec[-1]), float(-bvec[-1] / np.sqrt(abs(V[-1, -1]))), \
        float(t_logc ** 2), int(ok.sum()), float(cond_x), \
        float(np.sqrt(abs(V[-1, -1])))     # b=−系数(价格前负号) + HAC se


def dml_pliv(pk, seed=SEED):
    """DML-PLIV: lnQ = θ·P + g(X)+U, P = m(Z)+V, Z=ln(批发价)。
    交叉拟合(HistGradientBoosting),θ = Σ Đ·Ỹ / Σ Đ·P̃。
    返回 θ=−θ_PLIV(转成与 b 同号:b=−θ)。"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(seed)
    X = fe_matrix(pk["销售日期"])
    X["doy"] = pd.DatetimeIndex(pk["销售日期"]).dayofyear
    X["t"] = (pd.DatetimeIndex(pk["销售日期"]) - pd.Timestamp("2020-07-01")).days
    Xm = X.to_numpy(float)
    y = np.log(pk["Q_adj"].to_numpy())
    D = pk["p_list_obs"].to_numpy(float)
    Z = np.log(pk["c_whl"].to_numpy())
    ok = np.isfinite(Xm).all(1) & np.isfinite(y) & np.isfinite(D) & np.isfinite(Z)
    Xm, y, D, Z = Xm[ok], y[ok], D[ok], Z[ok]
    n = len(y)
    idx = rng.permutation(n)
    folds = np.array_split(idx, N_FOLDS)
    Dt = np.zeros(n)
    def mk():
        return HistGradientBoostingRegressor(max_iter=200, max_depth=3,
                                             learning_rate=0.06, random_state=SEED)
    Yt = np.zeros(n)
    Zt = np.zeros(n)
    for f in folds:
        mask = np.ones(n, bool); mask[f] = False
        Yt[f] = y[f] - mk().fit(Xm[mask], y[mask]).predict(Xm[f])
        Dt[f] = D[f] - mk().fit(Xm[mask], D[mask]).predict(Xm[f])
        Zt[f] = Z[f] - mk().fit(Xm[mask], Z[mask]).predict(Xm[f])
    theta = float((Dt @ Yt) / (Dt @ Zt))
    return -theta, n


def laspeyres_price(cat, sku, nd, panel):
    """固定权重(全期 SKU 正价销量份额)品类价格指数,缺失前向填充≤7天。"""
    w = (nd[nd["分类名称"] == cat].groupby("单品编码")["qn"].sum())
    w = w / w.sum()
    px = nd[nd["分类名称"] == cat].pivot_table(
        index="销售日期", columns="单品编码", values="pn")
    px = px.reindex(columns=w.index)
    idx_obs = px.notna().sum(axis=1)
    px = px.ffill(limit=7)
    keep = idx_obs >= max(3, int(0.3 * len(w)))
    px = px[keep]
    pl = (px * w).sum(axis=1, min_count=3)
    out = pl.rename("p_las").reset_index()
    return out


# ---------------- 主流程 ----------------
def main():
    print("加载数据 ...")
    panel, sku, nd = load_sku_panel()
    rows = []
    curves = {}
    for cat in CATS:
        p = panel[panel["分类名称"] == cat].sort_values("销售日期").copy()
        lo, hi = p["kappa_obs"].quantile([0.01, 0.99])
        p = p[p["kappa_obs"].between(lo, hi)].reset_index(drop=True)
        for col in ["p_list_obs", "c_whl", "S"]:
            p[col] = winsor(p[col])
        p, q10 = detect_censoring(p)
        yf, lncap, cens, sig = em_correct(p)
        p["Q_adj"] = np.exp(yf)                       # EM 补全后的需求

        # 主:半对数 2SLS(EM 补全);对照1:剔除删失日;对照2:DML-PLIV;
        # 对照3:Laspeyres 价格口径;对照4:不修正(对照 v2)
        b, t_b, F1, n, cond_x, se_hac = semilog_2sls(p)
        b_rm, t_rm, _, n_rm, _, _ = semilog_2sls(p[~p["censored"]].copy())
        b_dml, n_dml = dml_pliv(p)
        pl = laspeyres_price(cat, sku, nd, panel)
        p2 = p.merge(pl, on="销售日期", how="left").dropna(subset=["p_las"])
        p2 = p2[p2["p_las"] > 0].copy()
        p2["p_las"] = winsor(p2["p_las"])
        p2 = p2.drop(columns=["p_list_obs"]).rename(columns={"p_las": "p_list_obs"})
        b_las, t_las, _, n_las, _, _ = semilog_2sls(p2)
        b_raw, t_raw, _, _, _, _ = semilog_2sls(
            p.assign(Q_adj=p["S"]))                   # 不修正对照

        pbar = float(np.average(p["p_list_obs"], weights=p["S"]))
        cl, ch = float(p["p_list_obs"].quantile(0.005)), float(p["p_list_obs"].quantile(0.995))
        # 文献先验收缩(ADR-0003 口径C, empirical Bayes):
        # 品类级蔬菜弹性文献区间 −0.4~−1.0 → 先验 b0=0.7/P̄, τ0=0.3/P̄;
        # 后验 b = (b̂/σ̂² + b0/τ0²)/(1/σ̂² + 1/τ0²),权重=方差倒数(HAC se)。
        sigma_b = se_hac
        b0, tau0 = 0.7 / pbar, 0.3 / pbar
        w_data = 1 / sigma_b**2 / (1 / sigma_b**2 + 1 / tau0**2)
        b_post = (b / sigma_b**2 + b0 / tau0**2) / (1 / sigma_b**2 + 1 / tau0**2)
        rows.append({
            "品类": cat, "删失日占比": round(p["censored"].mean(), 3),
            "其中时点信号": round(p["cens_ev"].mean(), 3),
            "b(不修正)": round(b_raw, 4),
            "b(EM修正,2SLS主)": round(b, 4), "t(b)": round(t_b, 1),
            "se(b)HAC": round(sigma_b, 4),
            "一阶段F": round(F1, 0),
            "b(剔除删失日)": round(b_rm, 4),
            "b(DML-PLIV)": round(b_dml, 4),
            "b(Laspeyres口径)": round(b_las, 4),
            "先验b0": round(b0, 4), "先验τ0": round(tau0, 4),
            "数据权重": round(w_data, 3),
            "b(后验收缩)": round(b_post, 4),
            "隐含弹性@P̄(后验)": round(-b_post * pbar, 2),
            "内点价P*=1/b(后验)": round(1 / b_post, 2) if b_post > 0 else np.nan,
            "P̄": round(pbar, 2), "历史价域": f"[{cl:.2f},{ch:.2f}]",
            "样本日数": n})
        curves[cat] = dict(pbar=pbar, blo=cl, bhi=ch, b=b, b_raw=b_raw,
                           b_dml=b_dml, b_post=b_post)
        print(f"  {cat}: 删失率{p['censored'].mean():.1%} "
              f"b: 不修正{b_raw:.3f} → EM+2SLS {b:.3f} (HAC se={se_hac:.3f}) "
              f"| DML {b_dml:.3f} | 后验 b={b_post:.3f} (数据权重{w_data:.0%}, "
              f"先验b0={b0:.3f}) 隐含弹性@P̄={-b_post*pbar:.2f} "
              f"P*=1/b={1/b_post:.1f}")

    est = pd.DataFrame(rows)
    est.to_csv("p2o2_弹性参数.csv", index=False, encoding="utf-8-sig")
    print("\n=== 弹性参数汇总(p2o2_弹性参数.csv) ===")
    print(est.to_string(index=False))

    # ---------------- 图:需求曲线(历史域内) ----------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), dpi=150, facecolor=SURFACE)
    P = np.linspace(0.8, 1.0, 200)   # 占位,下面逐图重设
    for ax, cat in zip(axes.ravel(), CATS):
        style_ax(ax)
        c = curves[cat]
        pgrid = np.linspace(c["blo"], c["bhi"], 200)
        q_rel = np.exp(-c["b"] * (pgrid - c["pbar"]))
        q_rel_raw = np.exp(-c["b_raw"] * (pgrid - c["pbar"]))
        q_rel_post = np.exp(-c["b_post"] * (pgrid - c["pbar"]))
        ax.plot(pgrid, q_rel_raw, color=MUTED, lw=1.4, ls=(0, (4, 3)),
                label="修正前(对照)")
        ax.plot(pgrid, q_rel, color="#2a78d6", lw=1.8, label="EM+2SLS(数据)")
        ax.plot(pgrid, q_rel_post, color="#c0392b", lw=2.2,
                label="先验收缩后(主)")
        ax.axvline(c["pbar"], color=AXIS, lw=1, ls=":")
        ax.set_title(cat, fontsize=11.5, color=INK, pad=6)
        ax.set_xlabel("零售价(元/kg)", fontsize=10, color=INK)
        ax.set_ylabel("相对需求(Q/Q̄)", fontsize=10, color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("品类需求曲线:数据估计 → 文献先验收缩(主;历史价域内绘制)",
                 fontsize=13, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("p2o2_fig_需求曲线.png", facecolor=SURFACE)
    plt.close(fig)
    print("\n已输出:p2o2_弹性参数.csv / p2o2_fig_需求曲线.png")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
