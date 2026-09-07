# -*- coding: utf-8 -*-
"""
问题二·任务2b:品类定价(加成率)优化模型 (v3, 半对数响应)
====================================================================
口径(docs/adr/0003, 2026-09-06 用户确认):
  需求响应 = 半对数 lnQ = a − b·(P − P̄),b 取 p2o2_弹性参数.csv 的
  "文献先验收缩后"估计(EM删失修正 + 批发价IV 2SLS + HAC标准误 +
  品类蔬菜文献先验 empirical Bayes 收缩;数据与文献先验冲突 6–9σ,
  数据权重 90–98%,详见 p2/p2_弹性估计.py)。
  优化信任域 = 历史零售价域 [P_lo, P_hi](响应模型只在域内可信)。
  其余:θ=报废量损,Q=P50/(1−θ),可售上限=N50;P10/P90 → 需求对数正态
  抽样;逐窗清算 s_t=min(D·w_t,剩余);收档剩余零残值;19时后清仓档。
  收益 = E_D[Σ P·ρ_t·s_t] − ĉ·Q, P = ĉ·(1+κ)。

求解:一维稠密网格+黄金分割(保全局),6品类×7天独立;MILP 不必要的
     理由见论文求解章节(大M线性化是为用求解器而加假设)。

输入:附件1/2/3、p2_需求预测_202307.csv、p2_损耗率参数.csv、
     p2o2_弹性参数.csv(由 p2_弹性估计.py 生成)
输出:p2o3_定价结果.csv / p2o3_弹性灵敏度.csv
     p2o3_fig1_收益曲线.png / p2o3_fig2_销售轨迹.png
"""
import sys, os, warnings, traceback
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

plt.rcParams["font.sans-serif"] = ["Noto Serif CJK SC", "Noto Sans CJK SC",
                                   "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A1, A2, A3 = (os.path.join(_ROOT, f) for f in ("附件1.xlsx", "附件2.xlsx", "附件3.xlsx"))
FCST, THETA_CSV = "数据/p2_需求预测_202307.csv", "数据/p2_损耗率参数.csv"
ELAS_CSV = "p2o2_弹性参数.csv"
CATS = ["花叶类", "食用菌", "辣椒类", "水生根茎类", "茄类", "花菜类"]
HOUR_LO, HOUR_HI = 9, 22
CLEAR_HOUR = 19
SEASON_MONTHS = set(range(4, 11))
COST_WEEKS = 8
N_GRID = 201
N_DRAWS, SEED = 300, 20260906
Z90 = 1.2816

SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
ORANGE, RED = "#eb6834", "#c0392b"


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


# ---------------- 基础面板(成本锚点 + 窗口结构) ----------------
def load_panel():
    a1 = pd.read_excel(A1)[["单品编码", "分类名称"]]
    a2 = pd.read_excel(A2)
    a2 = a2[a2["销售类型"] == "销售"].copy()
    a2["销售日期"] = pd.to_datetime(a2["销售日期"])
    a2["hour"] = pd.to_datetime(a2["扫码销售时间"].astype(str), errors="coerce").dt.hour
    a2 = a2.merge(a1, on="单品编码", how="left")
    a2["is_disc"] = a2["是否打折销售"].astype(str).str.strip() == "是"
    a2 = a2[a2["分类名称"].isin(CATS)]
    a2["rev_"] = a2["销量(千克)"] * a2["销售单价(元/千克)"]
    nd = a2[~a2["is_disc"]].copy()
    nd["rp"] = nd["销量(千克)"] * nd["销售单价(元/千克)"]
    listp = (nd.groupby(["分类名称", "销售日期"])
               .agg(rp=("rp", "sum"), q=("销量(千克)", "sum"))
               .assign(p_list_obs=lambda d: d["rp"] / d["q"])["p_list_obs"]
               .reset_index())
    daily = a2.groupby(["分类名称", "销售日期"]).agg(
        S=("销量(千克)", "sum")).reset_index()
    a2w = a2[a2["hour"].between(HOUR_LO, HOUR_HI)].copy()
    hv = a2w.groupby(["分类名称", "销售日期", "hour"]).agg(
        q=("销量(千克)", "sum"), rev=("rev_", "sum")).reset_index()
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
                  .merge(wavg, on=["分类名称", "销售日期"]))
    return panel, hv


def window_structure(hv, cat):
    """旺季月份:窗口份额 w̄_t 与窗口实现价比 ρ_t(19时后=清仓档)。"""
    hours = list(range(HOUR_LO, HOUR_HI + 1))
    hh = hv[(hv["分类名称"] == cat) &
            (hv["销售日期"].dt.month.isin(SEASON_MONTHS))]
    wbar = (hh.groupby("hour")["q"].sum() / hh["q"].sum()).reindex(hours).fillna(0)
    h2 = hh.copy()
    ref = h2.groupby("销售日期")["rev"].transform("sum") / \
        h2.groupby("销售日期")["q"].transform("sum")
    h2["ref"] = ref            # 全渠道日均价的窗口占比参照(仅作比值分母量级)
    h2["ratio"] = (h2["rev"] / h2["q"]) / h2["ref"]
    h2 = h2.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio", "q"])
    h2 = h2[h2["ratio"].between(0.3, 1.2)]
    h2["rq"] = h2["ratio"] * h2["q"]
    g = h2.groupby("hour").agg(rq=("rq", "sum"), q=("q", "sum"))
    rho = (g["rq"] / g["q"]).reindex(hours)
    rho = rho.where(rho.notna(), 1.0)
    rho = rho / rho[rho.index < CLEAR_HOUR].mean()   # 以白天正价均值为 1
    rho = rho.clip(0.3, 1.0)
    rho[rho.index >= CLEAR_HOUR] = rho[rho.index >= CLEAR_HOUR].clip(upper=0.95)
    return dict(wbar=wbar.to_dict(), rho=rho.to_dict(), hours=hours)


# ---------------- 日模拟(对需求抽样取期望) ----------------
def simulate(par, N50, sigma_log, kappa, m):
    """半对数需求:Q(P)=N50·exp(−b·(P−P̄)),P=ĉ(1+κ)。"""
    rho = par["rho"]
    P = par["c_bar"] * (1 + kappa)
    N = N50 * np.exp(-par["b"] * (P - par["pbar"]))
    w = np.array([par["wbar"][t] for t in par["hours"]])
    cap = N50
    rem = np.full(len(m), float(cap))
    rev_ratio = np.zeros(len(m))
    sold = np.zeros(len(m))
    clear = np.zeros(len(m))
    for t, wt in zip(par["hours"], w):
        s = np.minimum(N * wt * m, rem)
        rem -= s
        rev_ratio += rho[t] * s
        sold += s
        if t >= CLEAR_HOUR:
            clear += s
    revenue = P * rev_ratio
    cost = par["c_bar"] * N50 / (1 - par["theta"])
    return dict(profit=revenue.mean() - cost, revenue=revenue.mean(),
                cost=cost, sold=sold.mean(), fill=sold.mean() / cap,
                clear_share=clear.mean() / max(sold.mean(), 1e-9))


def solve_kappa(par, N50, m):
    """信任域=历史零售价域 → κ域 [P_lo/ĉ−1, P_hi/ĉ−1]。网格+黄金分割。"""
    def prof(k):
        return simulate(par, N50, None, k, m)["profit"]

    klo = par["p_lo"] / par["c_bar"] - 1.0
    khi = par["p_hi"] / par["c_bar"] - 1.0
    ks = np.linspace(klo, khi, N_GRID)
    vals = np.array([prof(k) for k in ks])
    i = int(np.argmax(vals))
    a, b = ks[max(i - 1, 0)], ks[min(i + 1, len(ks) - 1)]
    gr = (np.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc_, fd_ = prof(c), prof(d)
    for _ in range(50):
        if fc_ > fd_:
            b, d, fd_ = d, c, fc_
            c = b - gr * (b - a)
            fc_ = prof(c)
        else:
            a, c, fc_ = c, d, fd_
            d = a + gr * (b - a)
            fd_ = prof(d)
    kstar = (a + b) / 2
    edge = 0.5 * (khi - klo) / N_GRID
    at_bound = min(abs(kstar - klo), abs(kstar - khi)) < edge
    pstar_theory = 1.0 / par["b"]          # 半对数无约束内点价
    return float(kstar), pd.DataFrame({"kappa": ks, "profit": vals}), \
        bool(at_bound), float(pstar_theory)


# ---------------- 主流程 ----------------
def main():
    print("加载数据 ...")
    panel, hv = load_panel()
    theta_t = pd.read_csv(THETA_CSV).set_index("品类")["θ_品类"]
    elas = pd.read_csv(ELAS_CSV).set_index("品类")
    fcst = pd.read_csv(FCST)
    fcst["销售日期"] = pd.to_datetime(fcst["销售日期"])

    rng = np.random.default_rng(SEED)
    pars = {}
    print("\n锚点(成本 ĉ 近8周;弹性=文献先验收缩后):")
    for cat in CATS:
        p = panel[panel["分类名称"] == cat]
        t_end = p["销售日期"].max()
        recent = p[p["销售日期"] >= t_end - pd.Timedelta(weeks=COST_WEEKS)]
        cbar = float(np.average(recent["c_whl"], weights=recent["S"]))
        erow = elas.loc[cat]
        plo, phi = (float(x) for x in
                    erow["历史价域"].strip("[]").split(","))
        par = dict(b=float(erow["b(后验收缩)"]),
                   b_2sls=float(erow["b(EM修正,2SLS主)"]),
                   b_dml=float(erow["b(DML-PLIV)"]),
                   pbar=float(erow["P̄"]), p_lo=plo, p_hi=phi,
                   c_bar=cbar, theta=float(theta_t[cat]),
                   **window_structure(hv, cat))
        pars[cat] = par
        print(f"  {cat}: b={par['b']:.4f}, P̄={par['pbar']:.2f}, "
              f"信任域P∈[{plo:.2f},{phi:.2f}], ĉ={cbar:.2f}, "
              f"κ域[{plo/cbar-1:.2f},{phi/cbar-1:.2f}]")

    print("\n求解 42 个一维问题 ...")
    results, curves = [], {}
    for cat in CATS:
        par = pars[cat]
        f = fcst[fcst["品类"] == cat].copy()
        f["sigma_log"] = (np.log(f["P90"]) - np.log(f["P10"])) / (2 * Z90)
        f["sigma_log"] = f["sigma_log"].bfill().ffill()
        for _, row in f.iterrows():
            N50 = float(row["P50"])
            m = np.exp(rng.normal(0.0, float(row["sigma_log"]), N_DRAWS))
            m = m / m.mean()
            kstar, curve, at_bound, p_theory = solve_kappa(par, N50, m)
            met = simulate(par, N50, None, kstar, m)
            kbar = par["pbar"] / par["c_bar"] - 1.0      # 维持现价 κ̄
            base = simulate(par, N50, None, kbar, m)
            results.append({
                "品类": cat, "日期": row["销售日期"].date(), "P50(kg)": N50,
                "补货Q(kg)": round(N50 / (1 - par["theta"]), 1),
                "κ*": round(kstar, 3), "κ*触界": "是" if at_bound else "否",
                "最优挂牌价(元/kg)": round(par["c_bar"] * (1 + kstar), 2),
                "无约束内点价P*=1/b": round(p_theory, 1),
                "期望销量(kg)": round(met["sold"], 1),
                "期望售罄率": round(met["fill"], 3),
                "清仓量占比": round(met["clear_share"], 3),
                "期望收益(元)": round(met["profit"], 0),
                "收益@κ̄(元)": round(base["profit"], 0),
                "增益(元)": round(met["profit"] - base["profit"], 0)})
            curves[(cat, str(row["销售日期"].date()))] = curve
    res = pd.DataFrame(results)
    res.to_csv("p2o3_定价结果.csv", index=False, encoding="utf-8-sig")
    print("\n=== 定价结果(全部 42 行见 p2o3_定价结果.csv) ===")
    print(res.drop(columns=["无约束内点价P*=1/b"]).to_string(index=False))

    print("\n=== 周汇总 ===")
    wk = res.groupby("品类").agg(周补货=("补货Q(kg)", "sum"),
                                周期望收益=("期望收益(元)", "sum"),
                                周增益=("增益(元)", "sum"),
                                平均κ=("κ*", "mean")).round(1)
    print(wk.to_string())
    print(f"\n全渠道周期望收益 {res['期望收益(元)'].sum():.0f} 元,"
          f"相对维持现价增益 {res['增益(元)'].sum():.0f} 元;"
          f"角点解 {int((res['κ*触界'] == '是').sum())}/42")

    # ---------------- 灵敏度:数据 / 后验 / DML 三口径 ----------------
    print("\n=== 弹性口径灵敏度(7/1):最优挂牌价与解形态 ===")
    sens = []
    r0map = {c: fcst[fcst["品类"] == c].iloc[0] for c in CATS}
    for cat in CATS:
        par0 = pars[cat]
        r0 = r0map[cat]
        N50 = float(r0["P50"])
        sl = float((np.log(r0["P90"]) - np.log(r0["P10"])) / (2 * Z90))
        m = np.exp(rng.normal(0.0, sl, N_DRAWS)); m = m / m.mean()
        row = {"品类": cat}
        for label, bkey in [("2SLS数据", "b_2sls"), ("后验收缩", "b"),
                            ("DML", "b_dml")]:
            par = dict(par0); par["b"] = par0[bkey]
            k, _, atb, pt = solve_kappa(par, N50, m)
            row[f"P*({label})"] = round(par["c_bar"] * (1 + k), 2)
            row[f"触界({label})"] = "是" if atb else "否"
            row[f"内点价({label})"] = round(1 / par0[bkey], 1)
        sens.append(row)
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv("p2o3_弹性灵敏度.csv", index=False, encoding="utf-8-sig")
    print(sens_df.to_string(index=False))

    # ---------------- 图 1:收益-κ 曲线 ----------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), dpi=150, facecolor=SURFACE)
    for ax, cat in zip(axes.ravel(), CATS):
        style_ax(ax)
        for d in range(1, 8):
            key = (cat, f"2023-07-0{d}")
            if key in curves:
                cv = curves[key]
                ax.plot(cv["kappa"], cv["profit"], lw=1.6,
                        label=f"7/{d}" if cat == CATS[0] else None)
        kbar = pars[cat]["pbar"] / pars[cat]["c_bar"] - 1.0
        ax.axvline(kbar, color=AXIS, ls=(0, (4, 3)), lw=1.2)
        ax.set_title(cat, fontsize=11.5, color=INK, pad=6)
        ax.set_xlabel("加成率 κ", fontsize=10, color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("期望日收益(元)", fontsize=10.5, color=INK)
    fig.suptitle("期望日收益关于加成率 κ 的曲线(半对数响应;竖虚线=维持现价 κ̄;横轴域=历史价信任域)",
                 fontsize=12.5, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("p2o3_fig1_收益曲线.png", facecolor=SURFACE)
    plt.close(fig)

    # ---------------- 图 2:销售轨迹对比 ----------------
    cat = "花叶类"
    r0 = r0map[cat]
    N50 = float(r0["P50"])
    sl = float((np.log(r0["P90"]) - np.log(r0["P10"])) / (2 * Z90))
    m = np.exp(rng.normal(0.0, sl, N_DRAWS)); m = m / m.mean()
    par = pars[cat]
    kstar, _, _, _ = solve_kappa(par, N50, m)
    kbar = par["pbar"] / par["c_bar"] - 1.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, facecolor=SURFACE)
    for ax, k, tag in [(axes[0], kstar, f"κ*={kstar:.2f}"),
                       (axes[1], kbar, f"κ̄={kbar:.2f}")]:
        style_ax(ax)
        P = par["c_bar"] * (1 + k)
        N = N50 * np.exp(-par["b"] * (P - par["pbar"]))
        w = np.array([par["wbar"][t] for t in par["hours"]])
        mean_traj = [float(np.minimum(N * w[i] * m, N50).mean())
                     for i in range(len(par["hours"]))]
        colors = [ORANGE if t < CLEAR_HOUR else RED for t in par["hours"]]
        ax.bar(par["hours"], mean_traj, color=colors, width=0.72, zorder=3)
        ax.axvline(CLEAR_HOUR - 0.5, color=AXIS, ls=(0, (4, 3)), lw=1.2)
        met = simulate(par, N50, None, k, m)
        ax.set_title(f"{cat} 2023-07-01  {tag}  期望售罄率={met['fill']:.0%},"
                     f"清仓占比={met['clear_share']:.0%}", fontsize=10.5, color=INK)
        ax.set_xlabel("小时", fontsize=10, color=INK)
        ax.set_ylabel("期望销量(kg)", fontsize=10.5, color=INK)
    fig.suptitle("销售轨迹对比:橙=正价/品相时段,红=清仓时段(19时后)",
                 fontsize=12.5, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig("p2o3_fig2_销售轨迹.png", facecolor=SURFACE)
    plt.close(fig)

    print("\n已输出:p2o3_定价结果.csv / p2o3_弹性灵敏度.csv / "
          "p2o3_fig1_收益曲线.png / p2o3_fig2_销售轨迹.png")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
