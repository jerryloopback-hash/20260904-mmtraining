# -*- coding: utf-8 -*-
"""08 严谨版完整流程：问题1后半部分（统计规律 + 风化程度 + 风化前预测）

相对 v1（分析脚本.py）的方法升级：
  1) 所有分析基于行归一化（闭合到100%）后的成分，消除总量漂移；
  2) 组间比较统一用 Mann-Whitney U + Cliff's delta 效应量 + Benjamini-Hochberg FDR
     多重检验校正，并以 CLR 对数比变换做成分数据敏感性分析；
  3) 风化程度指数新增稳健性检查（PC1 对照、留一排序稳定性、噪声地板对比）；
  4) 剂量-反应改用 Spearman(指数, 成分)+FDR 的正式趋势检验（三等分表仅作描述）；
  5) 预测主模型由 OLS 升级为 Theil-Sen 稳健回归，新增 B=1000 bootstrap 联合预测区间
     （先逐次闭合归一化再取分位数，区间与点预测自洽）；
  6) 验证：留一文物配对 + 预测值落入风化前经验区间比例 + 检出/还原排序一致性
     + 两方案间一致性。
输出目录：问题1后半部分/严谨版/（不覆盖 v1 旧结果，v1 可用 分析脚本.py 一键复现）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = r"D:\22年c问"
OUT = os.path.join(BASE, "问题1后半部分", "严谨版")
FIG = os.path.join(OUT, "图")
os.makedirs(FIG, exist_ok=True)

COMPS = ["二氧化硅(SiO2)", "氧化钠(Na2O)", "氧化钾(K2O)", "氧化钙(CaO)", "氧化镁(MgO)",
         "氧化铝(Al2O3)", "氧化铁(Fe2O3)", "氧化铜(CuO)", "氧化铅(PbO)", "氧化钡(BaO)",
         "五氧化二磷(P2O5)", "氧化锶(SrO)", "氧化锡(SnO2)", "二氧化硫(SO2)"]
SHORT = {c: c.split("(")[1].rstrip(")") for c in COMPS}
TYPES = ["高钾", "铅钡"]
PRE_STATES = ["风化前-无风化文物", "风化前-未风化点"]
RNG = np.random.default_rng(2022)


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR 校正，返回 q 值数组"""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    q = p[order] * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(q, 0, 1)
    return out


def cliffs_delta(x1, x2):
    """Cliff's delta（风化后-风化前方向）：δ>0 表示风化后普遍更高。
    |δ|<0.147 可忽略，<0.33 小，<0.474 中，否则大。"""
    n1, n2 = len(x1), len(x2)
    u = stats.mannwhitneyu(x1, x2, alternative="two-sided").statistic
    return 1 - 2 * u / (n1 * n2)


def delta_level(d):
    a = abs(d)
    return "可忽略" if a < 0.147 else ("小" if a < 0.33 else ("中" if a < 0.474 else "大"))


# ============ 1. 读取、清洗与闭合归一化 ============
f1 = pd.read_excel(os.path.join(BASE, "附件.xlsx"), sheet_name="表单1")
f2 = pd.read_excel(os.path.join(BASE, "附件.xlsx"), sheet_name="表单2")
f2["文物编号"] = f2["文物采样点"].astype(str).str.extract(r"^(\d+)").astype(float)
f2["标注"] = f2["文物采样点"].astype(str).str.replace(r"^\d+", "", regex=True)
f2 = f2.merge(f1, on="文物编号", how="left")
f2[COMPS] = f2[COMPS].fillna(0.0)
f2["累加和"] = f2[COMPS].sum(axis=1)
data = f2[(f2["累加和"] >= 85) & (f2["累加和"] <= 105)].copy()
print(f"[清洗] 有效数据 {len(data)}/{len(f2)}（剔除 {len(f2)-len(data)} 条累加和越界记录）")

norm = data[COMPS].div(data[COMPS].sum(axis=1), axis=0) * 100
for c in COMPS:
    data["n_" + SHORT[c]] = norm[c].values
# CLR 变换（敏感性分析用）：零以 0.005% 替代后取对数比
tmp = norm.mask(norm <= 0, 0.005)
data_clr = np.log(tmp.div(tmp.mean(axis=1), axis=0))
for c in COMPS:
    data["clr_" + SHORT[c]] = data_clr[c].values


def point_state(row):
    if row["表面风化"] == "无风化":
        return "风化前-无风化文物"
    if "未风化点" in row["标注"]:
        return "风化前-未风化点"
    if "严重风化点" in row["标注"]:
        return "严重风化点"
    return "一般风化点"


data["点状态"] = data.apply(point_state, axis=1)
data.to_excel(os.path.join(OUT, "01_清洗合并后数据.xlsx"), index=False)
print("[状态划分]", data.groupby(["类型", "点状态"]).size().to_dict())

# ============ 2. 等价性检验：未风化点 vs 无风化文物点（合并入"风化前"的依据） ============
chk = []
for c in COMPS:
    a = data[data["点状态"] == "风化前-无风化文物"]["n_" + SHORT[c]].values
    b = data[data["点状态"] == "风化前-未风化点"]["n_" + SHORT[c]].values
    if len(b) >= 3:
        chk.append({"成分": SHORT[c], "无风化文物中位数": round(np.median(a), 3),
                    "未风化点中位数": round(np.median(b), 3),
                    "MW p值": stats.mannwhitneyu(a, b).pvalue})
eq = pd.DataFrame(chk)
eq["FDR q值"] = bh_fdr(eq["MW p值"]).round(4)
eq["MW p值"] = eq["MW p值"].round(4)
eq.to_excel(os.path.join(OUT, "02b_未风化点等价性检验.xlsx"), index=False)
print(f"[等价性] FDR 后显著成分数：{(eq['FDR q值'] < 0.05).sum()}/14 → 可合并为风化前组")

# ============ 3. 统计规律：MW U + Cliff's delta + FDR（主分析），CLR 敏感性 ============
rows = []
for t in TYPES:
    d = data[data["类型"] == t]
    pre = d[d["点状态"].isin(PRE_STATES)]
    wth = d[d["点状态"].isin(["一般风化点", "严重风化点"])]
    for c in COMPS:
        s = SHORT[c]
        x1, x2 = pre["n_" + s].values, wth["n_" + s].values
        if len(x2) < 3:
            continue
        p = stats.mannwhitneyu(x1, x2, alternative="two-sided").pvalue
        delta = cliffs_delta(x1, x2)
        # CLR 敏感性
        p_clr = stats.mannwhitneyu(pre["clr_" + s], wth["clr_" + s],
                                   alternative="two-sided").pvalue
        rows.append({"类型": t, "成分": s,
                     "风化前均值": round(x1.mean(), 2), "风化后均值": round(x2.mean(), 2),
                     "风化前中位数": round(np.median(x1), 2), "风化后中位数": round(np.median(x2), 2),
                     "相对变化%": round((x2.mean() - x1.mean()) / x1.mean() * 100, 1) if x1.mean() > 0 else np.nan,
                     "Cliff's δ": round(delta, 3), "效应量等级": delta_level(delta),
                     "MW p值": round(p, 5), "方向": "上升" if delta > 0 else "下降",
                     "CLR敏感性 p值": round(p_clr, 5)})
rules = pd.DataFrame(rows)
for t in TYPES:
    m = rules["类型"] == t
    rules.loc[m, "FDR q值"] = bh_fdr(rules.loc[m, "MW p值"]).round(5)
    rules.loc[m, "CLR敏感性 q值"] = bh_fdr(rules.loc[m, "CLR敏感性 p值"]).round(5)
rules["显著(q<0.05)"] = np.where(rules["FDR q值"] < 0.05, "是", "否")
rules["CLR下结论一致"] = (rules["FDR q值"] < 0.05) == (rules["CLR敏感性 q值"] < 0.05)
rules.to_excel(os.path.join(OUT, "02_统计规律与检验.xlsx"), index=False)
sig = rules[rules["显著(q<0.05)"] == "是"]
print("\n[统计规律] FDR 后显著成分（类型, 成分, 相对变化%, δ, q, CLR一致）:")
print(sig[["类型", "成分", "相对变化%", "Cliff's δ", "FDR q值", "CLR下结论一致"]].to_string(index=False))
print("CLR 敏感性结论不一致条目数:", int((~rules["CLR下结论一致"]).sum()))

# ---- 箱线图 ----
def boxplot_type(t, comps_pick, fname):
    d = data[data["类型"] == t]
    pre = d[d["点状态"].isin(PRE_STATES)]
    wth = d[d["点状态"].isin(["一般风化点", "严重风化点"])]
    n = len(comps_pick)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(14, 7))
    axes = axes.flatten()
    for i, s in enumerate(comps_pick):
        ax = axes[i]
        col = s if s in d.columns else "n_" + s
        bp = ax.boxplot([pre[col].values, wth[col].values], tick_labels=["风化前", "风化后"],
                        widths=0.55, patch_artist=True, medianprops=dict(color="black"))
        for patch, c0 in zip(bp["boxes"], ["#B5D4F4", "#F5C4B3"]):
            patch.set_facecolor(c0)
        for j, grp in enumerate([pre[col].values, wth[col].values], start=1):
            ax.scatter(np.random.default_rng(7).normal(j, 0.06, len(grp)), grp,
                       s=14, color="#185FA5", alpha=0.6, zorder=3)
        r = rules[(rules["类型"] == t) & (rules["成分"] == s)]
        if len(r):
            qv = r["FDR q值"].iloc[0]
            star = "*" if qv < 0.05 else ""
            ax.set_title(f"{s}  (q={qv:.3g}{star})", fontsize=12)
        else:
            ax.set_title(s, fontsize=12)
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"{t}玻璃：风化前后主要化学成分对比（闭合归一化，*为FDR q<0.05）", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, fname), dpi=200)
    plt.close(fig)


boxplot_type("高钾", ["SiO2", "K2O", "CaO", "Al2O3", "Fe2O3", "MgO", "P2O5", "PbO"],
             "箱线图_高钾.png")
boxplot_type("铅钡", ["SiO2", "PbO", "BaO", "CaO", "Al2O3", "CuO", "P2O5", "SrO"],
             "箱线图_铅钡.png")

# ============ 4. 风化程度指数 + 稳健性 + 标签盲检验 ============
SIG = {t: rules[(rules["类型"] == t) & (rules["显著(q<0.05)"] == "是")]["成分"].tolist()
       for t in TYPES}
detail = data[["文物采样点", "文物编号", "类型", "纹饰", "颜色", "表面风化", "点状态"]].copy()
idx_cols = {}
for t in TYPES:
    d = data[data["类型"] == t]
    pre = d[d["点状态"].isin(PRE_STATES)]
    gen = d[d["点状态"] == "一般风化点"]
    comps = [c for c in SIG[t] if pre["n_" + c].std() >= 0.1 and gen["n_" + c].std() > 0]
    ncol = ["n_" + c for c in comps]
    mu, sd = pre[ncol].mean(), pre[ncol].std()
    Z = (d[ncol] - mu) / sd
    dvec = Z.loc[gen.index].mean().values
    idx = Z.values @ dvec / (dvec @ dvec)
    detail.loc[d.index, "风化程度指数"] = idx
    detail.loc[d.index, "入选成分"] = "+".join(comps)
    idx_cols[t] = idx

    # 稳健性①：PC1 对照指数
    Xw = Z.loc[gen.index].values
    Xc = Xw - Xw.mean(0)
    u, s_, vt = np.linalg.svd(Xc, full_matrices=False)
    load = vt[0]
    if load @ dvec < 0:
        load = -load
    idx_pc1 = Z.values @ load / (load @ load)
    rho_pc1 = stats.spearmanr(idx, idx_pc1).statistic
    # 稳健性②：留一排序稳定性（每次剔除一个一般风化点重建方向）
    rhos = []
    for i in gen.index:
        rest = gen.index.difference(pd.Index([i]))
        dv = Z.loc[rest].mean().values
        idx_i = Z.values @ dv / (dv @ dv)
        sub_all = idx[d.index.isin(gen.index)]
        sub_i = idx_i[d.index.isin(gen.index)]
        rhos.append(stats.spearmanr(sub_all, sub_i).statistic)
    # 稳健性③：噪声地板 vs 风化组离散度
    pre_idx = idx[d.index.isin(pre.index)]
    wth_idx = idx[d.index.isin(gen.index.append(data[(data["类型"] == t) &
                                                    (data["点状态"] == "严重风化点")].index))]
    floor95 = np.percentile(np.abs(pre_idx), 95)
    print(f"\n[{t}] 指数入选成分({len(comps)}): {comps}")
    print(f"[{t}] 稳健性: PC1对照 Spearman ρ={rho_pc1:.3f} | "
          f"留一排序ρ 最小={min(rhos):.3f} 中位={np.median(rhos):.3f} | "
          f"噪声地板|idx|95%={floor95:.3f} vs 风化组σ={wth_idx.std():.3f}")

detail["风化程度指数"] = detail["风化程度指数"].round(3)

# 标签盲检验：严重风化点排名（铅钡）
lb = detail[(detail["类型"] == "铅钡") &
            (detail["点状态"].isin(["一般风化点", "严重风化点"]))]
lb = lb.sort_values("风化程度指数", ascending=False).reset_index(drop=True)
lb["排名(1=最重)"] = lb.index + 1
print("\n[标签盲检验] 严重风化点在铅钡风化点中的排名:")
print(lb[lb["点状态"] == "严重风化点"][["排名(1=最重)", "文物采样点", "风化程度指数"]].to_string(index=False))

# 剂量-反应：Spearman(指数, 成分) + FDR（仅风化点，避免组间分离带来的虚假相关）
dose = []
for t in TYPES:
    d = detail[detail["类型"] == t]
    mask = d["点状态"].isin(["一般风化点", "严重风化点"])
    di = d[mask]
    ps, rhos_, names = [], [], []
    for c in SIG[t]:
        r = stats.spearmanr(di["风化程度指数"], data.loc[di.index, "n_" + c])
        ps.append(r.pvalue)
        rhos_.append(r.statistic)
        names.append(c)
    q = bh_fdr(ps)
    for j, c in enumerate(names):
        dose.append({"类型": t, "成分": c, "Spearman ρ": round(rhos_[j], 3),
                     "p值": round(ps[j], 5), "FDR q值": round(q[j], 5),
                     "单调趋势(q<0.05)": "是" if q[j] < 0.05 else "否"})
dose = pd.DataFrame(dose)
print("\n[剂量-反应] Spearman 单调趋势（q<0.05）:")
print(dose[dose["单调趋势(q<0.05)"] == "是"].to_string(index=False))

# 同文物配对对照
pairs = []
for art in [42, 49, 50]:
    sub = detail[(detail["文物编号"] == art)]
    for _, r in sub.iterrows():
        pairs.append({"文物编号": art, "采样点": r["文物采样点"],
                      "点状态": r["点状态"], "风化程度指数": r["风化程度指数"]})
pair_df = pd.DataFrame(pairs)

# 轨迹图
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
panels = [("铅钡", "SiO2", axes[0][0]), ("铅钡", "PbO", axes[0][1]),
          ("高钾", "K2O", axes[1][0]), ("高钾", "SiO2", axes[1][1])]
COLOR = {"风化前-无风化文物": "#9aa0a6", "风化前-未风化点": "#6c757d",
         "一般风化点": "#185FA5", "严重风化点": "#D85A30"}
for t, comp, ax in panels:
    d = detail[detail["类型"] == t]
    for st, g in d.groupby("点状态"):
        ax.scatter(g["风化程度指数"], data.loc[g.index, "n_" + comp],
                   s=34, alpha=0.85, color=COLOR[st], label=st, edgecolors="white", lw=0.5)
    ax.axvline(0, ls="--", lw=1, color="#bbbbbb")
    ax.axvline(1, ls="--", lw=1, color="#bbbbbb")
    ax.set_xlabel("风化程度指数（0=无风化均值，1=一般风化均值）")
    ax.set_ylabel(f"{comp} 含量 (%)")
    ax.set_title(f"{t}玻璃：{comp} 随风化程度的变化")
    ax.legend(fontsize=8)
fig.suptitle("风化程度指数与关键成分：由“两档分类”到“连续进程”", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FIG, "风化程度指数_轨迹.png"), dpi=200)
plt.close(fig)

# ============ 5. 预测：分位数配对 + Theil-Sen 稳健回归（主） / 均值比例（对照） + bootstrap 区间 ============
B_BOOT = 1000


def qq_pairs(x, y):
    K = max(len(x), len(y), 20)
    p = (np.arange(K) + 0.5) / K
    return np.quantile(x, p), np.quantile(y, p)


def fit_ts(xq, yq):
    if len(xq) < 3 or xq.std() < 1e-9 or yq.std() < 1e-9:
        return None
    res = stats.theilslopes(yq, xq)
    return float(res[0]), float(res[1])


model_info, detail_rows = [], []
boot_summary = {}
for t in TYPES:
    d = data[data["类型"] == t]
    pre = d[d["点状态"].isin(PRE_STATES)]
    wth = d[d["点状态"].isin(["一般风化点", "严重风化点"])]
    if len(wth) == 0:
        continue
    pts = wth["文物采样点"].tolist()
    nx = len(pts)
    boot_pred = np.empty((B_BOOT, nx, len(COMPS)))
    # 逐成分全样本拟合（点估计 + 模型参数）
    fits = {}
    for j, c in enumerate(COMPS):
        s = SHORT[c]
        x, y = wth["n_" + s].values, pre["n_" + s].values
        xq, yq = qq_pairs(x, y)
        f = fit_ts(xq, yq)
        fits[c] = f
        if f is not None:
            b, a = f
            r2 = np.corrcoef(xq, yq)[0, 1] ** 2
            yhat = a + b * xq
            ssr = ((yq - yhat) ** 2).sum()
            sst = ((yq - yq.mean()) ** 2).sum()
            r2 = 1 - ssr / sst if sst > 0 else np.nan
            model_info.append({"类型": t, "成分": s, "截距a": round(a, 4), "斜率b": round(b, 4),
                               "R2(QQ对)": round(r2, 3), "建模方式": "分位数配对+Theil-Sen"})
        else:
            model_info.append({"类型": t, "成分": s, "截距a": 0.0, "斜率b": 1.0,
                               "R2(QQ对)": None, "建模方式": "回退(保持原值)"})
        for i, pt in enumerate(pts):
            xv = wth["n_" + s].iloc[i]
            pred = max(0.0, a + b * xv) if f is not None else xv
            detail_rows.append({"采样点": pt, "类型": t, "点状态": wth["点状态"].iloc[i],
                                "成分": s, "风化后检测值": round(xv, 3),
                                "方案A_预测": pred, "方案B_预测": np.nan})
    # 均值比例法 k
    for j, c in enumerate(COMPS):
        s = SHORT[c]
        k = pre["n_" + s].mean() / wth["n_" + s].mean() if wth["n_" + s].mean() > 0.3 else 1.0
        for i, pt in enumerate(pts):
            m = [r for r in detail_rows if r["采样点"] == pt and r["成分"] == s][-1]
            m["方案B_预测"] = wth["n_" + s].iloc[i] * k
    # 联合 bootstrap：每次抽样后全成分拟合→预测→闭合归一化，再取分位数
    xv_all = np.array([wth["n_" + SHORT[c]].values for c in COMPS]).T  # (nx,14)
    for b in range(B_BOOT):
        widx = RNG.integers(0, len(wth), len(wth))
        pidx = RNG.integers(0, len(pre), len(pre))
        pv = np.empty((nx, len(COMPS)))
        ok_any = False
        for j, c in enumerate(COMPS):
            s = SHORT[c]
            x = wth["n_" + s].values[widx]
            y = pre["n_" + s].values[pidx]
            xq, yq = qq_pairs(x, y)
            f = fit_ts(xq, yq)
            if f is not None:
                pv[:, j] = np.clip(f[1] + f[0] * xv_all[:, j], 0, None)
                ok_any = True
            else:
                pv[:, j] = xv_all[:, j]
        tot = pv.sum(1, keepdims=True)
        tot[tot == 0] = 1
        boot_pred[b] = pv / tot * 100
    lo = np.percentile(boot_pred, 5, axis=0)
    hi = np.percentile(boot_pred, 95, axis=0)
    for i, pt in enumerate(pts):
        for j, c in enumerate(COMPS):
            m = [r for r in detail_rows if r["采样点"] == pt and r["成分"] == SHORT[c]][-1]
            m["A_boot下界"] = lo[i, j]
            m["A_boot上界"] = hi[i, j]
    boot_summary[t] = (pts, lo, hi)

det = pd.DataFrame(detail_rows)
# 点预测闭合归一化（方案A、B 分别进行）
for col in ["方案A_预测", "方案B_预测"]:
    det[col] = det.groupby("采样点")[col].transform(lambda s: np.clip(s, 0, None) / np.clip(s, 0, None).sum() * 100)
    det[col] = det[col].round(2)
for col in ["A_boot下界", "A_boot上界", "风化后检测值"]:
    det[col] = det[col].round(2)
det.to_excel(os.path.join(OUT, "03_风化前预测明细_含bootstrap区间.xlsx"), index=False)
pd.DataFrame(model_info).to_excel(os.path.join(OUT, "04_模型参数_方案A.xlsx"), index=False)

wideA = det.pivot_table(index=["采样点", "类型", "点状态"], columns="成分",
                        values="方案A_预测", aggfunc="first").reset_index()
wideB = det.pivot_table(index=["采样点", "类型", "点状态"], columns="成分",
                        values="方案B_预测", aggfunc="first").reset_index()
wideA.to_excel(os.path.join(OUT, "05_预测宽表_方案A.xlsx"), index=False)
wideB.to_excel(os.path.join(OUT, "05b_预测宽表_方案B.xlsx"), index=False)
print(f"\n[预测] 完成 {det['采样点'].nunique()} 个风化点 × 14 成分（方案A: Theil-Sen, 方案B: 比例缩放, bootstrap B={B_BOOT}）")

# ============ 6. 验证 ============
# 6.1 留一文物配对（49、50 号：同时有风化点与未风化点）
val_rows = []
for art in [49, 50]:
    d = data[data["类型"] == "铅钡"]
    wpick = d[(d["文物编号"] == art) & (d["点状态"].isin(["一般风化点", "严重风化点"]))]
    ppick = d[(d["文物编号"] == art) & (d["点状态"].str.startswith("风化前"))]
    if len(wpick) == 0 or len(ppick) == 0:
        continue
    pre_fit = d[(d["点状态"].isin(PRE_STATES)) & (d["文物编号"] != art)]
    wth_fit = d[(d["点状态"].isin(["一般风化点", "严重风化点"])) & (d["文物编号"] != art)]
    wrow, prow = wpick.iloc[0], ppick.iloc[0]
    for c in COMPS:
        s = SHORT[c]
        xq, yq = qq_pairs(wth_fit["n_" + s].values, pre_fit["n_" + s].values)
        f = fit_ts(xq, yq)
        predA = max(0.0, f[1] + f[0] * wrow["n_" + s]) if f else wrow["n_" + s]
        k = pre_fit["n_" + s].mean() / wth_fit["n_" + s].mean() if wth_fit["n_" + s].mean() > 0.3 else 1.0
        val_rows.append({"文物": art, "成分": s, "真实风化前值": prow["n_" + s],
                         "方案A预测": predA, "方案B预测": wrow["n_" + s] * k})
val = pd.DataFrame(val_rows)
for meth in ["方案A预测", "方案B预测"]:
    val[meth] = val.groupby("文物")[meth].transform(lambda s: s / s.sum() * 100)
val = val.round(2)
val["A误差"] = (val["方案A预测"] - val["真实风化前值"]).abs()
val["B误差"] = (val["方案B预测"] - val["真实风化前值"]).abs()
val.to_excel(os.path.join(OUT, "06_留一配对验证.xlsx"), index=False)
print(f"\n[留一验证] MAE 方案A={val['A误差'].mean():.2f} 方案B={val['B误差'].mean():.2f}（百分点）")
print(val.groupby("文物")[["A误差", "B误差"]].mean().round(2).to_string())

# 6.2 合理性检查：预测值落入风化前经验区间的比例（关键成分）
range_rows = []
for t in TYPES:
    d = data[data["类型"] == t]
    pre = d[d["点状态"].isin(PRE_STATES)]
    wth = det[(det["类型"] == t)]
    for s in ["SiO2", "PbO", "BaO", "K2O", "CaO", "P2O5"]:
        if s not in [SHORT[c] for c in COMPS]:
            continue
        lo_, hi_ = pre["n_" + s].min(), pre["n_" + s].max()
        pv = wth[wth["成分"] == s]["方案A_预测"]
        if len(pv) == 0:
            continue
        range_rows.append({"类型": t, "成分": s, "风化前经验区间": f"[{lo_:.1f}, {hi_:.1f}]",
                           "落入区间比例": round(((pv >= lo_) & (pv <= hi_)).mean(), 2),
                           "n风化点": len(pv)})
rng_df = pd.DataFrame(range_rows)
# 6.3 排序一致性：检出 SiO2 越高（风化越轻）→ 还原后 SiO2 越高
rank_rows = []
for t in TYPES:
    d = det[(det["类型"] == t) & (det["成分"] == "SiO2")]
    if len(d) >= 4:
        r = stats.spearmanr(d["风化后检测值"], d["方案A_预测"])
        rank_rows.append({"类型": t, "n": len(d), "Spearman ρ(检出SiO2, 还原SiO2)": round(r.statistic, 3),
                          "p值": round(r.pvalue, 4)})
# 6.4 方案间一致性
agree_abs = (det["方案A_预测"] - det["方案B_预测"]).abs().mean()
agree_rho = stats.spearmanr(det["方案A_预测"], det["方案B_预测"]).statistic
checks = pd.DataFrame(range_rows)
checks.to_excel(os.path.join(OUT, "07_合理性检查_经验区间.xlsx"), index=False)
pd.DataFrame(rank_rows).to_excel(os.path.join(OUT, "07b_排序一致性.xlsx"), index=False)
print(f"[合理性] 预测落入风化前经验区间比例（各关键成分）: {rng_df['落入区间比例'].tolist()}")
print(f"[排序一致] {rank_rows}")
print(f"[方案一致] A/B 平均绝对差 {agree_abs:.2f} 个百分点, Spearman ρ={agree_rho:.3f}")

# 验证散点图
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, meth, err in zip(axes, ["方案A预测", "方案B预测"], ["A误差", "B误差"]):
    ax.scatter(val["真实风化前值"], val[meth], s=28, color="#185FA5", alpha=0.75)
    lim = max(val["真实风化前值"].max(), val[meth].max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="#D85A30", lw=1.2)
    ax.set_xlabel("真实风化前含量 (%)")
    ax.set_ylabel("预测风化前含量 (%)")
    ax.set_title(f"{meth}（MAE={val[err].mean():.2f}）", fontsize=12)
fig.suptitle("留一文物配对验证：预测值 vs 真实值（49/50号，铅钡玻璃）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(FIG, "预测验证散点图.png"), dpi=200)
plt.close(fig)

# ============ 7. 指数与配对输出 ============
detail_sorted = detail.sort_values(["类型", "风化程度指数"], ascending=[True, False])
with pd.ExcelWriter(os.path.join(OUT, "08_风化程度指数.xlsx")) as w:
    detail_sorted.to_excel(w, sheet_name="指数明细", index=False)
    lb[["排名(1=最重)", "文物采样点", "点状态", "风化程度指数"]].to_excel(
        w, sheet_name="严重风化点盲检验", index=False)
    dose.to_excel(w, sheet_name="剂量反应_Spearman", index=False)
    pair_df.to_excel(w, sheet_name="同文物配对", index=False)

print("\n[完成] 严谨版结果已输出到", OUT)
for f in sorted(os.listdir(OUT)):
    print(" -", f)
for f in sorted(os.listdir(FIG)):
    print(" - 图/", f)
