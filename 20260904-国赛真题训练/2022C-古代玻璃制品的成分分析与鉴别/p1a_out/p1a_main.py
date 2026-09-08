# -*- coding: utf-8 -*-
"""
p1a_main.py — 问题一前半：表面风化与类型、纹饰、颜色的关联分析（三步联合框架）
执行第 1–5 步（第 6 步 MCA 按计划不做），输出表格(CSV)与图(PNG, 300dpi)到 p1a_out/。

依赖: numpy / pandas / scipy / matplotlib（无需 statsmodels、seaborn）
运行: python p1a_main.py        （附件路径见下方常量）
"""
import os
import re
import time

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullFormatter

# ==================== 顶部常量（小组规范：脚本只读原始附件） ====================
ATTACH = '附件.xlsx'
OUTDIR = 'p1a_out'
SEED = 2026          # 全局随机种子
N_BOOT = 5000        # V 的 Bootstrap 重抽样次数
N_PERM = 10000       # 置换检验次数
DPI = 300

# ==================== 图表样式（dataviz 参考调色板，已验证值） ====================
INK1, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, BASELINE, SURFACE = '#e1e0d9', '#c3c2b7', '#ffffff'
C1 = '#2a78d6'                                        # 分类色 slot1（蓝）
C_REF = '#898781'                                     # 参考系列（弱化灰）
ORD4 = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab']   # 序数蓝 4 步（熵阶梯）
WXDARK, WXLIGHT = '#1c5cab', '#86b6ef'                # 堆叠图：风化/无风化（同色系序数2步）
HEAT = LinearSegmentedColormap.from_list(
    'seq_blue', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])

plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'sans-serif'],
    'axes.unicode_minus': False,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': BASELINE, 'axes.linewidth': 0.8,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'text.color': INK1, 'axes.labelcolor': INK2, 'axes.titlecolor': INK1,
    'axes.titlesize': 11, 'axes.grid': False,
})

os.makedirs(OUTDIR, exist_ok=True)
LOG = []
def log(s=''):
    print(s)
    LOG.append(str(s))

def savefig(fig, name):
    fig.savefig(f'{OUTDIR}/{name}', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    log(f'  [图] {OUTDIR}/{name}')

# ==================== 第 1 步：数据底座 ====================
t0 = time.time()
log('=' * 72)
log('第 1 步  数据底座')
log('=' * 72)

f1 = pd.read_excel(ATTACH, sheet_name='表单1')
CMAP_COLOR = {'蓝绿': '蓝绿系', '绿': '蓝绿系', '浅绿': '蓝绿系', '深绿': '蓝绿系',
              '浅蓝': '蓝色系', '深蓝': '蓝色系', '紫': '深暗系', '黑': '深暗系'}
f1['色系'] = f1['颜色'].map(CMAP_COLOR).fillna('未记录')
f1['wx'] = (f1['表面风化'] == '风化').astype(int)     # 1=风化, 0=无风化
f1['ky'] = (f1['类型'] == '高钾').astype(int)
f1['wenB'] = (f1['纹饰'] == 'B').astype(int)
f1['wenC'] = (f1['纹饰'] == 'C').astype(int)
f1['sLl'] = (f1['色系'] == '蓝绿系').astype(int)
f1['sSr'] = (f1['色系'] == '深暗系').astype(int)
f1['sLs'] = (f1['色系'] == '蓝色系').astype(int)      # 参考组=蓝色系

base = f1[['文物编号', '纹饰', '类型', '颜色', '色系', '表面风化', 'wx', 'ky',
           'wenB', 'wenC', 'sLl', 'sSr', 'sLs']]
base.to_csv(f'{OUTDIR}/p1a_tab0_base.csv', index=False, encoding='utf-8-sig')
log(f'  [表] {OUTDIR}/p1a_tab0_base.csv  （58 件文物底表）')

# 点级后缀与文物级风化标签的一致性核查（读表单2 点名）
f2 = pd.read_excel(ATTACH, sheet_name='表单2')
wxmap = f1.set_index('文物编号')['表面风化']
bad = []
for name in f2['文物采样点'].astype(str):
    m = re.match(r'^(\d+)', name)
    if m and ('未风化点' in name or '严重风化点' in name) and wxmap[int(m.group(1))] == '无风化':
        bad.append(name)
log(f'  一致性核查: 带“未风化点/严重风化点”后缀且属无风化文物的点 = {len(bad)} 个（应为 0）')
miss_id = f1.loc[f1['色系'] == '未记录', '文物编号'].tolist()
log(f'  颜色缺失(未记录)文物: {miss_id}，风化状态 {f1.loc[f1["色系"] == "未记录", "表面风化"].tolist()}')
log(f'  分布: 类型 {dict(f1["类型"].value_counts())}；风化 {dict(f1["表面风化"].value_counts())}；'
    f'色系 {dict(f1["色系"].value_counts())}')

# ==================== 公共统计工具（numpy 加速版） ====================
def codes(s):
    return pd.factorize(s.to_numpy())[0], len(pd.unique(s.to_numpy()))

def chi2_np(ac, bc, Ka, Kb):
    """未校正 Pearson 卡方（numpy 版，供置换/自助循环使用）"""
    N = np.bincount(ac * Kb + bc, minlength=Ka * Kb).reshape(Ka, Kb).astype(float)
    n = N.sum()
    E = N.sum(1, keepdims=True) @ N.sum(0, keepdims=True) / n
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(E > 0, (N - E) ** 2 / E, 0.0)
    return float(t.sum()), N

def cramers_v_np(ac, bc, Ka, Kb):
    k = min(Ka, Kb) - 1
    if k < 1:
        return np.nan
    chi2, _ = chi2_np(ac, bc, Ka, Kb)
    return float(np.sqrt(chi2 / (len(ac) * k)))

def bh_adjust(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out

# ==================== 第 2 步：纠缠诊断（属性间 V 矩阵） ====================
log('')
log('=' * 72)
log('第 2 步  纠缠诊断（属性间 Cramér\'s V 矩阵）')
log('=' * 72)

ATTRS = ['表面风化', '类型', '纹饰', '色系']
codes_map = {a: codes(f1[a]) for a in ATTRS}
Vm = pd.DataFrame(1.0, index=ATTRS, columns=ATTRS)
for i, a in enumerate(ATTRS):
    for j, b in enumerate(ATTRS):
        if i < j:
            ac, Ka = codes_map[a]
            bc, Kb = codes_map[b]
            Vm.loc[a, b] = Vm.loc[b, a] = cramers_v_np(ac, bc, Ka, Kb)
Vm.round(3).to_csv(f'{OUTDIR}/p1a_tab0b_vmatrix.csv', encoding='utf-8-sig')
log(f'  [表] {OUTDIR}/p1a_tab0b_vmatrix.csv')
log(Vm.round(3).to_string())
log('  解读: 自变量间 V(类型,纹饰)=0.51、V(类型,色系)=0.41、V(纹饰,色系)=0.31，'
    '普遍强于三者与风化的 V(0.34/0.29/0.24) → 逐一比较不可靠，须联合分析。')

# ---- 图1：V 矩阵下三角热图 ----
fig, ax = plt.subplots(figsize=(5.0, 4.2))
M = Vm.to_numpy()
Mm = np.ma.masked_where(np.triu(np.ones_like(M, bool)), M)   # 隐藏上三角与对角
im = ax.imshow(Mm, cmap=HEAT, vmin=0.0, vmax=0.6)
for i in range(4):
    for j in range(i):
        v = M[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=11,
                color='#ffffff' if v > 0.38 else INK2,
                fontweight='bold' if v >= 0.4 else 'normal')
ax.set_xticks(range(4), ATTRS, fontsize=10)
ax.set_yticks(range(4), ATTRS, fontsize=10)
ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
ax.grid(which='minor', color=SURFACE, linewidth=2)
ax.tick_params(which='minor', length=0)
ax.tick_params(which='major', length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("Cramér's V", fontsize=9, color=INK2)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=8, color=MUTED)
ax.set_title("属性间 Cramér's V 矩阵（n=58）", loc='left', pad=10)
savefig(fig, 'p1a_fig1_v_heatmap.png')

# ==================== 第 3 步：边际筛查 ====================
log('')
log('=' * 72)
log(f'第 3 步  边际筛查（Fisher/置换检验 + OR + Bootstrap CI, {N_BOOT}/{N_PERM} 次）')
log('=' * 72)

WX = f1['wx'].to_numpy()
rows_tab1 = []

def marginal_row(label, var, kind, seed, dropna=False):
    """kind: '2x2' 走 Fisher；否则置换。var 为列名；dropna=True 时剔除该列缺失行。"""
    dfm = f1 if not dropna else f1[f1[var].notna()]
    a_codes, Ka = codes(dfm[var])
    b_codes, Kb = codes(dfm['表面风化'])
    chi2, p_chi2 = stats.chi2_contingency(pd.crosstab(dfm[var], dfm['表面风化']),
                                          correction=False)[:2]
    if kind == '2x2':
        # 中文按 Unicode 排序：铅钡 在 高钾 之前，须显式重排为 [高钾, 铅钡]
        ct = pd.crosstab(dfm[var], dfm['表面风化']).loc[['高钾', '铅钡'],
                                                        ['无风化', '风化']]
        (a11, a12), (a21, a22) = ct.to_numpy()
        fisher_or, p_main = stats.fisher_exact([[a11, a12], [a21, a22]])
        # 铅钡 vs 高钾 的风化优势比
        OR = (a22 / a21) / (a12 / a11)
        se = np.sqrt(1 / a11 + 1 / a12 + 1 / a21 + 1 / a22)
        lo, hi = np.exp(np.log(OR) - 1.96 * se), np.exp(np.log(OR) + 1.96 * se)
        yates = stats.chi2_contingency(ct, correction=True)
        extra = f'卡方未校正 χ²={chi2:.2f}/p={p_chi2:.4f}；Yates χ²={yates[0]:.2f}/p={yates[1]:.4f}'
        or_txt = f'{OR:.2f} ({lo:.2f}~{hi:.2f})'
        phi = np.sqrt(chi2 / len(dfm))
    else:
        obs, _ = chi2_np(a_codes, b_codes, Ka, Kb)
        r = np.random.default_rng(seed)
        bc = b_codes.copy()
        cnt = 0
        for _ in range(N_PERM):
            r.shuffle(bc)
            if chi2_np(a_codes, bc, Ka, Kb)[0] >= obs - 1e-9:
                cnt += 1
        p_main = (cnt + 1) / (N_PERM + 1)
        extra = f'置换检验（{N_PERM} 次）；卡方未校正 χ²={chi2:.2f}/p={p_chi2:.4f}'
        or_txt = '—'
        phi = np.nan
    # V 的 Bootstrap CI
    r = np.random.default_rng(seed + 1)
    n = len(dfm)
    vs, nfail = [], 0
    for _ in range(N_BOOT):
        idx = r.integers(0, n, n)
        v = cramers_v_np(a_codes[idx], b_codes[idx], Ka, Kb)
        if np.isnan(v):
            nfail += 1
        else:
            vs.append(v)
    lo_v, hi_v = np.percentile(vs, [2.5, 97.5])
    med_v = float(np.median(vs))
    v0 = cramers_v_np(a_codes, b_codes, Ka, Kb)
    # 风化率
    rate = dfm.groupby(var)['wx'].agg(['mean', 'count'])
    rate_txt = '；'.join(f'{i} {r["mean"] * 100:.1f}% (n={int(r["count"])})'
                         for i, r in rate.iterrows())
    rows_tab1.append(dict(因素=label, 表规格=kind.replace('x', '×') if kind == '2x2'
                          else f'2×{Ka}', 主检验=extra.split('；')[0], p主=p_main,
                          V点估计=round(v0, 3), V_CI=f'[{lo_v:.3f}, {hi_v:.3f}]',
                          V_CI_lo=lo_v, V_CI_hi=hi_v, V_boot中位=round(med_v, 3),
                          **{'OR(铅钡vs高钾)': or_txt},
                          phi=round(phi, 3),
                          交叉验证据=extra, 风化率=rate_txt))
    log(f'  {label}: p={p_main:.4f}  V={v0:.3f} [{lo_v:.3f}, {hi_v:.3f}]  '
        f'boot中位={med_v:.3f}  OR={or_txt}')
    return dict(label=label, v0=v0, lo=lo_v, hi=hi_v, p=p_main)

r_type = marginal_row('类型', '类型', '2x2', SEED + 100)
r_wen = marginal_row('纹饰', '纹饰', 'RxC', SEED + 200)
r_sex = marginal_row('色系', '色系', 'RxC', SEED + 300)
r_col8 = marginal_row('颜色(8类,参考)', '颜色', 'RxC', SEED + 400, dropna=True)

pb = bh_adjust([r_type['p'], r_wen['p'], r_sex['p']])
for row, adj in zip(rows_tab1[:3], pb):
    row['pBH'] = round(float(adj), 4)
rows_tab1[3]['pBH'] = None
tab1 = pd.DataFrame(rows_tab1)
tab1.to_csv(f'{OUTDIR}/p1a_tab1_marginal.csv', index=False, encoding='utf-8-sig')
log(f'  [表] {OUTDIR}/p1a_tab1_marginal.csv  （BH 校正: 类型 {pb[0]:.4f}, '
    f'纹饰 {pb[1]:.4f}, 色系 {pb[2]:.4f}）')

# ---- 图2：V 的 Bootstrap 95% CI 森林图 ----
fig, ax = plt.subplots(figsize=(6.4, 3.2))
fs = [(r_type, '类型 (2×2)'), (r_wen, '纹饰 (2×3)'), (r_sex, '色系 (2×4)'),
      (r_col8, '颜色·8类（参考）')]
ys = np.arange(len(fs))[::-1]
for (st, lab), y in zip(fs, ys):
    is_ref = '参考' in lab
    col = C_REF if is_ref else C1
    ax.hlines(y, st['lo'], st['hi'], color=col, lw=1.6, zorder=2)
    ax.plot(st['v0'], y, 'o', ms=7, mfc=col, mec=SURFACE, mew=1.6 if not is_ref else 0,
            zorder=3) if not is_ref else ax.plot(st['v0'], y, 'o', ms=7, mfc=SURFACE,
                                                 mec=col, mew=1.4, zorder=3)
    ax.text(0.665, y, f"{st['v0']:.2f} [{st['lo']:.2f}, {st['hi']:.2f}]",
            va='center', fontsize=8.5, color=INK2)
ax.set_yticks(ys, [x[1] for x in fs], fontsize=10)
ax.set_xlim(0, 0.66)
ax.set_xlabel("Cramér's V（点估计与 Bootstrap 95% CI）", fontsize=9)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.grid(axis='x', color=GRID, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for sp in ['top', 'right', 'left']:
    ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color(BASELINE)
ax.tick_params(length=0)
ax.set_title(f"关联强度的不确定性：V 的点估计与 Bootstrap 95% CI（{N_BOOT} 次重抽样）",
             loc='left', pad=10)
savefig(fig, 'p1a_fig2_v_forest.png')

# ---- 图3：分属性风化率 100% 堆叠条形（3 联图） ----
panels = [('类型', ['高钾', '铅钡'], '类型 (Fisher p=0.011)'),
          ('纹饰', ['A', 'B', 'C'], '纹饰 (置换 p=0.103)'),
          ('色系', ['蓝绿系', '蓝色系', '深暗系', '未记录'], '色系 (置换 p=0.364)')]
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharex=True)
for ax, (var, order, ttl) in zip(axes, panels):
    sub = f1.groupby(var)['wx'].agg(['sum', 'count']).loc[order]
    n = sub['count'].to_numpy()
    pwx = sub['sum'].to_numpy() / n * 100
    y = np.arange(len(order))[::-1]
    # 无风化（浅）在前，风化（深）在后；白色间隔由 edge 承担
    ax.barh(y, 100 - pwx, height=0.58, color=WXLIGHT, edgecolor=SURFACE, linewidth=1.4,
            label='无风化', zorder=2)
    ax.barh(y, pwx, left=100 - pwx, height=0.58, color=WXDARK, edgecolor=SURFACE,
            linewidth=1.4, label='风化', zorder=2)
    for yi, pw in zip(y, pwx):
        if pw > 10:
            ax.text(100 - pw / 2, yi, f'{pw:.0f}%', ha='center', va='center',
                    fontsize=8.5, color='#ffffff')
        if 100 - pw > 10:
            ax.text((100 - pw) / 2, yi, f'{100 - pw:.0f}%', ha='center', va='center',
                    fontsize=8.5, color=INK2)
    ax.set_yticks(y, [f'{o} (n={ni})' for o, ni in zip(order, n)], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.set_title(ttl, loc='left', fontsize=10)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(BASELINE)
    ax.tick_params(length=0)
axes[0].legend(loc='upper left', bbox_to_anchor=(0, -0.22), ncol=2, frameon=False,
               fontsize=9, handlelength=1.2, handleheight=0.9)
savefig(fig, 'p1a_fig3_wxrate_bars.png')

# ==================== 第 4 步：联合建模（Firth 惩罚逻辑回归） ====================
log('')
log('=' * 72)
log('第 4 步  联合建模（Firth 惩罚逻辑回归）')
log('=' * 72)

def firth_logit(X, y, max_iter=1000, tol=1e-9):
    """Firth 惩罚似然（修正评分法）：评分中加入 h_i*(0.5-mu_i) 修正项"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        W = np.clip(mu * (1 - mu), 1e-10, None)
        I = X.T @ (X * W[:, None])
        Iinv = np.linalg.pinv(I)
        XW = X * np.sqrt(W)[:, None]
        h = np.clip(np.diag(XW @ Iinv @ XW.T), 0, 1)
        U = X.T @ (y - mu + h * (0.5 - mu))
        step = Iinv @ U
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    mu = 1 / (1 + np.exp(-(X @ beta)))
    W = np.clip(mu * (1 - mu), 1e-10, None)
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ (X * W[:, None]))))
    return beta, se

def plain_logit(X, y, max_iter=500, tol=1e-10):
    """普通（无惩罚）logistic IRLS，用于对照与发散证据；返回 beta, se, converged"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    conv = False
    for _ in range(max_iter):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        W = np.clip(mu * (1 - mu), 1e-12, None)
        H = X.T @ (X * W[:, None])
        g = X.T @ (y - mu)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            conv = False
            break
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            conv = True
            break
    mu = 1 / (1 + np.exp(-(X @ beta)))
    W = np.clip(mu * (1 - mu), 1e-12, None)
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ (X * W[:, None]))))
    if not conv or np.max(np.abs(beta)) > 15:
        conv = False
    return beta, se, conv

def wald(beta, se):
    z = beta / se
    return 2 * stats.norm.sf(np.abs(z))

X_full = np.column_stack([np.ones(len(f1)), f1[['ky', 'wenB', 'wenC', 'sLl', 'sSr']].
                          to_numpy(float)])
names_full = ['截距', '高钾', '纹饰B', '纹饰C', '蓝绿系', '深暗系']
betaF, seF = firth_logit(X_full, WX)
betaP, seP_full, convP = plain_logit(X_full, WX)

X_ky = np.column_stack([np.ones(len(f1)), f1['ky'].to_numpy(float)])
betaKF, seKF = firth_logit(X_ky, WX)
betaKP, seKP, convKP = plain_logit(X_ky, WX)

rows_tab2 = []
for lab, b, s, model in zip(names_full, betaF, seF, ['Firth全模型'] * 6):
    p = wald(b, s)
    rows_tab2.append(dict(模型=model, 变量=lab, OR=round(float(np.exp(b)), 3),
                          CI_lo=round(float(np.exp(b - 1.96 * s)), 3),
                          CI_hi=round(float(np.exp(b + 1.96 * s)), 3),
                          Wald_p=round(float(p), 4)))
rows_tab2.append(dict(模型='对照·仅类型(Firth)', 变量='高钾',
                      OR=round(float(np.exp(betaKF[1])), 3),
                      CI_lo=round(float(np.exp(betaKF[1] - 1.96 * seKF[1])), 3),
                      CI_hi=round(float(np.exp(betaKF[1] + 1.96 * seKF[1])), 3),
                      Wald_p=round(float(wald(betaKF[1], seKF[1])), 4)))
rows_tab2.append(dict(模型='对照·仅类型(普通MLE)', 变量='高钾',
                      OR=round(float(np.exp(betaKP[1])), 3),
                      CI_lo=round(float(np.exp(betaKP[1] - 1.96 * seKP[1])), 3),
                      CI_hi=round(float(np.exp(betaKP[1] + 1.96 * seKP[1])), 3),
                      Wald_p=round(float(wald(betaKP[1], seKP[1])), 4)))
rows_tab2.append(dict(模型='对照·全模型(普通MLE)', 变量='—', OR='发散(不收敛)',
                      CI_lo='—', CI_hi='—', Wald_p=f'max|β|={np.max(np.abs(betaP)):.1f}'))
tab2 = pd.DataFrame(rows_tab2)
tab2.to_csv(f'{OUTDIR}/p1a_tab2_firth.csv', index=False, encoding='utf-8-sig')
log(f'  [表] {OUTDIR}/p1a_tab2_firth.csv')
log(tab2.to_string(index=False))
log(f'  解读: 全模型中高钾调整后 OR={np.exp(betaF[1]):.3f}（仅类型时 {np.exp(betaKF[1]):.3f}、'
    f'普通MLE {np.exp(betaKP[1]):.3f}），纹饰B 因 6/6 分离呈有限大 OR={np.exp(betaF[2]):.0f}。')

# 说明：Firth 全模型的完整数值见表 p1a_tab2_firth.csv（论文 1.6 节复核引用）。
# 论文简化版不再绘制 OR 森林图与条件熵阶梯图，相关高级方法数值保留在运行日志中备查。

# ==================== 第 5 步：条件信息增益（层内置换校准） ====================
log('')
log('=' * 72)
log(f'第 5 步  条件信息增益（层内置换 {N_PERM} 次）')
log('=' * 72)

y_all = WX.astype(float)
S = lambda cs: pd.factorize(f1[cs].astype(str).agg('|'.join, axis=1).to_numpy())[0] \
    if len(cs) else np.zeros(len(f1), dtype=int)

def H1(p):
    p = float(np.clip(p, 0, 1))
    return 0.0 if p <= 0 or p >= 1 else -(p * np.log(p) + (1 - p) * np.log(1 - p))

def cond_H(wx, sid):
    tot, n = 0.0, len(wx)
    for g in np.unique(sid):
        m = sid == g
        tot += m.sum() / n * H1(wx[m].mean())
    return float(tot)

def gain(xcols, given):
    sg, sgx = S(given), S(given + xcols)
    return cond_H(y_all, sg) - cond_H(y_all, sgx)

def perm_gain(xcols, given, seed):
    obs = gain(xcols, given)
    sg, sgx = S(given), S(given + xcols)
    r = np.random.default_rng(seed)
    wx = y_all.copy()
    idx_lists = [np.where(sg == g)[0] for g in np.unique(sg)]
    cnt = 0
    for _ in range(N_PERM):
        for ii in idx_lists:
            wx[ii] = wx[ii][r.permutation(ii.size)]
        if cond_H(wx, sg) - cond_H(wx, sgx) >= obs - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (N_PERM + 1)

TESTS = [('类型（边际）', ['ky'], [], SEED + 500),
         ('纹饰 | 给定类型', ['wenB', 'wenC'], ['ky'], SEED + 600),
         ('色系 | 给定类型', ['sLl', 'sSr'], ['ky'], SEED + 700),
         ('色系 | 给定类型+纹饰', ['sLl', 'sSr'], ['ky', 'wenB', 'wenC'], SEED + 800),
         ('纹饰+色系 | 给定类型', ['wenB', 'wenC', 'sLl', 'sSr'], ['ky'], SEED + 900),
         ('类型 | 给定纹饰', ['ky'], ['wenB', 'wenC'], SEED + 1000),
         ('类型 | 给定色系', ['ky'], ['sLl', 'sSr'], SEED + 1100)]
rows_tab3 = []
for lab, xc, gv, sd in TESTS:
    g, p = perm_gain(xc, gv, sd)
    rows_tab3.append(dict(条件增益=lab, 增益nat=round(g, 4), 置换p=round(p, 4)))
    log(f'  {lab:22s} 增益={g:.4f} nat   置换p={p:.4f}')
# 说明：条件增益数值仅写入运行日志备查（论文简化版未引用该 CSV）；
# 高钾层内的极端结构由下方分层分析以组合概率与层内卡方置换给出，为论文 1.6 节的证据。

# ==================== 稳健性：剔除未记录色系后的口径敏感性 ====================
log('')
log('=' * 72)
log('稳健性  剔除颜色未记录的 4 件后重跑边际检验')
log('=' * 72)

fs1 = f1[f1['色系'] != '未记录']
ctS = pd.crosstab(fs1['类型'], fs1['表面风化']).loc[['高钾', '铅钡'], ['无风化', '风化']]
(a11, a12), (a21, a22) = ctS.to_numpy()
pS = stats.fisher_exact([[a11, a12], [a21, a22]])[1]
log(f'  类型 2×2 (n={len(fs1)}): Fisher p={pS:.4f}（全样本 0.0113）')
for var, seed in [('纹饰', SEED + 1200), ('色系', SEED + 1300)]:
    ac, Ka = codes(fs1[var])
    bc, Kb = codes(fs1['表面风化'])
    obs, _ = chi2_np(ac, bc, Ka, Kb)
    r = np.random.default_rng(seed)
    bbc = bc.copy()
    cnt = 0
    for _ in range(N_PERM):
        r.shuffle(bbc)
        if chi2_np(ac, bbc, Ka, Kb)[0] >= obs - 1e-9:
            cnt += 1
    log(f'  {var} (n={len(fs1)}): 置换p={(cnt + 1) / (N_PERM + 1):.4f}')

# ==================== 按类型分层的纹饰/色系×风化 ====================
log('')
log('=' * 72)
log('分层分析  按类型分层后的纹饰、色系×风化')
log('=' * 72)

rows_tab5 = []
for layer, order in [('铅钡', ['A', 'C']), ('高钾', ['A', 'B', 'C'])]:
    sub = f1[f1['类型'] == layer]
    ct = pd.crosstab(sub['纹饰'], sub['表面风化']).loc[order, ['无风化', '风化']]
    for w, (n0, n1) in zip(order, ct.to_numpy()):
        rows_tab5.append(dict(层=layer, 纹饰档=w, n=n0 + n1, 无风化=n0, 风化=n1,
                              风化率=round(n1 / (n0 + n1) * 100, 1)))
    log(f'  {layer}层 (n={len(sub)}):\n' + ct.assign(风化率=lambda d: (d["风化"] / d.sum(1) * 100).round(1)).to_string())
tab5 = pd.DataFrame(rows_tab5)
tab5.to_csv(f'{OUTDIR}/p1a_tab5_stratified.csv', index=False, encoding='utf-8-sig')
log(f'  [表] {OUTDIR}/p1a_tab5_stratified.csv')

# 铅钡层内 A 与 C 的 2×2 Fisher
subq = f1[f1['类型'] == '铅钡']
ctQ = pd.crosstab(subq['纹饰'], subq['表面风化']).loc[['A', 'C'], ['无风化', '风化']]
(aq, bq), (cq, dq) = ctQ.to_numpy()
pQ = stats.fisher_exact([[aq, bq], [cq, dq]])[1]
log(f'  铅钡层内 A vs C 风化率 Fisher p={pQ:.4f}')

# 高钾层内 纹饰×风化：组合概率 + 层内置换（卡方统计量）
subg = f1[f1['类型'] == '高钾']
acg, Kag = codes(subg['纹饰'])
bcg, Kbg = codes(subg['表面风化'])
obsG, _ = chi2_np(acg, bcg, Kag, Kbg)
r = np.random.default_rng(SEED + 1400)
cnt = 0
for _ in range(N_PERM):
    r.shuffle(bcg)
    if chi2_np(acg, bcg, Kag, Kbg)[0] >= obsG - 1e-9:
        cnt += 1
pG = (cnt + 1) / (N_PERM + 1)
from math import comb
pComb = 1 / comb(18, 6)
log(f'  高钾层内 纹饰×风化: 观测χ²={obsG:.2f}, 层内置换p={pG:.4f}, 组合概率=1/C(18,6)={pComb:.2e}')

# 色系层内检验
subq3 = f1[f1['类型'] == '铅钡']
acq, Kq = codes(subq3['色系'])
bcq, Kbq = codes(subq3['表面风化'])
obsQ, _ = chi2_np(acq, bcq, Kq, Kbq)
r = np.random.default_rng(SEED + 1500)
cnt = 0
for _ in range(N_PERM):
    r.shuffle(bcq)
    if chi2_np(acq, bcq, Kq, Kbq)[0] >= obsQ - 1e-9:
        cnt += 1
pSexQ = (cnt + 1) / (N_PERM + 1)
log(f'  铅钡层内 色系×风化 置换p={pSexQ:.4f}')

subg2 = f1[f1['类型'] == '高钾']
ctG2 = pd.crosstab(subg2['色系'], subg2['表面风化']).loc[['蓝绿系', '蓝色系'],
                                                        ['无风化', '风化']]
pSexG = stats.fisher_exact(ctG2.to_numpy())[1]
log(f'  高钾层内 蓝绿系 vs 蓝色系 Fisher p={pSexG:.4f}（深暗系在高钾层无样本）')

# ---- 图：分层风化率 ----
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
for ax, (lname, order, ttl) in zip(axes, [
        ('铅钡', ['A', 'C'], f'铅钡层 (n=40)：A 与 C 无差异，Fisher p={pQ:.2f}'),
        ('高钾', ['A', 'B', 'C'], f'高钾层 (n=18)：B 档 6 件全部风化，组合概率 {pComb:.1e}')]):
    sub = f1[f1['类型'] == lname]
    g = sub.groupby('纹饰')['wx'].agg(['sum', 'count']).loc[order]
    rate = (g['sum'] / g['count'] * 100).to_numpy()
    y = np.arange(len(order))[::-1]
    ax.barh(y, rate, height=0.5, color=C1, zorder=2)
    for yi, rv in zip(y, rate):
        ax.text(rv + 2, yi, f'{rv:.1f}%', va='center', fontsize=9, color=INK2)
    ax.set_yticks(y, [f'{o} 档 (n={int(cn)})' for o, cn in zip(order, g['count'])],
                  fontsize=9.5)
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 50, 100])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.set_title(ttl, loc='left', fontsize=10)
    ax.grid(axis='x', color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(BASELINE)
    ax.tick_params(length=0)
savefig(fig, 'p1a_fig4_stratified.png')

# ==================== 收尾 ====================
log('')
log('=' * 72)
mins = (time.time() - t0) / 60
log(f'完成。耗时 {mins:.1f} min；随机种子 {SEED}；Bootstrap {N_BOOT} 次；置换 {N_PERM} 次。')
import sys
import scipy, matplotlib
log(f'环境: Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, '
    f'scipy {scipy.__version__}, matplotlib {matplotlib.__version__}')
with open(f'{OUTDIR}/p1a_run_log.txt', 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(LOG))
print(f'运行日志已写入 {OUTDIR}/p1a_run_log.txt')
