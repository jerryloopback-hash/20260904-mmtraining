# -*- coding: utf-8 -*-
"""
05_问题1_升级_相关归因与共现网络.py

在 04 的基础上继续深挖"相互关系"，并给出可直接服务问题3的选品依据：
  模块7  相关性归因分解：品类间日相关到底由什么驱动？
         三层相关(日 / 月序列 / 季节指纹)对比 + 去季节后的"纯协同"相关
  模块8  货位效率：单位毛利 × 周转速度，构建问题3(27-33单品)的选品候选池
  模块9  单品共现网络与社区发现：Louvain 社团 = 天然"蔬菜篮子"，
         与互信息对比，捕捉线性相关漏掉的非线性依赖
  模块10 替代关系网络：散装/份装的互斥强度量化

输入: data_clean/ 下的日粒度表与 04 产出的分层结果
输出: figs/Q1d_*.png + data_clean/选品候选池.csv + data_clean/共现社群.csv
"""
import os
import re
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
from sklearn.metrics import mutual_info_score

sns.set_style('whitegrid')
# 字体设置必须在 sns.set_style 之后, 否则会被重置导致中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FIG = 'figs'
OUT = 'data_clean'
os.makedirs(FIG, exist_ok=True)

CATS = ['花叶类', '花菜类', '水生根茎类', '茄类', '辣椒类', '食用菌']
CAT_C = {'花叶类': '#27ae60', '花菜类': '#16a085', '水生根茎类': '#2980b9',
         '茄类': '#8e44ad', '辣椒类': '#c0392b', '食用菌': '#e67e22'}
TIER_COLORS = {'热销': '#c0392b', '畅销': '#e67e22', '平销': '#27ae60', '滞销': '#95a5a6'}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f'{FIG}/{name}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [图] {name}')


def hr(t):
    print('\n' + '=' * 74); print(t); print('=' * 74)


daily = pd.read_pickle(f'{OUT}/日粒度_单品销量.pkl')
item = pd.read_csv(f'{OUT}/单品画像.csv', dtype={'单品编码': str})
tier_df = pd.read_csv(f'{OUT}/SKU分层结果.csv', dtype={'单品编码': str})
a1 = pd.read_csv(f'{OUT}/附件1_商品信息_clean.csv', dtype={'单品编码': str})

name_of = dict(zip(a1['单品编码'], a1['单品名称']))
cat_of = dict(zip(a1['单品编码'], a1['分类名称']))
tier_of = dict(zip(tier_df['单品编码'], tier_df['梯队']))


def base_name(nm):
    return re.sub(r'[（(][^（）()]*[)）]', '', str(nm)).strip()


# ==================================================================
# 模块 7  相关性归因分解
# ==================================================================
hr('模块7  品类相关性的归因分解：日相关到底由什么驱动？')

d = daily.copy()
d['月'] = d['销售日期'].dt.month
d['年月'] = d['销售日期'].dt.to_period('M')

P = d.pivot_table(index='销售日期', columns='分类名称', values='净销量_千克', aggfunc='sum')[CATS].fillna(0)
M = d.pivot_table(index='年月', columns='分类名称', values='净销量_千克', aggfunc='sum')[CATS].fillna(0)
mo = d.pivot_table(index='月', columns='分类名称', values='净销量_千克', aggfunc='sum')[CATS]
sh = mo / mo.sum()                      # 12维季节指纹

r_day = P.corr()
r_mon = M.corr()
r_sea = sh.corr()

iu = np.triu_indices(len(CATS), 1)
pairs = [f'{CATS[i]}-{CATS[j]}' for i, j in zip(*iu)]
cmp_df = pd.DataFrame({'日相关': r_day.values[iu], '月序列相关': r_mon.values[iu],
                       '季节指纹相似': r_sea.values[iu]}, index=pairs)

print('三层相关对比（15 个品类对）:')
print(cmp_df.round(3).to_string())
c1 = stats.pearsonr(cmp_df['日相关'], cmp_df['月序列相关'])
c2 = stats.pearsonr(cmp_df['日相关'], cmp_df['季节指纹相似'])
print(f'\n[归因1] 日相关 vs 月序列相关  : r={c1[0]:.3f} (p={c1[1]:.5f})')
print(f'[归因2] 日相关 vs 季节指纹相似: r={c2[0]:.3f} (p={c2[1]:.5f})')
print('  → 品类间日相关几乎完全由"月度层面的共同波动"决定, 日内噪声贡献很小。')

# ---- 去季节: 剔除季节相位后, 品类间还剩多少真实协同？ ----
seas_idx = sh / sh.mean()               # 月度季节指数(均值=1)
ds = P.copy()
for c in CATS:
    ds[c] = P[c] / P.index.month.map(seas_idx[c])
r_ds = ds.corr()
print('\n去季节后(日销量 ÷ 该品类月度季节指数)的相关矩阵:')
print(r_ds.round(3).to_string())
delta = pd.Series(r_day.values[iu] - r_ds.values[iu], index=pairs)
print('\n日相关 − 去季节相关 (差值越大 = 越依赖季节同步):')
print(delta.sort_values(ascending=False).round(3).to_string())
print(f'\n[归因3] 全部品类对: 日相关均值 {r_day.values[iu].mean():.3f} → 去季节后 {r_ds.values[iu].mean():.3f} '
      f'(变化 {r_ds.values[iu].mean() - r_day.values[iu].mean():+.3f})')
print('  → 总体上"不降反升": 去季节并没有摧毁品类相关, 而是改变了相关结构:')
for p_ in ['茄类-食用菌', '水生根茎类-茄类', '花叶类-茄类', '辣椒类-食用菌']:
    i_, j_ = CATS.index(p_.split('-')[0]), CATS.index(p_.split('-')[1])
    print(f'    {p_}: {r_day.iloc[i_, j_]:.3f} → {r_ds.iloc[i_, j_]:.3f} '
          f'({r_ds.iloc[i_, j_] - r_day.iloc[i_, j_]:+.3f})')
print('  → 对"冬菜-冬菜"组合, 季节同步与日内协同同向, 去季节后依旧强相关;')
print('    对"茄类(夏菜)-冬菜"组合, 反相季节掩盖了真实的日内协同, 去季节后相关显著恢复。')
print('  ★ 修正原结论: 茄类并非"需求独立", 其低相关是季节相位反相造成的表象;')
print('    剔除季节后茄类与所有品类均存在 0.30~0.40 的真实协同, 问题2 建品类联合补货模型时')
print('    应保留品类联动项, 只需显式引入各自的季节相位即可。')

# ---- 机制解释: 茄类为什么"独立" ----
print('\n[机制解释] 各品类旺季/淡季(圆形统计求得的季节重心):')
theta = 2 * np.pi * (np.arange(1, 13) - 1) / 12.0
cc_ = (sh.T.values * np.cos(theta)).sum(axis=1)
ss_ = (sh.T.values * np.sin(theta)).sum(axis=1)
peak = (np.degrees(np.arctan2(ss_, cc_)) % 360) / 30.0 + 1
# 注意: 圆形基波幅度 R 只刻画"单峰"季节性。本店月度销量呈"1月+8月"双峰,
# 基波会被抵消而趋近 0, 故此处改用"月度占比的变异系数"衡量季节强度(不限制峰形)。
seas_cv = sh.std() / sh.mean()
info = pd.DataFrame({'旺季月份(重心)': peak, '季节强度(CV)': seas_cv,
                     '旺季月(占比最大)': [int(sh[c].idxmax()) for c in CATS],
                     '淡季月(占比最小)': [int(sh[c].idxmin()) for c in CATS]}, index=CATS)
info['旺季占比'] = [sh[c].max() for c in CATS]
info['淡季占比'] = [sh[c].min() for c in CATS]
info['旺季/淡季倍数'] = info['旺季占比'] / info['淡季占比']
print(info.round(3).to_string())
_qz = info.loc['茄类']
print(f'\n  → 茄类是唯一的"夏菜"(重心 {_qz["旺季月份(重心)"]:.1f} 月, 旺季 7 月占 {_qz["旺季占比"] * 100:.1f}%, '
      f'淡季 12 月仅占 {_qz["淡季占比"] * 100:.1f}%, 旺淡倍数 {_qz["旺季/淡季倍数"]:.1f});')
print('    食用菌/水生根茎/辣椒是"冬菜"(重心 11~1 月, 6 月占比最低)。')
print('    二者季节相位接近反相, 这才是茄类与它们相关极低(月度口径甚至为负)的根本原因。')
print('  → 注意: 花叶类/辣椒类的圆形基波幅度很小(双峰: 1月与8月两个旺季), '
      '单纯的"重心月份"不足以刻画, 需结合旺淡倍数一起看。')
print(f'  → 季节强度排序(旺淡倍数): ' +
      ' > '.join(f'{c}({info.loc[c, "旺季/淡季倍数"]:.1f})'
                 for c in info['旺季/淡季倍数'].sort_values(ascending=False).index))

# ---- 图9: 三层相关对比 ----
fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
for ax, mat, t in zip(axes, [r_day, r_mon, r_sea, r_ds],
                      ['① 日销量相关', '② 月序列相关(36点)', '③ 季节指纹相似', '④ 去季节后日相关']):
    sns.heatmap(mat, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, ax=ax, cbar_kws={'shrink': .7}, annot_kws={'size': 8})
    ax.set_title(t, fontsize=10)
    ax.tick_params(axis='both', rotation=45, labelsize=8)
save(fig, 'Q1d_品类相关性三层归因.png')

# ---- 图10: 归因散点 ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(cmp_df['季节指纹相似'], cmp_df['日相关'], s=55, c='#2980b9', alpha=.75)
for k, v in cmp_df.iterrows():
    axes[0].annotate(k, (v['季节指纹相似'], v['日相关']), fontsize=7, xytext=(3, 3), textcoords='offset points')
z = np.polyfit(cmp_df['季节指纹相似'], cmp_df['日相关'], 1)
xs = np.linspace(cmp_df['季节指纹相似'].min(), cmp_df['季节指纹相似'].max(), 20)
axes[0].plot(xs, np.polyval(z, xs), 'r--', lw=1.5)
axes[0].set_xlabel('季节指纹相似度'); axes[0].set_ylabel('日销量相关系数')
axes[0].set_title(f'季节相位一致性 → 品类相关 (r={c2[0]:.3f})')

axes[1].barh(pairs, r_day.values[iu], color='#95a5a6', label='原始日相关', alpha=.85)
axes[1].barh(pairs, r_ds.values[iu], color='#c0392b', alpha=.85, label='去季节后相关')
axes[1].axvline(0, c='k', lw=.8)
axes[1].set_xlabel('相关系数'); axes[1].set_title('剔除季节性后，品类相关还剩多少？')
axes[1].legend(fontsize=9); axes[1].tick_params(labelsize=7.5)
save(fig, 'Q1d_相关性归因_季节贡献分解.png')

# ==================================================================
# 模块 8  货位效率 → 问题3 选品候选池
# ==================================================================
hr('模块8  货位效率：为问题3(27-33 个单品, 最小陈列 2.5kg)构建选品候选池')

q = pd.read_csv(f'{OUT}/SKU量利象限.csv', dtype={'单品编码': str})
lay = q.merge(tier_df[['单品编码', '梯队', '需求形态', '季节型', '生命周期状态', '动销率',
                       '日均销量_在架', '近半年有销售']],
              on='单品编码', how='left', suffixes=('', '_y'))
MIN_DISPLAY = 2.5   # 题干给定的最小陈列量(kg)

# 货位效率指标体系
#   日毛利产出速度 = 单位毛利(元/kg) × 日均销量_在架(kg/天)   → 该单品平均每天赚多少
#   货位周转速度   = 日均销量_在架 / 2.5                     → 2.5kg 货位平均每天周转几次
#   卖完一货位天数 = 2.5 / 日均销量_在架                      → 卖完一个最小货位需要几天
lay['日毛利产出'] = lay['单位毛利'] * lay['日均销量_在架']
lay['货位周转速度'] = lay['日均销量_在架'] / MIN_DISPLAY
lay['卖完一货位天数'] = MIN_DISPLAY / lay['日均销量_在架'].replace(0, np.nan)

# 只保留"近6个月仍有销售"的单品作为候选(已退市单品不适合作为长期选品)
cand = lay[lay['近半年有销售'] == True].copy()
print(f'候选池口径: 近6个月仍有销售的单品共 {len(cand)} 个'
      f'(从 {len(lay)} 个中剔除了 {len(lay) - len(cand)} 个已退市单品), '
      f'覆盖全部销量 {cand["总销量"].sum() / lay["总销量"].sum() * 100:.1f}%')

print('\n=== 按"日毛利产出"排序的 Top20（货位效率最高）===')
top = cand.sort_values('日毛利产出', ascending=False).head(20)
print(top[['单品名称', '分类名称', '单位毛利', '日均销量_在架', '日毛利产出',
           '货位周转速度', '卖完一货位天数', '梯队']].round(3).to_string(index=False))

print('\n=== 按"日毛利产出"排序的 Bottom10（货位效率最低，应避免占用货位）===')
bot = cand.sort_values('日毛利产出').head(10)
print(bot[['单品名称', '分类名称', '单位毛利', '日均销量_在架', '日毛利产出',
           '卖完一货位天数', '梯队']].round(3).to_string(index=False))

print('\n各梯队的货位效率(均值):')
print(cand.groupby('梯队').agg(单品数=('单品编码', 'count'), 日毛利产出=('日毛利产出', 'mean'),
                               单位毛利=('单位毛利', 'mean'), 货位周转速度=('货位周转速度', 'mean'),
                               卖完一货位天数=('卖完一货位天数', 'mean')
                               ).reindex(['热销', '畅销', '平销', '滞销']).round(3).to_string())

print('\n关键对比: 在候选池中, 单位毛利最高段 vs 最低段里"卖得最多"的单品')
K = max(8, len(cand) // 8)
hi = cand.sort_values('单位毛利', ascending=False).head(K)   # 只在高/低单位毛利段取
lo = cand.sort_values('单位毛利').head(K)
hi_v = hi.sort_values('总销量', ascending=False).iloc[0]
lo_v = lo.sort_values('总销量', ascending=False).iloc[0]
for tag, r in [('高毛利', hi_v), ('低毛利', lo_v)]:
    print(f'  [{tag}] {r["单品名称"]}({r["分类名称"]}): 单位毛利 {r["单位毛利"]:.2f} 元/kg, '
          f'日均 {r["日均销量_在架"]:.2f} kg/天, 日毛利产出 {r["日毛利产出"]:.2f} 元/天, '
          f'卖完 2.5kg 货位需 {r["卖完一货位天数"]:.2f} 天')
print(f'  → 两者单位毛利相差 {hi_v["单位毛利"] / max(lo_v["单位毛利"], 1e-9):.1f} 倍; '
      f'同样的 2.5kg 货位, 日毛利产出相差 {hi_v["日毛利产出"] / max(lo_v["日毛利产出"], 1e-9):.1f} 倍')

# 构建"推荐 30 个单品"：以日毛利产出为主, 同时保证品类覆盖与需求稳定性
print('\n=== 问题3 选品建议：贪心构建 30 个单品的推荐池 ===')
# 约束: 每个品类至少保留 2 个(保证品类齐全), 其余按日毛利产出从高到低补齐
N_TARGET, N_PER_CAT = 30, 2
picked, rest = [], []
for c in CATS:
    sub = cand[cand['分类名称'] == c].sort_values('日毛利产出', ascending=False)
    picked += sub.head(N_PER_CAT)['单品编码'].tolist()
    rest += sub.iloc[N_PER_CAT:]['单品编码'].tolist()
rest_df = cand.set_index('单品编码').loc[rest].sort_values('日毛利产出', ascending=False)
picked += rest_df.head(N_TARGET - len(picked)).index.tolist()
sel = cand.set_index('单品编码').loc[picked].sort_values('日毛利产出', ascending=False)
print(f'推荐池 {len(sel)} 个单品: 覆盖全部 6 品类, 销量覆盖 {sel["总销量"].sum() / lay["总销量"].sum() * 100:.1f}%, '
      f'毛利覆盖 {sel["总毛利"].sum() / lay["总毛利"].sum() * 100:.1f}%')
print('\n品类构成:')
print(sel.groupby('分类名称').agg(个数=('分类名称', 'count'),
                                 日毛利产出合计=('日毛利产出', 'sum')).round(2).to_string())
print('\n梯队构成: ' + str(sel['梯队'].value_counts().to_dict()))
print('\n推荐池明细:')
print(sel[['单品名称', '分类名称', '梯队', '需求形态', '单位毛利', '日均销量_在架',
           '日毛利产出', '卖完一货位天数']].round(3).to_string())

cand.to_csv(f'{OUT}/选品候选池.csv', index=False, encoding='utf-8-sig')
sel.reset_index().to_csv(f'{OUT}/问题3推荐选品30.csv', index=False, encoding='utf-8-sig')
print(f'\n已保存: {OUT}/选品候选池.csv, {OUT}/问题3推荐选品30.csv')

# ---- 图11: 货位效率 ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
axes[0].scatter(cand['日均销量_在架'], cand['单位毛利'], s=30, alpha=.65,
                c=[CAT_C.get(c, '#7f7f7f') for c in cand['分类名称']])
axes[0].set_xscale('log'); axes[0].set_xlabel('日均销量(kg/天, log)'); axes[0].set_ylabel('单位毛利(元/kg)')
axes[0].set_title('周转速度 × 单位毛利\n(右上=又快又赚, 左上=慢但赚)')
for nm in ['大白菜', '小米椒', '西兰花', '金针菇(盒)']:
    if nm in cand['单品名称'].values:
        r = cand[cand['单品名称'] == nm].iloc[0]
        axes[0].annotate(nm, (r['日均销量_在架'], r['单位毛利']), fontsize=8,
                         xytext=(4, 4), textcoords='offset points')

sc_ = axes[1].scatter(cand['卖完一货位天数'], cand['日毛利产出'], s=30, alpha=.7,
                      c=cand['单位毛利'], cmap='viridis')
axes[1].set_xscale('log'); axes[1].set_yscale('log')
axes[1].set_xlabel('卖完一个 2.5kg 货位所需天数(log)'); axes[1].set_ylabel('日毛利产出(元/天, log)')
axes[1].set_title('货位效率矩阵\n(左下=卖得快又赚得多, 最优)')
plt.colorbar(sc_, ax=axes[1], label='单位毛利(元/kg)')
axes[1].axvline(1, ls='--', c='r', lw=1, alpha=.6); axes[1].text(1.05, cand['日毛利产出'].max() * .5, '1天', fontsize=7)

sns.boxplot(data=cand, x='梯队', y='日毛利产出', order=['热销', '畅销', '平销', '滞销'],
            hue='梯队', palette=TIER_COLORS, legend=False, ax=axes[2], showfliers=False)
axes[2].set_yscale('log'); axes[2].set_ylabel('日毛利产出(元/天, log)'); axes[2].set_xlabel('')
axes[2].set_title('各梯队的日毛利产出分布')
save(fig, 'Q1d_货位效率与选品.png')

# ==================================================================
# 模块 9  共现网络与社区发现
# ==================================================================
hr('模块9  单品共现网络与社区发现（Louvain）')

Piz = daily.pivot_table(index='销售日期', columns='单品编码', values='净销量_千克',
                        aggfunc='sum').fillna(0.0)
# 注意: 必须用 int32/float。若用 int8, 共现天数最大可达 1085, 会溢出 int8 上限 127
# 而得到负数, 进而使 lift/Jaccard 出现荒谬的负值。
B = (Piz > 0).astype(np.int32)
N = B.shape[0]
codes = B.columns.tolist()
print(f'共现矩阵: {N} 天 × {len(codes)} 单品; 二值化后整体购买率 {B.values.mean():.3f}')

occ = B.values.sum(axis=0).astype(float)                 # 各单品出现的天数
Bv = B.values.astype(np.float64)
co = Bv.T @ Bv                                           # 共现天数矩阵
assert co.min() >= 0, '共现天数出现负值, 检查整数溢出'
with np.errstate(divide='ignore', invalid='ignore'):
    lift = co * N / np.outer(occ, occ)                   # lift = P(AB)/(P(A)P(B))
    jac = co / (occ[:, None] + occ[None, :] - co)        # Jaccard
np.fill_diagonal(lift, 0); np.fill_diagonal(jac, 0)
lift = np.nan_to_num(lift, nan=0, posinf=0, neginf=0)
jac = np.nan_to_num(jac, nan=0, posinf=0, neginf=0)
print(f'共现天数范围: [{co.min():.0f}, {co.max():.0f}]; lift 范围: [{lift.min():.3f}, {lift.max():.3f}]')

MIN_OCC, MIN_CO, MIN_LIFT = 60, 40, 1.35
keep = [i for i, c in enumerate(codes) if occ[i] >= MIN_OCC]
print(f'保留出现 ≥{MIN_OCC} 天的单品: {len(keep)} 个')
idx = np.ix_(keep, keep)
sub_codes = [codes[i] for i in keep]
co_s, lift_s, jac_s = co[idx], lift[idx], jac[idx]
n_k = len(keep)

# 建边: 共现天数足够 + lift 显著 >1
A = np.zeros((n_k, n_k))
for i in range(n_k):
    for j in range(i + 1, n_k):
        if co_s[i, j] >= MIN_CO and lift_s[i, j] >= MIN_LIFT:
            A[i, j] = A[j, i] = lift_s[i, j] - 1.0       # 边权 = 超出独立预期的部分
G = nx.from_numpy_array(A)
lab_map = {i: sub_codes[i] for i in range(n_k)}
G = nx.relabel_nodes(G, lab_map)
G.remove_nodes_from([n for n, deg in dict(G.degree()).items() if deg == 0])
print(f'共现网络: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边 '
      f'(建边条件: 共现≥{MIN_CO}天 且 lift≥{MIN_LIFT})')
if G.number_of_nodes() == 0:
    print('  [!] 建边条件过严, 自动放宽: 共现≥30天 且 lift≥1.2')
    MIN_CO, MIN_LIFT = 30, 1.2
    A = np.zeros((n_k, n_k))
    for i in range(n_k):
        for j in range(i + 1, n_k):
            if co_s[i, j] >= MIN_CO and lift_s[i, j] >= MIN_LIFT:
                A[i, j] = A[j, i] = lift_s[i, j] - 1.0
    G = nx.from_numpy_array(A)
    G = nx.relabel_nodes(G, lab_map)
    G.remove_nodes_from([n for n, deg in dict(G.degree()).items() if deg == 0])
    print(f'  放宽后: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边')

# ---- Louvain 社区发现 ----
comm = nx.community.louvain_communities(G, weight='weight', seed=42, resolution=1.0)
comm = sorted(comm, key=len, reverse=True)
mod = nx.community.modularity(G, comm, weight='weight')
print(f'\nLouvain 社区发现: {len(comm)} 个社团, 模块度 Q = {mod:.4f} '
      f'(Q>0.3 即认为社团结构显著)')

print('\n各社团构成:')
rows = []
for k, c in enumerate(comm):
    cs = list(c)
    cats = pd.Series([cat_of[x] for x in cs]).value_counts()
    vol = sum(float(cand.set_index('单品编码').loc[x, '总销量']) if x in set(cand['单品编码']) else 0.0 for x in cs)
    rows.append({'社团': k + 1, '规模': len(cs), '主导品类': cats.index[0],
                 '品类构成': ' '.join(f'{a}×{b}' for a, b in cats.items()),
                 '社团总销量': vol,
                 '代表单品': '、'.join(name_of[x] for x in cs[:5])})
cdf = pd.DataFrame(rows)
print(cdf[['社团', '规模', '主导品类', '品类构成', '代表单品']].to_string(index=False))

mem = {}
for k, c in enumerate(comm):
    for x in c:
        mem[x] = k + 1
pd.DataFrame({'单品编码': list(mem.keys()), '单品名称': [name_of[x] for x in mem],
              '分类名称': [cat_of[x] for x in mem], '社团': list(mem.values())}).to_csv(
    f'{OUT}/共现社群.csv', index=False, encoding='utf-8-sig')

# ---- 图12: 共现网络 ----
fig, ax = plt.subplots(figsize=(13, 11))
pos = nx.spring_layout(G, weight='weight', seed=42, k=1.1, iterations=60)
pal = sns.color_palette('tab10', max(len(comm), 3))
ncol = [pal[mem[n] - 1] for n in G.nodes()]
vol_all = cand.set_index('单品编码')['总销量'].to_dict()
nsz = [40 + 260 * np.sqrt(vol_all.get(n, 1) / max(vol_all.values())) for n in G.nodes()]
nx.draw_networkx_edges(G, pos, ax=ax, alpha=.28, edge_color='#7f8c8d', width=.7)
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=ncol, node_size=nsz, alpha=.88,
                       edgecolors='white', linewidths=.6)
nx.draw_networkx_labels(G, pos, ax=ax, labels={n: name_of[n] for n in G.nodes()},
                        font_size=6.5, font_family='Microsoft YaHei')
ax.set_title(f'单品共现网络（Louvain 社团, Q={mod:.3f}, 节点大小=销量, 颜色=社团）', fontsize=12)
ax.axis('off')
h = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pal[k], markersize=9,
                label=f'社团{k + 1} ({len(comm[k])}个)') for k in range(len(comm))]
ax.legend(handles=h, loc='upper left', fontsize=8, frameon=True)
save(fig, 'Q1d_单品共现网络与社团.png')

# ---- 图13: 社团 × 品类 ----
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ctb = pd.crosstab([cat_of[x] for x in mem], list(mem.values()))
sns.heatmap(ctb, annot=True, fmt='d', cmap='YlGnBu', ax=axes[0], cbar_kws={'label': '单品数'})
axes[0].set_xlabel('Louvain 社团编号'); axes[0].set_ylabel('品类')
axes[0].set_title('社团 × 品类 交叉（检验社团是否=品类）')

deg = pd.Series(dict(G.degree(weight='weight'))).sort_values(ascending=False)
top15 = deg.head(15)
axes[1].barh([name_of[x] for x in top15.index][::-1], top15.values[::-1], color='#16a085', alpha=.85)
axes[1].set_xlabel('加权度(与他人的共现强度之和)'); axes[1].set_title('网络枢纽单品 Top15\n(枢纽=购物篮中心，断货影响面最大)')
save(fig, 'Q1d_社团构成与枢纽单品.png')

# ==================================================================
# 模块 10  互信息 vs 线性相关 + 替代关系
# ==================================================================
hr('模块10  互信息(非线性依赖) vs 线性相关，以及替代关系量化')

# 对 Top40 单品计算互信息, 与 Pearson 对比
top40 = cand.sort_values('总销量', ascending=False).head(40)['单品编码'].tolist()
top40 = [c for c in top40 if c in Piz.columns]
Bt = B[top40].values
mi = np.zeros((len(top40), len(top40)))


def _entropy(v):
    p = np.bincount(v) / len(v)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


H = np.array([_entropy(Bt[:, i]) for i in range(len(top40))])
for i in range(len(top40)):
    for j in range(i + 1, len(top40)):
        m = mutual_info_score(Bt[:, i], Bt[:, j])
        # 归一化互信息 NMI = MI / sqrt(H(A)H(B))。
        # 原始 MI 受限于两个变量的边际熵: 高频单品几乎天天有售, H 很小,
        # 即使完全相关 MI 也接近 0, 会严重低估依赖, 必须归一化后才可比。
        den = np.sqrt(H[i] * H[j])
        mi[i, j] = mi[j, i] = m / den if den > 1e-9 else 0.0
pr = np.corrcoef(Piz[top40].values.T)
iut = np.triu_indices(len(top40), 1)
mi_v, pr_v = mi[iut], pr[iut]
rmi = stats.spearmanr(mi_v, pr_v)
print(f'Top40 单品({len(top40)}个): 归一化互信息(NMI) vs Pearson 的 Spearman ρ = {rmi[0]:.3f} (p={rmi[1]:.2e})')
print('  → NMI 衡量"是否在同一天出现"的依赖; Pearson 衡量"销量数值是否同涨同跌"。')
print('    两者回答的是不同问题, ρ 不接近 1 是正常且有信息量的结果。')

df_mp = pd.DataFrame({'互信息': mi_v, 'Pearson': pr_v},
                     index=[f'{name_of[top40[i]]} | {name_of[top40[j]]}' for i, j in zip(*iut)])
print('\n线性相关弱但互信息高的组合(非线性依赖，线性方法会漏掉):')
df_mp['残差'] = df_mp['互信息'].rank(pct=True) - df_mp['Pearson'].rank(pct=True)
print(df_mp.sort_values('残差', ascending=False).head(10).round(4).to_string())
print('\n互信息低但线性相关高的组合(需警惕伪相关):')
print(df_mp.sort_values('残差').head(8).round(4).to_string())

# ---- 替代关系: 散装 vs 份装 ----
print('\n[替代关系] 同款不同包装的"互斥强度":')
lp = np.log1p(Piz)
sub_rows = []
bn_of = {c: base_name(name_of[c]) for c in Piz.columns if c in name_of}
by_base = {}
for c, bn in bn_of.items():
    by_base.setdefault(bn, []).append(c)
for bn, cs in by_base.items():
    if len(cs) < 2 or bn == '':
        continue
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            a, b = cs[i], cs[j]
            if a not in lp.columns or b not in lp.columns:
                continue
            both = (Piz[a] > 0) & (Piz[b] > 0)
            if both.sum() < 20:
                continue
            r_all = stats.pearsonr(lp[a], lp[b])[0]
            r_co = stats.pearsonr(lp.loc[both, a], lp.loc[both, b])[0]
            sub_rows.append({'基名': bn, '组合': f'{name_of[a]} ↔ {name_of[b]}',
                             '共现天数': int(both.sum()),
                             'lift': lift[codes.index(a), codes.index(b)] if a in codes and b in codes else np.nan,
                             'Jaccard': jac[codes.index(a), codes.index(b)] if a in codes and b in codes else np.nan,
                             '全样本r': r_all, '共售日条件r': r_co})
sub_df = pd.DataFrame(sub_rows).sort_values('lift')
if len(sub_df):
    print(sub_df.round(3).to_string(index=False))
    print(f'\n其中 lift < 1 的有 {(sub_df["lift"] < 1).sum()} / {len(sub_df)} 组 '
          f'→ lift<1 表示两者"倾向于不同时出现", 即替代关系的直接证据')
    print(f'共售日条件相关为负的有 {(sub_df["共售日条件r"] < 0).sum()} / {len(sub_df)} 组')

# ---- 图14: 互信息 vs Pearson ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
axes[0].scatter(pr_v, mi_v, s=30, alpha=.6, c='#8e44ad')
axes[0].set_xlabel('Pearson 线性相关'); axes[0].set_ylabel('互信息(非线性依赖)')
axes[0].set_title(f'Top40 单品: 线性相关 vs 互信息 (ρ={rmi[0]:.3f})')
sns.kdeplot(x=pr_v, y=mi_v, ax=axes[0], levels=5, color='#2c3e50', linewidths=.8)

sns.heatmap(pd.DataFrame(mi, index=[name_of[c][:9] for c in top40],
                         columns=[name_of[c][:9] for c in top40]),
            cmap='magma', ax=axes[1], cbar_kws={'label': '互信息', 'shrink': .8})
axes[1].set_title('Top40 单品互信息矩阵'); axes[1].tick_params(labelsize=5.5)
save(fig, 'Q1d_互信息与线性相关对比.png')

hr('模块7-10 完成')
print(f'  {OUT}/选品候选池.csv      货位效率指标(日毛利产出/周转速度)')
print(f'  {OUT}/问题3推荐选品30.csv  贪心构建的 30 单品推荐池')
print(f'  {OUT}/共现社群.csv         Louvain 社团归属')
