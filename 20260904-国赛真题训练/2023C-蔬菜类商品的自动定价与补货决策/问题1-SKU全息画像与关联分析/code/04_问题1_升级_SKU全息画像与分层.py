# -*- coding: utf-8 -*-
"""
04_问题1_升级_SKU全息画像与分层.py

对问题1做"从销量排序到结构诊断"的升级分析：
  模块1  SKU 全息画像：对齐批发价(附件3)与损耗率(附件4)，推导真实毛利
  模块2  ABC-XYZ 二维分层矩阵（销售额累计 × 需求波动）
  模块3  熵权-TOPSIS 综合评分 → 热销/畅销/平销/滞销 四梯队
  模块4  Syntetos-Boylan 需求形态分类（ADI × CV²）
  模块5  量-利四象限矩阵（销量贡献 × 毛利贡献）
  模块6  季节指纹（圆形统计）+ 生命周期

输入: data_clean/ 下的日粒度表、单品画像、附件3、附件4
输出: figs/Q1c_*.png + data_clean/SKU全息画像.csv + 若干分层结果表
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
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

sns.set_style('whitegrid')
# 注意: sns.set_style 会重置字体, 字体设置必须在其后
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FIG = 'figs'
OUT = 'data_clean'
os.makedirs(FIG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

CATS = ['花叶类', '花菜类', '水生根茎类', '茄类', '辣椒类', '食用菌']
TIER_COLORS = {'热销': '#c0392b', '畅销': '#e67e22', '平销': '#27ae60', '滞销': '#95a5a6'}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f'{FIG}/{name}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [图] {name}')


def hr(title):
    print('\n' + '=' * 74)
    print(title)
    print('=' * 74)


# ==================================================================
# 模块 1  SKU 全息画像：把"量"升级为"量 + 利"
# ==================================================================
hr('模块1  SKU 全息画像（对齐批发价与损耗率，推导真实毛利）')

a1 = pd.read_csv(f'{OUT}/附件1_商品信息_clean.csv', dtype={'单品编码': str, '分类编码': str})
a3 = pd.read_csv(f'{OUT}/附件3_批发价_clean.csv', dtype={'单品编码': str})
a4 = pd.read_csv(f'{OUT}/附件4_损耗率_clean.csv', dtype={'单品编码': str})
daily = pd.read_pickle(f'{OUT}/日粒度_单品销量.pkl')
item = pd.read_csv(f'{OUT}/单品画像.csv', dtype={'单品编码': str, '分类编码': str})

a3['日期'] = pd.to_datetime(a3['日期'])
a3 = a3.dropna(subset=['批发价格(元/千克)'])
a3 = a3[a3['批发价格(元/千克)'] > 0]
print(f'附件3 有效批发价记录: {len(a3)} 条, 覆盖单品 {a3["单品编码"].nunique()} 个, '
      f'日期 {a3["日期"].min().date()} ~ {a3["日期"].max().date()}')

# ---- 构造 日期×单品 的批发价面板（前向填充补齐）----
wp = a3.pivot_table(index='日期', columns='单品编码', values='批发价格(元/千克)', aggfunc='mean')
full_dates = pd.date_range(daily['销售日期'].min(), daily['销售日期'].max(), freq='D')
wp = wp.reindex(full_dates)
item_med = wp.median()
n_missing_before = int(wp.isna().sum().sum())
# 先按单品中位数兜底(该单品可能从未出现在附件3), 再时间前向/后向填充
wp = wp.fillna(item_med).ffill().bfill()
wp = wp.fillna(item_med).ffill().bfill()
glob_med = float(a3['批发价格(元/千克)'].median())
wp = wp.fillna(glob_med)
print(f'批发价面板: {wp.shape[0]}天 × {wp.shape[1]}单品, 原始缺失 {n_missing_before} 格 '
      f'({n_missing_before / (wp.shape[0] * wp.shape[1]) * 100:.1f}%), 已用 中位数+前后向填充 补齐')

wpl = wp.stack(future_stack=True).rename('批发价')
wpl.index.names = ['销售日期', '单品编码']

d = daily.merge(wpl.reset_index(), on=['销售日期', '单品编码'], how='left')
d['批发价'] = d['批发价'].fillna(d['单品编码'].map(item_med)).fillna(glob_med)
print(f'流水-批发价匹配率: {d["批发价"].notna().mean() * 100:.2f}%')

# ---- 成本与毛利模型 ----
# 设进货 Q 千克, 损耗率 λ, 可售 Q(1-λ); 为卖出 S 千克需进货 S/(1-λ)
# 收入 = S·p ; 成本 = S·w/(1-λ) ; 毛利 = S·( p - w/(1-λ) )
loss = dict(zip(a4['单品编码'], a4['损耗率(%)'] / 100.0))
d['损耗率'] = d['单品编码'].map(loss).fillna(0.0)
d['有效成本'] = d['批发价'] / (1.0 - d['损耗率'])          # 每卖出 1kg 的真实进货成本
d['单位毛利'] = d['加权销售均价'] - d['有效成本']
d['毛利额'] = d['净销量_千克'] * d['单位毛利']
d['损耗成本'] = d['净销量_千克'] * d['批发价'] * d['损耗率'] / (1.0 - d['损耗率'])

# ---- SKU 级画像 ----
def build_profile(g):
    g = g.sort_values('销售日期')
    q = g['净销量_千克'].values
    nz = q[q > 0]
    span_days = (g['销售日期'].max() - g['销售日期'].min()).days + 1
    act_days = int((q > 0).sum())
    return pd.Series({
        '首销日': g['销售日期'].min(),
        '末销日': g['销售日期'].max(),
        '在架天数': span_days,
        '有销售天数': act_days,
        '总销量': q.sum(),
        '总销售额': g['净销售额_元'].sum(),
        '总毛利': g['毛利额'].sum(),
        '日均销量_在架': q.sum() / span_days,
        '日均销量_有售': q.sum() / act_days if act_days else 0.0,
        '日销量标准差': q.std(ddof=1) if span_days > 1 else 0.0,
        '非零日均': nz.mean() if len(nz) else 0.0,
        '非零日标准差': nz.std(ddof=1) if len(nz) > 1 else 0.0,
        '单日最大销量': q.max(),
        '平均批发价': g['批发价'].mean(),
        '平均售价': g['净销售额_元'].sum() / q.sum() if q.sum() else np.nan,
        '损耗率': g['损耗率'].iloc[0],
        '单位毛利': g['毛利额'].sum() / q.sum() if q.sum() else np.nan,
        '交易笔数': g['交易笔数'].sum(),
    })


prof = d.groupby('单品编码').apply(build_profile, include_groups=False).reset_index()
prof = prof.merge(a1[['单品编码', '单品名称', '分类名称']], on='单品编码', how='left')

# ---- 派生比率指标 ----
prof['动销率'] = prof['有销售天数'] / prof['在架天数']           # 广度: 被购买的频繁程度
prof['变异系数CV'] = prof['日销量标准差'] / prof['日均销量_在架'].replace(0, np.nan)  # 波动(含零)
prof['非零CV'] = prof['非零日标准差'] / prof['非零日均'].replace(0, np.nan)           # 波动(不含零)
prof['ADI'] = prof['在架天数'] / prof['有销售天数'].replace(0, np.nan)                # 平均需求间隔
prof['CV2'] = prof['非零CV'] ** 2
prof['稳定性'] = 1.0 / (1.0 + prof['变异系数CV'])
prof['毛利率'] = prof['总毛利'] / prof['总销售额'].replace(0, np.nan)
prof['笔均重量'] = prof['总销量'] / prof['交易笔数'].replace(0, np.nan)

prof = prof.replace([np.inf, -np.inf], np.nan)
sold = prof[prof['总销量'] > 0].copy()
print(f'\nSKU 画像构建完成: {len(prof)} 个单品, 其中 {len(sold)} 个有实际销售')
print(f'毛利率中位数 {prof["毛利率"].median():.3f}, 单位毛利中位数 {prof["单位毛利"].median():.3f} 元/kg')
print(f'亏损销售(单位毛利<0)的单品数: {(prof["单位毛利"] < 0).sum()} 个')

hr('模块1 产出：各品类的量-价-本-利结构')
struct = prof.groupby('分类名称').agg(
    单品数=('单品编码', 'count'),
    总销量=('总销量', 'sum'),
    总销售额=('总销售额', 'sum'),
    总毛利=('总毛利', 'sum'),
    平均损耗率=('损耗率', 'mean'),
    毛利率=('毛利率', 'median'),
).reindex(CATS)
struct['销量占比'] = struct['总销量'] / struct['总销量'].sum()
struct['销售额占比'] = struct['总销售额'] / struct['总销售额'].sum()
struct['毛利占比'] = struct['总毛利'] / struct['总毛利'].sum()
struct['毛利率_整体'] = struct['总毛利'] / struct['总销售额']
print(struct[['单品数', '销量占比', '销售额占比', '毛利占比', '毛利率_整体', '平均损耗率']].round(4).to_string())
mis = struct['毛利占比'] - struct['销售额占比']
print('\n毛利占比 - 销售额占比(>0 说明该品类"更赚钱"):')
print(mis.round(4).to_string())
prof.to_csv(f'{OUT}/SKU全息画像.csv', index=False, encoding='utf-8-sig')
print(f'\n已保存: {OUT}/SKU全息画像.csv')

# ---- 图1: 量-额-利 三者的品类错位 ----
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, col, t in zip(axes, ['销量占比', '销售额占比', '毛利占比'],
                      ['销量份额', '销售额份额', '毛利份额']):
    v = struct[col].sort_values(ascending=False)
    ax.bar(v.index, v.values * 100, color=sns.color_palette('Set2', len(v)))
    ax.set_title(f'品类{t}'); ax.set_ylabel('%'); ax.tick_params(axis='x', rotation=30)
    for i, x in enumerate(v.values * 100):
        ax.text(i, x + 0.8, f'{x:.1f}', ha='center', fontsize=9)
    ax.set_ylim(0, max(v.values * 100) * 1.18)
save(fig, 'Q1c_品类_销量_销售额_毛利_三者错位.png')

# ==================================================================
# 模块 2  ABC-XYZ 二维分层矩阵
# ==================================================================
hr('模块2  ABC-XYZ 二维分层矩阵（销售额累计占比 × 需求波动性）')

abc = sold.sort_values('总销售额', ascending=False).copy()
cum = abc['总销售额'].cumsum() / abc['总销售额'].sum()


def abc_tag(c):
    if c <= 0.70:
        return 'A'
    if c <= 0.90:
        return 'B'
    return 'C'


abc['ABC'] = cum.map(abc_tag)

# XYZ 的需求波动应建立在"补货决策粒度"上。日粒度需求高度零膨胀,
# 直接算日 CV 会使几乎所有单品都落入 Z 类(实测日 CV 均值高达 6.9),
# 故按"周"聚合后再计算变异系数 —— 这是 ABC-XYZ 的标准做法。
d['周'] = d['销售日期'].dt.to_period('W').dt.start_time
wk = d.groupby(['单品编码', '周'])['净销量_千克'].sum().reset_index()
# 只统计单品在架期内的周(首尾销售日之间), 避免把未上架期算成零需求
span = prof.set_index('单品编码')[['首销日', '末销日']]
wk = wk.merge(span, left_on='单品编码', right_index=True, how='left')
wk = wk[(wk['周'] >= wk['首销日']) & (wk['周'] <= wk['末销日'])]
wstat = wk.groupby('单品编码')['净销量_千克'].agg(['mean', 'std', 'count'])
wstat['周CV'] = wstat['std'] / wstat['mean'].replace(0, np.nan)
sold = sold.merge(wstat[['周CV']], left_on='单品编码', right_index=True, how='left')
print(f'周聚合后需求变异系数 CV 中位数 = {sold["周CV"].median():.3f} '
      f'(日粒度 CV 中位数 = {sold["变异系数CV"].median():.3f}, 说明日粒度 CV 被零膨胀严重放大)')


def xyz_tag(cv):
    if pd.isna(cv):
        return 'Z'
    if cv < 0.5:
        return 'X'
    if cv < 1.0:
        return 'Y'
    return 'Z'


abc['XYZ'] = abc['变异系数CV'].map(xyz_tag)
abc = abc.merge(sold[['单品编码', '周CV']], on='单品编码', how='left')
abc['XYZ'] = abc['周CV'].map(xyz_tag)
abc['ABCXYZ'] = abc['ABC'] + abc['XYZ']

print('ABC 分层:')
print(abc.groupby('ABC').agg(单品数=('单品编码', 'count'), 销售额占比=('总销售额', lambda s: s.sum()),
                             销量占比=('总销量', lambda s: s.sum())).assign(
    销售额占比=lambda x: x['销售额占比'] / abc['总销售额'].sum(),
    销量占比=lambda x: x['销量占比'] / abc['总销量'].sum()).round(4).to_string())
print('\nXYZ 分层(按周聚合需求的变异系数 CV):')
print(abc.groupby('XYZ').agg(单品数=('单品编码', 'count'), 周CV均值=('周CV', 'mean'),
                             动销率均值=('动销率', 'mean')).round(3).to_string())

ct = pd.crosstab(abc['ABC'], abc['XYZ']).reindex(index=['A', 'B', 'C'], columns=['X', 'Y', 'Z']).fillna(0).astype(int)
print('\nABC × XYZ 交叉表(单元格=单品数):')
print(ct.to_string())
print('\n各格销售额占比(%):')
cs = pd.crosstab(abc['ABC'], abc['XYZ'], values=abc['总销售额'], aggfunc='sum').reindex(
    index=['A', 'B', 'C'], columns=['X', 'Y', 'Z']).fillna(0)
print((cs / abc['总销售额'].sum() * 100).round(2).to_string())

STRATEGY = {
    'AX': '核心主力·需求平稳 → 高服务水平, 定量补货(ROP), 可签长期供货协议',
    'AY': '核心主力·需求波动 → 提高安全库存, 动态补货, 优先保障不缺货',
    'AZ': '核心主力·需求剧烈波动 → 快速响应+柔性补货, 高价时段限量, 严控损耗',
    'BX': '中坚商品·需求平稳 → 常规周期补货, 维持基准库存',
    'BY': '中坚商品·需求波动 → 滚动预测补货, 适度安全库存',
    'BZ': '中坚商品·需求剧烈波动 → 小批量多频次, 降价清货为主',
    'CX': '长尾商品·需求平稳 → 低量常备, 满足一站式购物即可',
    'CY': '长尾商品·需求波动 → 按需订购, 减少陈列面积',
    'CZ': '长尾商品·需求剧烈波动 → 末位候选淘汰, 仅在旺季临时上架',
}
print('\n九宫格补货/定价策略映射:')
for k in ['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ', 'CX', 'CY', 'CZ']:
    n = int(ct.loc[k[0], k[1]])
    print(f'  {k}(n={n:3d}): {STRATEGY[k]}')

# ---- 图2: ABC-XYZ 热力图 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar_kws={'label': '单品数'})
axes[0].set_title('ABC × XYZ 单品数分布'); axes[0].set_ylabel('ABC(销售额贡献)'); axes[0].set_xlabel('XYZ(需求波动)')
sns.heatmap(cs / abc['总销售额'].sum() * 100, annot=True, fmt='.1f', cmap='Oranges', ax=axes[1],
            cbar_kws={'label': '销售额占比(%)'})
axes[1].set_title('ABC × XYZ 销售额贡献(%)'); axes[1].set_ylabel(''); axes[1].set_xlabel('XYZ(需求波动)')
save(fig, 'Q1c_ABCXYZ二维分层矩阵.png')

# ==================================================================
# 模块 3  熵权-TOPSIS 四梯队：热销 / 畅销 / 平销 / 滞销
# ==================================================================
hr('模块3  熵权-TOPSIS 综合评分 → 热销/畅销/平销/滞销 四梯队')

# 四个正交维度: 规模 / 广度 / 强度 / 稳定 (先对数化压缩长尾, 避免被极端值主导)
F = pd.DataFrame(index=sold.index)
F['规模'] = np.log1p(sold['总销量'])
F['广度'] = sold['动销率']
F['强度'] = np.log1p(sold['日均销量_有售'])
F['稳定'] = sold['稳定性']
F = F.replace([np.inf, -np.inf], np.nan).fillna(0.0)

print('四个维度的相关矩阵(检验是否正交/共线):')
print(F.corr().round(3).to_string())


def entropy_weight_topsis(X):
    """熵权法确定权重 + TOPSIS 综合评分。X: DataFrame, 指标均为正向(越大越好)。"""
    Z = (X - X.min()) / (X.max() - X.min())
    Z = Z.clip(lower=1e-6)
    P = Z.div(Z.sum(axis=0), axis=1)
    k = 1.0 / np.log(len(Z))
    E = -k * (P * np.log(P)).sum(axis=0)
    d = 1.0 - E
    W = d / d.sum()
    V = Z * W
    vpos, vneg = V.max(), V.min()
    Dp = np.sqrt(((V - vpos) ** 2).sum(axis=1))
    Dn = np.sqrt(((V - vneg) ** 2).sum(axis=1))
    C = Dn / (Dp + Dn)
    return C, W


score, W = entropy_weight_topsis(F)
sold['TOPSIS得分'] = score
print('\n熵权法得到的指标权重(基于各维度信息熵的差异程度):')
for k, v in W.items():
    print(f'  {k}: 权重 {v:.4f}')

# 按得分降序划分四梯队 (10% / 20% / 35% / 35%)
sold = sold.sort_values('TOPSIS得分', ascending=False).reset_index(drop=True)
n = len(sold)
cuts = [int(n * 0.10), int(n * 0.30), int(n * 0.65)]
tiers = np.array(['平销'] * n, dtype=object)
tiers[:cuts[0]] = '热销'
tiers[cuts[0]:cuts[1]] = '畅销'
tiers[cuts[1]:cuts[2]] = '平销'
tiers[cuts[2]:] = '滞销'
sold['梯队'] = tiers
sold['排名'] = np.arange(1, n + 1)

print('\n四梯队画像:')
tp = sold.groupby('梯队').agg(
    单品数=('单品编码', 'count'),
    销量占比=('总销量', lambda s: s.sum() / sold['总销量'].sum()),
    销售额占比=('总销售额', lambda s: s.sum() / sold['总销售额'].sum()),
    毛利占比=('总毛利', lambda s: s.sum() / sold['总毛利'].sum()),
    总销量均值=('总销量', 'mean'),
    动销率均值=('动销率', 'mean'),
    日均销量均值=('日均销量_有售', 'mean'),
    CV均值=('变异系数CV', 'mean'),
    TOPSIS均值=('TOPSIS得分', 'mean'),
).reindex(['热销', '畅销', '平销', '滞销'])
print(tp.round(4).to_string())

# 单调性检验: 梯队是否随销售表现单调
rho, p = stats.spearmanr(sold['TOPSIS得分'], sold['总销量'])
print(f'\n[检验] TOPSIS得分 vs 总销量 Spearman ρ={rho:.3f} (p={p:.2e}) '
      f'→ 相关但不是同一件事, 分层不是"按销量重排"')
rho2, p2 = stats.spearmanr(sold['TOPSIS得分'], sold['动销率'])
print(f'[检验] TOPSIS得分 vs 动销率   Spearman ρ={rho2:.3f} (p={p2:.2e})')
for c in ['规模', '广度', '强度', '稳定']:
    f = [sold[sold['梯队'] == t][c].mean() for t in ['热销', '畅销', '平销', '滞销']] if c in sold.columns else None
    if f is not None:
        print(f'  维度"{c}" 四梯队均值: ' + ' > '.join(f'{x:.3f}' for x in f))

print('\n各梯队代表单品(每梯队取 TOPSIS 首尾各5个):')
for t in ['热销', '畅销', '平销', '滞销']:
    sub = sold[sold['梯队'] == t]
    print(f'\n-- {t} (n={len(sub)}) 头部5 --')
    print(sub.head(5)[['单品名称', '分类名称', '总销量', '动销率', '日均销量_有售', '变异系数CV', 'TOPSIS得分']]
          .round(3).to_string(index=False))

# 梯队 × 品类 交叉
print('\n梯队 × 品类 交叉表(单品数):')
print(pd.crosstab(sold['梯队'], sold['分类名称']).reindex(
    index=['热销', '畅销', '平销', '滞销'])[CATS].to_string())

# ---- 图3: 四梯队可视化 ----
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for t in ['热销', '畅销', '平销', '滞销']:
    sub = sold[sold['梯队'] == t]
    axes[0].scatter(sub['总销量'], sub['日均销量_有售'], s=32, alpha=.75, c=TIER_COLORS[t], label=f'{t}(n={len(sub)})')
axes[0].set_xscale('log'); axes[0].set_yscale('log')
axes[0].set_xlabel('总销量(kg, log)'); axes[0].set_ylabel('日均销量(有售日, kg, log)')
axes[0].set_title('① 规模 × 强度'); axes[0].legend(fontsize=9)

for t in ['热销', '畅销', '平销', '滞销']:
    sub = sold[sold['梯队'] == t]
    axes[1].scatter(sub['动销率'], sub['变异系数CV'], s=32, alpha=.75, c=TIER_COLORS[t], label=t)
axes[1].set_xlabel('动销率(有销售天数/在架天数)'); axes[1].set_ylabel('需求变异系数 CV')
axes[1].set_title('② 广度 × 波动'); axes[1].set_yscale('log'); axes[1].legend(fontsize=9)

sh = tp[['销量占比', '销售额占比', '毛利占比']] * 100
sh.plot(kind='bar', ax=axes[2], color=['#3498db', '#9b59b6', '#f1c40f'], rot=0)
axes[2].set_title('③ 各梯队的销量/销售额/毛利贡献占比'); axes[2].set_ylabel('%')
save(fig, 'Q1c_四梯队_规模强度广度波动与贡献.png')

# ---- 图4: 帕累托 + 梯队分界 ----
fig, ax = plt.subplots(figsize=(11, 5.2))
cs_ratio = sold['总销量'].cumsum() / sold['总销量'].sum() * 100
ax.plot(np.arange(1, n + 1), cs_ratio, color='#c0392b', lw=2, label='销量累计占比')
ax.fill_between(np.arange(1, n + 1), 0, cs_ratio, color='#c0392b', alpha=.12)
for c, lab in zip(cuts, ['热销|畅销', '畅销|平销', '平销|滞销']):
    ax.axvline(c, ls='--', lw=1.2, color='#2c3e50')
    ax.text(c, 3, f' {lab}\n ({c})', fontsize=8, rotation=0, color='#2c3e50')
ax.axhline(80, ls=':', color='gray', lw=1); ax.text(n * .55, 82, '80% 销量线', fontsize=8, color='gray')
ax.set_xlabel('单品按 TOPSIS 得分降序排名'); ax.set_ylabel('累计销量占比(%)')
ax.set_title('SKU 帕累托曲线与四梯队分界'); ax.legend(loc='lower right'); ax.set_xlim(0, n)
save(fig, 'Q1c_四梯队_帕累托曲线.png')

print('\n[洞察] 前 %.0f%% 的"热销"单品贡献 %.1f%% 销量、%.1f%% 销售额、%.1f%% 毛利'
      % (len(sold[sold['梯队'] == '热销']) / n * 100,
         tp.loc['热销', '销量占比'] * 100, tp.loc['热销', '销售额占比'] * 100, tp.loc['热销', '毛利占比'] * 100))

# ==================================================================
# 模块 4  Syntetos-Boylan 需求形态分类
# ==================================================================
hr('模块4  Syntetos-Boylan 需求形态分类（ADI × CV²）')

ADI_CUT, CV2_CUT = 1.32, 0.49


def sb_tag(row):
    adi, cv2 = row['ADI'], row['CV2']
    if pd.isna(adi) or pd.isna(cv2):
        return 'Lumpy'
    if adi < ADI_CUT and cv2 < CV2_CUT:
        return 'Smooth'
    if adi >= ADI_CUT and cv2 < CV2_CUT:
        return 'Intermittent'
    if adi < ADI_CUT and cv2 >= CV2_CUT:
        return 'Erratic'
    return 'Lumpy'


sold['需求形态'] = sold.apply(sb_tag, axis=1)
SB_NAME = {'Smooth': '平滑型', 'Intermittent': '间断型', 'Erratic': ' erratic型', 'Lumpy': '块状型'}
SB_MODEL = {
    'Smooth': '移动平均 / 指数平滑(Holt), 定量补货模型(ROP)',
    'Intermittent': 'Croston 法 / SBA, 按需小批量补货',
    'Erratic': '带安全库存的指数平滑, 提高服务水平应对波动',
    'Lumpy': 'Croston / Bootstrap, 短保质期下应"少备勤补", 优先降价清货',
}
print('需求形态分布:')
sb = sold.groupby('需求形态').agg(
    单品数=('单品编码', 'count'),
    ADI均值=('ADI', 'mean'),
    CV2均值=('CV2', 'mean'),
    销量占比=('总销量', lambda s: s.sum() / sold['总销量'].sum()),
    动销率均值=('动销率', 'mean'),
)
sb['占比'] = sb['单品数'] / sb['单品数'].sum()
print(sb.round(3).to_string())
print('\n各形态对应的预测/补货模型:')
for k, v in SB_MODEL.items():
    nk = sb.loc[k, '单品数'] if k in sb.index else 0
    print(f'  {k}({SB_NAME[k]}, n={nk}): {v}')

print('\n需求形态 × 梯队 交叉(单品数):')
print(pd.crosstab(sold['需求形态'], sold['梯队']).reindex(
    index=['Smooth', 'Intermittent', 'Erratic', 'Lumpy'],
    columns=['热销', '畅销', '平销', '滞销']).fillna(0).astype(int).to_string())

print('\n各品类的需求形态构成(%):')
sbc = pd.crosstab(sold['分类名称'], sold['需求形态'], normalize='index').reindex(
    index=CATS, columns=['Smooth', 'Intermittent', 'Erratic', 'Lumpy']).fillna(0) * 100
print(sbc.round(1).to_string())

# ---- 图5: ADI × CV² 四象限散点 ----
fig, ax = plt.subplots(figsize=(10, 6.5))
cc = {'Smooth': '#27ae60', 'Intermittent': '#2980b9', 'Erratic': '#e67e22', 'Lumpy': '#c0392b'}
sub2 = sold[sold['CV2'] > 0]
for k in ['Smooth', 'Intermittent', 'Erratic', 'Lumpy']:
    s3 = sub2[sub2['需求形态'] == k]
    ax.scatter(s3['ADI'], s3['CV2'], s=34, alpha=.7, c=cc[k], label=f'{k}({SB_NAME[k]}, n={len(s3)})')
ax.axvline(ADI_CUT, ls='--', c='k', lw=1.1)
ax.axhline(CV2_CUT, ls='--', c='k', lw=1.1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('ADI 平均需求间隔(在架天数/有销售天数, log)')
ax.set_ylabel('CV² 非零需求变异系数平方(log)')
ax.set_title('Syntetos-Boylan 需求形态四象限（决定补货模型选择）')
ax.legend(loc='upper left', fontsize=9)
for k, (x, y) in {'平滑: 天天卖、量稳': (1.05, 0.12), '间断: 隔几天卖、量稳': (2.2, 0.12),
                  '波动: 天天卖、量忽高忽低': (1.05, 1.2), '块状: 隔几天卖、量忽高忽低': (2.2, 1.2)}.items():
    ax.text(x, y, k, fontsize=7.5, color='#34495e', alpha=.85)
save(fig, 'Q1c_需求形态_SyntetosBoylan四象限.png')

# ==================================================================
# 模块 5  量-利四象限矩阵
# ==================================================================
hr('模块5  量-利四象限矩阵（销量贡献 × 盈利能力）')

q = sold[sold['总毛利'].notna()].copy()
# 关键设计: 两轴必须解耦。绝对销量与绝对毛利高度共线(都会被少数爆品主导),
# 故 X 轴用"销量份额"(规模), Y 轴用"毛利率"(单位盈利能力), 二者正交。
q['销量份额'] = q['总销量'] / q['总销量'].sum()
qx = q['销量份额'].median()
qy = q['毛利率'].median()


def quad(r):
    if r['销量份额'] >= qx and r['毛利率'] >= qy:
        return '明星品(量大且利厚)'
    if r['销量份额'] >= qx and r['毛利率'] < qy:
        return '引流品(量大但利薄)'
    if r['销量份额'] < qx and r['毛利率'] >= qy:
        return '利润品(量小但利厚)'
    return '问题品(量小且利薄)'


q['象限'] = q.apply(quad, axis=1)
QUAD_STR = {
    '明星品(量大且利厚)': '核心基本盘: 优先保障货架与补货, 定价稳健, 是门店量利双主力',
    '引流品(量大但利薄)': '引流担当: 价格敏感, 承担带动客流职能; 应与高毛利品做关联陈列/组合定价',
    '利润品(量小但利厚)': '利润担当: 需求未被充分挖掘, 应加大陈列与推荐, 借引流品带动其销量',
    '问题品(量小且利薄)': '末位候选: 压缩陈列面积, 或作为淘汰/替换的首要对象',
}
qs = q.groupby('象限').agg(
    单品数=('单品编码', 'count'),
    销量占比=('总销量', lambda s: s.sum() / q['总销量'].sum()),
    毛利占比=('总毛利', lambda s: s.sum() / q['总毛利'].sum()),
    毛利率均值=('毛利率', 'mean'),
    单位毛利均值=('单位毛利', 'mean'),
)
qs['单品数占比'] = qs['单品数'] / qs['单品数'].sum()
print(f'分界线: 销量份额中位数={qx * 100:.3f}%, 毛利率中位数={qy * 100:.2f}%')
print(qs.round(4).to_string())
print('\n各象限策略:')
for k, v in QUAD_STR.items():
    nk = int(qs.loc[k, '单品数']) if k in qs.index else 0
    print(f'  {k}(n={nk}): {v}')

print('\n明星品 Top12(量利双主力):')
print(q[q['象限'] == '明星品(量大且利厚)'].sort_values('总毛利', ascending=False).head(12)[
    ['单品名称', '分类名称', '总销量', '总毛利', '毛利率', '单位毛利', '梯队']].round(3).to_string(index=False))
print('\n利润品 Top10(量小但毛利率高, 最值得扶持):')
print(q[q['象限'] == '利润品(量小但利厚)'].sort_values('总毛利', ascending=False).head(10)[
    ['单品名称', '分类名称', '总销量', '总毛利', '毛利率', '单位毛利', '梯队']].round(3).to_string(index=False))
print('\n引流品 Top10(走量但毛利率薄):')
print(q[q['象限'] == '引流品(量大但利薄)'].sort_values('总销量', ascending=False).head(10)[
    ['单品名称', '分类名称', '总销量', '总毛利', '毛利率', '单位毛利']].round(3).to_string(index=False))

# ---- 5.2 贡献错位: 毛利份额 - 销量份额 ----
print('\n[贡献错位分析] 毛利份额 - 销量份额 (>0 = 以较少的量贡献了更多的利):')
q['毛利份额'] = q['总毛利'] / q['总毛利'].sum()
q['贡献错位'] = q['毛利份额'] - q['销量份额']
mis_top = q.sort_values('贡献错位', ascending=False).head(12)
mis_bot = q.sort_values('贡献错位').head(12)
print('\n被低估的"隐形利润品"(错位最大, 建议增加陈列):')
print(mis_top[['单品名称', '分类名称', '销量份额', '毛利份额', '贡献错位', '毛利率']].round(5).to_string(index=False))
print('\n"虚胖"的走量品(占用大量销量却贡献不足, 注意其引流价值):')
print(mis_bot[['单品名称', '分类名称', '销量份额', '毛利份额', '贡献错位', '毛利率']].round(5).to_string(index=False))

# ---- 图6: 量-利四象限 ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
qc = {'明星品(量大且利厚)': '#c0392b', '引流品(量大但利薄)': '#2980b9',
      '利润品(量小但利厚)': '#f39c12', '问题品(量小且利薄)': '#95a5a6'}
ax = axes[0]
for k in ['明星品(量大且利厚)', '引流品(量大但利薄)', '利润品(量小但利厚)', '问题品(量小且利薄)']:
    s4 = q[q['象限'] == k]
    ax.scatter(s4['销量份额'] * 100, s4['毛利率'] * 100, s=42, alpha=.72, c=qc[k], label=f'{k} (n={len(s4)})')
ax.axvline(qx * 100, ls='--', c='k', lw=1.1, alpha=.6)
ax.axhline(qy * 100, ls='--', c='k', lw=1.1, alpha=.6)
ax.set_xscale('log')
ax.set_xlabel('销量份额(%, log)'); ax.set_ylabel('毛利率(%)')
ax.set_title('量-利四象限矩阵（已扣除批发成本与损耗）')
ax.legend(fontsize=9, loc='best')
for k, (xx, yy) in {'明星品': (qx * 260, qy * 128), '引流品': (qx * 260, qy * 62),
                    '利润品': (qx * 22, qy * 128), '问题品': (qx * 22, qy * 62)}.items():
    ax.text(xx, yy, k, fontsize=12, fontweight='bold', color='#2c3e50', alpha=.45)

ax2b = axes[1]
ms = q.sort_values('贡献错位')
ax2b.barh(np.arange(len(ms)), ms['贡献错位'] * 100,
          color=['#c0392b' if v > 0 else '#2980b9' for v in ms['贡献错位']], alpha=.75)
ax2b.axvline(0, c='k', lw=1)
ax2b.set_yticks([]); ax2b.set_xlabel('毛利份额 − 销量份额 (百分点)')
ax2b.set_title('贡献错位谱（红=隐形利润品, 蓝=虚胖走量品）')
save(fig, 'Q1c_量利四象限矩阵.png')

# ---- 图7: 单位毛利 vs 损耗率 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(prof['损耗率'] * 100, prof['单位毛利'], s=26, alpha=.65, c='#16a085')
axes[0].axhline(0, ls='--', c='r', lw=1)
axes[0].set_xlabel('损耗率(%)'); axes[0].set_ylabel('单位毛利(元/kg)')
axes[0].set_title('损耗率 × 单位毛利')
r, pv = stats.pearsonr(prof['损耗率'], prof['单位毛利'])
axes[0].text(0.02, 0.94, f'Pearson r={r:.3f} (p={pv:.2e})', transform=axes[0].transAxes, fontsize=9)
axes[1].scatter(prof['平均售价'], prof['单位毛利'], s=26, alpha=.65, c='#8e44ad')
axes[1].axhline(0, ls='--', c='r', lw=1)
axes[1].set_xlabel('平均售价(元/kg)'); axes[1].set_ylabel('单位毛利(元/kg)')
axes[1].set_title('售价 × 单位毛利（高价≠高毛利）')
r2, pv2 = stats.pearsonr(prof['平均售价'].fillna(0), prof['单位毛利'].fillna(0))
axes[1].text(0.02, 0.94, f'Pearson r={r2:.3f} (p={pv2:.2e})', transform=axes[1].transAxes, fontsize=9)
save(fig, 'Q1c_损耗率与售价对单位毛利的影响.png')

# ==================================================================
# 模块 6  季节指纹（圆形统计）+ 生命周期
# ==================================================================
hr('模块6  季节指纹（圆形统计）与生命周期')

d['月'] = d['销售日期'].dt.month
mon = d.pivot_table(index='单品编码', columns='月', values='净销量_千克', aggfunc='sum').reindex(
    columns=range(1, 13)).fillna(0.0)
mon_share = mon.div(mon.sum(axis=1).replace(0, np.nan), axis=0)

# 归一化季节熵: 1=全年均匀, 0=极度集中
def norm_entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(12))


# 圆形统计: 把月份映射到圆周求"季节重心"与"季节集中度"
theta = 2 * np.pi * (np.arange(1, 13) - 1) / 12.0
cosv = (mon_share.fillna(0).values * np.cos(theta)).sum(axis=1)
sinv = (mon_share.fillna(0).values * np.sin(theta)).sum(axis=1)
R = np.sqrt(cosv ** 2 + sinv ** 2)                      # 季节集中度 0~1
peak = (np.degrees(np.arctan2(sinv, cosv)) % 360) / 30.0 + 1   # 旺季月份 1~12

seas = pd.DataFrame({
    '单品编码': mon.index,
    '季节集中度': R,
    '旺季月份': peak,
    '季节均衡度': [norm_entropy(mon_share.loc[c].values) for c in mon.index],
})
sold = sold.merge(seas, on='单品编码', how='left')

print('季节集中度分布(0=全年均匀, 1=完全集中在单月):')
print(sold['季节集中度'].describe().round(3).to_string())
print('\n各品类的平均季节集中度与旺季月份:')
sc = sold.groupby('分类名称').agg(季节集中度均值=('季节集中度', 'mean'),
                                 季节均衡度均值=('季节均衡度', 'mean')).reindex(CATS)
# 品类层面旺季
cat_mon = d.pivot_table(index='月', columns='分类名称', values='净销量_千克', aggfunc='sum')[CATS]
cat_share = cat_mon.div(cat_mon.sum(axis=0), axis=1)
cc_ = (cat_share.T.values * np.cos(theta)).sum(axis=1)
ss_ = (cat_share.T.values * np.sin(theta)).sum(axis=1)
sc['旺季月份'] = (np.degrees(np.arctan2(ss_, cc_)) % 360) / 30.0 + 1
sc['旺季销量占比'] = [cat_share[c].max() for c in CATS]
sc['淡季销量占比'] = [cat_share[c].min() for c in CATS]
print(sc.round(3).to_string())

# 单品季节型聚类(Ward, 基于12维月度占比)
Xs = mon_share.loc[[c for c in sold['单品编码'] if c in mon_share.index]].fillna(0)
Zl = linkage(Xs.values, method='ward')
for k in [3, 4, 5]:
    lab = fcluster(Zl, k, criterion='maxclust')
    print(f'  季节型聚类 k={k}: ' + str(dict(zip(*np.unique(lab, return_counts=True)))))
K = 5
lab = fcluster(Zl, K, criterion='maxclust')
smap = dict(zip(Xs.index, lab))
sold['季节型'] = sold['单品编码'].map(smap)
# 依据每簇的"季节集中度"与"峰值月份"命名。
# 命名直接取簇均值曲线的 argmax(与图中画出的曲线峰值一致, 圆形均值会因卷绕偏移一个月, 不用于命名)。
peak_of, conc_of, argmax_of, curve_of = {}, {}, {}, {}
for L in np.unique(lab):
    idx = Xs.index[lab == L]
    m = Xs.loc[idx].mean(axis=0).values
    c_ = (m * np.cos(theta)).sum(); s_ = (m * np.sin(theta)).sum()
    peak_of[L] = (np.degrees(np.arctan2(s_, c_)) % 360) / 30.0 + 1
    conc_of[L] = np.sqrt(c_ ** 2 + s_ ** 2)
    argmax_of[L] = int(np.argmax(m)) + 1
    curve_of[L] = m


def season_label(L):
    if conc_of[L] < 0.55:
        return '弱季节型(全年均衡)'
    return f'强季节-{argmax_of[L]}月峰'


name_lab = {L: season_label(L) for L in np.unique(lab)}
sold['季节型'] = sold['季节型'].map(name_lab)
print(f'\n季节型聚类结果(k={K}, 按簇均值曲线的峰值月份自动命名):')
st = sold.groupby('季节型').agg(单品数=('单品编码', 'count'), 季节集中度=('季节集中度', 'mean'),
                               峰值月份=('旺季月份', 'mean'),
                               销量占比=('总销量', lambda s: s.sum() / sold['总销量'].sum()))
nm2arg = {name_lab[L]: argmax_of[L] for L in np.unique(lab)}
st['曲线峰值月'] = st.index.map(nm2arg)
print(st.sort_values('单品数', ascending=False).round(3).to_string())
print('\n季节型 × 品类 交叉(单品数):')
print(pd.crosstab(sold['季节型'], sold['分类名称'])[CATS].to_string())

# 生命周期
# 决策相关口径: "近6个月是否还有销售" —— 决定该单品当下还值不值得纳入补货/选品范围
end_date = d['销售日期'].max()
sold['退市距今天数'] = (end_date - sold['末销日']).dt.days
sold['近半年有销售'] = sold['退市距今天数'] <= 180
sold['生命周期状态'] = np.where(sold['近半年有销售'], '活跃在架', '已退市')
print('\n生命周期状态分布(按"近6个月是否有销售"划分):')
print(sold.groupby('生命周期状态').agg(单品数=('单品编码', 'count'),
                                       销量占比=('总销量', lambda s: s.sum() / sold['总销量'].sum()),
                                       在架天数均值=('在架天数', 'mean')).round(3).to_string())
print(f'\n  活跃在架单品占全部销量 {sold[sold["近半年有销售"]]["总销量"].sum() / sold["总销量"].sum() * 100:.1f}%'
      f' —— 这是问题3选品的合理候选池')
print('  各品类的活跃在架单品数(供问题3保证品类覆盖时参考):')
print(sold.groupby('分类名称')['近半年有销售'].agg(活跃数='sum', 总数='count').reindex(CATS).to_string())

# ---- 图8: 季节指纹 ----
fig = plt.figure(figsize=(15, 9))
ax0 = fig.add_subplot(2, 2, 1, projection='polar')
bars = ax0.bar(theta, cat_share['花叶类'].values * 100, width=2 * np.pi / 12 * .85, color='#27ae60', alpha=.8)
ax0.set_xticks(theta); ax0.set_xticklabels([f'{m}月' for m in range(1, 13)], fontsize=8)
ax0.set_title('示例：花叶类月度销量占比(极坐标)', pad=14)

ax1 = fig.add_subplot(2, 2, 2)
cm = cat_share.copy(); cm.index = [f'{m}月' for m in cm.index]
sns.heatmap(cm.T, cmap='YlGnBu', annot=True, fmt='.2f', ax=ax1, cbar_kws={'label': '该月销量占全年比重'})
ax1.set_title('品类 × 月份 季节指纹'); ax1.set_ylabel(''); ax1.tick_params(labelsize=8)

ax2 = fig.add_subplot(2, 2, 3)
for L in sorted(np.unique(lab), key=lambda L: argmax_of[L]):
    idx = Xs.index[lab == L]
    ax2.plot(range(1, 13), Xs.loc[idx].mean(axis=0).values * 100, lw=2.0,
             label=f'{name_lab[L]}(n={len(idx)})', marker='o', ms=3.5)
ax2.set_xticks(range(1, 13)); ax2.set_xticklabels([f'{m}' for m in range(1, 13)])
ax2.set_xlabel('月份'); ax2.set_ylabel('该月销量占全年比重(%)')
ax2.set_title(f'单品季节型聚类的月度均值曲线(k={K})'); ax2.legend(fontsize=8)

ax3 = fig.add_subplot(2, 2, 4)
for st_, col in zip(['活跃在架', '已退市'], ['#27ae60', '#95a5a6']):
    s5 = sold[sold['生命周期状态'] == st_]
    ax3.scatter(s5['旺季月份'], s5['季节集中度'], s=34, alpha=.72, c=col, label=f'{st_}(n={len(s5)})')
ax3.set_xlabel('旺季月份(圆形统计季节重心, 1~12月)'); ax3.set_ylabel('季节集中度(0=全年均匀)')
ax3.set_title('季节重心 × 季节集中度 × 生命周期'); ax3.legend(fontsize=8)
save(fig, 'Q1c_季节指纹与生命周期.png')

# ==================================================================
# 输出
# ==================================================================
sold.to_csv(f'{OUT}/SKU分层结果.csv', index=False, encoding='utf-8-sig')
q.to_csv(f'{OUT}/SKU量利象限.csv', index=False, encoding='utf-8-sig')
abc[['单品编码', '单品名称', '分类名称', 'ABC', 'XYZ', 'ABCXYZ']].to_csv(
    f'{OUT}/ABCXYZ分层.csv', index=False, encoding='utf-8-sig')

hr('模块1-6 完成，产出文件')
print(f'  {OUT}/SKU全息画像.csv   全量单品量-价-本-利指标')
print(f'  {OUT}/SKU分层结果.csv    四梯队+需求形态+季节型+生命周期')
print(f'  {OUT}/SKU量利象限.csv    量-利四象限归属')
print(f'  {OUT}/ABCXYZ分层.csv     ABC-XYZ 九宫格归属')
