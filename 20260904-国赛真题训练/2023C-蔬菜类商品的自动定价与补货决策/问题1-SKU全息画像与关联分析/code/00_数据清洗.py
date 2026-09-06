# -*- coding: utf-8 -*-
"""
00_数据清洗.py
2023 高教社杯 C题 —— 蔬菜类商品自动定价与补货决策
数据预处理：对附件1~4进行缺失值、重复值、异常值、类型转换、口径统一与衍生字段构建，
参照国赛论文的数据清洗范式，最终输出可直接用于后续建模的"干净"数据。
输出目录: data_clean/
"""
import os
import numpy as np
import pandas as pd

pd.set_option('display.width', 220)
pd.set_option('display.unicode.east_asian_width', True)

OUT = 'data_clean'
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# 0. 读入
# ---------------------------------------------------------------
print('=' * 70)
print('读取数据 ...')
a1 = pd.read_excel('附件1.xlsx')

# 附件2较大，优先复用缓存，避免重复读 Excel
cache = '_cache_a2.pkl'
if os.path.exists(cache):
    a2 = pd.read_pickle(cache)
    print('附件2 从缓存读取:', a2.shape)
else:
    a2 = pd.read_excel('附件2.xlsx')
    a2.to_pickle(cache)
    print('附件2 从 Excel 读取:', a2.shape)

a3 = pd.read_excel('附件3.xlsx')
a4 = pd.read_excel('附件4.xlsx', sheet_name='Sheet1')
print('附件1/3/4 形状:', a1.shape, a3.shape, a4.shape)

# ---------------------------------------------------------------
# 1. 附件1 商品信息(字典表)
# ---------------------------------------------------------------
print('\n' + '=' * 70)
print('【附件1 商品信息】')
a1 = a1.rename(columns=str.strip)
# 类型统一
a1['单品编码'] = a1['单品编码'].astype(str).str.strip()
a1['单品名称'] = a1['单品名称'].astype(str).str.strip()
a1['分类编码'] = a1['分类编码'].astype(str).str.strip()
a1['分类名称'] = a1['分类名称'].astype(str).str.strip()

print('单品数:', a1['单品编码'].nunique(), ' 缺失值合计:', a1.isna().sum().sum(),
      ' 重复单品编码:', a1['单品编码'].duplicated().sum())

# 品类字典(编码 -> 名称)，用于后续把 6 大品类作为分析维度
cat_map = a1.drop_duplicates('分类编码').set_index('分类编码')['分类名称'].to_dict()
print('品类字典:', cat_map)

a1.to_csv(f'{OUT}/附件1_商品信息_clean.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------
# 2. 附件2 销售流水
# ---------------------------------------------------------------
print('\n' + '=' * 70)
print('【附件2 销售流水】')
a2 = a2.rename(columns=str.strip)
a2['单品编码'] = a2['单品编码'].astype(str).str.strip()
a2['销售类型'] = a2['销售类型'].astype(str).str.strip()
a2['是否打折销售'] = a2['是否打折销售'].astype(str).str.strip()

# 2.1 类型转换
a2['销售日期'] = pd.to_datetime(a2['销售日期'])
# 扫码时间存在多种格式(HH:MM:SS 与 HH:MM:SS.mmm)，用 mixed 解析
t = pd.to_datetime(a2['扫码销售时间'], errors='coerce', format='mixed')
n_fail = t.isna().sum()
print(f'扫码销售时间解析失败 {n_fail} 条 -> 置 00:00:00 (占比 {n_fail/len(a2)*100:.4f}%)')
a2['扫码时间'] = t.dt.time.fillna(pd.Timestamp('00:00:00').time())
a2['小时'] = t.dt.hour.fillna(0).astype(int)

# 2.2 数值字段
a2['销量(千克)'] = pd.to_numeric(a2['销量(千克)'], errors='coerce')
a2['销售单价(元/千克)'] = pd.to_numeric(a2['销售单价(元/千克)'], errors='coerce')

# 2.3 退货与负销量口径
# 观察: 销量<0 的行数 == 销售类型=退货 的行数，说明负销量即退货冲减量
n_ret = (a2['销售类型'] == '退货').sum()
n_neg = (a2['销量(千克)'] < 0).sum()
print(f'退货条数 {n_ret}，负销量条数 {n_neg}，一致={n_ret==n_neg}')
print(f'退货占比 {n_ret/len(a2)*100:.4f}%（极小，后续以"净销量"口径聚合，即正销量减去退货量）')

# 2.4 衍生字段
a2['销售额(元)'] = a2['销量(千克)'] * a2['销售单价(元/千克)']  # 带符号，退货为负
a2['年'] = a2['销售日期'].dt.year
a2['月'] = a2['销售日期'].dt.month
a2['日'] = a2['销售日期'].dt.day
a2['星期'] = a2['销售日期'].dt.weekday  # 0=周一
a2['是否周末'] = (a2['星期'] >= 5).astype(int)

# 2.5 异常值识别(IQR 边界，只标记不删除——大销量/高价多为真实团购或高档单品)
def iqr_bounds(s, k=3.0):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

for col in ['销量(千克)', '销售单价(元/千克)']:
    s = a2.loc[a2['销量(千克)'] > 0, col] if col == '销量(千克)' else a2[col]
    lo, hi = iqr_bounds(s)
    n_out = ((s < lo) | (s > hi)).sum()
    print(f'{col} IQR(3.0) 异常点 {n_out} 个 (上界 {hi:.2f})，保留并做稳健处理')

# 保存清洗后的流水(轻量：仅保存必要列)
cols_keep = ['销售日期', '扫码时间', '单品编码', '销量(千克)', '销售单价(元/千克)',
             '销售类型', '是否打折销售', '销售额(元)', '年', '月', '日', '星期', '是否周末', '小时']
a2[cols_keep].to_csv(f'{OUT}/附件2_销售流水_clean.csv', index=False, encoding='utf-8-sig')
a2[cols_keep].to_pickle(f'{OUT}/附件2_销售流水_clean.pkl')
print('附件2 清洗后保存:', a2[cols_keep].shape)

# ---------------------------------------------------------------
# 3. 附件3 批发价格
# ---------------------------------------------------------------
print('\n' + '=' * 70)
print('【附件3 批发价格】')
a3 = a3.rename(columns=str.strip)
a3['单品编码'] = a3['单品编码'].astype(str).str.strip()
a3['日期'] = pd.to_datetime(a3['日期'])
a3['批发价格(元/千克)'] = pd.to_numeric(a3['批发价格(元/千克)'], errors='coerce')
print('缺失:', a3.isna().sum().sum(), ' 重复(日期,单品):', a3.duplicated(['日期', '单品编码']).sum())
# 异常值识别
lo, hi = iqr_bounds(a3['批发价格(元/千克)'])
print(f'批发价 IQR 异常点 {((a3["批发价格(元/千克)"]<lo)|(a3["批发价格(元/千克)"]>hi)).sum()} 个，'
      f'min={a3["批发价格(元/千克)"].min():.3f}, max={a3["批发价格(元/千克)"].max():.3f}')
a3.to_csv(f'{OUT}/附件3_批发价_clean.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------
# 4. 附件4 损耗率
# ---------------------------------------------------------------
print('\n' + '=' * 70)
print('【附件4 损耗率】')
a4 = a4.rename(columns=str.strip)
a4['单品编码'] = a4['单品编码'].astype(str).str.strip()
a4['损耗率(%)'] = pd.to_numeric(a4['损耗率(%)'], errors='coerce')
print('缺失:', a4.isna().sum().sum(), ' 重复单品:', a4['单品编码'].duplicated().sum(),
      ' 范围:', a4['损耗率(%)'].min(), '~', a4['损耗率(%)'].max())
a4.to_csv(f'{OUT}/附件4_损耗率_clean.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------
# 5. 主表合并：单品维度画像(用于后续分析)
# ---------------------------------------------------------------
print('\n' + '=' * 70)
print('【合并单品画像】')
item = a1.copy()
item = item.merge(a4[['单品编码', '损耗率(%)']], on='单品编码', how='left')
# 每个单品的销售汇总
sales_sum = a2.groupby('单品编码').agg(
    总销量_千克=('销量(千克)', 'sum'),
    总销售额_元=('销售额(元)', 'sum'),
    有销售天数=('销售日期', 'nunique'),
).reset_index()
item = item.merge(sales_sum, on='单品编码', how='left')
item['总销量_千克'] = item['总销量_千克'].fillna(0)
item['总销售额_元'] = item['总销售额_元'].fillna(0)
item['有销售天数'] = item['有销售天数'].fillna(0).astype(int)
item['均价_元每千克'] = item['总销售额_元'] / item['总销量_千克'].replace(0, np.nan)
item['品类编码'] = item['分类编码']
item.to_csv(f'{OUT}/单品画像.csv', index=False, encoding='utf-8-sig')
print('单品画像行数:', item.shape[0], ' 从未销售单品数:', (item['有销售天数'] == 0).sum())

# ---------------------------------------------------------------
# 6. 日粒度聚合表(问题1~3核心工作表)
# ---------------------------------------------------------------
print('\n' + '=' * 70)
print('【日粒度聚合】')
daily = a2.groupby(['销售日期', '单品编码']).agg(
    净销量_千克=('销量(千克)', 'sum'),      # 正销量 - 退货量
    净销售额_元=('销售额(元)', 'sum'),
    毛销量_千克=('销量(千克)', lambda s: s[s > 0].sum()),
    是否打折_次=('是否打折销售', lambda s: (s == '是').sum()),
    交易笔数=('销量(千克)', 'size'),
).reset_index()
daily['加权销售均价'] = daily['净销售额_元'] / daily['净销量_千克'].replace(0, np.nan)
daily = daily.merge(a1[['单品编码', '分类编码', '分类名称']], on='单品编码', how='left')
daily.to_pickle(f'{OUT}/日粒度_单品销量.pkl')
print('日粒度聚合表形状:', daily.shape)

# 品类日汇总
cat_daily = daily.groupby(['销售日期', '分类编码', '分类名称']).agg(
    净销量_千克=('净销量_千克', 'sum'),
    净销售额_元=('净销售额_元', 'sum'),
    单品数=('单品编码', 'nunique'),
).reset_index()
cat_daily['加权均价'] = cat_daily['净销售额_元'] / cat_daily['净销量_千克'].replace(0, np.nan)
cat_daily.to_pickle(f'{OUT}/日粒度_品类销量.pkl')
print('品类日汇总形状:', cat_daily.shape)

print('\n清洗完成，输出目录:', OUT)
print('文件清单:', os.listdir(OUT))
