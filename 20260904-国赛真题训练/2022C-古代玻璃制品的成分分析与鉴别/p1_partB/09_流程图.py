# -*- coding: utf-8 -*-
"""09 流程图：问题一后半部分（严谨版）建模流程"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FIG = r"D:\22年c问\问题1后半部分\严谨版\图"

TITLE_FS, BODY_FS, LINE_H, TITLE_OFF = 11.5, 8.4, 1.66, 1.75


def box_h(lines):
    return TITLE_OFF + 1.3 + (len(lines) - 1) * LINE_H + 1.35


fig, ax = plt.subplots(figsize=(13, 12.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x0, x1, y_top, title, lines, fc, ec):
    h = box_h(lines)
    ax.add_patch(FancyBboxPatch((x0, y_top - h), x1 - x0, h,
                                boxstyle="round,pad=0.35,rounding_size=1.1",
                                fc=fc, ec=ec, lw=1.7))
    cx = (x0 + x1) / 2
    ax.text(cx, y_top - TITLE_OFF, title, ha="center", va="center",
            fontsize=TITLE_FS, fontweight="bold", color="#1a1a1a")
    for k, ln in enumerate(lines):
        ax.text(cx, y_top - TITLE_OFF - 1.55 - k * LINE_H, ln, ha="center",
                va="center", fontsize=BODY_FS, color="#333333")
    return y_top - h  # 底边


def down(x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.8))


def rail_arrow(y, x0=57.2, x1=62.8):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.4,
                                linestyle="--"))


MX0, MX1 = 3, 57
RX0, RX1 = 63, 97
C_IN = ("#F0F0F8", "#555f7d")
C_PRE = ("#EAF3FB", "#185FA5")
C_STAT = ("#FDEEE7", "#D85A30")
C_IDX = ("#FFF8E1", "#B8860B")
C_PRED = ("#EAF7EE", "#2E8B57")
C_RAIL = ("#F7F7F7", "#8a8a8a")

rows = [
    ("数据输入",
     ["表单1：58件文物（类型、纹饰、颜色、表面风化）", "表单2：69个采样点 × 14种化学成分"],
     None, C_IN),
    ("① 数据预处理",
     ["有效性筛选：累加和 85%~105%，保留 67 条", "未检出成分记 0；行归一化闭合至 100%",
      "CLR 对数比变换备用（敏感性分析）"],
     None, C_PRE),
    ("② 采样点状态划分",
     ["无风化文物点 25 ｜ 未风化点 10", "一般风化点 29 ｜ 严重风化点 3",
      "合并为：风化前组 35 vs 风化组 32"],
     ("等价性检验（合并依据）",
      ["未风化点 vs 无风化文物点", "MW 检验×14 + FDR：0 项显著", "→ 二者无差异，可合并"]), C_PRE),
    ("③ 风化统计规律（问题1-2）",
     ["类型内 MW U 检验 + Cliff's δ 效应量", "BH-FDR 多重校正（q<0.05）",
      "高钾：SiO2↑，K2O/MgO/Al2O3/P2O5↓", "铅钡：SiO2↓，PbO/CaO/P2O5↑，Na2O↓"],
     ("CLR 敏感性分析",
      ["对数比变换下重跑全部检验", "显著成分结论基本一致",
       "（例外：铅钡 Na2O，含量<2%）", "→ 规律对闭合效应稳健"]), C_STAT),
    ("④ 风化程度指数（连续进程条）",
     ["显著成分标准化 → 质心方向投影", "刻度：0=风化前均值，1=一般风化均值",
      "逐点量化风化深浅（严重点不参与定义）", "→ 由“两档分类”升级为“连续程度”"],
     ("指数稳健性与检验",
      ["PC1对照 ρ=0.72~0.92；留一排序 ρ≥0.996", "标签盲检验：54号第1，08/26号中等偏重",
       "同文物配对（49/50号）风向一致", "→ “严重”标签≠成分偏移最大"]), C_IDX),
    ("⑤ 剂量-反应检验",
     ["Spearman(指数, 成分) + FDR", "7 项成分随风化程度单调渐变（q<0.05）",
      "→ 程度越深、变化越大，规律连续"],
     None, C_IDX),
    ("⑥ 风化前成分预测（问题1-3）",
     ["方案A：分位数配对 + Theil-Sen 稳健回归（主）", "方案B：均值比例缩放（对照）",
      "B=1000 联合 bootstrap → 90% 预测区间", "闭合归一化 → 32 个风化点全部还原"],
     ("模型验证",
      ["留一文物配对 MAE：A 2.20 / B 1.51", "预测落入风化前经验区间 81%~100%",
       "排序一致性 ρ≥0.94（检SiO2高→还原高）", "方案间一致：平均差 1.06，ρ=0.89"]), C_PRED),
]

GAP = 2.2
y_top = 95.5
bottoms = []
for title, lines, rail, colors in rows:
    b_main = box(MX0, MX1, y_top, title, lines, *colors)
    if rail is not None:
        box(RX0, RX1, y_top, rail[0], rail[1], *C_RAIL)
        rail_arrow(y_top - TITLE_OFF - 1.55)
    bottoms.append((b_main, y_top))
    y_top = b_main - GAP

b_out = box(MX0, 97, y_top,
            "输 出",
            ["统计规律表（FDR 显著 + CLR 稳健） ｜ 风化程度指数 ｜ 32 点风化前成分预测（含区间）",
             "→ 为问题二“先还原、再判型”提供输入"], "#F5F0FA", "#6a4fa3")

for i in range(len(rows)):
    down((MX0 + MX1) / 2, bottoms[i][0], bottoms[i + 1][1] if i + 1 < len(rows) else y_top)

ax.text(50, 98.6, "2022 国赛 C 题 问题一（后半部分）· 严谨版建模流程", ha="center",
        fontsize=15.5, fontweight="bold", color="#1a1a1a")
ax.set_ylim(b_out - 3.5, 100)

fig.savefig(os.path.join(FIG, "建模流程图.png"), dpi=200, bbox_inches="tight")
print("saved:", os.path.join(FIG, "建模流程图.png"))
