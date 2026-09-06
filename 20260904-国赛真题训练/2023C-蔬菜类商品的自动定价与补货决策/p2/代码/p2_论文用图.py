# -*- coding: utf-8 -*-
"""论文专用图:p2_paper_fig1~fig5(PDF 矢量 + PNG 预览)
fig1 水生根茎类日内销量结构 / fig2 六品类渠道结构 / fig3 花叶类折扣深度曲线
fig4 花叶类需求预测 / fig5 分品类 WAPE 主对照
运行目录:p2/(原始附件在 ../)"""
import sys
import warnings

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

CATS = ["花叶类", "食用菌", "辣椒类", "水生根茎类", "茄类", "花菜类"]
CH_ORDER = ["正价", "品相折扣", "清仓"]
CH_COLOR = {"正价": "#2a78d6", "品相折扣": "#eb6834", "清仓": "#1baf7a"}
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, GRID, AXIS = "#ffffff", "#e1e0d9", "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
ENGINE = "LightGBM"


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(f"paper_{name}.pdf")
    fig.savefig(f"paper_{name}.png", dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print("已保存", name)


a1 = pd.read_excel("../附件1.xlsx")[["单品编码", "分类名称"]]
a2 = pd.read_excel("../附件2.xlsx")
a2 = a2[a2["销售类型"] == "销售"].copy()
a2["销售日期"] = pd.to_datetime(a2["销售日期"])
a2["hour"] = pd.to_datetime(a2["扫码销售时间"].astype(str), errors="coerce").dt.hour
a2 = a2.merge(a1, on="单品编码", how="left")
a2["is_disc"] = a2["是否打折销售"].astype(str).str.strip() == "是"
a2["渠道"] = np.where(~a2["is_disc"], "正价",
             np.where(a2["hour"] < 19, "品相折扣", "清仓"))

# ---------- fig1 水生根茎类:日内销量结构 ----------
hours = list(range(8, 23))
sub = a2[a2["分类名称"] == "水生根茎类"]
piv = sub.groupby(["hour", "渠道"])["销量(千克)"].sum().unstack()
for c in CH_ORDER:
    if c not in piv.columns:
        piv[c] = 0.0
piv = piv.reindex(hours, fill_value=0.0).fillna(0.0)[CH_ORDER]
tot = piv.values.sum()
fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=150, facecolor=SURFACE)
style_ax(ax)
bottom = np.zeros(len(hours))
for ch in CH_ORDER:
    v = piv[ch].values / tot * 100
    ax.bar(hours, v, bottom=bottom, width=0.72, color=CH_COLOR[ch],
           edgecolor=SURFACE, linewidth=0.6, zorder=3, label=ch)
    bottom += v
ax.set_ylim(0, bottom.max() * 1.22)
ax.set_xticks(hours[::2])
ax.set_xlabel("销售时间(时)", fontsize=10.5, color=INK)
ax.set_ylabel("占全期销量(%)", fontsize=10.5, color=INK)
ax.legend(ncol=3, loc="upper center", frameon=False, fontsize=9.5,
          bbox_to_anchor=(0.5, 1.14))
fig.tight_layout()
save(fig, "fig1")

# ---------- fig2 六品类渠道结构 ----------
tot2 = (a2.groupby(["分类名称", "渠道"])["销量(千克)"].sum()
        .unstack().reindex(index=CATS))
for c in CH_ORDER:
    if c not in tot2.columns:
        tot2[c] = 0.0
tot2 = tot2[CH_ORDER].fillna(0.0)
pct = tot2.div(tot2.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(6.8, 3.4), dpi=150, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
y = np.arange(len(CATS))[::-1]
left = np.zeros(len(CATS))
for ch in CH_ORDER:
    v = pct[ch].values
    ax.barh(y, v, left=left, height=0.6, color=CH_COLOR[ch],
            edgecolor=SURFACE, linewidth=0.8, label=ch, zorder=3)
    for i, vv in enumerate(v):
        if vv >= 4:
            ax.text(left[i] + vv / 2, y[i], f"{vv:.1f}%", ha="center",
                    va="center", fontsize=8.5, color="#ffffff")
    left += v
ax.set_yticks(y)
ax.set_yticklabels(CATS, fontsize=10, color=INK)
ax.set_xlim(0, 100)
ax.set_xlabel("占全期销量比重(%)", fontsize=10.5, color=INK)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.tick_params(colors=MUTED, labelcolor=INK2)
ax.grid(True, axis="x", color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02),
          frameon=False, fontsize=9.5)
fig.tight_layout()
save(fig, "fig2")

# ---------- fig3 花叶类:日内折扣深度曲线 ----------
ref = (a2[~a2["is_disc"]].groupby(["单品编码", "销售日期"])["销售单价(元/千克)"]
       .median().rename("ref_price").reset_index().sort_values("销售日期"))
disc = a2[a2["is_disc"]].sort_values("销售日期").copy()
disc = pd.merge_asof(disc, ref, on="销售日期", by="单品编码",
                     direction="backward", allow_exact_matches=True,
                     tolerance=pd.Timedelta("7D"))
disc["ratio"] = disc["销售单价(元/千克)"] / disc["ref_price"]
disc = disc[disc["ratio"].between(0.2, 1.5)]
d1 = disc[disc["分类名称"] == "花叶类"].groupby("hour")["ratio"].median()
fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=150, facecolor=SURFACE)
style_ax(ax)
ax.axhline(1.0, color=AXIS, lw=1, zorder=1)
ax.axvline(19, color=AXIS, lw=1.2, ls=(0, (4, 3)), zorder=2)
ax.plot(d1.index, d1.values, color=ORANGE, marker="o", ms=4.5, lw=1.8, zorder=3)
ax.set_ylim(0.5, 1.05)
ax.set_xticks(hours[::2])
ax.set_xlim(7.5, 22.5)
ax.annotate("19时清仓档生效", xy=(19.2, 0.53), fontsize=9, color=INK2)
ax.set_xlabel("销售时间(时)", fontsize=10.5, color=INK)
ax.set_ylabel("折后价/原价(中位)", fontsize=10.5, color=INK)
fig.tight_layout()
save(fig, "fig3")

# ---------- fig4 花叶类:需求预测 ----------
fc = pd.read_csv("p2_需求预测_202307.csv", parse_dates=["销售日期"])
fdates = pd.date_range("2023-07-01", periods=7)
P = fc[fc["品类"] == "花叶类"].set_index("销售日期")
gq = a2.groupby(["分类名称", "销售日期"])["销量(千克)"].sum()
cal = pd.date_range(a2["销售日期"].min(), a2["销售日期"].max(), freq="D")
h0 = cal.get_loc(pd.Timestamp("2023-05-15"))
hist = pd.Series([gq.get(("花叶类", d), 0.0) for d in cal[h0:]], index=cal[h0:])
fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=150, facecolor=SURFACE)
style_ax(ax)
ax.plot(hist.index, hist.values, color=BLUE, lw=1.5, zorder=3, label="历史销量")
ax.fill_between(fdates, P["P10"], P["P90"], color=BLUE, alpha=0.18,
                zorder=2, label="P10-P90 区间")
ax.plot(fdates, P["P50"], color=ORANGE, lw=1.8, marker="o", ms=4,
        zorder=4, label="P50 预测")
ax.axvline(fdates[0], color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)
ax.set_xticks(list(hist.index[::10]) + [fdates[3]])
ax.tick_params(axis="x", rotation=30, labelsize=8)
ax.set_ylabel("日需求量(kg)", fontsize=10.5, color=INK)
ax.legend(ncol=3, loc="upper center", frameon=False, fontsize=9.5,
          bbox_to_anchor=(0.5, 1.16))
fig.tight_layout()
save(fig, "fig4")

# ---------- fig5 分品类 WAPE 主对照 ----------
res = pd.read_csv("p2_回测指标汇总.csv")
main3 = ["季节朴素", "STL+岭回归", f"STL+{ENGINE}(断货修复)"]
lab3 = {"季节朴素": "季节朴素", "STL+岭回归": "STL+岭回归",
        f"STL+{ENGINE}(断货修复)": "STL+LightGBM(断货修复)"}
cols = {"季节朴素": "#898781", "STL+岭回归": ORANGE, f"STL+{ENGINE}(断货修复)": BLUE}
fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=150, facecolor=SURFACE)
style_ax(ax)
w = 0.26
for j, m in enumerate(main3):
    sub = res[res["模型"] == m].set_index("品类")["WAPE"].reindex(CATS)
    ax.bar(np.arange(6) + (j - 1) * w, sub.values, width=w,
           color=cols[m], label=lab3[m], zorder=3)
ax.set_xticks(range(6))
ax.set_xticklabels(CATS, fontsize=9.5, color=INK2)
ax.set_ylabel("WAPE", fontsize=10.5, color=INK)
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02),
          frameon=False, fontsize=9)
fig.tight_layout()
save(fig, "fig5")
