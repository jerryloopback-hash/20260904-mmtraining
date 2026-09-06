# -*- coding: utf-8 -*-
"""
问题二·任务1:品类日内销售结构模型 —— 参数化"日内画像" + 日销售模拟器
====================================================================
模型口径(小组定稿 2026-09-06:机制用题面、数量用实测):
  机制(题面原文):"商超对运损和品相变差的商品通常进行打折销售" —— 打折
  是针对运损/品相受损商品的残值回收机制,该叙事写入论文;
  数量(实测):折扣渠道量按附件2 实测统计,不与 θ 挂钩。θ(附件4,盘点
  口径)= 从未售出的报废部分,只用于补货换算:补货总量 = 预测销量/(1-θ)。
  渠道划分(按时段×打折标记,19 时为清仓时点;茄类例外约 0.83 档):
    正价      未打折销售;
    品相折扣  19 时前打折 —— 题面所述"运损/品相受损但仍可售"渠道,
              其每日量即任务目标"因不新鲜等打折卖出的量(不含损耗)";
    清仓      19 时后打折 —— 日终未售完清仓(0.600 档为主)。
  曾检验"损耗全部进入折扣渠道"的假设(口径A):六品类实测折扣占比
  (2.7~9.0%)均低于 θ(6.7~15.5%),数量不成立,故弃用;标定表保留在
  控制台输出与论文的假设检验段落。simulate_day(mode=...) 仍可复现:
  mode="empirical" 为定稿口径,mode="theta" 为该对照假设。

输入:附件1/2/4(原始数据;队友预处理完成后,改下方路径常量重跑即可)
输出:
  p2_品类日内画像.csv       品类×小时×渠道:销量占比、订单占比、销量加权折后价比值
  p2_渠道占比_逐日.csv      品类×日期:正价/品相/清仓 销量与占比(目标②每日量按口径取"清仓"或"品相折扣"列)
  p2_损耗率参数.csv         品类 θ(附件4 Sheet1)+ 单品 θ 品类均值(Sheet2,对照)
  p2_fig1_日内销量结构.png  六品类小图:逐小时销量按三渠道堆叠(统一坐标)
  p2_fig2_渠道结构.png      分品类渠道销量占比(100% 堆叠横条)
  p2_fig3_实现价格比值.png  日内折扣深度连续曲线(按小时中位;19 时清仓档竖线标注)
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
from matplotlib.patches import Patch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 配置(队友换清洗后数据,改这三个路径即可) ----------------
A1, A2, A4 = "../附件1.xlsx", "../附件2.xlsx", "../附件4.xlsx"   # 原始附件在上级目录
CLEAR_HOUR = 19                  # ≥19 时 → 清仓渠道;<19 时的打折 → 品相渠道
HOUR_LO, HOUR_HI = 8, 22         # 分析窗口(窗口外记录占比在控制台报告)
RATIO_LO, RATIO_HI = 0.2, 1.5    # 折后价/原价 合理域,域外剔除(仅影响价格统计)
CATS = ["花叶类", "食用菌", "辣椒类", "水生根茎类", "茄类", "花菜类"]
CH_ORDER = ["正价", "品相折扣", "清仓"]          # 堆叠/图例固定顺序
CH_COLOR = {"正价": "#2a78d6", "品相折扣": "#eb6834", "清仓": "#1baf7a"}

# 图面样式
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


# ---------------- 数据准备 ----------------
def load_records():
    a1 = pd.read_excel(A1)[["单品编码", "分类名称"]]
    a2 = pd.read_excel(A2)
    a2 = a2[a2["销售类型"] == "销售"].copy()          # 剔除退货
    a2["销售日期"] = pd.to_datetime(a2["销售日期"])
    a2["hour"] = pd.to_datetime(a2["扫码销售时间"].astype(str), errors="coerce").dt.hour
    a2 = a2.merge(a1, on="单品编码", how="left")
    a2["is_disc"] = a2["是否打折销售"].astype(str).str.strip() == "是"
    n0 = len(a2)
    a2 = a2[a2["hour"].between(HOUR_LO, HOUR_HI)].copy()
    print(f"销售流水 {n0} 条(已剔除退货);小时窗口 [{HOUR_LO},{HOUR_HI}] 内 "
          f"{len(a2)} 条(窗口外 {n0 - len(a2)} 条,{(n0 - len(a2)) / n0:.2%},不计入)")
    a2["渠道"] = np.where(~a2["is_disc"], "正价",
                 np.where(a2["hour"] < CLEAR_HOUR, "品相折扣", "清仓"))
    return a2


def add_ratio(a2):
    """打折记录的 折后价/原价 比值。原价基准=同单品同日未打折价中位数,
    缺失回退该单品近 7 天原价中位数(与前期折扣深度分析同口径)。"""
    ref = (a2[~a2["is_disc"]].groupby(["单品编码", "销售日期"])["销售单价(元/千克)"]
           .median().rename("ref_price").reset_index().sort_values("销售日期"))
    disc = a2[a2["is_disc"]].sort_values("销售日期").copy()
    n_all = len(disc)
    disc = pd.merge_asof(disc, ref, on="销售日期", by="单品编码",
                         direction="backward", allow_exact_matches=True,
                         tolerance=pd.Timedelta("7D"))
    disc["ratio"] = disc["销售单价(元/千克)"] / disc["ref_price"]
    ok = disc["ratio"].between(RATIO_LO, RATIO_HI)
    print(f"打折记录 {n_all} 条;匹配到原价基准 {disc['ref_price'].notna().sum()} 条"
          f"({disc['ref_price'].notna().mean():.1%});比值在 [{RATIO_LO},{RATIO_HI}] 内 "
          f"{ok.sum()} 条({ok.mean():.1%},价格类统计仅用这部分)")
    return disc[ok].copy()


# ---------------- 模型参数表 ----------------
def daily_channel(a2):
    """品类×日期×渠道 销量表(目标②:每日品相渠道量即 '品相折扣' 列)"""
    g = a2.groupby(["分类名称", "销售日期", "渠道"])["销量(千克)"].sum()
    d = g.unstack("渠道")
    for c in CH_ORDER:
        if c not in d.columns:
            d[c] = 0.0
    d = d[CH_ORDER]
    d["总销量"] = d[CH_ORDER].sum(axis=1)
    for c in CH_ORDER:
        d[f"{c}占比"] = d[c] / d["总销量"]
    d = d.reset_index()
    d.to_csv("p2_渠道占比_逐日.csv", index=False, encoding="utf-8-sig")
    return d


def build_profile(a2, disc_ok):
    """品类×小时×渠道 画像:销量占比 / 订单占比 / 销量加权折后价比值(占全期口径)"""
    g = a2.groupby(["分类名称", "hour", "渠道"]).agg(
        销量=("销量(千克)", "sum"), 订单=("销量(千克)", "size")).reset_index()
    tot = a2.groupby("分类名称").agg(总销量=("销量(千克)", "sum"),
                                   总订单=("销量(千克)", "size"))
    g = g.merge(tot, on="分类名称")
    g["销量占比"] = g["销量"] / g["总销量"]
    g["订单占比"] = g["订单"] / g["总订单"]
    dw = disc_ok.assign(wr=disc_ok["ratio"] * disc_ok["销量(千克)"])
    dw = dw.groupby(["分类名称", "hour", "渠道"]).agg(
        wsum=("wr", "sum"), q=("销量(千克)", "sum")).reset_index()
    dw["折后价比值_加权"] = dw["wsum"] / dw["q"]
    prof = g.merge(dw[["分类名称", "hour", "渠道", "折后价比值_加权"]],
                   on=["分类名称", "hour", "渠道"], how="left")
    prof = prof[["分类名称", "hour", "渠道", "销量", "销量占比",
                 "订单", "订单占比", "折后价比值_加权"]]
    prof.to_csv("p2_品类日内画像.csv", index=False, encoding="utf-8-sig",
                float_format="%.6f")
    return prof


def loss_table():
    """品类 θ:附件4 Sheet1(品类平均损耗率;Sheet1 的品类列为分类编码,
    需经附件1 映射回品类名);Sheet2 单品 θ 按品类平均作对照。
    若编码映射失败,回退用 Sheet2 品类均值作为 θ_品类。"""
    a1_full = pd.read_excel(A1)
    a4_cat = pd.read_excel(A4, sheet_name=0)
    a4_cat.columns = [str(c).strip() for c in a4_cat.columns]
    lcol = next((c for c in a4_cat.columns if "损耗率" in c), a4_cat.columns[-1])
    ccol = next((c for c in a4_cat.columns if c != lcol), a4_cat.columns[0])
    cat_map = (a1_full[["分类编码", "分类名称"]].drop_duplicates()
               .assign(k=lambda d: d["分类编码"].astype(str).str.strip()))
    t = a4_cat[[ccol, lcol]].copy()
    t.columns = ["code", "损耗率%_Sheet1"]
    t["k"] = t["code"].astype(str).str.strip()
    t = t.merge(cat_map[["k", "分类名称"]], on="k", how="left")
    t["品类"] = t["分类名称"]
    t = t.drop(columns="分类名称")
    t["θ_品类"] = t["损耗率%_Sheet1"] / 100.0
    a4_item = pd.read_excel(A4, sheet_name=1)
    m = a4_item.merge(a1_full[["单品编码", "分类名称"]], on="单品编码", how="left")
    t2 = m.groupby("分类名称")["损耗率(%)"].mean().rename("θ_单品均值").reset_index()
    t = t.merge(t2, left_on="品类", right_on="分类名称", how="outer").drop(
        columns="分类名称")
    if t["品类"].isna().any():          # 编码映射失败 → 回退 Sheet2 品类均值
        t = t2.rename(columns={"θ_单品均值": "θ_品类"}).assign(**{"损耗率%_Sheet1": np.nan})
    t["_o"] = t["品类"].map({c: i for i, c in enumerate(CATS)})
    t = (t.sort_values("_o").drop(columns="_o")
         .reindex(columns=["品类", "损耗率%_Sheet1", "θ_品类", "θ_单品均值"])
         .reset_index(drop=True))
    t.to_csv("p2_损耗率参数.csv", index=False, encoding="utf-8-sig")
    return t


def base_prices(a2):
    """各品类正价销量加权均价(模拟器中的原价基准 p0)"""
    nd = a2[~a2["is_disc"]]
    out = {}
    for c in CATS:
        d = nd[nd["分类名称"] == c]
        out[c] = float(np.average(d["销售单价(元/千克)"], weights=d["销量(千克)"]))
    return out


# ---------------- 图像 ----------------
def fig1_structure(a2):
    hours = list(range(HOUR_LO, HOUR_HI + 1))
    piv = (a2.groupby(["分类名称", "hour", "渠道"])["销量(千克)"].sum()
             .unstack("渠道"))
    for c in CH_ORDER:
        if c not in piv.columns:
            piv[c] = 0.0
    piv = piv[CH_ORDER].fillna(0.0)
    ymax = 0.0
    for c in CATS:
        sub = piv.loc[c].reindex(hours, fill_value=0.0)
        ymax = max(ymax, sub.values.max() / sub.values.sum() * 100)
    ymax *= 1.18
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=150,
                             facecolor=SURFACE, sharex=True, sharey=True)
    for ax, c in zip(axes.ravel(), CATS):
        sub = piv.loc[c].reindex(hours, fill_value=0.0)
        tot = sub.values.sum()
        style_ax(ax)
        bottom = np.zeros(len(hours))
        for ch in CH_ORDER:
            v = sub[ch].values / tot * 100
            ax.bar(hours, v, bottom=bottom, width=0.72, color=CH_COLOR[ch],
                   edgecolor=SURFACE, linewidth=0.6, zorder=3)
            bottom += v
        ax.set_ylim(0, ymax)
        ax.set_xlim(HOUR_LO - 0.6, HOUR_HI + 0.6)
        ax.set_xticks(list(range(HOUR_LO, HOUR_HI + 1, 2)))
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("销售时间(时)", fontsize=10.5, color=INK)
        ax.set_title(c, fontsize=11.5, color=INK, pad=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("占全期销量(%)", fontsize=10.5, color=INK)
    handles = [Patch(facecolor=CH_COLOR[ch], label=ch) for ch in CH_ORDER]
    fig.legend(handles=handles, ncol=3, loc="upper center", frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("品类日内销量结构:逐小时销量按渠道堆叠", fontsize=13.5,
                 color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("p2_fig1_日内销量结构.png", facecolor=SURFACE)
    plt.close(fig)


def fig2_channels(a2):
    tot = (a2.groupby(["分类名称", "渠道"])["销量(千克)"].sum()
             .unstack("渠道").reindex(index=CATS))
    for c in CH_ORDER:
        if c not in tot.columns:
            tot[c] = 0.0
    tot = tot[CH_ORDER].fillna(0.0)
    pct = tot.div(tot.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=150, facecolor=SURFACE)
    y = np.arange(len(CATS))[::-1]
    left = np.zeros(len(CATS))
    for ch in CH_ORDER:
        v = pct[ch].values
        ax.barh(y, v, left=left, height=0.6, color=CH_COLOR[ch],
                edgecolor=SURFACE, linewidth=0.8, label=ch, zorder=3)
        for i, vv in enumerate(v):
            if vv >= 4:
                ax.text(left[i] + vv / 2, y[i], f"{vv:.1f}%", ha="center",
                        va="center", fontsize=9, color="#ffffff")
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels(CATS, fontsize=10.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("占全期销量比重(%)", fontsize=11, color=INK)
    ax.set_facecolor(SURFACE)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              frameon=False, fontsize=10.5)
    ax.set_title("渠道结构:正价 / 品相折扣(19 时前)/ 清仓(19 时后)",
                 fontsize=12.5, color=INK, pad=30)
    fig.tight_layout()
    fig.savefig("p2_fig2_渠道结构.png", facecolor=SURFACE)
    plt.close(fig)


def fig3_price_ratio(disc_ok):
    """连续的日内折扣深度曲线(全部打折记录,按小时中位),供分时段定价使用。
    19 时清仓时点以竖线标注;19 时后的水平跳变是真实的清仓档位,非数据缺失。"""
    med = disc_ok.groupby(["分类名称", "hour"])["ratio"].median().reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=150,
                             facecolor=SURFACE, sharex=True, sharey=True)
    for ax, c in zip(axes.ravel(), CATS):
        style_ax(ax)
        d = med[med["分类名称"] == c]
        ax.axhline(1.0, color=AXIS, lw=1, zorder=1)
        ax.axvline(CLEAR_HOUR, color=AXIS, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.plot(d["hour"], d["ratio"], color=CH_COLOR["品相折扣"], marker="o",
                ms=4, lw=1.8, zorder=3)
        ax.set_ylim(0.5, 1.05)
        ax.set_xticks(list(range(HOUR_LO, HOUR_HI + 1, 2)))
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("销售时间(时)", fontsize=10.5, color=INK)
        ax.set_title(c, fontsize=11.5, color=INK, pad=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("折后价/原价(中位)", fontsize=10.5, color=INK)
    axes[0, 0].annotate("19时清仓档生效", xy=(CLEAR_HOUR + 0.3, 0.53),
                        fontsize=9, color=INK2)
    fig.suptitle("日内折扣深度曲线:折后价/原价(1.0 = 原价;连续曲线,19 时跳变为清仓档位)",
                 fontsize=13.5, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("p2_fig3_实现价格比值.png", facecolor=SURFACE)
    plt.close(fig)


# ---------------- 日销售模拟器 ----------------
def simulate_day(prof, prices, theta, cat, Q_total, mode="empirical"):
    """给定品类与当日总销量 Q_total(kg),生成逐小时×渠道的销量与收入。
    mode="empirical"(定稿口径):三渠道量按实测日内画像占比;θ 仅用于
        补货换算,补货总量 = Q_total/(1-θ)。
    mode="theta"(对照假设,标定未通过):损耗 θ·Q_total 直接进入折扣渠道,
        按实测折扣日内形状分配到品相/清仓;其余 (1-θ)·Q_total 为正价;
        补货总量 = Q_total。
    收入 = 销量 × p0 × 折后价比值(正价=1;缺失按原价计)。"""
    p = prof[prof["分类名称"] == cat]
    p0 = prices[cat]
    th = float(theta)
    # 渠道内日内相对形状
    shape = {}
    for ch in CH_ORDER:
        s = p[p["渠道"] == ch].set_index("hour")["销量占比"]
        shape[ch] = s / s.sum() if s.sum() > 0 else s
    if mode == "theta":
        pw = p.loc[p["渠道"] == "品相折扣", "销量占比"].sum()
        cw = p.loc[p["渠道"] == "清仓", "销量占比"].sum()
        disc_total = th * Q_total
        w = pw / (pw + cw) if (pw + cw) > 0 else 1.0
        amounts = {"正价": (1 - th) * Q_total, "品相折扣": disc_total * w,
                   "清仓": disc_total * (1 - w)}
    else:
        amounts = {ch: float(p.loc[p["渠道"] == ch, "销量占比"].sum()) * Q_total
                   for ch in CH_ORDER}
    # 渠道级折后价比值(小时级缺失时回退)
    ch_ratio = {}
    for ch in ["品相折扣", "清仓"]:
        v = p[p["渠道"] == ch].dropna(subset=["折后价比值_加权"])
        ch_ratio[ch] = (float(np.average(v["折后价比值_加权"], weights=v["销量"]))
                        if len(v) else 1.0)
    rows = []
    for ch in CH_ORDER:
        r0 = 1.0 if ch == "正价" else ch_ratio[ch]
        for h, w in shape[ch].items():
            r = p[(p["渠道"] == ch) & (p["hour"] == h)]
            ratio = r["折后价比值_加权"].iloc[0] if len(r) else np.nan
            ratio = float(ratio) if pd.notna(ratio) else r0
            q = float(w) * amounts[ch]
            rows.append({"小时": int(h), "渠道": ch, "销量kg": q,
                         "实现单价": p0 * ratio, "收入元": q * p0 * ratio})
    return pd.DataFrame(rows), amounts


# ---------------- 主流程 ----------------
def main():
    a2 = load_records()
    disc_ok = add_ratio(a2)

    daily = daily_channel(a2)
    prof = build_profile(a2, disc_ok)
    theta_t = loss_table()
    prices = base_prices(a2)

    print("\n=== 品类 θ(附件4;口径A:折扣渠道量锚定 / 口径B:补货换算) ===")
    print(theta_t.round(4).to_string(index=False))

    print("\n=== 全期渠道结构(占各品类销量比重 %) ===")
    tot = (a2.groupby(["分类名称", "渠道"])["销量(千克)"].sum()
             .unstack("渠道"))
    for c in CH_ORDER:
        if c not in tot.columns:
            tot[c] = 0.0
    tot = tot[CH_ORDER]
    pct = tot.div(tot[CH_ORDER].sum(axis=1), axis=0) * 100
    show = pd.DataFrame({
        f"{ch}%": pct[ch].round(2) for ch in CH_ORDER})
    show["清仓/品相 量比"] = (tot["清仓"] / tot["品相折扣"].replace(0, np.nan)).round(2)
    print(show.to_string())

    print("\n=== 标定检查:口径A(损耗全部进折扣渠道)要求 折扣总量占比 ≈ θ ===")
    cal = pd.DataFrame({
        "θ%": (theta_t.set_index("品类")["θ_品类"] * 100).round(2),
        "实测折扣占比%": (tot[["品相折扣", "清仓"]].sum(axis=1)
                    / tot[CH_ORDER].sum(axis=1) * 100).round(2)})
    cal["差异(实测-θ)"] = (cal["实测折扣占比%"] - cal["θ%"]).round(2)
    print(cal.to_string())
    print("六品类实测均低于 θ ⇒ '损耗全部进折扣'假设(口径A)数量不成立;\n"
          "小组定稿:机制叙事用题面,数量用实测(实测口径),θ 仅用于补货换算。")

    print("\n=== 每日品相渠道量(目标②:因不新鲜等打折卖出的量,不含完全损耗) ===")
    d = daily.groupby("分类名称")["品相折扣"].agg(["mean", "std", "max"]).round(2)
    d.columns = ["日均kg", "标准差", "单日最大"]
    print(d.to_string())

    print("\n=== 各渠道实现价格(销量加权折后价比值,全期) ===")
    wr = disc_ok.assign(wr=disc_ok["ratio"] * disc_ok["销量(千克)"])
    g = wr.groupby(["分类名称", "渠道"]).agg(wsum=("wr", "sum"),
                                             q=("销量(千克)", "sum"))
    wavg = (g["wsum"] / g["q"]).unstack("渠道")
    print(wavg[["品相折扣", "清仓"]].round(3).to_string())

    fig1_structure(a2)
    fig2_channels(a2)
    fig3_price_ratio(disc_ok)
    print("\n已保存 3 张图:p2_fig1_日内销量结构 / p2_fig2_渠道结构 / "
          "p2_fig3_实现价格比值")

    print("\n=== 模拟器示例:花叶类某日总销量 200 kg(定稿口径:数量用实测) ===")
    th = dict(zip(theta_t["品类"], theta_t["θ_品类"]))
    sim, _ = simulate_day(prof, prices, th["花叶类"], "花叶类", 200.0, mode="empirical")
    s = sim.groupby("渠道").agg(销量kg=("销量kg", "sum"), 收入元=("收入元", "sum"))
    s["收入占比%"] = (s["收入元"] / s["收入元"].sum() * 100).round(2)
    print(s.round(2).to_string())
    print(f"定稿口径:补货 = 200/(1-θ={th['花叶类']:.4f}) = {200 / (1 - th['花叶类']):.1f} kg,"
          f"模拟总收入 {sim['收入元'].sum():.0f} 元")
    sim_a, _ = simulate_day(prof, prices, th["花叶类"], "花叶类", 200.0, mode="theta")
    print(f"口径A(已否定的对照假设):补货 = 200 kg,总收入 {sim_a['收入元'].sum():.0f} 元")
    print("(正式使用:Q_total 取问题二需求预测的日销量,逐品类逐日调用 simulate_day)")

    print("\n已保存:p2_品类日内画像.csv / p2_渠道占比_逐日.csv / p2_损耗率参数.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("\n[运行出错] 请把上面完整报错原样贴回给 Claude。")
