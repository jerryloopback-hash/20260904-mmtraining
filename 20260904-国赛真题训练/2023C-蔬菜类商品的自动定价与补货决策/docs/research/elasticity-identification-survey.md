# 弹性识别方法调研笔记（2023C 蔬菜定价与补货）

> 调研日期：2026-09-06。用途：支撑问题一"价格弹性识别"的方法选择与稳健性论证。
> 本笔记基于联网检索的一手文献（期刊页面 / arXiv / NBER / 开放获取 PDF，已存 `refs/`）。
> 注意：本笔记不参考"优秀论文C050"样文（用户要求独立建模）。

我们已有的设定：log(日销量) ~ E·log(日零售价) + 年月FE + 星期FE，品类加权批发价对数作 IV 的 2SLS，一阶段 F≈2200–4900，初步 E=−0.06~−0.43（全部缺乏弹性）。数据：单店、6 品类、约 1043 天品类级日度数据、日度批发价、无竞争者/顾客数据、库存近似=P50 预测。

---

## 1. 零售/农产品场景价格弹性的识别方法谱系

### 1.1 OLS 的内生性来源

观测价格不是随机实验的结果，估计量通常向 0（更缺乏弹性）方向有偏：

1. **同时性（simultaneity）**：零售商会根据未观测的需求冲击（天气、客流量、口碑）调价——需求高时提价，量价同向变动，把 |E| 往 0 压。这是需求估计最经典的偏差来源（Ackerberg 课程讲义对 cost-shifter IV 与传递偏差的系统梳理：[Ackerberg, *Estimating Price Elasticities in Differentiated Product Demand Systems*](https://www.princeton.edu/~erp/erp%20seminar%20pdfs/papersspring09/ackerberg.pdf)）。
2. **遗漏变量**：促销/陈列/季节性同时影响价格与销量；可用控制变量吸收（Rossi 2014 一系）。
3. **测量误差（对我们尤其重要）**：品类"价格"是销量加权均价，受单品构成（mix）变化污染——促销时便宜单品占比上升，均价下降是构成效应而非同一商品降价。这类误差同样把弹性向 0 衰减。DellaVigna & Gentzkow (2019) 明确指出：在扫描数据上做加权聚合会"过度给低价加权"，因为消费者在低价时买得更多（[DellaVigna & Gentzkow 2019, NBER w23996](https://www.nber.org/system/files/working_papers/w23996/w23996.pdf)，已下载）。
4. **断货删失**（见第 2 节）：高需求日销量被封顶，同样把 |E| 往 0 压。

**重要警示（直接相关）**：营销科学研究所（MSI）2024 年报告在一家有批发价数据的真实零售商上，把实验提价得到的弹性与所有主流观测数据修正方法（Hausman 工具、批发价 cost-shifter、滞后价格、PPI、事件研究）逐一对比，**没有任何一个 IV 能收敛实验与观测弹性之间的差距**，观测弹性系统性偏弱（[MSI Report 24-139, *Observational Price Variation in Scanner Data Cannot…*, 2024](https://thearf-org-unified-admin.s3.amazonaws.com/MSI_Report_24-139.pdf)，已下载）。含义：我们的 IV 估计应作为 **|E| 的下界** 来论证与呈现，而不是精确真值——这本身可以写成论文的方法论局限/稳健性讨论。

### 1.2 工具变量选择谱系

| IV 类型 | 代表文献 | 在我们数据上的适配 |
|---|---|---|
| **成本转移变量 cost shifters**（批发价/进货成本） | Fong et al. 2010；MSI 24-139 将其列为标准做法；Ackerberg 讲义 | **最适配**：我们有日度品类批发价，正是文献中"最干净的扫描数据 IV"（成本只通过零售价影响销量） |
| **Hausman 工具**（同一商品其他市场/门店的价格） | Hausman 1996；Nevo 2001；DellaVigna & Gentzkow 2019 | **不可用**：单店，无其他门店/市场数据 |
| **促销日历/滞后价格** | Villas-Boas & Winer 1999（滞后价格）；促销哑变量作外生控制 | 可用作控制变量（星期/节假日/促销代理），我们已做星期 FE |
| **PPI/大类价格指数** | Chintagunta et al. 2005 | 我们没有外购指数，但批发价本身就是更好的成本 IV |

注意 Hausman 类工具的思想有一个单店变体：**用批发价的滞后项或品类内其他商品的批发价均值**作工具的稳健性检验（仍属 cost-shifter 家族），可写进稳健性小节。

### 1.3 2SLS 小样本注意事项

- 弱工具：我们一阶段 F≈2200–4900，远超 Stock-Yogo 临界值，弱工具不是问题；反而要警惕**工具有效性（排他性）无法检验**——只能在论文中论证：批发价→零售价（一阶段强）、批发价与当日需求冲击近似无关（蔬菜批发价由产地供给/气候决定，对单店当日客流冲击不敏感）。
- 标准误应使用**对异方差与时间序列自相关稳健**的形式（日度数据 HAC / Newey-West，或按周聚类）；论文中报告 AR 诊断比只报 F 更稳。
- 逐品类估计只有 ~1000 个有效观测（扣 FE 后更少），应避免在 2SLS 中堆太多控制变量；固定效应层数（年月×品类）与样本量比例要报告。

### 1.4 因果机器学习：Double/Debiased ML

- **Chernozhukov et al. 2018**（*Double/Debiased Machine Learning for Treatment and Structural Parameters*, Econometrics Journal 21; [arXiv:1608.00060](https://arxiv.org/pdf/1608.00060)，已下载）：对部分线性模型 log Q = θ·log P + g(X) + ζ，用任意 ML 拟合 nuisance 函数 g、m，经 **Neyman 正交化（Frisch-Waugh 残差化）+ 交叉拟合（cross-fitting）** 后残差对残差回归，得到 √n 一致的 θ。正交化的本质是把"控制变量的误差"对目标参数的一阶影响消掉（partialling out + Neyman 正交）。
- **带 IV 的版本（PLIV/DML-IV）**：同框架下价格方程用残差化的 IV，正好对应我们的设定；官方包 DoubleML 有完整的"价格弹性"教程（基于 Roemheld 2021 零售数据的 Kaggle 公开数据集，log-log + DML）：[DoubleML 官方教程 *Estimation of Price Elasticities with DML*](https://docs.doubleml.org/tutorial/stable/notebooks/py_elasticity_analysis_tools.html)。
- **~1000 观测上是否可行**：DML 的理论要求 nuisance 估计以 n^(-1/4) 收敛且目标参数维度低（我们每品类只有 1 个 θ）。样本量没有硬性下限，实践判断标准是"残差化的价格是否有足够变异"（[Revology 实务指南：DML price elasticity](https://revologyanalytics.com/articles/machine-learning-price-elasticity)：B2B 月度数据 18–24 个月、高频零售 12 个月周数据即可；我们 1043 天日度数据、价格日度变动，变异充足）。n≈1000 对"标量弹性 + 中低维控制"完全可行；**不适合**的是在 n≈1000 上做高维异质性森林（见下）。
- DML **不能替代 IV**：若价格与不可观测需求冲击相关，DML 只靠控制变量救不了内生性；正确用法是 DML-IV（控制用 ML，识别仍靠批发价工具）。

### 1.5 因果森林 / 异质性处理效应

- Wager & Athey 2018（JASA; [arXiv:1504.01132](https://arxiv.org/pdf/1504.01132)，已下载）与 Athey–Tibshirani–Wager 2019（Annals of Statistics; [GRF, arXiv:1610.01271](https://arxiv.org/pdf/1610.01271)，已下载）：honest 分裂 + 交叉拟合的 CATE 估计，GRF 原生支持 IV 情形。
- 小样本证据：综述与模拟显示小样本下森林"无法正确给出估计，但能捕捉潜在异质性方向"（[Semantic Scholar 摘要汇总](https://www.semanticscholar.org/paper/Estimating-Treatment-Effects-with-Causal-Forests:-Athey-Wager/ee07ca60a0619f2f0cb72c7fc690c14a6598801e)）；置信区间对细粒度分组偏保守（[Modified Causal Forests, IZA DP 12040](https://docs.iza.org/dp12040.pdf)）。
- **对我们**：n≈1000/品类、单一连续处理 log P，CATE 森林的检出力不足。可行替代：在 DML 中用 log P × 季节/节假日交互，检验弹性是否随季节漂移（参数化异质性），比森林更稳、论文同样有亮点。GRF-IV 只作为附录稳健性。

---

## 2. 断货/删失需求修正

### 2.1 问题刻画

日销售 = min(真实需求, 当日可售库存上限)。高需求（通常伴随好天气/节假日/低价格）日被封顶 → 观测协方差被压缩 → |E| 向 0 衰减；这正是我们 E≈−0.06 这类"接近零"估计的首要嫌疑（问题 (a)）。零售 AI 数据集工作对此有直接陈述：销量数据把断货日当作真实需求处理会产生**系统性低估偏差**（[FreshRetailNet-LT, arXiv:2505.16319, 2025](https://arxiv.org/html/2505.16319v3)，已下载：两阶段"先恢复断货期潜需求、再训练预测"使预测精度提升 1.63%、系统性低估从 6.69% 降到近 0；其续作（[MDPI Sustainability 2025](https://www.mdpi.com/2071-1050/18/15/7642)）报告把断货观测作为下界事件处理，WAPE 改善约 2.7 个百分点、需求低估从约 8% 收窄到 1%）。

### 2.2 方法清单

1. **Tobit 类（时间变上限删失回归）**：经典删失似然（Tobin 1958 起源）。我们的上限逐日不同（≈P50 预测量），应使用**逐观测上限的 censored regression**；与 IV 结合即 Tobit-IV（健康经济学中有成熟应用：[Censored Quantile IV, NBER w15085](https://www.nber.org/system/files/working_papers/w15085/w15085.pdf)：同一数据 OLS/Tobit 与 CQIV 的弹性差异可达数倍）。局限：高斯误差假设强；删失比例高时估计不稳。
2. **EM 算法**：把断货日的潜需求当缺失数据，E 步在给定参数下对截断部分取条件期望（可用分布尾部均值），M 步重估需求方程参数；这是删失回归的标准替代，航空收益管理中最常用的两个 unconstraining 方法之一就是 EM（另一为投影去截断 PD）。对照评测见 [Skurla-Babic, *Evaluation of Unconstraining Methods in Airlines' RM Systems*](http://www.emc-review.com/sites/default/files/2019-2/Ruzica%20Skurla%20Babic.pdf)（8 种方法在仿真上比较：从"丢弃删失观测"的天真法到 EM/PD）。
3. **收益管理 demand unconstraining**：booking curve（把未受限历史期的需求到达曲线外推到被截断期）、naive/averaging、PD、EM、指数平滑外推（Queenan et al. 2007）。综述：[Guo, Xiao & Li 2012, *Unconstraining Methods in Revenue Management Systems*, Advances in Operations Research](https://onlinelibrary.wiley.com/doi/10.1155/2012/270910)（已下载镜像）；小样本场景专门研究：[Kourentzes 2017, *Unconstraining Methods for RM Systems under Small Data*](https://kourentzes.com/forecasting/wp-content/uploads/2017/09/Kourentzes_2017_Unconstraining.pdf)（已下载：历史数据少或全部受限时，指数平滑类外推优于 EM/PD——与我们"每品类 1043 天、断货日占比不高"的情形部分吻合：**混合法**可能最优）。
4. **断货时点信息（stock-out timing）**：Jain, Rudi & Wang 2015（Operations Research 63(1), *Demand Estimation and Ordering Under Censoring: Stock-Out Timing Is (Almost) All You Need*）：知道断货**何时发生**比知道删失观测的数量信息大得多（其报告时点观测可消除 76.1% 的学习损失；[INFORMS 摘要页](https://pubsonline.informs.org/doi/10.1287/opre.2014.1326)）。我们的"库存近似=P50 预测"正是这类信息：**当日销量≈库存上限即可判定为删失日，删失阈值=该日可售量**。
5. **非参数删失估计**：Kaplan-Meier 型估计、USVT 矩阵补全（[Censored Demand Estimation in Retail, ACM SIGMETRICS PER 2017](https://par.nsf.gov/servlets/purl/10066022)，已下载，Walmart 数据验证）；删失越重估计越差的解析刻画见该文。
6. **库存记录不准（我们的 P50 只是近似）**：Mersereau 2015（MSOM 17(3)）证明忽略库存记录不准会让删失需求估计系统性偏差（[摘要](https://pubsonline.informs.org/doi/10.1287/msom.2015.0520)）→ 论文中应做删失阈值 ±X% 的敏感性分析。
7. **动态定价视角**（可选亮点）：Chen, Wang & Zhou 2023（Management Science，*Optimal Policies for Dynamic Pricing and Inventory Control with Nonparametric Censored Demands*）与 SSRN 2024/25（*Dynamic Pricing and Learning with Censored Demand*）把"边定价边从删失销量学习"形式化——可作为论文讨论节的理论支撑（[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6843200)）。

### 2.3 修正后弹性通常变化多大

- **没有统一结论**：文献多为"修正后需求均值/预测偏差显著下降"（FreshRetailNet：低估 6.69%→≈0），直接量化"弹性校正幅度"的实证很少；健康保险删失场景中 OLS→Tobit-IV 弹性从弱/不可靠变为 −2.3~−3.2（NBER w15085），提示删失+内生性联合修正后 |E| 可上升数倍。
- 对我们的可行做法：先诊断删失程度（销量贴 P50 上限的天数占比；蔬菜断货通常不严重，估计 5–15%），再在自有数据上做仿真校验——用未删失样本估计"真"弹性，人为封顶后看 2SLS 的衰减幅度，作为论文的方法论小实验（无需外部文献背书）。

---

## 3. 文献中蔬菜/生鲜/食品杂货的弹性数值区间

| 来源 | 商品类别 | 弹性 | 方法/数据规模 |
|---|---|---|---|
| Andreyeva, Long & Brownell 2010 系统综述（163 项研究） | 食品总体 | \|E\| 均值 0.27–0.81，蔬果类估计仅占 11% | 时间序列/家计/扫描数据混合（[PMC2804646](https://pmc.ncbi.nlm.nih.gov/articles/PMC2804646)，已下载） |
| Rickard（Cornell，行业访谈引用） | 水果/蔬菜平均 | 约 −0.6 ~ −0.7 | 多项学术研究综合（[Produce Business](https://producebusiness.com/the-science-of-pricing)） |
| 同上（Spezzano，Vons 前 VP） | 香蕉 | −0.9 ~ −0.98（最不敏感） | 连锁超市单品促销数据 |
| 同上 | 葡萄 | −1.62 ~ −1.67（最敏感） | 两家门店（作者自称"heroic"） |
| 同上（Ortega/Ward） | 芒果 | −1.3 ~ −1.8（随价格变动） | 每月约 1000 名消费者、10 万+ 数据点 |
| Glaser & Thompson 1998（USDA ERS / AAEA） | 冷冻蔬菜（西兰花/豆/豌豆/玉米） | 常规较小，有机 −1.63 ~ −2.27 | AIDS 需求系统 + 全国超市扫描数据（[AgEcon PDF](https://ageconsearch.umn.edu/record/21583/files/sp99gl01.pdf)，已下载） |
| Nature Food 2025（澳大利亚 NielsenIQ Homescan） | 18 个食品大类 | 饮料最弹性（−1.2~−1.5），92% 交叉弹性 \|·\|<0.2 | QUAIDS/AIDS + 约 10,000 户面板、66 万价格-需求观测（[Nature Food](https://www.nature.com/articles/s43016-025-01184-1)） |
| USDA ERS Food Demand Analysis | 苹果等单品示例 | 例如苹果 −0.58 | [USDA ERS](https://www.ers.usda.gov/topics/food-choices-health/food-consumption-demand/food-demand-analysis) |
| DellaVigna & Gentzkow 2019 | 美国连锁超市扫描数据（品类/模块层级） | 店级弹性中位数约 −2，分布 10 分位至 90 分位约 −1.4 ~ −2.8（含促销深的 SKU/模块） | Nielsen 2006–14、73 链、23,715 商品，Hausman 式 IV（NBER w23996，已下载） |
| 田静、吕平 2024（*应用数学进展*，基于同一 2023C 数据） | 6 个蔬菜品类（花叶/花菜/水生根茎/茄/辣椒/食用菌） | 5 类 0<ε<1，仅水生根茎类 ε>1；并对弹性系数做敏感性分析 | log-log OLS（无 IV、无删失修正）对销售总量与成本加成定价（[Hanspub 开放获取](https://pdf.hanspub.org/aam2024136_22623865.pdf)，已下载） |

**对比判断**：
- 文献共识是**生鲜/蔬菜品类级弹性普遍缺乏弹性但非零**：品类聚合层面约 −0.4 ~ −1.0，单品+促销层面可到 −1.5 ~ −2.5。聚合越粗、日度频次越高、促销越浅，|E| 越小——方向上与我们 E=−0.06~−0.43 一致。
- 我们的下端（−0.06）显著低于文献同类下界（约 −0.4），**合理怀疑被两股力量压扁**：断货删失（第 2 节）+ 加权均价的构成污染（DellaVigna-Gentzkow 指出的聚合效应）。修正后若落到 −0.4 ~ −1.0 区间，即可用上表作为外部效度证据（"落在文献区间内"），这比单纯报 −0.06 有说服力得多。
- 参考文献中 2023C 亲自过的同行（田静、吕平 2024）在无 IV、无修正的 log-log OLS 下也得到"5/6 品类缺乏弹性（0<ε<1）"，与我们方向一致，可作中文文献对照（注意其用"成本加成定价"直接当价格，且无内生性处理，精度存疑）。

---

## 4. 综合建议：方法排序（可行性 × 对论文亮点贡献）

前提约束：单店、品类级、~1043 天/品类、日度批发价、库存近似=P50。

| 排名 | 方法 | 可行性 | 亮点贡献 | 实施要点 |
|---|---|---|---|---|
| 1 | **保留 2SLS（批发价 IV）为主干 + 稳健性组** | 极高 | 识别策略交代 + 有文献背书（cost-shifter 是扫描数据标准 IV） | (i) HAC 标准误；(ii) 工具变体稳健性：批发价滞后一期、品类内批发价均值（Hausman 思想的单店变体）；(iii) 明确把 |E| 解读为下界，引用 MSI 24-139 支撑 |
| 2 | **删失修正后再估弹性（两步法）** | 中高 | **最大增量**：直接回应 E≈−0.06 的嫌疑，问题 (a) | 用 P50 上限判定删失日（销量≥阈值×0.98 视为删失）；删失占比低时用 EM/Tobit（逐日变上限）重估，删失严重或全受限时退化为指数平滑外推（Kourentzes 2017）；修正后销量重新跑 2SLS；对阈值 ±20% 做敏感性；附一个小仿真展示删失把 E 往 0 压的幅度 |
| 3 | **固定加权价格指数替换销量加权均价** | 中 | 回应问题 (b)，简单可复现 | 用附件 2 SKU 级数据构造 Laspeyres 式固定权重品类均价（基期为期初），与现均价并报；两者弹性差异本身就是 mix 污染的证据 |
| 4 | **DML-IV（DoubleML PLIV）** | 中高 | 方法现代性亮点，n≈1000 可行 | 控制变量=年月 FE+星期 FE+节假日+批发价滞后；交叉拟合 5 折；与 2SLS 点估计对比，差异小说明结果稳健；代码用 doubleml 包 elasticity 教程改 |
| 5 | **非线性/参数域内弹性（回应角点解）** | 中 | 回应问题 (c)：优化必须在价格支撑域内做 | (i) 弹性按价格分箱局部估计（price-bin 回归），检查弹性随价格漂移；(ii) 或 log Q 对 log P 及其平方项；(iii) 优化层面对未来定价加"信任域"（不超出历史价格分位域 ±15%），引用"历史域外线性外推不可信"的标准论证 |
| 6 | **参数化弹性异质性（季节×价格交互 DML）** | 中 | 增量亮点：提出"旺季/淡季弹性不同→分季定价" | 在 DML 中加 log P × 季节交互；避免在 n≈1000 上跑 causal forest（森林小样本不可靠，GRF-IV 只进附录） |
| 7 | **动态定价/ Thompson 采样讨论**（可选） | 低-中 | 与"自动定价"题目呼应 | 只作讨论节：引用 censored-demand dynamic pricing 文献，说明我们离线估计+信任域是它的静态近似 |

**不建议**：因果森林全量异质性（样本不足）、Hausman 跨市场工具（无数据）、结构 BLP 式需求系统（6 品类替代品矩阵 + 单店数据支撑不了）。

---

## 已下载文献（`docs/research/refs/`）

| 文件 | 文献 |
|---|---|
| Chernozhukov2018-DoubleDebiasedML.pdf | Chernozhukov et al. 2018, Double/Debiased ML, Econometrics Journal（arXiv:1608.00060） |
| WagerAthey2018-RandomForestsHTE.pdf | Wager & Athey 2018, JASA（arXiv:1504.01132） |
| AtheyTibshiraniWager2019-GeneralizedRandomForests.pdf | Athey, Tibshirani & Wager 2019, Annals of Statistics（arXiv:1610.01271） |
| DellaVignaGentzkow2019-UniformPricing.pdf | DellaVigna & Gentzkow 2019, QJE / NBER w23996（扫描数据弹性估计与聚合偏差） |
| MSI2024-ScannerDataObservationalPriceElasticity.pdf | MSI Report 24-139, 2024（观测 IV 与实验弹性的差距，含批发价 IV 评测） |
| GuoXiaoLi2012-UnconstrainingMethodsRM.pdf | Guo, Xiao & Li 2012, Advances in OR（收益管理 unconstraining 综述） |
| Kourentzes2017-UnconstrainingSmallSamples.pdf | Kourentzes et al. 2017（小样本 unconstraining：指数平滑外推） |
| Wilkie2020-CensoredDemandEstimationReview.pdf | Wilkie 2020, Lancaster RT1（零售删失需求估计方法综述：Tobit/EM/KM） |
| Andreyeva2010-FoodPriceElasticityReview.pdf | Andreyeva, Long & Brownell 2010（163 项食品价格弹性系统综述） |
| GlaserThompson1998-FrozenVegetablesAIDS.pdf | Glaser & Thompson 1998（冷冻蔬菜 AIDS 弹性，USDA） |
| CensoredDemandEstimationRetail2017-USVT.pdf | Censored Demand Estimation in Retail, SIGMETRICS PER 2017（USVT，Walmart） |
| FreshRetailNet2025-StockoutCensoredDemand.pdf | FreshRetailNet-LT 2025（断货标注的生鲜删失需求数据集） |
| NBERw15085-CensoredQuantileIV.pdf（如缺见下方清单） | Censored Quantile IV 弹性（健康场景，Tobit-IV 参照） |
| 田静吕平2024-蔬菜补货定价时序（TianLyu2024-vegetable-replenishment-pricing-timeseries.pdf） | 基于 2023C 数据的中文开放获取对照 |
| （目录中另有其他代理此前下载的运营管理类 PDF，如 BesbesZeevi、Ferreira2016 等，与本题动态定价背景相关） | |

> 注：上一轮批次中 NBER w15085（CQIV）尚未确认下载成功，若 `refs/` 无该文件，见下方人工清单。

## 需人工下载清单（付费墙/需登录）

1. **Jain, Rudi & Wang 2015**, *Demand Estimation and Ordering Under Censoring: Stock-Out Timing Is (Almost) All You Need*, Operations Research 63(1):134–150. DOI: 10.1287/opre.2014.1326 —— 删失需求"时点信息"最重要实证，直接支撑我们用 P50 上限做删失判定。
2. **Nie（聂森）等 2024**, *基于价格弹性的蔬菜类商品自动定价与补货决策*, 《数学建模及其应用》13(2):66–72, DOI: 10.19943/j.2095-3070.jmmia.2024.02.08 —— 官方期刊上针对本题的弹性+LSTM+NSGA-II 方案，了解已发表处理方式（需注册登录）。
3. **Hausman 1996**, *Valuation of New Goods under Perfect and Incomplete Competition*（NBER 章节）—— Hausman 工具原始出处，引用时最好看原文。
4. **Nevo 2001**, *Measuring Market Power in the Ready-to-Eat Cereal Industry*, Econometrica 69(2) —— Hausman 工具操作化标准文献。
5. **Mersereau 2015**, *Demand Estimation from Censored Observations with Inventory Record Inaccuracy*, MSOM 17(3) —— 对应我们"P50 只是近似上限"的敏感性论证。
6. **Agrawal & Smith 1996**, *Estimating Negative Binomial Demand for Retail Inventory Management with Unobservable Lost Sales*, NRL 43(6) —— 删失需求经典。
7. **Lau & Lau 1996**, *Estimating the Demand Distributions of Single-Period Items Having Frequent Stockouts*, EJOR 92(2)。
8. **Chen, Wang & Zhou 2023**, *Optimal Policies for Dynamic Pricing and Inventory Control with Nonparametric Censored Demands*, Management Science —— 删失需求下动态定价理论支撑。
9. **Nature Food 2025**, *Food price elasticity estimates in Australia*（DOI 见 [文章页](https://www.nature.com/articles/s43016-025-01184-1)）—— 最新家庭扫描数据大类弹性基准。
10. **NBER w15085**, *Censored Quantile Instrumental Variable Estimates of the Price Elasticity…*（如上表未成功落盘）：https://www.nber.org/system/files/working_papers/w15085/w15085.pdf
11. **Liu, Smith & Orkin 2002**, *Estimating Unconstrained Hotel Demand Based on Censored Booking Data*, Journal of Revenue and Pricing Management 1(2) —— 酒店 unconstraining 实务。

---

### 关键结论一句话

我们的 E=−0.06~−0.43 在"品类级聚合 + 日度 + 无删失修正"的文献坐标里**偏低但不离谱**；论文最有价值的一步是把断货删失（用 P50 上限 + EM/Tobit）与价格构成污染（固定权重指数）修掉后重估，若 |E| 落到 −0.4~−1.0，即可同时完成内部一致性检验与外部文献对标；并把观测 IV 估计明确表述为 |E| 下界（MSI 2024 的实验对照证据）。
