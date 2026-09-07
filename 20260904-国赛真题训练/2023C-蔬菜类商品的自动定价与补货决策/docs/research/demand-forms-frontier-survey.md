# 需求-价格响应关系构建方式：文献调研笔记

> 调研日期：2026-09-06。目的：为 2023 国赛 C 题（蔬菜类商品自动定价与补货）寻找比 log-log 等弹性形式更好、更有亮点的"需求响应"构建方式。
> 背景：6 个蔬菜品类、约 1043 天日度数据（销量 + 品类批发价），当前 log Q = a + E·log P + 协变量，E≈−0.06~−0.43（缺乏弹性）；log-log 下 |E|<1 时收益 R(p)=A·p^{1+E} 随价格单调上升，优化只有角点解，且 log-log 外推到历史价格域之外不可信。
> 说明：按用户要求，本笔记未阅读仓库中"优秀论文C050"样文，全部内容来自公开文献检索。

---

## 1. 需求函数形式的谱系与优劣

### 1.1 价格响应函数的分类框架

收益管理（RM）文献把"价格响应函数"（price-response function）d(p) 作为定价优化的一等公民，常用三种刻画：斜率、风险率（hazard rate）与弹性。常见候选为**线性、常数弹性（log-log）、logit** 三类（Phillips《Pricing and Revenue Optimization》第 3 章；DTU 收益管理课程讲义对其有系统的归纳，含"logit 是反 S 形、更接近意愿支付分布"的论述）。
- 来源：Pricing and Revenue Optimization, ch.3 "Models of Demand"（https://erenow.org/common/pricing-and-revenue-optimization/3.php）
- 来源：DTU 02735 收益管理讲义 "Basic price optimization"（http://www2.imm.dtu.dk/courses/02735/RMlecture04.pdf，本地 refs/DTU02735-price-response-functions-lecture-notes.pdf）——讲义中明确给出：常数弹性需求 d(p)=A·p^{−ε} 时 R'(p) 方向由 (1−ε) 决定，**ε<1（缺乏弹性）时收益随价格单调上升**——这正是我们遇到的角点解问题的教科书式表述。

### 1.2 各形式逐一评述（含收益曲线形态与内点最优）

| 形式 | 表达 | 弹性 E(p) | 收益曲线 | 内点最优？ |
|---|---|---|---|---|
| 线性 | Q=a−bP | −bP/a⁻¹·P，随价格增大 | 抛物线， concave | 有：p*=a/(2b)（且 P→a/b 处需求归零，天然 choke price） |
| 半对数 semilog | ln Q=a−bP（或 Q=a−b·lnP） | E(p)=−b·p，随价格线性增大 | R=p·e^{a−bP} 单峰 | 有：p*=1/b（ln Q=a−bP 版本） |
| log-log 等弹性 | ln Q=a−E·lnP | 常数 E | R=A·p^{1+E} 单调（\|E\|≠1） | **无**（\|E\|<1 时顶到价格上缘；\|E\|>1 时顶到下缘） |
| Box-Cox | Q^{(λ)}=a+b·P^{(λ)}，λ∈[0,1] | 随 λ 与价格变化，λ=0 退化为 log-log、λ=1 为线性 | 介于两者之间，λ>0 一般可恢复内点最优 | 视 λ；可由数据"投票"选出形态 |
| translog / AIDS / QUAIDS | 预算份额方程组 | 随价格/支出变化，满足可加性、齐次性、对称性 | 多品类联合，单品类弹性随份额变化 | 通常可产生内点解 |
| MNL / logit 需求 | s(p)=e^{v(p)}/(1+Σe^{v})，含"不买"选项 | 随价格增大（反 S 形） | 准凹，存在内点最优价格 | 有（MNL 定价/品类优化的标准结论，见 Rusmevichientong 等） |
| mixed logit / BLP | 价格系数随机分布 αᵢ | 个体异质，聚合弹性随价格变化 | 内点最优；拟合灵活 | 有，但估计成本高 |

要点展开：

- **线性与半对数都天然内点最优**。半对数（ln Q = a − bP）的弹性随价格线性增长，价格足够高时 |E(p)|>1，收益自动出现内点峰值 p*=1/b；同时它保持了"需求非负、随价格下降"的经济学性质（对数侧无界但可截断）。这与 log-log 形成互补：**log-log 把弹性冻结为常数，是角点解的根源**。
- 来源：Data 88E 教材 "Log-log and Semi-log Demand Curves"（https://data88e.org/textbook/content/01-demand/03-log-log.html）——讨论两式对"消费者按比例还是按绝对量响应价格"的不同隐含假设。
- 来源：Statworx 博客 "Food for Regression"（https://www.statworx.com/en/content-hub/blog/food-for-regression-using-sales-data-to-identify-price-elasticity）——用真实食品销售数据演示 linear / log-level / log-log 三种设定得到的弹性差异巨大，选错形式会直接传导到定价决策。
- **logit 价格响应是 RM 实务的"第三标准形"**：d(p)=C·e^{−b(p−p₀)}/(1+e^{−b(p−p₀)})，有需求上限 C（市场规模）与隐含的意愿支付分布，反 S 形，收益内点最优。Phillips 明确指出 logit 比 probit 更易处理、使用远更广泛。
- 来源：Pricing and Revenue Optimization ch.3（https://erenow.org/common/pricing-and-revenue-optimization/3.php）；DTU 讲义（同上）。
- **MNL 离散选择模型天然带"不买"选项与选择概率上限**：单品类退化为 share 形式 s(p)=e^{α−βp}/(1+e^{α−βp})，需求被市场规模截断，收益函数在 MNL 定价问题中被证明是准凹/可高效优化的。
- 来源：Rusmevichientong, Van Ryzin & ... "Assortment Optimization and Pricing under the Multinomial Logit Model with Impatient Customers"（http://faculty.marshall.usc.edu/Paat-Rusmevichientong/psfiles/impatient_mnl.pdf，本地 refs/Rusmevichientong2014-mnl-assortment-pricing-impatient.pdf）
- 来源：Kök & Fisher 等零售品类需求估计综述（https://faculty.wharton.upenn.edu/wp-content/uploads/2012/07/Demand-estimation---assortment-optimization.pdf）
- **Box-Cox 是"log-log vs 半对数"之争的正规化解决**：λ 由数据估计，λ=0 → log-log、λ=1 → 线性；Spitzer 的 Monte Carlo 研究系统考察了小样本下 Box-Cox 估计的性质。小样本（~1000 天）下 λ 的估计有噪声，但作为"形式选择"的稳健性检验非常合适。
- 来源：Sakia 1992 综述与相关文献汇编 "The Use of Box-Cox Transformation Technique in Economic and Statistical Analyses"（https://www.scholarlinkinstitute.org/jetems/articles/The%20Use%20of%20Box-Cox%20Transformation%20Technique%20in%20Economic%20and%20Statistical%20Analyses.pdf）
- 来源：Sydney 大学工作论文 "Does the Box-Cox transformation help in forecasting?"（https://ses.library.usyd.edu.au/bitstream/handle/2123/8167/OMWP_2011_08.pdf）——结论：多数情况下 log 变换（λ=0）已是不错选择。
- **需求系统（AIDS/QUAIDS/LES/translog）适合多品类联合**：以预算份额为因变量，天然满足可加性（份额和为 1）、齐次性、对称性约束，能同时给出自家价格弹性与交叉弹性（品类替代关系），与 C 题问题 1"品类间关联"呼应。食品场景标准应用：Peltner & Thiele (2021) 用 censored QUAIDS + 两阶段预算法在德国家庭扫描数据上估计各食品品类弹性（蔬菜水果等）。
- 来源：Peltner & Thiele 2021, "Elasticities of Food Demand in Germany – A Demand System Analysis Using Disaggregated Household Scanner Data", GJAE 70(1)（https://ageconsearch.umn.edu/record/343287/files/Elasticities%20of%20Food%20Demand%20in%20Germany.pdf，本地 refs/PeltnerThiele2021-quaids-food-demand-germany.pdf）
- 来源：Stata `demandsys` 功能页（Cobb-Douglas/LES/translog/AIDS/QUAIDS 一站式估计）（https://www.stata.com/features/overview/flexible-demand-system-estimation）
- 来源：IFPRI E-FooD 数据集方法文档（QUAIDS 弹性推导细节）（https://cgspace.cgiar.org/bitstreams/ee86912f-00db-4c3f-b3e2-d3c90bcc7357/download）
- **小样本可估性判断**（对应 ~1043 天、6 品类、品类级）：log-log、半对数、线性、Box-Cox 都是 2–4 参数/品类，OLS/NLS 即可，非常稳妥；AIDS/QUAIDS 是 6 个份额方程的 SUR 联立，参数量数十个，1043 个日度观测可行，但识别依赖价格变异——品类批发价日度有变异，可估；MNL/mixed logit 的难点在于"不买份额"不可观测，需要外部校准或假设市场规模 C，竞赛场景下可作为"建模选择"讲清即可；BLP 风格随机系数需要微观个体或跨市场数据，**不适合本数据**。
- 佐证（需求系统在时间序列长度有限时的使用）：US 食品需求综述用 BEA/BLS 年度/月度时间序列估计差分需求系统（Rotterdam/FDLAIDS 等）（https://s.giannini.ucop.edu/uploads/giannini_public/54/99/54994570-7739-4f54-99d7-91dd4aca3762/48-fooddemand.pdf）

### 1.3 对我们问题的直接含义

收益管理文献（Phillips/DTU 讲义）的结论可直接引用到论文里：**常数弹性需求 + |E|<1 ⇒ 收益随价格单调增 ⇒ 最优解是价格上界的角点解，这不是 bug 而是模型形态的必然**。摆脱角点解的三条正路：(a) 换成弹性随价格变化的形态（线性/半对数/logit/Box-Cox/需求系统）；(b) 承认价格可行域即历史价格域，把"顶到上缘"重新解释为"在可信外推域内提价"，并给出弹性变化的证据；(c) 用非参数/单调约束方法让数据决定曲线形状（见 §2、§3）。

---

## 2. 非参数/非线需求与动态定价文献

### 2.1 非参数需求学习（nonparametric demand learning）

- **Besbes & Zeevi (2009, Operations Research)** "Dynamic Pricing Without Knowing the Demand Function: Risk Bounds and Near-Optimal Algorithms"：需求函数完全未知（非参数），先"学习阶段"试探 κ 个价格、后"定价阶段"锁定好价格的 two-phase 策略，并给出 regret 下界。这是非参数需求学习的奠基文。
  - 来源/PDF：https://faculty.wharton.upenn.edu/wp-content/uploads/2008/06/Dp_wo_demand_risk_ob_az_posted.pdf（本地 refs/BesbesZeevi2009-dynamic-pricing-nonparametric-demand.pdf）
- **Besbes & Zeevi (2015, Management Science)** "On the (Surprising) Sufficiency of Linear Models for Dynamic Pricing with Demand Learning"：即使真实需求非参数，用**线性需求模型**做学习与定价也近乎最优（模型误设下依然稳健）。这对竞赛论文是极好的引用点：**为"用简单参数形式（线性/半对数）+ 在历史价格域内优化"提供理论背书**。
  - 来源/PDF：https://business.columbia.edu/sites/default/files-efs/imce-uploads/CPRM/2012-3-sufficiency_linear_models.pdf（本地 refs/BesbesZeevi2015-sufficiency-linear-models-dynamic-pricing.pdf）
- **Ye 等 (2024, ICML/PMLR)** "Smoothness-Adaptive Dynamic Pricing with Nonparametric Demand Learning"：假设需求函数 f(p) 属于 Hölder 光滑类，k 阶可导时最优 regret 为 Õ(T^{(k+1)/(2k+1)})；直接把"需求是价格的光滑未知函数"形式化——支持我们用样条/核平滑在历史价格域内估曲线。
  - 来源/PDF：https://proceedings.mlr.press/v238/ye24b/ye24b.pdf（本地 refs/Ye2024-smoothness-adaptive-dynamic-pricing-nonparametric.pdf）
- 相关：Bu, Simchi-Levi & Wang (2022, NeurIPS) "Context-Based Dynamic Pricing with Partially Linear Demand Model"（部分线性：参数价格项 + 非参数协变量项，https://papers.neurips.cc/paper_files/paper/2022/file/964892fb1437e73ef14f305df9bf5e7b-Paper-Conference.pdf）——与我们"log 价格项 + 协变量"的结构最接近的学术版；Chen & Gallego (2021, OR) "Nonparametric pricing analytics with customer covariates"（带协变量的非参数定价）。

### 2.2 形状约束（shape-constrained）回归：让需求"单调递减 + 凹"

- 非参数需求估计若无约束会"抖动、非单调"：Blundell, Horowitz & Parey 的实证发现，施加 Slutsky/形状约束后估计显著稳定。cemmap 工作论文系统讲了**形状约束下的非参数估计与一致置信带（uniform confidence band）**——可用于给需求曲线画置信带，进而给下游优化做"置信带内最坏/最好收益"分析。
  - 来源：Chernozhukov 等（cemmap CWP29/16）"Nonparametric estimation and inference under shape restrictions"（https://cemmap.ac.uk/wp-content/uploads/2020/08/CWP2916.pdf，本地 refs/Chernozhukov2016-shape-restrictions-nonparametric-inference.pdf）
  - 来源：Guntuboyina & Sen 综述 "Nonparametric Shape-restricted Regression"（https://arxiv.org/pdf/1709.05707）
  - 来源：UC Davis 讲义/论文示例——"constrained estimator 单调且凹，多项式与 smoothing spline 均违反单调/凹性"（https://arefiles.ucdavis.edu/uploads/filer_public/2014/03/27/jan_7_2013.pdf）
- 样条方面：DTU 团队提出适合需求模型的 cost-damping 样条函数类（效用对成本单调递减、边际敏感度递减，Box-Cox 为特例），可用于运输/零售需求。
  - 来源："A spline function class suitable for demand models"（https://backend.orbit.dtu.dk/ws/files/150862756/melju_1_s2.0_S2452306218300078_main.pdf）

### 2.3 Choke price（需求归零价格）

- 定义：choke price 是使需求降为零的最低价格（更高价格"扼杀"全部需求）。在定价模型中它充当价格可行域的自然上缘：线性需求 choke price = a/b；半对数与 log-log 形式的 choke price 为无穷大（log-log 尤其糟：任何有限价格下需求都为正，外推时需求随降价指数爆炸）。
  - 来源：Competera 术语表 "What is the Choke Price?"（https://competera.ai/resources/glossary/choke-price）
  - 来源：学术论文 "Pricing with the Help of the Choke Price"（对 choke price 在定价中的作用有专门讨论）（https://www.researchgate.net/publication/228256948_Pricing_with_the_Help_of_the_Choke_Price）
  - 来源：ScienceDirect 主题页 "Price Response Function"（指出 choke price 可随时间变化，常用于资源/易逝品定价）（https://www.sciencedirect.com/topics/economics-econometrics-and-finance/price-response-function）
- 对我们的用法：可把 choke price 作为**外推保护机制**——在优化时把价格网格截断在 [min 历史价, choke 估计价] 内，或选择内含有限 choke 的形式（线性、logit），从根上避免 log-log 外推爆炸。

### 2.4 Markdown / 清仓动态定价中的需求设定（易腐品）

- **Caro & Gallien (2012, Operations Research)** Zara 清仓定价：需求用**幂函数（类似常数弹性）F(p)=C(p/p_T)^{−β}** 加"货架老化折扣因子 κ"（价格随上架周数衰减），预测模型喂给价格优化模型。说明**清仓 markdown 文献大量用乘法/幂形式 + 时间衰减因子**，而不是纯 log-log 外推。
  - 来源：https://escholarship.org/uc/item/0fm8d8sv（条目页，见文末"需人工下载清单"）；后续工作 Caro et al. "Coordination of Inventory Distribution and Price Markdowns"（PDF：https://www.anderson.ucla.edu/faculty_pages/felipe.caro/papers/pdf_FC29.pdf，本地 refs/Caro2018-coordination-inventory-distribution-price-markdowns.pdf，其中复述了 Caro-Gallien 的 κ、β 设定）
- **易腐品库存+定价联合控制的解析文献几乎清一色用"加法需求 D(p)+ε 或乘法需求 D(p)·ε"**：
  - Chen, Pang & Pan (2014, M&SOM) "Coordinating Inventory Control and Pricing Strategies for Perishable Products"：加法形式 dt=D(p)+εt，D 严格递减，测试函数用线性 α−βp（PDF：https://publish.illinois.edu/xinchen/files/2014/04/perishableinventory2014-Final-proof.pdf，本地 refs/ChenPangPan2014-coordinating-inventory-pricing-perishable.pdf）
  - Herbon & Khmelnitsky (2017, EJOR) "Optimal dynamic pricing and ordering of a perishable product under additive effects of price and time on demand"（价格与时间对需求的加性效应；https://www.sciencedirect.com/science/article/abs/pii/S0377221716310670）
- **清仓/临期折扣与浪费**：Smith & Achabal (1998) 清仓定价（销售率依赖价格、季节与剩余品类）；SSRN 论文在 **MNL 需求**设定下证明对临期品打折的 markdown 政策能减少浪费（https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4738284_code3131236.pdf?abstractid=4151451）；Adenso-Díaz et al. (2017, Applied Mathematical Modelling) 量化易腐品动态定价对收入与浪费的影响；Buisman et al. (2019, IJPE) 零售生鲜折扣+动态货架期减少浪费。
- **质量/新鲜度衰减的常见设定**：价值/质量 r(t)=e^{−αt}（指数衰减），需求再对 r 做响应（JASSS 论文综述了线性/势/对数/指数等需求-价格函数族）（https://www.jasss.org/21/2/12.html）。

**小结**：生鲜/易腐场景文献的需求响应 = 价格项（线性/幂/指数/logit）×（或 +）新鲜度-时间项，**几乎没人把单一 log-log 直接外推出历史价格域**；"价格项 + 衰减项 + 加法或乘法噪声"是主流模板。

---

## 3. 机器学习需求模型与优化的结合

### 3.1 Predict-then-optimize 与一体化

- **Ferreira 等 (2016, HBS/Management Science)** "Analytics for an Online Retailer: Demand Forecasting and Price Optimization"（Rue La La 案例）：**回归树预测需求 → 多品价格优化**（含参考价格效应），是"ML 预测 + 数值优化定价"的标杆案例；他们特别讨论了非参数需求模型（树）预测如何转化为定价——在价格网格上做反事实模拟求解。
  - 来源/PDF：https://www.hbs.edu/ris/Publication%20Files/kris%20Analytics%20for%20an%20Online%20Retailer_6ef5f3e6-48e7-4923-a2d4-607d3a3d943c.pdf（本地 refs/Ferreira2016-analytics-online-retailer-pricing.pdf）
- **Oroojlooyjadid, Snyder & Takáč (2020, IISE Trans.)** "Applying Deep Learning to the Newsvendor Problem"：指出"分离式估计-优化（SEO）"会因只优化预测精度而次优，改用以报童成本为损失函数的深度学习直接出订货量——论文里可引来说明我们"预测-优化分离 + 网格搜索"的取舍理由。
  - 来源/PDF：https://engineering.lehigh.edu/sites/engineering.lehigh.edu/files/_DEPARTMENTS/ise/pdf/tech-papers/17/17T_004_0.pdf（本地 refs/Oroojlooyjadid2020-deep-learning-newsvendor.pdf）
- 更一般的"预测→处方"框架：Bertsimas & Kallus (2020, Management Science) "From Predictive to Prescriptive Analytics"。

### 3.2 因果/反事实价格响应

- 价格是连续"处理"，销量响应即连续处理效应：**S-learner**（把价格作为特征放进单一模型，反事实改价格预测差分）与 **T-learner**（分价格段建模）是最直接的两类 meta-learner；S-learner 在近年大规模营销 uplift 基准中表现最好（UpliftBench：S-learner Qini 0.376 优于 T/X-learner 与因果森林）（https://arxiv.org/html/2604.06123v1）。
  - meta-learner 奠基文：Künzel, Sekhon, Bickel & Yu (2019, PNAS) "Metalearners for estimating heterogeneous treatment effects using machine learning"。
  - 教学性综述：Stata 官方博客 "Heterogeneous treatment-effect estimation with S-, T-, and X-learners"（https://blog.stata.com/2025/10/22/heterogeneous-treatment-effect-estimation-with-s-t-and-x-learners-using-h2oml）；MDPI 对比研究（S-learner 偏差最小、T-learner 适合响应函数差异大的情形，https://www.mdpi.com/2297-8747/30/6/139）
- **单调约束梯度提升（monotone constraints）**：LightGBM/XGBoost/CatBoost 都支持对特征施加单调约束（split 时若破坏单调性则放弃该切分）。对价格特征施加"−1"约束可**硬性保证销量预测随价格单调不增**——这恰好治好树模型在数据稀疏区"价格越高销量反升"的反直觉局部翻转；基准研究显示约束在大数据上精度损失 <0.2%，小数据+强约束下约 2–3%，是"几乎免费"的可解释性/稳健性投资。
  - 来源：datadive 博客 "Monotonicity constraints in machine learning"（https://blog.datadive.net/monotonicity-constraints-in-machine-learning）
  - 来源：arXiv 基准 "What's the Price of Monotonicity? A Multi-Dataset Benchmark"（https://arxiv.org/pdf/2512.17945）
  - 来源：XGBoost/LightGBM 单调约束机制解析（https://medium.com/data-science/how-does-the-popular-xgboost-and-lightgbm-algorithms-enforce-monotonic-constraint-cf8fce797acb）
- 价格弹性异质性：因果森林/GRF（Athey, Tibshirani, Wager 2019, "Generalized Random Forests"，本地 refs 已有）可对"日/季节"特征估条件弹性，回应"弹性是否随季节漂移"。

### 3.3 对数线性 vs ML 响应曲线的权衡

- **可外推性**：log-log 全域有解析式但历史域外不可信（且 |E|<1 时单调收益导致角点解）；树/GBM 响应在历史域外退化为常数（外推=不响应），**因此 ML 路线应把优化网格限制在历史价格域内**——这与 Besbes & Zeevi (2015)"线性模型+域内优化足够好"的精神一致。
- **可解释性/评委友好**：log-log 弹性一句话讲清；GBM 需要 PDP（部分依赖图）+ 单调约束 + 特征重要性来讲"数据驱动的需求曲线"。折中方案：**参数价格核（半对数/logit）+ ML 协变量修正**，或"双模型对照"（参数基线 vs 单调 GBM），既保可解释又有亮点。
- 来源：Google Cloud "Price Optimization Using Vertex AI Forecast"（工业界同样采用"预测模型 → 价格情景模拟 → 优化"的 predict-then-optimize 流水线，https://cloud.google.com/blog/products/ai-machine-learning/price-optimization-using-vertex-ai-forecast）

---

## 4. 生鲜蔬菜场景的典型做法

- **与 C 题同题的公开论文**：田静、吕平 (2024)《基于时间序列模型的蔬菜类商品的补货和定价策略分析》（应用数学进展）——对 6 品类"取对数后的日销量对日售价"做线性回归得到价格需求弹性（即 log-log），再乘积 ARIMA 预测销量、按损耗率折算补货量、最后数学规划最大化日净收益；并做了弹性系数灵敏度分析（结论：替代品存在使模型稳健）。
  - 来源/PDF：https://pdf.hanspub.org/aam2024136_22623865.pdf（本地 refs/TianLyu2024-vegetable-replenishment-pricing-timeseries.pdf）
- **零售生鲜定价/补货+定价联合优化的学术文献（Omega/EJOR/IJPE 等）需求设定归纳**：
  - 加法线性：Chen, Pang & Pan (2014, M&SOM)（易腐品库存+定价，D(p)=α−βp 测试）——见 §2.4。
  - 加法（价格+时间/新鲜度）：Herbon & Khmelnitsky (2017, EJOR)。
  - 乘法/幂（含老化折扣）：Caro & Gallien (2012, OR, Zara 清仓)。
  - MNL/logit：SSRN 临期品 markdown 减 waste（§2.4）；"Joint dynamic pricing of multiple perishable products under consumer choice" (Management Science)。
  - 指数需求：Zhang & Chen (2013, IJPE) "Dynamic pricing for seasonal products with price-dependent demand"；Liu, Zhang & Tang (2015, EJOR) "Joint dynamic pricing and inventory control for perishable products"（易腐品联合动态定价与库存）。
  - 新鲜度-价格-库存依赖需求：Li & Teng (2018, EJOR)（需求依赖售价、参考价、新鲜度与陈列库存）。
- **中国超市生鲜弹性实证**：Liu, Zhang & Li (2020, Journal of Retailing) "Price elasticity modeling of fresh products based on Chinese supermarket data"（基于中国超市数据的生鲜价格弹性建模；见 HBEM 论文引文列表 https://hbem.org/index.php/OJS/article/view/323）。
- **中文期刊/学位论文**：康莎《不同需求特征下生鲜农产品双渠道定价与库存补货联合决策研究》（需求依赖价格与库存水平，分线上线下需求率）；潘小飞等 (2022) 考虑损失厌恶的生鲜保鲜努力与定价优化（公路交通科技）。
- **清仓/报损的处理惯例**：多数解析文献把损耗/清仓并入残值 s（salvage）或折扣价 tier（Caro-Gallien 的 κ 老化因子；Bitran, Caldentey & Mondschein 1998 的清仓 DP），而非显式对报损建模——竞赛论文可沿用"损耗率→补货量放大 + 残值/打折"两条通道。
- **数据规模对照**：上述实证类论文多用 1–3 年的门店/品类级面板（与我们 ~1043 天同量级），需求形式多为 2–5 参数的线性/对数/指数族——**说明我们的样本量支持"灵活但低维"的价格响应设定**。

---

## 5. 综合建议：候选需求响应设定排序

按"可行性 × 亮点 × 与下游一维 κ 网格优化兼容性"排序（下游目标：以成本加成率 κ 为决策变量最大化期望收益；数据：6 品类 × ~1043 天）：

### 候选 A（首推）：半对数需求 ln Q = a − b·κ·C + 协变量（价格用加成后售价的水平值）
- **函数形式**：ln Q = a − b·P + X'γ（P=批发价×(1+κ)）；弹性 E(P)=−b·P 随价格递增，收益 R(P)=P·e^{a−bP+Xγ} 单峰，**内点最优 P*=1/b 必然存在**，κ 网格搜索直接命中内点。
- **估计**：品类级 OLS/WLS，与现有 log-log 管线只差一列变换；并行报 log-log 作对照。
- **安全外推**：对 P>历史域上缘，用预测区间截断 + 声明"价格域=历史域±20%"；可再配有限 choke（若同时对比线性形式 choke=a/b）。
- **亮点**：一句话讲清"为什么换形式"——文献标准结论（常数弹性 + 低弹性 ⇒ 角点解；DTU 讲义/Phillips）+ 半对数恢复内点最优；Besbes & Zeevi (2015) 背书简单参数形式在定价学习中的稳健性。

### 候选 B：logit 价格响应（含"不买"选项的市场饱和模型）
- **形式**：Q(p)=C·σ(a−b·p)·X-修正（σ 为 logistic），C=品类饱和销量；需求有上界 C、反 S 形，收益内点最优。
- **估计**：NLS 或以"无购买份额"参数化的 MNL（需校准 C 或设为待估参数，说明建模假设即可）。
- **安全外推**：天然有界 [0,C]——**所有候选中外推行为最安全**，历史域外需求趋于 0/C 而非爆炸。
- **亮点**：离散选择/意愿支付分布的故事（logit = WTP 分布积分），与"蔬菜是高频必需品但高价时消费者转向替代品类"的直觉吻合；MNL 定价的准凹性有成熟理论（Rusmevichientong 等）。

### 候选 C：Box-Cox / 幂-指数混合族，让数据选形式
- **形式**：Q^{(λ)}=a+b·P^{(λ)}，或 Box-Cox 价格项嵌进 ln Q 模型；λ 数据驱动。
- **估计**：MLE/NLS 网格搜 λ∈[0,1]，报告 λ 的似然曲线；λ 落在 (0,1) 即说明 log-log 与线性都不对，是形式选择的实证证据。
- **安全外推**：同 A（域内优化+截断），λ 估计的不确定性可做成 λ∈{0,0.5,1} 的情景分析。
- **亮点**："我们不预设函数形式，而用 Box-Cox 检验之"——方法严谨性加分；Spitzer 小样本 Monte Carlo 文献支撑。

### 候选 D（ML 亮点线）：单调约束 GBM 需求曲线 + 域内反事实网格
- **形式**：LightGBM/XGBoost 预测 ln Q，特征含价格（monotone_constraints=−1）、星期、月份、批发价、滞后项等；价格 PDP 即需求响应曲线。
- **估计**：时间序列切分验证（最后 60 天留出），单调约束硬保证价格响应递减；可加 S-learner 式反事实（同模型改价格预测差分）。
- **安全外推**：**把 κ 网格限制在历史价格域内**（树模型域外输出常数，等于"不响应"，物理上安全）；再给 PDP 加 bootstrap 置信带。
- **亮点**：causal-ML 语言（连续处理、反事实响应、单调性先验），对比参数基线展示"响应曲线形状随季节变化"；Ferreira et al. (2016) 与 UpliftBench 支撑。

### 候选 E（系统/关系线，与问题 1 联动）：6 品类 AIDS/QUAIDS 需求系统
- **形式**：预算份额 w_i 对 ln p_j、总支出 ln X 回归（AIDS 一阶；QUAIDS 加二次项），满足可加/齐次/对称约束，得自家+交叉弹性矩阵。
- **估计**：SUR 联立（Stata demandsys 或手写），1043 天日度面板足够；价格用品类批发价指数。
- **安全外推**：弹性矩阵仅在历史份额/价格点附近有效，下游优化时对每个品类仍需一维响应（可用 AIDS 弹性在当前点局部线性化）。
- **亮点**：与问题 1"品类关联/替代互补"直接打通——问题 1 的相关性分析变成问题 2 的交叉弹性结构，全卷逻辑闭环；Peltner & Thiele (2021) 是现成的方法模板。风险：参数多、日度价格可能共线（各品类批发价同受季节驱动），需检查。

### 推荐组合
主线用 **A（半对数，恢复内点最优）**，稳健性/亮点用 **C（Box-Cox 检验形式）+ D（单调 GBM 对照曲线）**；若问题 1 已做足品类关联分析，可把 **E（QUAIDS 交叉弹性）**作为第 4 问"还需要什么数据/模型"的延伸讨论。log-log 保留为基线并用 DTU/Phillips 的标准结论解释角点解现象，转为"模型诊断"素材。

---

## 附：已下载 PDF（docs/research/refs/）

1. TianLyu2024-vegetable-replenishment-pricing-timeseries.pdf（同题中文论文，log-log+ARIMA+规划）
2. BesbesZeevi2009-dynamic-pricing-nonparametric-demand.pdf（非参数需求学习奠基文，OR 2009）
3. BesbesZeevi2015-sufficiency-linear-models-dynamic-pricing.pdf（线性模型对动态定价"出奇地够用"，Mgmt Sci 2015）
4. Ye2024-smoothness-adaptive-dynamic-pricing-nonparametric.pdf（光滑自适应非参数动态定价，ICML 2024）
5. ChenPangPan2014-coordinating-inventory-pricing-perishable.pdf（易腐品库存+定价，加法需求，M&SOM）
6. Caro2018-coordination-inventory-distribution-price-markdowns.pdf（清仓 markdown 协调，含 Caro-Gallien 需求设定）
7. PeltnerThiele2021-quaids-food-demand-germany.pdf（食品 QUAIDS 需求系统弹性，德国扫描数据）
8. Rusmevichientong2014-mnl-assortment-pricing-impatient.pdf（MNL 定价/品类优化理论）
9. Chernozhukov2016-shape-restrictions-nonparametric-inference.pdf（形状约束非参数估计+一致置信带，cemmap）
10. DTU02735-price-response-functions-lecture-notes.pdf（价格响应函数/常数弹性角点解的标准教材表述）
11. Ferreira2016-analytics-online-retailer-pricing.pdf（Rue La La：ML 预测+定价优化，Mgmt Sci/HBS）
12. Oroojlooyjadid2020-deep-learning-newsvendor.pdf（深度学习报童，预测-优化一体化）
13. 另：refs 目录中已有此前会话下载的 AtheyTibshiraniWager2019、WagerAthey2018（因果森林/GRF）、Chernozhukov2018-DoubleDebiasedML（DML）、GlaserThompson1998-FrozenVegetablesAIDS（蔬菜 AIDS 应用）、MSI2024-ScannerDataObservationalPriceElasticity、DellaVignaGentzkow2019-UniformPricing、CensoredDemandEstimationRetail2017、FreshRetailNet2025-StockoutCensoredDemand 等，可配合本笔记使用。

## 附：需人工下载清单（付费墙/未拿到 PDF）

| 标题 | 作者/年份 | 链接 | 为何重要 |
|---|---|---|---|
| Clearance Pricing Optimization for a Fast-Fashion Retailer | Caro & Gallien, 2012, Operations Research 60(6) | https://escholarship.org/uc/item/0fm8d8sv （eScholarship 条目页可能有全文）；INFORMS 正式版付费 | 清仓 markdown 的标杆实证，幂函数+老化因子 κ 的需求设定是易腐品定价模板 |
| Testing the Validity of a Demand Model: An Operations Perspective | Besbes, Phillips & Zeevi, 2010, M&SOM | SSRN 检索 | "模型有效性检验"直接对应我们"形式选择"章节 |
| Nonparametric Pricing Analytics with Customer Covariates | Chen & Gallego, 2021, Operations Research 69(3) | INFORMS 付费 | 带协变量的非参数定价，候选 D 的理论支撑 |
| Joint dynamic pricing and inventory control for perishable products | Liu, Zhang & Tang, 2015, EJOR 245(3) | ScienceDirect 付费 | 易腐品定价+库存联合优化的需求形式参照 |
| Optimal dynamic pricing and ordering of a perishable product under additive effects of price and time on demand | Herbon & Khmelnitsky, 2017, EJOR 260(2) | ScienceDirect 付费 | 价格+时间加性需求设定 |
| Dynamic pricing for seasonal products with price-dependent demand | Zhang & Chen, 2013, IJPE 145(2) | ScienceDirect 付费 | 指数/线性需求形式的季节品定价 |
| Price elasticity modeling of fresh products based on Chinese supermarket data | Liu, Zhang & Li, 2020, Journal of Retailing 96(3) | ScienceDirect 付费 | 中国超市生鲜弹性实证（与我们数据最接近） |
| Clearance Pricing and Inventory Policies for Retail Chains | Smith & Achabal, 1998, Management Science | INFORMS 付费 | 清仓定价经典：销售率依赖价格/季节/剩余品类 |
| Discounting and dynamic shelf life to reduce fresh food waste at retailers | Buisman, Haijema & Bloemhof-Ruwaard, 2019, IJPE 209 | ScienceDirect 付费 | 生鲜折扣与损耗 |
| Effects of dynamic pricing of perishable products on revenue and waste | Adenso-Díaz, Lozano & Palacio, 2017, Applied Mathematical Modelling 45 | ScienceDirect 付费 | 动态定价-收入-浪费三角 |
| Waste Reduction of Perishable Products through Markdowns (MNL 需求设定) | SSRN 工作论文 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4151451 | MNL 需求下的临期打折，候选 B 的易腐场景参照 |
| Almost Ideal Demand System / Quadratic AIDS 原始文献 | Deaton & Muellbauer 1980; Banks, Blundell & Lewbel 1997, J. Applied Econometrics | Wiley/JSTOR | QUAIDS 原始出处（方法模板 Peltner & Thiele 2021 已有） |
| Pricing and Revenue Optimization（专著 ch.3 Models of Demand） | Phillips, 2005, Stanford University Press | 图书馆/出版礼 | 价格响应函数三种标准形与收益形态的权威表述（DTU 讲义已覆盖核心内容） |
