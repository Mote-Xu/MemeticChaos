# MemeticChaos — 文献/方法参考手册

> 生成日期: 2026-07-13 (进行中, 持续更新)
> 会话: MemeticChaos 文献参考 (67da905a)
> 方法论铁律: **对象是新的, 方法必须踩已验证的成熟石头, 别自制方法。** 某方法无成熟先例 → 诚实标"无成熟方法 (UNDERPOWERED)"。

---

## 1. 信息性采样 / MNAR / 选择偏差校正

> 用途: Wayback 存档密度混杂 (密度是信号的函数) — 被存档多的时期≠更活跃, 可能是更"值得存档"。

### 1.1 Heckman 两阶段 (样本选择模型)

**Heckman, J.J. (1979).** "Sample Selection Bias as a Specification Error." *Econometrica*, 47(1), 153–161. ⭐ 经典两阶段论文。

- **给我们什么**: 当选择机制依赖不可观测变量 (MNAR), 可通过 probit 选择方程 → 逆 Mills 比率 λ → 代入主方程校正。要求**排除约束** (至少一个变量在选择方程中但不在主方程中)。
- **成熟度**: 经典 (Nobel 奖工作)。局限: 依赖联合正态假设; 无有效排除约束时模型脆弱 (共线性 λ 与 X)。半参数扩展 (Ahn & Powell 1993; Das, Newey & Vella 2003) 放松正态假设但排除约束仍是硬的。

**Vella, F. (1998).** "Estimating Models with Sample Selection Bias: A Survey." *Journal of Human Resources*, 33(1), 127–169. ⭐ 优秀综述。

**Wooldridge, J.M. (2010).** *Econometric Analysis of Cross Section and Panel Data* (2nd ed.), MIT Press. Ch. 17 & 19. 标准研究生教材。

### 1.2 IPW (逆概率加权)

**Horvitz, D.G. & Thompson, D.J. (1952).** "A Generalization of Sampling Without Replacement from a Finite Universe." *JASA*, 47(260), 663–685. ⭐ 原始 IPW。

**Robins, J.M. & Rotnitzky, A. (1995).** "Semiparametric Efficiency in Multivariate Regression Models with Missing Data." *JASA*, 90(429), 122–129. 缺失数据上下文的 IPW。

**Hirano, K., Imbens, G.W. & Ridder, G. (2003).** "Efficient Estimation of Average Treatment Effects Using the Estimated Propensity Score." *Econometrica*, 71(4), 1161–1189. ⭐ 证明估计的 propensity score 达到半参数效率界。

- **给我们什么**: 用已知选择概率对观测加权, 重建无偏总体。**前提是 MAR (选择依赖可观测变量)**。对 Wayback 存档: 若可建模"什么让某天更可能被存档" (基于 Twitter 热度/新闻事件/时段), 则可 IPW 重加权。但若**被存档本身就是信号的函数** (即信号越"值得记录"越被存档), 则 MAR 不成立 → 需要 MNAR 方法。
- **成熟度**: 经典 (流行病学/生物统计标准工具)。局限: 对倾向模型误指定敏感; 极端权重导致高方差 → 需 trimming/stabilization。

### 1.3 敏感性界: Rosenbaum Bounds

**Rosenbaum, P.R. (2002).** *Observational Studies* (2nd ed.), Springer. Ch. 4 ⭐ 权威教材。

- **核心**: Γ = 两个观测相同协变量的个体因未测量混杂而获得处理的**最大 odds ratio**。从 Γ=1 (无隐藏偏差) 开始逐次增加, 检验 p 值何时变成不显著。**Γ 越大 p 值仍显著 → 结论越稳健。**

**Rosenbaum, P.R. (2004).** "Design Sensitivity in Observational Studies." *Biometrika*, 91, 153–164. 引入设计灵敏度概念。

- **给我们什么**: 不回答"偏差是否存在", 回答"需要多大的隐藏偏差才能推翻结论"。对于 Wayback 存档: 若 IPW 校正后某结论成立, Rosenbaum bounds 告诉我们"存档倾向需要与未测信号多大相关才能推翻"。

- **成熟度**: 主流 (观测研究标准敏感性分析)。局限: 检验 sharp null (无个体效应), 对异质效应保守。

### 1.4 E-value

**VanderWeele, T.J. & Ding, P. (2017).** "Sensitivity Analysis in Observational Research: Introducing the E-value." *Annals of Internal Medicine*, 167, 268–274. ⭐ 经典 E-value 论文。

**Ding, P. & VanderWeele, T.J. (2016).** "Sensitivity Analysis Without Assumptions." *Epidemiology*, 27(3), 368–377. 形式推导。

- **给我们什么**: 比 Rosenbaum Γ 更直观 — E-value = "一个未测量混杂因子需要同时与 exposure 和 outcome 至少有多强关联 (risk ratio 尺度), 才能解释掉观测到的效应"。**E-value 低 (如 1.3) = 脆弱; 高 (5+) = 稳健。**

- **成熟度**: 主流 (流行病学新标准, 2017 年后广泛采用)。局限: 基于风险比尺度; 对连续暴露/结局需做二分化或参数化扩展。

---

## 2. 天文/生态的选择函数 / 等密度重采样 (Flux-Limited Sampling)

> 用途: 与 §1 平行的武器 —— Wayback 存档密度类似天文的"越亮的星越容易被看到"。

### 2.1 Malmquist 偏差

**Malmquist, K.G. (1922).** "On some relations in stellar statistics." *Medd. Lund. Astron. Obs.*, Ser. I, 100, 1. ⭐ 原始论文。

- **核心**: 在流量受限样本中, 给定距离处观测到的天体平均光度系统性地**偏亮**于真实总体均值 (因为暗的被截掉了)。两个变体: 经典 (均匀空间密度) 和距离依赖 (随距离放大偏倚)。

- **给我们什么**: Wayback 等价: 给定时间距离 (距今), "被存档"的叙事片段平均上有更高的"叙事能量" (因低能的没触发存档)。Malmquist 的框架直接提供如何在密度估计中校正的数学。

- **成熟度**: 经典 (近 100 年, 天文学入门知识)。局限: 需知道流量限制 (存档阈值) 和内禀散布 (叙事能量的 σ); 对非高斯散布需做分布假设。

### 2.2 V/Vmax — 完整度校正

**Schmidt, M. (1968).** "Space Distribution and Luminosity Functions of Quasi-Stellar Radio Sources." *ApJ*, 151, 393. ⭐ **天文学最经典的完整度校正方法**。

- **核心**: 对每个天体计算 Vmax (该天体在仍能被探测到的最大体积), 权重 = 1/Vmax。若 V/Vmax ~ 0.5 → 均匀完整样本; 偏离 → 演化或不完整。

- **给我们什么**: 对 Wayback 存档直接类比: 每条被存档的文本 → 计算"最远多久之前它仍能被存档" (archive depth) → 以 1/Vmax 加权。**当且仅当存档概率单调依赖于某些可测特征时有效。**

- **成熟度**: 经典 (50+ 年)。局限: 需要知道选择函数的参数形式; 对多元选择不直接推广。

### 2.3 现代最大似然框架

**Sandage, A., Tammann, G.A. & Yahil, A. (1979).** "The Velocity Field of Bright Nearby Galaxies." *ApJ*, 232, 352. 引入选择函数进入似然函数。

**Efstathiou, G., Ellis, R.S. & Peterson, B.A. (1988).** "The Evolution of Active Galactic Nuclei." *MNRAS*, 232, 431. SWML 非参数光度函数估计, 自然纳入流量限制选择函数。

**Willick, J.A. (1994).** "The Las Campanas Observatory/Palomar 10,000 km/s Cluster Survey." *ApJS*, 92, 1. MLE 纳入距离依赖选择函数。

**Wall, J.V. & Jenkins, C.R. (2012).** *Practical Statistics for Astronomers* (2nd ed.), Cambridge Univ. Press. Ch. 6–7: 流量受限样本的实用处理。

- **给我们什么**: **这些方法的统一洞见**: 不直接分析"观测到的分布"作真实分布, 而在似然中显式建模"选择函数乘真实分布 = 观测分布"。对 Wayback: archive(d) × true_density(d) → 以去卷积或 MLE 推断 true_density。

- **成熟度**: 主流 (天文学标准工具链)。局限: 需参数化选择函数; 若选择函数未知且不可估计则降级为未校正描述。

---

## 3. Google Trends 方法学

> 用途: 长跨度日级拼接、量化噪声、geo 代表性/覆盖偏差的已知批评。

### 3.1 日级数据拼接方法

**Medic, D. & Schuster, T. (2019).** "Knitting Daily Google Trends." SSRN 3126324. ⭐ **专门讨论日级拼接方法的论文** — 利用 pytrends 的 `dailydata` 功能, 通过不同长度的时间窗口查询再将每日数据归一化拼接 (重叠窗口锚定)。

- **给我们什么**: 直接可用的日级拼接工程方法: 分月查询 → 选取锚定窗口归一 → 拼接长序列。自带方法约束 (采样误差、锚定窗口选择敏感性)。

- **成熟度**: 新兴/应用 (非理论贡献, 工程方法)。局限: 依赖 Google 内部归一化算法 (黑箱); 不同窗口查询的归一化基不同导致拼接误差。

### 3.2 Google Trends 的已知批评与选择偏差

**Mellon, J. (2013).** "Internet Search Data and Issue Salience: The Properties of Google Trends as a Measure of Issue Salience." *Journal of Elections, Public Opinion & Parties*, 24(1), 45–72.

- **给我们什么**: 系统检验 Google Trends 作为"议题显著性"度量的统计属性。指出: 搜索量 ≠ 态度; 重复搜索同一用户无法去重; 归一化 (0–100) 使跨词/跨期比较不直接可比。

**Carneiro, H.A. & Mylonakis, E. (2009).** "Google Trends: A Web-Based Tool for Real-Time Surveillance of Disease Outbreaks." *Clinical Infectious Diseases*, 49(10), 1557–1564.

- 早期讨论 Google Trends 噪声和采样偏差的经验报告。

**Butler, D. (2013).** "When Google Got Flu Wrong." *Nature*, 494(7436), 155–156. ⭐ 标志性批评: Google Flu Trends 高估流感发病率, 部分因搜索行为受媒体恐慌驱动而非实际发病率。

**Lazer, D., Kennedy, R., King, G., & Vespignani, A. (2014).** "The Parable of Google Flu: Traps in Big Data Analysis." *Science*, 343(6176), 1203–1205. ⭐ **最权威的大数据批评**: Google Flu Trends 失败的根本原因是 "算法动力学" (Google 自己的搜索推荐改变用户搜索行为) 混淆了信号。

- **给我们什么 (汇总)**: Google Trends 的三个结构性偏差 — ① **覆盖偏差**: geo=CN 的用户是"能访问 Google 的中国用户", 不是中国人口随机样本 (VPN 用户/学术/外贸相关人群为主); ② **算法漂移**: Google 搜索算法本身改变搜索结果推荐, 从而改变用户搜索行为, 这不是被观测对象变了而是观测仪器变了; ③ **归一化黑箱**: 0–100 归一化使绝对搜索量不可得, 跨时间窗口不可比。★★ 对于本项目, 最关键的批评是: **Google Trends geo=CN 测量的是"搜索被 Google 归属到 CN 的接入", 不是中国大陆互联网叙事场**。已在 PROJECT_STATUS §7 标注为 [Operator: GoogleTrends-CN]。

- **成熟度**: 主流批评 (2014 Science 文章被引 1800+)。

### 3.3 Google Trends 的统计属性 (正面视角)

**Stephens-Davidowitz, S. (2014).** "The Cost of Racial Animus on a Black Candidate: Evidence Using Google Search Data." *Journal of Public Economics*, 118, 26–40. 用 Google Trends 揭示调查问卷测不到的隐性态度 — 证明其作为"revealed preference"的价值。

**Choi, H. & Varian, H. (2012).** "Predicting the Present with Google Trends." *Economic Record*, 88(s1), 2–9. ⭐ Google 首席经济学家写的入门方法指南, 讨论 Trends 数据在 nowcasting 中的标准用法和局限。

- **给我们什么**: 搜索数据测量的是"revealed attention"不是"stated opinion" — 这正是本项目要利用的性质。但"revealed by whom" (覆盖偏差) 必须正视。

---

## 4. 参数恒定性 / 时变参数 (TVP)

> 用途: 检验预设 2 (时不变生成机制) — 系统参数是否随时间变化。

### 4.1 Nyblom-Hansen 参数恒定性检验

**Nyblom, J. (1989).** "Testing for the Constancy of Parameters Over Time." *JASA*, 84(405), 223–230. ⭐ 原始论文。

- **核心**: Lagrange Multiplier (LM) 型检验。H0: 参数在样本期内恒定; H1: 参数遵循随机游走 (martingale)。**不预设断点位置**, 检验"参数是否整体恒定"。需独立同分布假设 (残差 i.i.d.)。

**Hansen, B.E. (1992).** "Testing for Parameter Instability in Linear Models." *Journal of Policy Modeling*, 14(4), 517–533. 扩展 Nyblom 到协整回归。

- **给我们什么**: 本项目已用: `tvp_ar_stationarity.py` 做了 Nyblom 检验 (PC2: p=0.0065, PC1: p>0.05)。**关键约束**: Nyblom 是"仪器的灵敏度" — 它告诉你是否有证据拒绝恒定, **不告诉你替代假设具体是什么** (随机游走? 结构突变? 慢漂?)。H0 堆积率高 (本项目 boot 结果 51.9%) → 功率不足 → "不拒绝"≠"通过检验"。

- **成熟度**: 经典 (30+ 年)。局限: **本质上 UNDERPOWERED 在短序列上** (本项目 127 月对于参数恒定性检验是微数据); 对 i.i.d. 残差的依赖在时间序列中常不满足。

### 4.2 TVP-AR — Durbin-Koopman 状态空间方法

**Durbin, J. & Koopman, S.J. (2012).** *Time Series Analysis by State Space Methods* (2nd ed.), Oxford University Press. ⭐ **标准教材**。

- **核心**: 状态空间模型 + Kalman 滤波 + 扰动平滑 (disturbance smoother)。将时变参数视作隐状态, 用 Kalman 递推估计每时点的参数值。Frequentist (MLE 超参数 + Kalman/RTS) 和 Bayesian (MCMC/particle filter) 两种范式。

**Kim, C.J. & Nelson, C.R. (1999).** *State-Space Models with Regime Switching*, MIT Press. 结合 Markov 机制切换与时变参数的经典教材。

**Primiceri, G.E. (2005).** "Time Varying Structural Vector Autoregressions and Monetary Policy." *Review of Economic Studies*, 72(3), 821–852. ⭐ 最经典的 TVP-VAR 应用论文 (宏观经济学, 被引 2500+)。

- **给我们什么**: 本项目已用简化版 (随机游走 β 的 Kalman 估计): `tvp_ar_stationarity.py` 估计了 PC1/PC2 的 AR(2) 系数漂移轨迹。**关键局限**: 127 月的 TVP 估计本质上是"平滑先验" (smoothness prior), 信号-噪声比低时 Kalman 平滑回归先验 (趋近常数)。解决需要: ① 更长序列; ② 高分辨率 (日级) 回填; ③ Bayesian 先验敏感性分析。

- **成熟度**: 主流 (宏观经济学标准工具)。局限: 信号-噪声比决定估计质量; 短序列上 TVP 本质上插值而非估计。

### 4.3 结构突变检验

**Bai, J. & Perron, P. (1998).** "Estimating and Testing Linear Models with Multiple Structural Changes." *Econometrica*, 66(1), 47–78. ⭐ 多结构断点检验的经典。

**Bai, J. & Perron, P. (2003).** "Computation and Analysis of Multiple Structural Change Models." *Journal of Applied Econometrics*, 18(1), 1–22. 计算实现。

**Andrews, D.W.K. (1993).** "Tests for Parameter Instability and Structural Change with Unknown Change Point." *Econometrica*, 61(4), 821–856. 单未知断点的 sup-Wald 检验。

- **给我们什么**: 与 Nyblom 互补 — Nyblom 检验"总体是否有变化", Bai-Perron 检验"如果有变化, 在哪里断, 断了几次"。本项目时段审计 (三段切分) 是"人定断点", Bai-Perron 可提供**数据驱动的断点位置**作对照。

- **成熟度**: 经典。局限: 需预设最大断点数; 127 月的功率不足以可靠定位多断点。

---

## 5. 聚类有效性 / 离散 vs 连续 / "何时不该分簇"

> 用途: Regime Discreteness 审计 (2026-07-12) — GMM 4 簇是真实结构还是强制切割连续数据。

### 5.1 Gap Statistic

**Tibshirani, R., Walther, G. & Hastie, T. (2001).** "Estimating the Number of Clusters in a Data Set via the Gap Statistic." *JRSS-B*, 63(2), 411–423. ⭐ 经典论文。

- **核心**: 对每个 k, 计算 log(W_k) 与 null 参考分布 (均匀分布) 下 log(W_k) 的期望差 (gap)。选 gap 不再显著增大的 k。**不硬编码簇数, 由数据与 null 对比决定。**

- **给我们什么**: 本项目 RD-b 审计用 gap statistic → k=1 (不是 4)。直接可引用的方法验证。**这是"何时不该分簇"的标准答案之一。**

- **成熟度**: 经典 (被引 7000+)。局限: 对非球形簇 null 参考需调整; gap 曲线在"连续渐变"数据上可能无清晰最大值。

### 5.2 Silverman 多峰检验

**Silverman, B.W. (1981).** "Using Kernel Density Estimates to Investigate Multimodality." *JRSS-B*, 43(1), 97–99. ⭐ 原始论文。

- **核心**: 对 KDE 带宽从窄到宽扫描, 计算在什么带宽下分布从 k 个峰变成 k-1 个。过量平滑才消除的多峰是"真实"的 (非采样噪声)。用 bootstrap 生成 null (单峰) 分布, 检验临界带宽。

- **给我们什么**: 本项目 RD-a 审计用 Silverman test → PC1 多峰 p=0。**直接检验"一个分布是否真的多峰"** 还是 "KDE 带宽选择造出的假峰"。是离散安检的第一关。

- **成熟度**: 经典 (40+ 年)。局限: 对高维 (>2D) 需降至低维; bootstrap null 单峰假设在非单峰定义上有歧义。

### 5.3 Bootstrap 簇稳定性

**Hennig, C. (2007).** "Cluster-wise Assessment of Cluster Stability." *Computational Statistics & Data Analysis*, 52(1), 258–271. ⭐ 核心论文。

**von Luxburg, U. (2010).** "Clustering Stability: An Overview." *Foundations and Trends in Machine Learning*, 2(3), 235–274. 综述。

**Ben-Hur, A., Elisseeff, A. & Guyon, I. (2002).** "A Stability Based Method for Discovering Structure in Clustered Data." *Pacific Symposium on Biocomputing*, 7, 6–17.

- **核心**: 对数据做 bootstrap 重采样, 重跑聚类, 度量"同一簇"在不同重采样下的 Jaccard 重叠度。稳定的簇应该在不同重采样中反复出现; 不稳定的簇是噪声切割。

- **给我们什么**: 本项目 RD-b 审计用 Jaccard 稳定性 → GMM(4) 全簇 Jaccard 0.20-0.56, R2 仅 0.48 — **聚类的簇并不稳定**。已有成熟方法论支撑。

- **成熟度**: 主流。局限: 聚类算法本身的选择影响稳定性; 需定义"同一簇"的匹配规则 (匈牙利匹配)。

### 5.4 "何时不该分簇" — 方法论原则

**Ketchen, D.J. & Shook, C.L. (1996).** "The Application of Cluster Analysis in Strategic Management Research: An Analysis and Critique." *Strategic Management Journal*, 17(6), 441–458. 管理学中滥用聚类的早期系统批评。

**Steinley, D. (2003).** "Local Optima in K-Means Clustering: What You Don't Know May Hurt You." *Psychological Methods*, 8(3), 294–304. K-means 局部最优的危险。

- **给我们什么**: 方法论提醒 — 聚类只是数据简化工具, 不是"发现真实分组"的证据。**"聚类做了 ≠ 真的有簇"** — 这是本项目 RD 审计的哲学基础。

---

## 6. 递归量化分析 (RQA) & 临界转变早期预警

### 6.1 RQA 基础

**Eckmann, J.P., Kamphorst, S.O. & Ruelle, D. (1987).** "Recurrence Plots of Dynamical Systems." *Europhysics Letters*, 4(9), 973–977. ⭐ 引入 recurrence plot (图形)。

**Zbilut, J.P. & Webber, C.L. (1992).** "Embeddings and Delays as Derived from Quantification of Recurrence Plots." *Physics Letters A*, 171(3-4), 199–203. ⭐ **RQA 诞生的论文** — 从定性图形到定量指标。

**Webber, C.L. & Zbilut, J.P. (2005).** "Recurrence Quantification Analysis of Nonlinear Dynamical Systems." In *Tutorials in Contemporary Nonlinear Methods for the Behavioral Sciences*. 广泛引用的教程。

**Webber, C.L. & Marwan, N. (eds.) (2015).** *Recurrence Quantification Analysis: Theory and Best Practices*. Springer. ⭐ 现代综合教材, 覆盖最佳实践。

- **核心 RQA 指标**: %REC (复发率), %DET (确定性 = 对角线结构比例), ENTR (对角线长度 Shannon 熵), MAXLINE (最长对角线 = 1/最大 Lyapunov), TREND (偏离对角线 = 非平稳性/漂移)。

- **给我们什么**: 本项目已用: RQA 确认 R2 为真实结构分离 (零跨相区复发), 非 GMM artifact。**RQA 对非平稳、噪声、短序列不敏感** — 不做线性假设, 不做分布假设, 不需要平稳性。**特别适合本项目的叙事状态序列。**

- **成熟度**: 主流 (30+ 年, 跨学科: 生理学→工程→金融→气候)。局限: 嵌入参数 (m, τ) 和半径 ε 的选择影响结果; 参数选择已有成熟准则 (FNN 定 m, AMI 定 τ)。

### 6.2 临界减慢 (Critical Slowing Down) — 早期预警信号

**Scheffer, M. et al. (2009).** "Early-warning Signals for Critical Transitions." *Nature*, 461(7260), 53–59. ⭐ **分岔前预警的经典 Nature 综述** (被引 3000+)。

- **核心**: 系统接近 fold 分岔 (catastrophic bifurcation) 时, 回复速度减慢 → **自相关增大 + 方差增大**。这是跨系统通用的统计信号 (生态/气候/金融/生理)。不是预测"何时跳", 是预测"跳的概率在升高"。

**Scheffer, M. et al. (2012).** "Anticipating Critical Transitions." *Science*, 338(6105), 344–348. ⭐ 更新: 结合网络结构特征 + 统计预警信号。

**Dakos, V., Carpenter, S.R., van Nes, E.H. & Scheffer, M. (2015).** "Resilience Indicators: Prospects and Limitations for Early Warnings of Regime Shifts." *Phil. Trans. R. Soc. B*, 370(1659). ⭐ **重要的方法论警示** — 何时这些信号管用, 何时不管用。

- **关键局限**: ① 不是所有 regime shift 都涉及分岔; ② 临界减慢也可能在非灾难性转变前出现 (Kéfi et al. 2013); ③ **噪声驱动的转变 (noise-induced transition) 不显示临界减慢**。

**Kéfi, S., Dakos, V., Scheffer, M., van Nes, E.H. & Rietkerk, M. (2013).** "Early Warning Signals Also Precede Non-Catastrophic Transitions." *Oikos*, 122, 641–648.

**Lenton, T.M. et al. (2008).** "Tipping Elements in the Earth's Climate System." *PNAS*, 105(6), 1786–1793. 气候倾覆元素中的早期预警应用。

- **给我们什么 (汇总)**:
  1. **已用的临界减慢**: 本项目的 Sensitivity 指标 (Critical Slowing = 0.76) 基于自相关-方差上升, 有 Scheffer 框架支撑。
  2. **检验方法成熟**: 自相关 (lag-1 AC)、方差、回归至均值的速率 — 三个指标一起看, 而非单项。
  3. **重要的否定**: "有临界减慢信号 ≠ 一定临近分岔" — 可能只是系统对噪声的反应变了, 或噪声变了。
  4. **对项目最有价值的洞见**: Scheffer 2015 的警示直接可用于本项目 Sensitivity 指标的 caveat 表述。

- **成熟度**: 主流 (Nature+Science 两篇, 生态/气候/金融广泛应用)。局限: 对非分岔转变不敏感; 需滚动窗口估计 (窗口选择影响结果)。

---

## 7. 梗/信息扩散动力学

> 用途: SIR 变体、Hawkes 过程、信息级联、reflexivity/performativity。

### 7.1 SIR 流行病模型 → 梗扩散

**Wang, L. & Wood, B.C. (2011).** "An Epidemiological Approach to Model the Viral Propagation of Memes." *Applied Mathematical Modelling*, 35(11), 5442–5447.

- **给我们什么**: 直接对"梗"用 SIR 建模的先例。S=susceptible (未接触), I=infected (在传播), R=recovered (已遗忘)。对多梗竞争场景需扩展到 SIS/SIRS (可重复感染)。

**Lonnberg, A., Xiao, P. & Wolfinger, K. (2020).** "The Growth, Spread, and Mutation of Internet Phenomena: A Study of Memes." *Results in Applied Mathematics*, 6, 100092. ⭐ SIR 随机微分方程 (SDE) 对梗扩散建模, 包含突变。

**Mussumeci, E. & Coelho, F.C. (2017).** "Modeling News Spread as an SIR Process over Temporal Networks." arXiv:1701.07853. NLP (Word2Vec+TF-IDF) 构建时序传播网 + SIR 近似。

- **给我们什么 (汇总)**:
  1. SIR 是梗扩散的基本框架 (成熟先例)。
  2. 关键限制: SIR 假设同质混合 (homogeneous mixing) — 网络拓扑被忽略; 梗的"语义变异" (mutation) 需要 SIR 变体 (随机微分方程含突变项)。
  3. 对本项目: 已有一个基础 `sir_meme.py` (见 `src/models/sir_meme.py`), 可引用 Wang & Wood (2011) 作为成熟方法背书。
  4. 对多梗竞争 (本项目的叙事生态系统), SIR 单梗模型不够 → 需多菌株 SIR 或 Hawkes 互激励。

- **成熟度**: 经典 (流行病学 100 年; 应用到梗 ~15 年)。局限: 同质混合假设对社交网络不现实; 参数 (β, γ) 在非平稳叙事环境中可能时变。

### 7.2 Hawkes 过程 — 自激励点过程

**Hawkes, A.G. (1971).** "Spectra of Some Self-Exciting and Mutually Exciting Point Processes." *Biometrika*, 58(1), 83–90. ⭐ 原始论文。

- **核心**: λ(t) = μ + Σ φ(t - tⱼ) — 每个过去事件通过内存核 φ 提升未来事件的发生强度。"rich-get-richer" 的数学形式化。地震学 (主震→余震) 到社交媒体 (首发帖→转发雪崩) 通用。

**Rizoiu, M.A. et al. (2017).** "A Tutorial on Hawkes Processes for Events in Social Media." arXiv:1708.06401. ⭐ 面向社媒研究者的 Hawkes 入门教程 — 用转发级联做例子, 可直接上手。推荐为本项目的首选入门。

**Rizoiu, M.A. et al. (2018).** "SIR-Hawkes: Linking Epidemic Models and Hawkes Processes to Model Diffusions in Finite Populations." *WWW 2018*. ⭐ **桥梁论文** — 证明 SIR 和 Hawkes 在边缘化掉恢复事件后数学等价。给出有限人口版本 (HawkesN) 和级联大小分布 (Borel 分布)。

**Kong, Q. et al. (2020).** "Describing and Predicting Online Items with Reshare Cascades via Dual Mixture Self-Exciting Processes." *CIKM 2020*. 双混合自激过程, 区分极右/阴谋论/正规新闻的扩散模式 (F1=0.945)。

- **给我们什么**:
  1. Hawkes 比 SIR 更强: 不需要同质混合假设, 自然处理级联的幂律大小分布, 可调参 (μ=基线强度, α=分支因子/传染力, θ=记忆衰减)。
  2. **对本项目最核心的洞见**: 互激励 Hawkes (mutually exciting) 可建模"梗 A 的爆发出现在梗 B 之后"的跨梗传染 — 这是直接可用的叙事生态系统动力学的数学框架。
  3. 参数 α (分支因子): α<1 亚临界 (梗会消退), α=1 临界 (持久级联), α>1 超临界 (爆炸性病毒传播)。

- **成熟度**: 主流 (地震学→金融→社媒, 50+ 年由理论到应用)。局限: 内存核 φ 的形式 (指数/power-law/Rayleigh) 需假设; 非参数 Hawkes 计算量大。

### 7.3 信息级联 — "fads and fashion" 基础

**Bikhchandani, S., Hirshleifer, D. & Welch, I. (1992).** "A Theory of Fads, Fashion, Custom, and Cultural Change as Informational Cascades." *Journal of Political Economy*, 100(5), 992–1026. ⭐ **信息级联的经典论文**。

- **核心**: 序贯个体观察前人**行动** (非信号), 在仅 2-3 人一致后, 所有后续者理性地忽略自己的私人信息而跟从 — 级联形成。级联概率→1, 但可以是**错误的** (bad cascade)。极其脆弱 — 少量新公开信息可瞬间打破。

**Banerjee, A.V. (1992).** "A Simple Model of Herd Behavior." *Quarterly Journal of Economics*, 107(3), 797–817. 独立提出相近概念 (herd behavior)。

- **给我们什么**:
  1. 梗的"突然爆发 + 突然遗忘"不是非理性的 — 是理性个体在信息稀缺时的最优行为 (局部最优→全局次优)。
  2. 级联的脆弱性 = 叙事"相变"的微观机制候选 — 一个足够强的"反叙事"公开信号可瞬间瓦解已形成的共识级联。
  3. **对本项目的关键**: 这提供了"R2 是共识收敛"假说的微观基础 — Fixation 可能是级联锁死 (所有人看别人都没动, 自己也不动); Peak/Origin 可能是级联崩溃后的探索期。

- **成熟度**: 经典 (被引 5000+)。局限: 级联模型假设严格序贯 — 社交媒体是并发的; 扩展需要结合网络拓扑。

### 7.4 Reflexivity / Performativity — 测量反作用于被测对象

**MacKenzie, D. (2006).** *An Engine, Not a Camera: How Financial Models Shape Markets*. MIT Press. ⭐ 经典书籍 — 金融模型不是"相机"(被动记录), 而是"引擎"(主动塑造)。

**MacKenzie, D. & Millo, Y. (2003).** "Constructing a Market, Performing Theory: The Historical Sociology of a Financial Derivatives Exchange." *American Journal of Sociology*, 109(1), 107–145. ⭐ 经典实证论文 — Black-Scholes 期权定价模型从不准确→被交易员使用→市场向模型预测收敛 (Barnesian performativity)。

- **核心区分** (MacKenzie):
  - Generic performativity: 经济理论被使用但未改变实际过程
  - Effective performativity: 使用理论实际影响了结果
  - **Barnesian performativity**: 使用使结果**更像理论的预测** (反馈闭环)
  - Counter-performativity: 使用使结果不像预测

**Soros, G. (1987).** *The Alchemy of Finance*. 引入 reflexivity: 参与者的认知偏差 → 影响事件 → 事件反哺错误认知 → 正反馈远离均衡。

- **给我们什么 (对本项目至关重要)**:
  1. 这是本项目的**认识论前提之一的形式化表述**: 测量的行为改变被测量对象。叙事分析被传播→反馈回叙事场。
  2. 本项目 FR31 (战略对弈顾问) 的"输出会影响用户的叙事选择"不是 bug, 是 performativity — 需要显式标注, 而不是假装没有。
  3. **直接可用的分析框架**: 当系统的"观测者"也是参与者时 (本项目 Dashboard 的访问者可能同时也是叙事生态的一部分), MacKenzie 的类型学告诉我们如何区分不同程度的反馈。
  4. **与 §5.4 "聚类做了 ≠ 真的有簇"的共鸣**: 方法的选择 (P 选 10 维/GMM 4 簇) 本身就是 performative — 方法框架改变我们对系统的认知, 认知反向影响方法迭代。

- **成熟度**: 主流 (经济社会学 STS, 20+ 年; 近年进入 ML 领域 — Perdomo et al. 2020 "Performative Prediction")。局限: 难以定量测量 performativity 的程度; 本质上是一个质性分析的框架, 量化扩展仍在发展中。

### 7.5 汇总: 梗扩散的方法工具箱

| 需求 | 推荐方法 | 参考文献 |
|------|---------|---------|
| 单梗传播曲线拟合 | SIR ODE/SDE | Wang & Wood (2011) |
| 多梗竞争/跨梗传染 | 互激励 Hawkes 过程 | Hawkes (1971); Rizoiu et al. (2018) |
| 突然爆发的微观机制 | 信息级联 BHW | Bikhchandani et al. (1992) |
| 测量→反馈到系统的认知风险 | Performativity 框架 | MacKenzie & Millo (2003) |
| 级联大小的不确定性 | 分支过程 → Borel 分布 | Rizoiu et al. (2018) |

---

## 8. 多观测者 / 跨算子不变性 / CSS 测量模型

> 用途: 多数据源 (Trends/Scraper/百度指数/Wayback) 的跨源一致性检验 — 不同"观测算子"测的是不是同一个底层叙事状态。

### 8.1 测量不变性 (Measurement Invariance) — 心理测量学经典框架

**Meredith, W. (1993).** "Measurement Invariance, Factor Analysis and Factorial Invariance." *Psychometrika*, 58(4), 525–543. ⭐ **测量不变性的奠基论文** (心理测量学会主席演讲)。

- **核心**: 多组 CFA (验证性因子分析) 的层级检验 — 从最松到最紧, 逐级检验不同数据源 (或不同人群/时点) 测量的是否是同一个潜变量:

| 层级 | 约束 | 含义 |
|------|------|------|
| **Configural** | 同样因子结构 (哪些指标加载到哪些因子) | "不同源看到的因子结构一样" |
| **Metric (Weak)** | + 因子载荷相等 | "一单位潜变量变化在不同源中对应同样的指标变化" |
| **Scalar (Strong)** | + 截距相等 | "潜变量=0 时, 不同源的指标基线相同" |
| **Strict** | + 残差方差相等 | "测量误差在不同源中同样大" |

- **给我们什么**:
  1. **直接的跨算子检验框架**: 用 Trends 测的 x(t) 和用 Scraper embedding 测的 x'(t) — 它们达到哪一级不变性? Configural? Metric? Scalar? 这就是"跨算子不变性"的操作化定义。
  2. 通常 cross-cultural 研究要求至少 **Metric invariance** 才可比均值; 要求 **Scalar invariance** 才可比潜变量分数。本项目: 至少需要 Metric (同方向的协变), Scalar 可能是过于严苛的目标 (不同源的"基线"本就不可能相同)。
  3. **未达到 invariance 不是灾难** — 它意味着不同算子测量的是**不同方面** (不是"同一个 X 的不同视角"), 这会杀掉"存在唯一真实叙事状态"的预设 — 而这正是本项目第四预设 (Competing Explanatory Layer) 的内容。

**Vandenberg, R.J. & Lance, C.E. (2000).** "A Review and Synthesis of the Measurement Invariance Literature: Suggestions, Practices, and Recommendations for Organizational Research." *Organizational Research Methods*, 3(1), 4–70. ⭐ 应用综述 — 如何在组织/跨文化研究中实施测量不变性检验。

**Millsap, R.E. (2011).** *Statistical Approaches to Measurement Invariance*. Routledge. 现代教材。

- **成熟度**: 经典 (心理测量学标准框架, 30+ 年)。局限: 依赖 CFA 假设 (线性指标-因子关系 + 多元正态); 对非正态需稳健标准误; 大样本会 trivial rejection (任何微小偏差都显著) → 需结合 fit 指数 (ΔCFI < 0.01 的实用判据)。

### 8.2 典型相关分析 (CCA) — 多视图学习的经典方法

**Hotelling, H. (1936).** "Relations Between Two Sets of Variates." *Biometrika*, 28(3-4), 321–377. ⭐ CCA 的原始论文。

- **核心**: 给定两视图 X₁ 和 X₂ (如 Trends 51 维 vs Scraper 384 维 embedding), 找投影方向 w₁, w₂ 使投影后的 corr(X₁w₁, X₂w₂) 最大化。可推广到 >2 视图 (Generalized CCA / MCCA)。

**Bach, F.R. & Jordan, M.I. (2005).** "A Probabilistic Interpretation of Canonical Correlation Analysis." *UC Berkeley Technical Report 688*. ⭐ 概率 CCA — 两视图条件独立给定共享潜变量 z。这直接等同于"不同观测算子测量同一潜状态"的概率模型。

**Foster, D.P., Kakade, S.M. & Zhang, T. (2008).** "Multi-View Dimensionality Reduction via Canonical Correlation Analysis." *TTI Technical Report TR-2008-4*. 两视图下 CCA 的维度约简保证: 若视图条件独立于潜状态 H, CCA 恢复 H 张成的子空间。

- **给我们什么**:
  1. 概率 CCA (Bach & Jordan) 是"跨算子测量同一状态"的直接概率模型— 如果 X_trends 和 X_scraper 确实条件独立于共享叙事状态 x(t), 则 CCA 能恢复 x(t)。
  2. 规范相关值 ρ (canonical correlation) 本身是"两视图共享多少信息"的度量 — ρ 若接近 1 说明两个算子高度一致; ρ 接近 0 说明它们测不同东西 → 可能不存在单一的共同潜变量。
  3. **可操作的检验**: 对不同算子对的 CCA 第一规范相关 → 若都不高, 则"跨算子测量同一系统"的预设被质疑; 若对某些对高、某些低 → 某些算子本质上测不同侧面, 应分到不同潜变量。

- **成熟度**: 经典 (Hotelling 1936) / 主流 (概率 CCA 2005)。局限: 假设线性关系; 对高维 sparse 数据需 regularized/sparse CCA; 条件独立假设强 (在测量共享底层机制时可能不成立)。

### 8.3 跨算子不变性 — 本项目专用整合

**没有标准论文** (因为"多观测算子交叉验证叙事状态"这个具体问题没有教科书解决方案)。但上述两个成熟框架 + 本项目已经有的工具可以搭建:

| 检验 | 对应方法 | 问的问题 |
|------|---------|---------|
| **Operational Consistency** | CCA ρ₁ (第一规范相关) | 两算子共享多少一维结构? |
| **Structural Equivalence** | Meredith Metric Invariance (因子载荷相等) | 共享结构中的"因子负载"在算子间是否一致? |
| **Rank Stability** | Cross-operator 状态排序的 Spearman ρ / Kendall τ | 月级状态的排序在不同算子间是否一致? |
| **Disagreement Structure** | CCA 残差 vs 时间/事件 | 两算子分歧时 — 什么时候分歧大? (噪声特征? 选择性覆盖差异?) |

- **成熟度**: **无成熟先例 (针对本项目的具体问题)** — 上述框架的拼接是新组合, 引用 Meredith (1993) + Bach & Jordan (2005) 作为各构件的方法背书。**诚实标注: 跨算子交叉验证在"叙事状态"的上下文中是 UNDERPOWERED/方法先行 — 数据还不够 (不同算子的时间重叠窗口短)。**

---

## 9. 中文社媒历史数据集

> 用途: 列出可得、带许可/出处的历史数据集, 标时间跨度 + 覆盖人群。用于历史高分辨率重建 (破月度分辨率墙)。

### 9.1 Weibo (新浪微博) — 最大的中文社媒平台

**Weiboscope Open Data (香港大学)**

- **论文**: Fu, K.W., Chan, C.H. & Chau, M. (2013). "Assessing Censorship on Microblogs in China: Discriminatory Keyword Analysis and the Real-Name Registration Policy." *IEEE Internet Computing*, 17(3), 42–50.
- **数据**: ~2.27 亿帖, 1,440 万用户。2011–2012 开始持续采集。
- **特色**: 包含被删帖记录 (censorship tracking), 用户/消息 ID 匿名化。
- **获取**: [datahub.hku.hk](https://datahub.hku.hk/articles/dataset/Weiboscope_Open_Data/16674565) — 开放获取。
- **覆盖**: 微博活跃用户子集 — 偏城市/年轻/教育程度较高的中文互联网用户。不是中国人口随机样本。
- **给我们什么**: ★★ **对本项目最珍贵的资源** — 2011 年起的历史微博文本, 可直接回溯"梗"在微博上的发展轨迹。被删帖记录更是一个独特的"叙事压力"指标。

**Weibo-COV**

- **论文**: Hu, Y., Huang, H., Chen, A. & Mao, X.L. (2020). "Weibo-COV: A Large-Scale COVID-19 Social Media Dataset from Weibo." arXiv:2005.09174.
- **数据**: ~4,090 万帖。2019-12 至 2020-04。179 个 COVID-19 关键词。20M 活跃用户池。
- **获取**: [GitHub](https://github.com/nghuyong/weibo-public-opinion-datasets) — 需数据使用协议。
- **覆盖**: 疫情初期, 微博活跃用户。时间窗短但分辨率极高 (天/小时级)。
- **给我们什么**: 高分辨率 (小时级) 的单一事件叙事演化 — 可作为"叙事相变"的高分辨率证据, 提供月度分析无法看到的快速动力学。

**Multi-Domain False News Dataset**

- **论文**: 发表于 *Information Processing & Management* (2022)。
- **数据**: 44,728 帖, 9 领域, 40,215 用户, 340 万转发。2009–2019 (10 年)。
- **获取**: [GitHub](https://github.com/ICTMCG/Characterizing-Weibo-Multi-Domain-False-News)。
- **给我们什么**: 10 年跨度的虚假信息扩散网络 — 可作为"叙事污染"的标记。

**Weibo NER Corpus**

- **论文**: Peng, N. & Dredze, M. (EMNLP 2015; ACL 2016).
- **数据**: 1,890 标注消息。2013-11 至 2014-12。CC BY-SA 3.0 许可。
- **获取**: HuggingFace `minskiter/weibo` 或 [openi.pcl.ac.cn](https://openi.pcl.ac.cn/kewei/golden-horse)。
- **给我们什么**: 中文社媒 NLP 的标准基准 — 可用来训练/评估本项目 embedding 管道在中文社媒上的 NER 性能。

### 9.2 新闻文本数据集

**GDELT (Global Database of Events, Language, and Tone)**

- **论文**: Leetaru, K. & Schrodt, P.A. (2013). "GDELT: Global Data on Events, Location, and Tone, 1979–2012." *ISA Annual Convention*.
- **数据**: 1979–今, ~15 分钟更新, 含大量中文新闻源 (人民日报/新华社等)。
- **获取**: [gdeltproject.org](https://www.gdeltproject.org/) — 开放下载 (Google BigQuery / 原始 CSV)。
- **覆盖**: 全球新闻机构, 含中国。**不是中文互联网原生叙事, 是机构新闻的叙事框架。**
- **给我们什么**: ★★ 可作为"外部控制变量" — GDELT 的 Tone/event counts 能做 Google Trends 的平行对照; 1979 年起的历史深度可检验"互联网之前"的叙事基线。
- **已知局限** (对本项目): ① 来源是机构新闻而非社交媒体 — 测量的是"精英叙事框架", 不是"大众叙事反应"。② GDELT 的事件编码 (CAMEO) 对中文新闻的覆盖误差未知。③ 用于本项目时应标注 [Operator: GDELT-news], 与 [Operator: GoogleTrends-CN] 平行的观测算子。

**中文新闻分类数据集**

- **THUCNews**: 清华 NLP 组, 74 万条新闻标题+正文, 14 类。2005–2011 新浪 RSS。获取: [thunlp.org](http://thunlp.org/)。
- **SogouCA / SogouCS**: 搜狗实验室, 大规模新闻全文语料 (2012 年)。获取: [sogou.com/labs](https://www.sogou.com/labs/resource/ca.php) (可能已下线, 需确认)。
- **给我们什么**: 训练中文新闻分类/embedding 的基础语料 — 间接有用, 但不直接提供历史叙事动力学数据。

### 9.3 其他数据源

**百度指数 (Baidu Index)**

- 类似 Google Trends 但针对中国大陆。覆盖比 Google Trends geo=CN 广但**无公开下载** — 需爬取或学术合作。
- **已知限制**: 百度指数的用户画像报告 (年龄/性别/地域) 不公开; 指数 0–100 归一化 + 关键词限制。

**Wayback Machine (Internet Archive)**

- 1996–今的网页快照。是**不可再生的高分辨率历史痕迹** (本项目已标注)。
- 获取: 通过 `archive.org` API / CDX 索引。
- 已知偏差: archive density 混杂 (§1, §2 的方法即为此设计)。

**国家基础科学数据中心 — 5M+ 内容安全语料**

- 2020–2024, 514 万+ 文本记录, 2.74GB。三分类 (正面/中性/有害)。
- 获取: [nbsdc.cn](https://cstr.cn/16666.11.nbsdc.zYso7hP5) — 国家科技资源共享服务平台。
- **给我们什么**: 内容安全分类的官方标注 — 对"叙事被监管干预"的分析有价值, 但时间跨度短。

### 9.4 数据集汇总

| 数据集 | 时间跨度 | 大小 | 获取难度 | 对本项目最直接的用途 |
|--------|---------|------|:---:|------|
| **Weiboscope** | 2011–2012+ | 227M 帖 | 开放 | ★ 历史微博文本 — 回溯梗轨迹 |
| **Weibo-COV** | 2019–2020 | 41M 帖 | 需申请 | 高分辨率单一事件叙事演化 |
| **False News (2009-2019)** | 2009–2019 | 45K 帖+转发网络 | 开放 | 叙事污染的传播网络 |
| **GDELT** | 1979–今 | 全量 | 开放 | ★ 外部控制变量 — 机构新闻叙事框架 |
| **THUCNews** | 2005–2011 | 74 万条 | 开放 | 基础 NLP 训练语料 |
| **SogouCA/CS** | 2012 | 大 | 确认状态 | 基础 NLP 训练语料 |
| **Wayback Machine** | 1996–今 | N/A | API 开放 | ★ 不可再生的网页历史快照 |
| **百度指数** | 2011–今 | N/A | 无公开下载 | GDP 对应物 — 覆盖比 Google 广但不可复制 |
| **B站 (bilibili)** | 2009–今 | — | 本项目已采集部分 | 已采集 22 条叙事档案 |

> **★ 标注 = 高优先级回填源。** 其余为辅助/训练语料。

### 9.5 重要警示: 覆盖 ≠ 代表

上述所有数据集的共同局限:

1. **城市/年轻/教育偏差**: 社媒用户不等于中国人口 — 微博/B站/知乎的用户画像偏城市中产和年轻人。
2. **平台选择偏差**: 每个平台是一个独立生态 (微博=公共广场, B站=亚文化, 知乎=长文辩论) — 单一平台的叙事不等于"中文互联网叙事"。
3. **审查/内容删除偏差**: 微博和 B站存在内容删除 — 被删的内容恰恰可能是叙事压力最强的信号 (censorship as signal)。
4. **归档不完整**: Weiboscope 虽持续采集但不是全量 — 存在缺失期和 API 限制。
5. **这就是 §1 和 §2 方法要解决的核心问题**: 对所有数据集的选择偏差 (覆盖、删除、归档密度) 都要做标注, 不上报"这代表中国互联网叙事", 而报"这是在 [Operator: X] 的 [Known bias: Y] 下观测到的东西"。

---

> **状态**: 全部 9 题初稿完成 (§1–9)。最后更新: 2026-07-13。
