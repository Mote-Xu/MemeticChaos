"""
Evidence Ledger — 证据分级账本 (第八轮外部 AI 共识, 2026-07-09)

把项目所有结论按 epistemic 等级归档, 并显式记录每条结论挂靠哪些底层假设。
核心价值 (GPT): 当某个假设被证伪, **只有挂靠它的高等级条目 (E3/E4) 失效,
低等级观测 (E0/E1/E2) 继续保留** —— 项目不会因单个假设倒塌而整盘崩。

这不是又一个解释框架, 是**方法论资产** (长期保留)。它同时是假设生命周期的载体:
一个结论/假说何时出生 (active)、降级 (suspended)、退休 (retired), 有据可查。

等级 (E0→E4, 越高越依赖解释):
  E0 原始观测   — 未加工数据
  E1 统计描述   — 数据在说什么, 不加解释
  E2 操作结果   — 依赖算法/参数选择, 换工具可变
  E3 条件解释   — 有条件推断, 依赖具体假设
  E4 机制假说   — 竞争并存, 待证据淘汰 (Competing Explanatory Layer)

用法:
  conda run -n MemeticChaos python src/analysis/evidence_ledger.py            # 完整账本报告
  conda run -n MemeticChaos python src/analysis/evidence_ledger.py --if-fails time-invariance  # 级联失效查询
  conda run -n MemeticChaos python src/analysis/evidence_ledger.py --grade E4  # 只看机制假说
  conda run -n MemeticChaos python src/analysis/evidence_ledger.py --json      # 导出 JSON
"""

import json, sys, argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
OUTPUT_PATH = ROOT / "data" / "processed" / "evidence_ledger.json"

# ═══════════════════════════════════════════════
# 底层假设注册表 — 每条结论挂靠的前提
# ═══════════════════════════════════════════════
# status: holds(当前采纳) / untestable(不可检验) / falsified(已证伪) / suspect(存疑)

ASSUMPTIONS = {
    "time-invariance": {
        "desc": "存在时不变的生成机制 (2015-2025 同一系统)",
        "status": "split",
        "note": ("第九轮拆三层 (GPT): P2a 所有低维参数平稳 = 对 PC2 REJECTED "
                 "(Nyblom AR(2), 残差白噪声, p=0.0065); P2b 全局机制平稳 = 未决; "
                 "P2c 观测算子平稳 = 未决。低维层非无功率, 但 PC2 非恒定性对因果三支简并。"),
    },
    "obs-operator-stationarity": {
        "desc": "观测算子时不变 (梗'变异'/'漂移'的含义跨时代不变)",
        "status": "suspect",
        "note": "P2c。PC2 载 mutation/semantic_drift; 若含义随时代变(2020真实迁移 vs 2023表达形式), 自相关上升可能是测量变非系统变。接 Q8 (narrative-as-observation)。relaxation_probe 碰不到此支。",
    },
    "regime-discretization": {
        "desc": "状态空间可切成离散相区 (GMM) — 母假设, 第十一轮已拆 RD-a/b/c",
        "status": "split",
        "note": ("regime_discreteness.py (gap/Jaccard/Silverman + RQA 交叉, 审计层 2026-07-12) 镜像 "
                 "time-invariance 的 P2a/b/c 拆法, 拆三层: RD-a [[regime-lowdim-structure]] 低维有 label-free "
                 "结构 = holds; RD-b [[regime-count]] 分成 K 个离散相区 = suspect; RD-c [[r2-basin]] R2 是离散"
                 "分离盆地 = not-supported。★两头都不获支持★: 4-clean-regimes 与 pure-continuum 皆无支撑——有真"
                 "低维结构(RD-a, PC1 多峰), 但 4-相区切分不稳(RD-b), R2 盆地vs漂移区 underpowered(RD-c)。"
                 "挂靠者已改挂对应子假设, 按子假设复核 (suspect/not-supported ≠ falsified, 不自动失效)。"),
    },
    "regime-lowdim-structure": {
        "desc": "RD-a: 低维空间存在 label-free 结构 (非纯连续均匀)",
        "status": "holds",
        "note": ("PC1 Silverman 多峰 p=0 (轴由数据定义, 非标签 → 非循环); 功率自检通过 (uni0.37/bi0.0)。"
                 "这是**反 pure-continuum** 的硬证据 —— 阻止把 RD 读成'纯连续'。"),
    },
    "regime-count": {
        "desc": "RD-b: 状态空间分成 K 个离散相区 (K=4 计数/切分是稳健结构)",
        "status": "suspect",
        "note": ("gap statistic 最优 k=1 (BIC 只从 k≥3 搜, 结构性排除 k=1/2); GMM(4) 全簇自举 Jaccard "
                 "0.20-0.56 (R2 亦仅 0.48); curation 众数 7、仅 4% 复现 k=4。计数非稳健结构, 是操作切分。"
                 "★raw E1 统计 (0复发/切换率/方差比) 作为数存活, 仅'跨相区/regime'框定被 flag。★"),
    },
    "r2-basin": {
        "desc": "RD-c: R2 是离散分离盆地 (而非连续流形上的高驻留漂移区)",
        "status": "not-supported",
        "note": ("★两读法并存 (Competing Explanatory Layer), 数据分不开 (UNDERPOWERED)★: (I) 连续 drift-dwelling "
                 "(慢漂入新区并驻留); (II) 真实离散盆地 + 检测 underpowered。二者对同一证据 (RQA 零复发 / weak-irrev "
                 "/ PC2 持久性升) reconcile 得一样好——而这三路都在 regime 框架内算, 打不破平局。★不 crown "
                 "drift-dwelling★ (那是给单一图景加冕)。注: R2 as **高驻留区** (无盆地本体承诺) = holds; "
                 "R2 as **离散分离盆地** = not-supported。R2_axis Silverman 因轴由标签定义=循环, 不作证据。"),
    },
    "aggregation": {
        "desc": "宏观状态 = 微观数据聚合, 无反因性损失",
        "status": "suspect",
        "note": "架构级预设3。聚合反因性可能塌缩个体强结构 (Q8)。未检验。",
    },
    "attention-proxy": {
        "desc": "Google Trends 月搜索量 = [Operator: GoogleTrends-CN] 接入定义框的注意力",
        "status": "holds",
        "note": ("★观测算子重构 (2026-07-12): 不是'集体/大众注意力', 是'搜索被 Google 归属到 CN 的查询'的注意力 "
                 "—— 由'能否够到 Google'技术门槛定义, 成分异质 UNKNOWN (禁标精英/VPN少数)。127月 H 侧结论"
                 "全部 conditional on 此框。★历史段无干净'国内大众个体'算子 (见 operator_ledger honesty_ceiling)★。"
                 "3-7天滞后 + 仅已知关键词; 可回填日级但只给注意力、给不了叙事特征 → 拼不出日级 x(t), 不重跑 regime。"),
    },
    "narrative-as-observation": {
        "desc": "采集数据 = 对现实结构的观测",
        "status": "falsified",
        "note": "Q8 翻转: 采集的是人群叙事化反应, 不是现实观测。但叙事化倾向本身是观测目标。",
    },
    "markov": {
        "desc": "x(t+1)=F(x(t),u,y) 一阶马尔可夫, 无长记忆",
        "status": "suspect",
        "note": "非线性/长记忆情况从未检验。",
    },
    "stage-ontology": {
        "desc": "五阶段 (origin/emergence/peak/controversy/fixation) 是真实生命周期结构",
        "status": "suspect",
        "note": "本体离散化 (危险)。人工阈值划分, 和数据驱动原则矛盾。ARI=0.27: 机器见3类非5类。",
    },
    "meme-homogeneous": {
        "desc": "'梗' 是同质分析单元",
        "status": "suspect",
        "note": "'躺平'(3年情绪) 和 '鸡你太美'(2月娱乐) 被当同类数据点。",
    },
    "single-picture": {
        "desc": "存在唯一正确的物理图景",
        "status": "falsified",
        "note": "第八轮 GPT: R2 的多个解释可能是不同统计量的不同投影, 未必互斥。→ Competing Explanatory Layer。",
    },
}

# ═══════════════════════════════════════════════
# 账本条目 — 从现有结论归档
# ═══════════════════════════════════════════════
# grade: E0-E4; depends: 挂靠的 assumption keys; status: active/suspended/retired

LEDGER = [
    # ── E1 统计描述 ──
    {"id": "r2-persistence", "grade": "E1", "value": "0.973",
     "statement": "R2 相区自持概率 97.3%",
     "depends": ["regime-count", "time-invariance"],
     "source": "regime_detector.py / regime_map.json", "status": "active",
     "note": "raw E1 (给定分区的自持率是数, 存活); 'R2 是盆地' 框定挂 [[r2-basin]] not-supported。"},
    {"id": "r2-variance-inflation", "grade": "E1", "value": "2.31x",
     "statement": "R2 内部状态方差从 early→late 放大 2.31×",
     "depends": ["regime-count"],
     "source": "ms_ar_first_cut.py", "status": "active",
     "note": "raw E1 存活; 方差放大兼容 basin-底变平 与 drift-dwelling-内表达多样化 两读法, 不判偏。"},
    {"id": "r2-pc-coupling", "grade": "E1", "value": "PC4 r=+0.85, PC5 r=-0.87",
     "statement": "R2 内部 PC4/PC5 与 z1 强相关 (p≈0, Bonferroni 后显著)",
     "depends": ["regime-count"],
     "source": "ms_ar_first_cut.py", "status": "active",
     "note": "raw E1 存活; R2 分组框定挂 [[regime-count]] suspect。"},
    {"id": "rqa-zero-recurrence", "grade": "E1", "value": "0 对",
     "statement": "R2 与 R1/R3 零跨相区 ε-复发 (≥12月间隔)",
     "depends": ["regime-count"],
     "source": "irreversibility_test.py", "status": "active",
     "note": ("★raw E1 (0复发是事实) 存活★; '跨相区分离' 的解释挂 [[r2-basin]] not-supported —— "
              "连续慢漂移入新区并驻留同样产生零复发, 不独证盆地。")},
    {"id": "slice-subspace-angles", "grade": "E1", "value": "40.9/48.3/52.7°",
     "statement": "三时段两两子空间均值夹角, 全低于连续区块噪声 p95=70.7°",
     "depends": [],
     "source": "temporal_slice_audit.py", "status": "active"},
    {"id": "switch-rate", "grade": "E1", "value": "0.111/月",
     "statement": "regime 切换率 0.111/月 (127月14次), 归一化熵 0.932",
     "depends": ["regime-count"],
     "source": "regime_detector.py", "status": "active",
     "note": "★raw E1 (切换计数是事实) 存活★; 'regime 切换' 框定挂 [[regime-count]] suspect。"},

    # ── E2 操作结果 ──
    {"id": "pca-d90", "grade": "E2", "value": "d90=10",
     "statement": "18维特征 PCA 90%方差需 10 维",
     "depends": ["aggregation"],
     "source": "representation_learning.py", "status": "active"},
    {"id": "counterfactual-platform", "grade": "E2", "value": "16.5x > 1.9x",
     "statement": "Counterfactual: 平台组漂移比 16.5× > AI 组 1.9×",
     "depends": ["attention-proxy"],
     "source": "control_manifold.py", "status": "active"},
    {"id": "gmm-regimes", "grade": "E2", "value": "4簇→3相区",
     "statement": "GMM+BIC 选出 4 观测簇, 合并 3 物理相区",
     "depends": ["regime-count", "aggregation"],
     "source": "regime_detector.py", "status": "active",
     "note": "★'4簇' 计数非稳健 (gap→k=1, Jaccard 全<0.56) → 降级为操作切分, 非世界的相区数。★"},
    {"id": "regime-discreteness-weak", "grade": "E2", "value": "gap→k=1; R2 Jaccard 0.48; PC1 多峰",
     "statement": ("regime 离散性审计: gap 最优 k=1 (BIC 只搜 k≥3), GMM(4) 全簇 Jaccard 0.20-0.56 (R2=0.48); "
                   "唯 PC1 (label-free) Silverman 多峰 p=0; PC2/PC3 单峰"),
     "depends": [],
     "source": "regime_discreteness.py", "status": "active",
     "note": ("★不 crown 任一极★: 4-clean-regimes 与 pure-continuum 皆不获支持。RD-a 真结构存在 (PC1 多峰, 非循环); "
              "RD-b 计数脆弱; RD-c R2 盆地 vs 漂移驻留 UNDERPOWERED。★R2_axis Silverman p=0 是循环★ (轴由 GMM "
              "标签定义再验分离 = 同数据既提又验; 单 Gaussian blob 也 p=0) → 仅上界, 不作证据。resolve "
              "RQA-vs-curation 张力 (二者分别成立于不同粒度, 非矛盾) 是真进展; 但 basin-vs-drift 平局未破。")},
    {"id": "pc1-labelfree-multimodal", "grade": "E1", "value": "Silverman p=0",
     "statement": "PC1 (label-free 轴) Silverman 单峰检验拒绝: p_multimodal=0, 多峰 (功率自检通过)",
     "depends": [],
     "source": "regime_discreteness.py", "status": "active",
     "note": "非循环 (PC1 数据定义, 非标签); 支持 [[regime-lowdim-structure]] holds —— 反 pure-continuum 的硬证据。"},
    {"id": "narrative-ari", "grade": "E2", "value": "ARI=0.27",
     "statement": "叙事聚类 ARI=0.27: 机器见 3 类, 非人工 5 类",
     "depends": ["stage-ontology", "meme-homogeneous"],
     "source": "narrative_clustering.py", "status": "active"},
    {"id": "structure-r2", "grade": "E2", "value": "R²=0.79",
     "statement": "注意力结构 (HHI+叙事熵) 可从外部场预测 R²=0.79",
     "depends": ["attention-proxy"],
     "source": "order_form_predictor.py", "status": "active"},
    {"id": "monthly-semantic-2026", "grade": "E2", "value": "2026-06/07",
     "statement": "月度语义状态 (cov_trace/各向异性/漂移) 已建, 覆盖 2026-06/07",
     "depends": ["narrative-as-observation"],
     "source": "monthly_aggregator.py / monthly_semantic_state.json", "status": "active"},

    # ── E3 条件解释 ──
    {"id": "r2-real-cluster", "grade": "E3", "value": "分离 CONTESTED",
     "statement": "R2 是状态空间真实结构分离, 非 GMM artifact",
     "depends": ["r2-basin", "time-invariance"],
     "source": "irreversibility_test.py (RQA)", "status": "active",
     "note": ("★RD-c not-supported★: R2 as 真实分离盆地不获支持; R2 as 高驻留区 holds。"
              "drift-dwelling vs underpowered-basin 竞争未破。active+flag, 非 falsified (suspect≠证伪)。")},
    {"id": "weak-irreversibility", "grade": "E3", "value": "WEAK",
     "statement": "不可逆性来自外部场慢漂移, 非系统内禀 (Time-Reversal 对称)",
     "depends": ["regime-count", "time-invariance"],
     "source": "irreversibility_test.py", "status": "active",
     "note": "在 regime 框架内算; '慢漂入盆地' 同样兼容 → 打不破 basin-vs-drift 平局。active+flag。"},
    {"id": "h1a-lowdim", "grade": "E3", "value": "SUPPORTED",
     "statement": "H1a: 叙事状态存在低维表示",
     "depends": ["aggregation"],
     "source": "representation_learning.py", "status": "active"},
    {"id": "h1b-rejected", "grade": "E3", "value": "REJECTED",
     "statement": "H1b: 月度点预测不可行, 状态近似随机游走 (VARX R²=-0.32)",
     "depends": ["markov"],
     "source": "representation_learning.py", "status": "active"},
    {"id": "p2-underpowered", "grade": "E3", "value": "UNDERPOWERED",
     "statement": "时不变性在月度分辨率下 not identifiable (resolution-dependent)",
     "depends": ["time-invariance"],
     "source": "temporal_slice_audit.py", "status": "active"},
    {"id": "p2a-rejected-pc2", "grade": "E3", "value": "P2a REJECTED (PC2)",
     "statement": "低维参数平稳性 (P2a) 对 PC2 被拒: AR(2) 参数在127月内非恒定",
     "depends": [],
     "source": "nyblom_stationarity.py", "status": "active",
     "note": "Nyblom L=1.27, null p95=0.91, p=0.0065; 残差过 Ljung-Box(0.15) 白噪声; 反驳'分辨率墙杀死一切'。"},
    {"id": "relaxation-weakened", "grade": "E3", "value": "LEAN_SLOWING (exploratory)",
     "statement": "PC2 relaxation 结构改变: ρ与var同升(corr0.78)、ρ领先var",
     "depends": ["obs-operator-stationarity"],
     "source": "relaxation_probe.py", "status": "active",
     "note": "exploratory (有效独立窗~2, 同分辨率墙); 倾向slowing但排除不了(c)观测算子改变。"},

    # ── E1 (Nyblom 新增) ──
    {"id": "pc2-ar-nonconstancy", "grade": "E1", "value": "p=0.0065",
     "statement": "PC2 (churn/mutation轴) AR(2) 参数非恒定 (Nyblom, 残差白噪声)",
     "depends": [],
     "source": "nyblom_stationarity.py", "status": "active"},
    {"id": "pc2-persistence-var-rise", "grade": "E1", "value": "ρ0.57→0.94, var0.53→2.17",
     "statement": "PC2 AR自相关前半0.57→后半0.94, 方差0.53→2.17",
     "depends": [],
     "source": "nyblom_stationarity.py + relaxation_probe.py", "status": "active"},
    {"id": "pc2-tvp-drift", "grade": "E1", "value": "σ̂²_η=0.00155, boot_p=0.009",
     "statement": "PC2 TVP-AR 估到非零系数漂移: q̂=0.0072, σ̂²_η=0.00155, LR=7.77, boot_p=0.009; AR(2) persistence 主根模 |ρ| 0.42→0.86 (Σφ 0.39→0.83) 逐月上升",
     "depends": [],
     "source": "tvp_ar_stationarity.py", "status": "active",
     "note": ("Nyblom L (score test, p=0.0065) 的**估计版对应物, 非独立通道** —— 同一 σ²_η=0 假设的检验版 vs 估计版, "
              "两估计器结论一致 (boot_p 0.009 vs Nyblom 0.0065)。增量=漂移的幅度 σ²_η + β(t) 轨迹。"
              "★不 resolve P2★: β 时变对归因三支 (机制改变/critical slowing/观测算子改变) 简并, TVP-AR 分不开。"
              "★persistence 口径★: PC2 是 AR(2), 用主根模/Σφ (非单 φ1); 与 [[pc2-persistence-var-rise]] "
              "relaxation_probe 的 0.57→0.94 是同一定性发现 (PC2 持久性上升) 的不同参数化 (AR2 主根 vs AR1 系数), 非矛盾。")},
    {"id": "pc1-tvp-underpowered", "grade": "E1", "value": "q̂=0(边界), boot_p=1.0",
     "statement": "PC1 TVP-AR 未估到漂移 (q̂ 堆在边界 0); 但 H0(恒定系数) 下 MLE 37.5% 堆积在 0",
     "depends": [],
     "source": "tvp_ar_stationarity.py", "status": "active",
     "note": ("★σ̂²_η≈0 ≠ 时不变★: N=127 下 σ²_η MLE pile-up-at-zero (真无漂移时也 37.5% 堆 0) + 低功率 "
              "(正对照 AR φ 0.2→0.9 有 ~1/6 漏检)。无漂移是 UNDERPOWERED, 不坐实 P2。合成正/负对照已验估计器不假阳。")},

    # ── E4 机制假说 (Competing Explanatory Layer — 并存待淘汰) ──
    {"id": "r2-hysteresis", "grade": "E4", "value": "候选",
     "statement": "R2 = hysteresis basin (滞回势阱), 系统被外部约束困住",
     "depends": ["r2-basin", "time-invariance", "single-picture"],
     "source": "早期物理隐喻", "status": "active",
     "note": "结构崩塌图景。逃逸需外部冲击。RD-c not-supported 削弱之; E4 竞争层保留, 对手 = [[r2-drift-dwelling]]。"},
    {"id": "r2-drift-dwelling", "grade": "E4", "value": "候选(与盆地并列)",
     "statement": "R2 = 连续流形上的高驻留漂移区 (慢漂入新区并驻留), 非离散分离盆地",
     "depends": ["regime-lowdim-structure"],
     "source": "regime_discreteness.py (RD-c 竞争读法)", "status": "active",
     "note": ("反盆地假说: 挂 [[regime-lowdim-structure]] (RD-a holds, 有真实高驻留区可漂入), ★不挂 r2-basin★ "
              "—— 否则 r2-basin not-supported 时本假说会被误列级联失效 (方向反了)。与 [[r2-hysteresis]](盆地) 是 "
              "Competing pair, 由 [[r2-basin]] 的 status 区分二者 (basin holds→hysteresis favored; "
              "not-supported→本读法 favored), 而非 depends 级联。⚠对称约束: 不 crown 本读法也不 crown 盆地。")},
    {"id": "r2-consensus", "grade": "E4", "value": "候选",
     "statement": "R2 = consensus convergence, 百万个体独立收敛到同一应对策略",
     "depends": ["aggregation", "single-picture", "narrative-as-observation"],
     "source": "Q9 替代假说", "status": "active",
     "note": "共识收敛图景。方差放大=同一策略内表达创新。⚠ Gemini 主张彻底采纳=预设犯案。"},
    {"id": "r2-protocol", "grade": "E4", "value": "候选",
     "statement": "R2 = communication protocol 收敛 (黑话/梗脱离观点本身)",
     "depends": ["single-picture", "narrative-as-observation"],
     "source": "第七轮 GPT", "status": "active",
     "note": "通信编码协议图景。"},
    {"id": "r2-damper", "grade": "E4", "value": "候选",
     "statement": "R2 = narrative damper, 算法与人群反馈回路相干增强",
     "depends": ["single-picture", "attention-proxy"],
     "source": "第七轮 Gemini", "status": "active",
     "note": "叙事阻尼器图景。"},

    # ── E4 PC2 relaxation 变化的三支归因 (第九轮, 简并并列 pending) ──
    {"id": "pc2-cause-slowing", "grade": "E4", "value": "候选(倾向)",
     "statement": "PC2 relaxation 减弱 = 固定机制逼近分岔 (critical slowing)",
     "depends": ["time-invariance"],
     "source": "relaxation_probe (LEAN_SLOWING)", "status": "active",
     "note": "exploratory 支持 (ρ领先var), 但不 confirmatory; 与 WEAK_IRREVERSIBILITY 兼容。"},
    {"id": "pc2-cause-mechanism", "grade": "E4", "value": "候选",
     "statement": "PC2 relaxation 变化 = 生成机制真的改变 (F_t≠F)",
     "depends": [],
     "source": "Nyblom 简并支(a)", "status": "active"},
    {"id": "pc2-cause-obsop", "grade": "E4", "value": "候选",
     "statement": "PC2 变化 = 观测算子改变 (梗'变异'含义随时代变), 非系统变",
     "depends": ["obs-operator-stationarity"],
     "source": "Nyblom 简并支(c), GPT 补", "status": "active",
     "note": "relaxation_probe 碰不到此支; 接 Q8 narrative-as-observation。"},
]

GRADE_DESC = {
    "E0": "原始观测", "E1": "统计描述", "E2": "操作结果",
    "E3": "条件解释", "E4": "机制假说",
}


def build_ledger() -> dict:
    return {
        "source": "evidence_ledger.py",
        "principle": "假设倒了只失效挂靠它的高级条目 (E3/E4); E0/E1/E2 保留",
        "grades": GRADE_DESC,
        "assumptions": ASSUMPTIONS,
        "entries": LEDGER,
    }


def cascade_query(assumption: str) -> dict:
    """若某假设倒了, 三桶分流 (审计层 spec, 2026-07-12):

      invalidated     = 挂靠该假设的 E3/E4 (条件解释/机制假说 — 失效)
      survive_flagged = 挂靠该假设的 E0/E1/E2 (数存活, 但'挂靠该假设的框定'被 flag)
      survive_clean   = 未挂靠该假设 (不受影响)

    这样 code 与条目 note 一致: raw E1 (0复发/切换计数/GMM 输出4簇) 是事实 → 存活,
    只是'跨相区/physical regime' 框定被 flag, 而非笼统'失效'。
    (两桶版会把 raw E1 误列为失效, 与其 note 'survive' 自相矛盾。)
    """
    if assumption not in ASSUMPTIONS:
        return {"error": f"未知假设 '{assumption}'。可选: {list(ASSUMPTIONS.keys())}"}
    invalidated, survive_flagged, survive_clean = [], [], []
    for e in LEDGER:
        if assumption in e["depends"]:
            (invalidated if e["grade"] in ("E3", "E4") else survive_flagged).append(e)
        else:
            survive_clean.append(e)
    return {
        "assumption": assumption,
        "assumption_desc": ASSUMPTIONS[assumption]["desc"],
        "assumption_status": ASSUMPTIONS[assumption]["status"],
        "invalidated": invalidated,
        "survive_flagged": survive_flagged,
        "survive_clean": survive_clean,
    }


def print_report():
    print("=" * 64)
    print("Evidence Ledger — 证据分级账本")
    print("=" * 64)

    # 按等级分组
    for g in ["E0", "E1", "E2", "E3", "E4"]:
        entries = [e for e in LEDGER if e["grade"] == g]
        if not entries:
            continue
        print(f"\n── {g} {GRADE_DESC[g]} ({len(entries)}) ──")
        for e in entries:
            dep = ", ".join(e["depends"]) if e["depends"] else "(无假设依赖)"
            print(f"  [{e['id']}] {e['statement']}")
            print(f"      value={e['value']}  挂靠: {dep}")
            if e.get("note"):
                print(f"      note: {e['note']}")

    # 假设健康度
    print(f"\n{'─'*64}\n假设注册表 (底层前提健康度):")
    icon = {"holds": "✅", "untestable": "❓", "suspect": "⚠️", "falsified": "❌",
            "split": "🔀", "not-supported": "🚫"}
    for k, a in ASSUMPTIONS.items():
        n_dep = sum(1 for e in LEDGER if k in e["depends"])
        print(f"  {icon.get(a['status'],'?')} {k:<24s} [{a['status']:<13s}] {n_dep} 条挂靠 — {a['desc']}")

    # 风险提示: 挂靠 falsified/untestable/not-supported 假设的 E3/E4 (flag, 非自动失效)
    print(f"\n{'─'*64}\n⚠ 脆弱条目 (挂靠 falsified/untestable/not-supported 假设的 E3/E4):")
    for e in LEDGER:
        if e["grade"] in ("E3", "E4"):
            bad = [d for d in e["depends"]
                   if ASSUMPTIONS[d]["status"] in ("falsified", "untestable", "not-supported")]
            if bad:
                print(f"  [{e['grade']}] {e['id']:<20s} 挂靠 {bad}")


def main():
    ap = argparse.ArgumentParser(description="Evidence Ledger")
    ap.add_argument("--if-fails", type=str, help="级联失效查询: 某假设被证伪后果")
    ap.add_argument("--grade", type=str, help="只看某等级 (E0-E4)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--save", action="store_true", default=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.if_fails:
        r = cascade_query(args.if_fails)
        if "error" in r:
            print(r["error"]); return
        print("=" * 64)
        print(f"级联失效查询: 若 '{r['assumption']}' 被证伪")
        print(f"  ({r['assumption_desc']}, 当前状态={r['assumption_status']})")
        print("=" * 64)
        print(f"\n❌ 失效 ({len(r['invalidated'])} 条 E3/E4 — 条件解释/机制假说):")
        for e in r["invalidated"]:
            print(f"  [{e['grade']}] {e['id']}: {e['statement']}")
        print(f"\n🚩 数存活但框定被 flag ({len(r['survive_flagged'])} 条 E0/E1/E2 — "
              f"统计量是事实, 挂靠该假设的框定被 flag):")
        for e in r["survive_flagged"]:
            print(f"  [{e['grade']}] {e['id']}: {e['statement']}")
        print(f"\n✅ 完全不受影响 ({len(r['survive_clean'])} 条, 未挂靠该假设):")
        for e in r["survive_clean"]:
            if e["grade"] in ("E0", "E1", "E2"):
                print(f"  [{e['grade']}] {e['id']}: {e['statement']}")
        return

    if args.grade:
        g = args.grade.upper()
        print(f"── {g} {GRADE_DESC.get(g,'?')} ──")
        for e in LEDGER:
            if e["grade"] == g:
                print(f"  [{e['id']}] {e['statement']} (挂靠: {', '.join(e['depends']) or '无'})")
        return

    print_report()

    if args.save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(build_ledger(), f, ensure_ascii=False, indent=2)
        print(f"\n已保存 → {OUTPUT_PATH}")

    if args.json:
        print(json.dumps(build_ledger(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
