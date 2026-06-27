"""
集成两层预测模型 — 外部场 + 内部叙事约束 → 秩序形态预测

Architecture:
  Layer 0 (Data):  每月构建特征 — 外部场(51维) + 集体约束场(5维) + 混沌轴 + 类别分布
  Layer 1 (External):  外部场 → Ridge → 基线混沌轴预测
  Layer 2 (Internal):  叙事约束场 → Ridge → 混沌轴修正 + 约束场预测
  Output:  秩序形态 = f(混沌轴, 约束场, 类别分布) → 聚类命名 + NL描述

用法:
    python src/models/order_form_predictor.py              # 训练+评估+报告
    python src/models/order_form_predictor.py --forecast 6  # 预测未来6个月
    python src/models/order_form_predictor.py --report      # 仅生成当前报告
"""

import json, sys, os, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

warnings.filterwarnings("ignore")
np.random.seed(42)

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ═══════════════════════════════════════════════
# Paths & Config
# ═══════════════════════════════════════════════

ROOT = Path(__file__).parent.parent.parent  # MemeticChaos/
DATA_DIR = ROOT / "data/collector"
PROCESSED_DIR = ROOT / "data/processed"
NARRATIVE_DIRS = [
    PROCESSED_DIR / "narratives",
    PROCESSED_DIR / "narratives_from_trends",
]

# 5 类别的混沌轴位置 (人工标注, 来自策展数据)
CATEGORY_CHAOS = {
    "解构自嘲": -0.33, "攻击发泄": -0.62, "虚无退却": -0.59,
    "身份认同": +0.34, "纯粹娱乐": +0.19,
}

# Google Trends keyword → (display_name, category)
# 基于 curator.py 的 29 条策展数据 + 手动扩展
TREND_TO_MEME = {
    # ── 有完整 B站 narrative 的 ──
    "打工人": ("打工人", "解构自嘲"),
    "内卷": ("内卷 / 卷", "身份认同"),
    "躺平": ("躺平", "虚无退却"),
    "普信男": ("普信男", "攻击发泄"),
    "普信": ("普信男", "攻击发泄"),
    "小镇做题家": ("小镇做题家", "身份认同"),
    "摆烂": ("摆烂", "虚无退却"),
    "润": ("润", "虚无退却"),
    "后浪": ("后浪", "身份认同"),
    "鸡你太美": ("鸡你太美", "纯粹娱乐"),
    "科目三": ("科目三", "纯粹娱乐"),
    "孔乙己的长衫": ("孔乙己的长衫", "身份认同"),
    "精神状态": ("精神状态良好", "解构自嘲"),
    "雪糕刺客": ("XX刺客", "攻击发泄"),
    "遥遥领先": ("遥遥领先", "纯粹娱乐"),
    "遥遥领先 华为": ("遥遥领先", "纯粹娱乐"),
    # ── 有 trends 生成 narrative 的 ──
    "吗喽": ("吗喽", "解构自嘲"),
    "鼠鼠": ("鼠鼠", "解构自嘲"),
    "牛马": ("牛马", "解构自嘲"),
    "i人 e人": ("i人/e人", "身份认同"),
    "情绪价值": ("情绪价值", "身份认同"),
    "原生家庭": ("原生家庭", "身份认同"),
    "尊嘟假嘟": ("尊嘟假嘟", "纯粹娱乐"),
    "凡尔赛": ("凡尔赛", "纯粹娱乐"),
    "元宇宙": ("元宇宙", "纯粹娱乐"),
    "citywalk": ("citywalk", "纯粹娱乐"),
    "芭比Q": ("芭比Q", "纯粹娱乐"),
    "栓Q": ("栓Q", "纯粹娱乐"),
    "美拉德": ("美拉德", "纯粹娱乐"),
    "南方小土豆": ("南方小土豆", "纯粹娱乐"),
    "破防": ("破防", "解构自嘲"),
    "社恐": ("社恐", "解构自嘲"),
    "社死": ("社死", "解构自嘲"),
    "精神内耗": ("精神内耗", "虚无退却"),
    "不结婚": ("四不/不婚不育", "虚无退却"),
    "不婚不育": ("四不/不婚不育", "虚无退却"),
    "专家建议": ("建议专家不要建议", "攻击发泄"),
    "建议专家不要建议": ("建议专家不要建议", "攻击发泄"),
    "显眼": ("显眼包", "纯粹娱乐"),
    "显眼包": ("显眼包", "纯粹娱乐"),
    "发疯文学 梗": ("发疯文学", "解构自嘲"),
    "多巴胺穿搭": ("多巴胺穿搭", "纯粹娱乐"),
}

# 约束场 5 维名称
CONSTRAINT_LABELS = ["Identity", "Humor/Decon", "Conflict", "Novelty", "Accessibility"]
CATEGORY_NAMES = list(CATEGORY_CHAOS.keys())

# ═══════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════

def load_external_field() -> dict:
    """加载 51 维外部场 (Google Trends 外部关键词)."""
    with open(DATA_DIR / "external_field_2015_2025.json", "r", encoding="utf-8") as f:
        return json.load(f)["data"]  # {keyword: {month: value}}


def load_meme_trends() -> dict:
    """加载 43 梗 Google Trends 数据."""
    with open(DATA_DIR / "google_trends_2015_2025.json", "r", encoding="utf-8") as f:
        return json.load(f)["memes"]  # {keyword: {month: value}}


def load_all_narratives() -> dict:
    """加载全部 58-87 条叙事 (B站 + trends生成)."""
    narratives = {}
    for nar_dir in NARRATIVE_DIRS:
        if not nar_dir.exists():
            continue
        for fp in nar_dir.glob("*.json"):
            if fp.name.startswith("_"):
                continue
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    narratives[fp.stem] = json.load(f)
            except Exception:
                pass
    return narratives


def _match_narrative(trend_kw: str, meme_name: str, narratives: dict) -> Optional[dict]:
    """为 Google Trends 关键词匹配对应的 narrative JSON."""
    # 直接按 meme_name 匹配
    candidates = [
        meme_name.replace(" ", "_").replace("/", "_"),
        meme_name.replace(" ", "").replace("/", ""),
    ]
    for c in candidates:
        if c in narratives:
            return narratives[c]
    # 模糊匹配: 检查 narrative keys
    for nar_key, nar_val in narratives.items():
        nar_clean = nar_key.replace("_", " ").lower()
        kw_terms = trend_kw.lower().split()
        if all(t in nar_clean for t in kw_terms if len(t) > 1):
            return nar_val
    return None


# ═══════════════════════════════════════════════
# Concept Bottleneck (lightweight inline copy for independence)
# ═══════════════════════════════════════════════

# 35 可观察概念 + 关键词锚点 (from concept_bottleneck.py)
CONCEPT_ANCHORS = {
    "official_release": "官方 发布 B站 宣传片 政府 主流 媒体 央视 新闻",
    "grassroots": "贴吧 社区 自发 网民 网友 草根 段子 论坛",
    "celebrity_trigger": "明星 演员 名人 网红 偶像 艺人 蔡徐坤",
    "accident_trigger": "偶然 意外 突然 走红 误传 乌龙",
    "policy_trigger": "政策 制度 政府 官方 文件 法规 改革",
    "platform_event": "平台 活动 B站 微博 抖音 官方 活动",
    "KOL_amplification": "博主 UP主 KOL 大V 传播 放大 转发",
    "algorithm_push": "算法 推荐 推送 流量 热搜 排名",
    "cross_platform": "跨平台 微博 知乎 抖音 B站 小红书 豆瓣 多个 平台",
    "mainstream_media": "媒体 新闻 央视 报道 主流 报纸 电视",
    "brand_hijack": "品牌 营销 蹭 热点 企业 广告 商业",
    "class_conflict": "阶层 贫富 阶级 底层 打工 人上人 优越 普通人",
    "gender_conflict": "性别 男女 女性 男性 普信 女权 男权",
    "generation_conflict": "青年 老一辈 代际 父母 后浪 前浪 年轻人",
    "political_conflict": "政治 敏感 审查 官方 意识形态 批判 五四",
    "value_conflict": "价值 传统 道德 消费 主义 批判 反思 争议",
    "parody": "戏仿 恶搞 模仿 鬼畜 搞笑 滑稽 整活",
    "irony": "反讽 解构 自嘲 黑色幽默 讽刺 批判 重新定义",
    "semantic_drift": "语义 漂移 泛化 衍生 原义 新义 含义 变化",
    "remix": "二次创作 模板 段子 变体 改编 翻拍 复刻",
    "institutionalization": "制度化 收编 官方 主流化 正常化 日常",
    "anger": "愤怒 攻击 宣泄 骂 对立 冲突 撕裂",
    "humor_laugh": "幽默 搞笑 自嘲 好玩 有趣 娱乐 梗 玩笑",
    "schadenfreude": "幸灾乐祸 看戏 吃瓜 围观 讽刺",
    "identity_belonging": "身份 认同 归属 共鸣 我们 大家都是 共同体",
    "hope": "希望 向往 美好 梦想 未来 激励 鼓舞",
    "nihilism": "虚无 无力 躺平 退出 放弃 摆烂 无所谓",
    "anxiety": "焦虑 压力 竞争 内卷 生存 就业 经济",
    "nostalgia": "怀旧 回忆 童年 曾经 过去 经典",
    "youth_dominant": "青年 年轻人 大学生 学生 B站 校园",
    "white_collar": "职场 白领 程序员 上班 打工 社畜",
    "student": "学生 学校 考试 教育 做题 论文",
    "rural": "下沉 农村 乡镇 县城 基层",
    "elite": "精英 知识分子 学术 高大上 985 名校",
}
CONCEPT_NAMES = list(CONCEPT_ANCHORS.keys())
N_CONCEPTS = len(CONCEPT_NAMES)  # 33

# 5D 约束映射权重 (from concept_bottleneck.py)
IDENTITY_W = {"identity_belonging": 0.4, "youth_dominant": 0.15, "white_collar": 0.1,
              "student": 0.1, "official_release": -0.2, "grassroots": 0.2,
              "hope": 0.15, "mainstream_media": 0.1, "KOL_amplification": 0.15}
HUMOR_W = {"humor_laugh": 0.5, "irony": 0.3, "parody": 0.25, "remix": 0.15,
           "anger": -0.2, "nihilism": -0.1, "semantic_drift": 0.1, "cross_platform": 0.1}
CONFLICT_W = {"class_conflict": 0.35, "gender_conflict": 0.3, "political_conflict": 0.3,
              "generation_conflict": 0.25, "value_conflict": 0.25, "anger": 0.2,
              "algorithm_push": -0.1, "identity_belonging": -0.15}
NOVELTY_W = {"parody": 0.3, "remix": 0.3, "semantic_drift": 0.25, "irony": 0.2,
             "accident_trigger": 0.2, "grassroots": 0.2, "official_release": -0.15,
             "cross_platform": 0.15, "KOL_amplification": 0.1}
ACCESS_W = {"humor_laugh": 0.3, "remix": 0.25, "youth_dominant": 0.2,
            "algorithm_push": 0.25, "cross_platform": 0.2, "brand_hijack": 0.15,
            "political_conflict": -0.3, "class_conflict": -0.1, "elite": -0.15}
CONSTRAINT_WEIGHTS = [IDENTITY_W, HUMOR_W, CONFLICT_W, NOVELTY_W, ACCESS_W]


def _bigrams(text: str) -> set:
    return set(text[i:i+2] for i in range(len(text)-1))


def _soft_score(text: str, anchor_keywords: str) -> float:
    """中文关键词命中评分 — 替代 bigram Jaccard (对中文无效).

    对每个锚点关键词检查是否在叙事文本中出现.
    命中率 + 连续命中 bonus.
    """
    if not text.strip():
        return 0.0
    # 分词: 按空格和中英文边界拆分
    keywords = [kw.strip() for kw in anchor_keywords.replace(",", " ").split() if kw.strip()]
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw in text)
    # 连续命中: 如果多个关键词连续出现, 加bonus
    consecutive_bonus = 0.0
    for i in range(len(keywords) - 1):
        if keywords[i] in text and keywords[i+1] in text:
            consecutive_bonus += 0.05
    base_score = hits / len(keywords)
    return min(1.0, base_score + min(0.2, consecutive_bonus))


def narrative_to_concept_vec(narrative: dict) -> np.ndarray:
    """从 narrative JSON 提取 33 维可观察概念向量."""
    origin = narrative.get("origin") or {}
    social = narrative.get("social_context") or {}
    drift = narrative.get("semantic_drift") or {}
    phases = narrative.get("spread_phases") or []
    mutations = narrative.get("mutations") or []

    all_desc = " ".join([str(p.get("description") or "") for p in phases])
    all_figures = " ".join([str(f) for p in phases for f in (p.get("key_figures") or [])])
    triggers_text = " ".join([str(t) for t in (social.get("triggers") or [])])
    backlash_text = " ".join([str(b) for b in (social.get("backlash_events") or [])])
    mut_text = " ".join([str(m.get("relationship") or "") for m in mutations])

    narrative_full = " ".join([
        str(origin.get("trigger_event") or ""),
        str(origin.get("platform") or ""),
        str(origin.get("precursor") or ""),
        all_desc, all_figures, triggers_text, backlash_text,
        str(drift.get("drift_direction") or ""),
        str(social.get("target_audience") or ""),
        str(social.get("political_sensitivity") or ""),
        mut_text,
        narrative.get("narrative_summary", ""),
        narrative.get("curve_shape", ""),
        narrative.get("social_context_hint", ""),
    ])

    v = np.zeros(N_CONCEPTS)
    for i, name in enumerate(CONCEPT_NAMES):
        v[i] = _soft_score(narrative_full, CONCEPT_ANCHORS[name])
    return v


def concept_to_constraint(concept_vec: np.ndarray) -> np.ndarray:
    """33 维概念 → 5 维约束 (加权映射 + 线性缩放, 保留方差)."""
    p = np.zeros(5)
    for dim, weights in enumerate(CONSTRAINT_WEIGHTS):
        total = 0.0
        weight_sum = 0.0
        for name, w in weights.items():
            if name in CONCEPT_NAMES:
                total += w * concept_vec[CONCEPT_NAMES.index(name)]
                weight_sum += abs(w)
        # Normalize by weight sum to get [0, 1] range, then scale
        if weight_sum > 0:
            raw = total / weight_sum
        else:
            raw = 0.0
        # Linear scaling: map [-1, +1] → [0.1, 0.9]
        p[dim] = float(np.clip(0.5 + raw * 0.4, 0.1, 0.9))
    return p


def _load_llm_scores() -> dict:
    """加载 LLM 概念打分结果 (Step 2)."""
    scores_path = ROOT / "data/processed/llm_concept_scores.json"
    if scores_path.exists():
        with open(scores_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_meme_constraint(trend_kw: str, meme_name: str, narratives: dict,
                        llm_scores: dict = None) -> np.ndarray:
    """获取一个梗的 5 维约束向量.

    优先级: LLM 打分 > 软匹配 > 类别默认值
    """
    if llm_scores is None:
        llm_scores = _load_llm_scores()

    # Step 1: Try LLM scores
    if llm_scores and meme_name in llm_scores:
        constraint = llm_scores[meme_name].get("constraint")
        if constraint and len(constraint) == 5:
            return np.array(constraint)

    # Step 2: Try soft matching (fallback)
    nar = _match_narrative(trend_kw, meme_name, narratives)
    if nar is not None:
        concept = narrative_to_concept_vec(nar)
        c = concept_to_constraint(concept)
        # Check if constraint has meaningful variance (not all ~0.5)
        if np.std(c) > 0.03:
            return c

    # Step 3: Category defaults
    cat = TREND_TO_MEME.get(trend_kw, (meme_name, "纯粹娱乐"))[1]
    cat_defaults = {
        "解构自嘲": [0.40, 0.55, 0.30, 0.45, 0.50],
        "攻击发泄": [0.30, 0.20, 0.75, 0.30, 0.30],
        "虚无退却": [0.25, 0.25, 0.45, 0.25, 0.25],
        "身份认同": [0.70, 0.30, 0.35, 0.30, 0.50],
        "纯粹娱乐": [0.30, 0.70, 0.15, 0.60, 0.75],
    }
    return np.array(cat_defaults.get(cat, [0.4, 0.4, 0.4, 0.4, 0.4]))


# ═══════════════════════════════════════════════
# Monthly Feature Builder
# ═══════════════════════════════════════════════

@dataclass
class MonthlyState:
    """一个月的完整集体状态."""
    month: str
    ext_field: np.ndarray        # 51-dim 外部场
    chaos_axis: float            # 集体混沌轴
    constraint: np.ndarray       # 5-dim 注意力加权约束 (Step 2 LLM替换)
    cat_dist: np.ndarray         # 5-dim 类别分布
    total_attention: float       # 总梗注意力
    active_meme_count: int       # 活跃梗数
    attention_hhi: float         # 注意力集中度 (HHI指数, 反多样化)
    cat_entropy: float           # 类别分布熵 (叙事多样性)
    dominant_cat: str            # 主导类别


def build_monthly_states(
    ext_field: dict,
    meme_trends: dict,
    narratives: dict,
    months: list[str],
) -> list[MonthlyState]:
    """为每个月构建完整的状态向量.

    对每个月:
    1. 外部场向量 (51-dim)
    2. 混沌轴: 注意力加权类别混沌
    3. 约束场: 注意力加权 5D 约束 (从 narrative 提取)
    4. 类别分布: 5-dim
    """
    ext_keywords = sorted(ext_field.keys())
    states = []

    # 预计算每个 meme 的约束向量 (优先 LLM 打分)
    llm_scores = _load_llm_scores()
    meme_constraints = {}
    for trend_kw in meme_trends:
        if trend_kw in TREND_TO_MEME:
            meme_name, cat = TREND_TO_MEME[trend_kw]
            meme_constraints[trend_kw] = get_meme_constraint(
                trend_kw, meme_name, narratives, llm_scores)
        else:
            meme_constraints[trend_kw] = np.array([0.4, 0.4, 0.4, 0.4, 0.4])

    for month in months:
        # 1. External field
        ext_vec = np.array([ext_field[kw].get(month, 0.0) for kw in ext_keywords])

        # 2. Meme attention + chaos + constraint + category
        chaos_sum = 0.0
        weight_sum = 0.0
        constraint_sum = np.zeros(5)
        cat_sum = np.zeros(5)
        active_count = 0

        for trend_kw, monthly_data in meme_trends.items():
            w = monthly_data.get(month, 0.0)
            if w <= 0:
                continue

            active_count += 1
            weight_sum += w

            # Chaos contribution
            if trend_kw in TREND_TO_MEME:
                cat = TREND_TO_MEME[trend_kw][1]
                cat_chaos = CATEGORY_CHAOS.get(cat, 0.0)
                chaos_sum += cat_chaos * w
                if cat in CATEGORY_NAMES:
                    cat_sum[CATEGORY_NAMES.index(cat)] += w

            # Constraint contribution
            c = meme_constraints.get(trend_kw, np.zeros(5))
            constraint_sum += c * w

        # Normalize
        chaos = chaos_sum / weight_sum if weight_sum > 0 else 0.0
        constraint = constraint_sum / weight_sum if weight_sum > 0 else np.full(5, 0.4)
        cat_dist = cat_sum / cat_sum.sum() if cat_sum.sum() > 0 else np.ones(5) / 5

        # Attention HHI (Herfindahl-Hirschman Index: 注意力集中度)
        if weight_sum > 0:
            shares = []
            for trend_kw, monthly_data in meme_trends.items():
                w = monthly_data.get(month, 0.0)
                if w > 0:
                    shares.append(w / weight_sum)
            hhi = sum(s**2 for s in shares) if shares else 0.0
        else:
            hhi = 0.0

        # Category entropy (叙事多样性)
        if cat_sum.sum() > 0:
            p = cat_dist[cat_dist > 0]
            cat_entropy = float(-np.sum(p * np.log(p)))
        else:
            cat_entropy = 0.0

        dom_cat = CATEGORY_NAMES[int(np.argmax(cat_dist))] if cat_dist.sum() > 0 else "none"

        states.append(MonthlyState(
            month=month,
            ext_field=ext_vec,
            chaos_axis=float(chaos),
            constraint=constraint,
            cat_dist=cat_dist,
            total_attention=float(weight_sum),
            active_meme_count=active_count,
            attention_hhi=float(hhi),
            cat_entropy=cat_entropy,
            dominant_cat=dom_cat,
        ))

    return states


# ═══════════════════════════════════════════════
# Two-Layer Prediction Model
# ═══════════════════════════════════════════════

class OrderFormPredictor:
    """两层集成预测模型.

    Layer 1: 外部场 → 基线预测 (chaos, constraint, cat_dist)
    Layer 2: 叙事约束历史 → 残差修正

    最终输出: 秩序形态 = f(chaos, constraint, cat_dist)
    """

    def __init__(self, lookback: int = 6, horizon: int = 1):
        self.lookback = lookback
        self.horizon = horizon
        self.scaler: Optional[StandardScaler] = None
        self.model_ext: Optional[Ridge] = None          # 外部场 → chaos
        self.model_constraint: Optional[Ridge] = None   # 外部场 → constraint (5D)
        self.model_cat: Optional[Ridge] = None          # 外部场 → cat_dist (5D)
        self.model_residual: Optional[Ridge] = None     # 约束历史 → chaos 残差修正
        self.pca_ext: Optional[PCA] = None              # 外部场 51dim → top-N PCs
        self._order_forms: Optional[dict] = None
        self._kmeans: Optional[KMeans] = None

    # ── Feature building (降维版: 外部场通过PCA压缩) ──

    def _build_features(self, states: list[MonthlyState],
                        ext_pc: int = 8) -> tuple[np.ndarray, np.ndarray]:
        """构建训练数据. 外部场 51-dim → PCA → ext_pc 维.

        每月特征: ext_pc(PCA) + chaos(1) + constraint(5) + cat_dist(5)
                  + hhi(1) + entropy(1) + attention(1)
        = ext_pc + 14 维. Flatten lookback 个月.
        y: 目标月的 [chaos, constraint_5d, cat_dist_5d, hhi, entropy] = 13-dim
        """
        n = len(states)

        # Fit PCA on external field
        all_ext = np.array([s.ext_field for s in states])
        self.pca_ext = PCA(n_components=ext_pc)
        ext_pcs = self.pca_ext.fit_transform(all_ext)
        ext_var = self.pca_ext.explained_variance_ratio_.sum()
        print(f"  外部场 PCA({ext_pc}): 解释 {ext_var:.1%} 方差")

        X_list, y_list = [], []
        for i in range(self.lookback, n - self.horizon):
            feats = []
            for t in range(i - self.lookback, i):
                s = states[t]
                feats.extend(ext_pcs[t].tolist())            # ext_pc
                feats.append(s.chaos_axis)                    # 1
                feats.extend(s.constraint.tolist())            # 5 ← LLM约束,有方差
                feats.extend(s.cat_dist.tolist())              # 5
                feats.append(s.attention_hhi)                  # 1
                feats.append(s.cat_entropy)                    # 1
                feats.append(np.log1p(s.total_attention))      # 1
            X_list.append(feats)

            target = states[i + self.horizon - 1]
            y_list.append(np.concatenate([
                [target.chaos_axis],
                target.constraint.tolist(),      # 5
                target.cat_dist.tolist(),        # 5
                [target.attention_hhi],           # 1
                [target.cat_entropy],             # 1
            ]))

        feat_dim = ext_pc + 1 + 5 + 5 + 1 + 1 + 1  # per month
        print(f"  每月特征: {feat_dim} 维 × {self.lookback} 月 = {feat_dim * self.lookback} 维")
        return np.array(X_list), np.array(y_list)

    # ── Training ──

    def fit(self, states: list[MonthlyState], ext_pc: int = 8) -> dict:
        """训练两层模型.

        Layer 1: 外部场(PCA) + chaos + constraint + cat_dist + HHI + entropy → baseline
        Layer 2: constraint历史 → chaos 残差修正
        Target: [chaos, constraint(5), cat_dist(5), hhi, entropy] = 13-dim
        """
        X, y = self._build_features(states, ext_pc=ext_pc)
        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        print(f"\n[训练数据] {n_samples} 样本, {n_features} 特征 → {n_targets} 目标")
        print(f"  特征/样本比: {n_features/n_samples:.1f}")

        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)

        alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

        # Chaos
        self.model_ext = RidgeCV(alphas=alphas, cv=5)
        self.model_ext.fit(X_s, y[:, 0])
        chaos_pred = self.model_ext.predict(X_s)
        r2_chaos = r2_score(y[:, 0], chaos_pred)

        # Constraint (5D) — NOW HAS VARIANCE
        self.model_constraint = Ridge(alpha=self.model_ext.alpha_)
        self.model_constraint.fit(X_s, y[:, 1:6])
        constraint_pred = self.model_constraint.predict(X_s)
        r2_constraint = r2_score(y[:, 1:6], constraint_pred)

        # Category (5D)
        self.model_cat = Ridge(alpha=self.model_ext.alpha_)
        self.model_cat.fit(X_s, y[:, 6:11])
        cat_pred = self.model_cat.predict(X_s)
        r2_cat = r2_score(y[:, 6:11], cat_pred)

        # HHI + Entropy
        self.model_struct = Ridge(alpha=self.model_ext.alpha_)
        self.model_struct.fit(X_s, y[:, 11:13])
        struct_pred = self.model_struct.predict(X_s)
        r2_struct = r2_score(y[:, 11:13], struct_pred)

        # ── Layer 2: constraint history → chaos residual ──
        chaos_residual = y[:, 0] - chaos_pred
        X_res = np.array([
            np.concatenate([states[i+t].constraint.tolist() + [states[i+t].chaos_axis]
                          for t in range(-self.lookback, 0)])
            for i in range(self.lookback, len(states) - self.horizon)
        ])
        X_res_s = StandardScaler().fit_transform(X_res)
        self.model_residual = RidgeCV(alphas=alphas, cv=5)
        self.model_residual.fit(X_res_s, chaos_residual)
        residual_pred = self.model_residual.predict(X_res_s)
        r2_residual = r2_score(chaos_residual, residual_pred)

        chaos_combined = chaos_pred + residual_pred
        r2_combined = r2_score(y[:, 0], chaos_combined)

        results = {
            "n_samples": n_samples, "n_features": n_features,
            "alpha_ext": float(self.model_ext.alpha_),
            "alpha_res": float(self.model_residual.alpha_),
            "r2_chaos_external": float(r2_chaos),
            "r2_chaos_two_layer": float(r2_combined),
            "r2_constraint": float(r2_constraint),
            "r2_cat_dist": float(r2_cat),
            "r2_structure": float(r2_struct),
            "r2_residual": float(r2_residual),
            "residual_std": float(np.std(chaos_residual)),
            "internal_layer_benefit": float(r2_combined - r2_chaos),
        }

        print(f"\n[Layer 1 — 外部场基线 (Ridge α={self.model_ext.alpha_:.1f})]")
        print(f"  Chaos R²:       {r2_chaos:.4f}")
        print(f"  Constraint R²:  {r2_constraint:.4f}  ★ 有方差!")
        print(f"  Category R²:    {r2_cat:.4f}")
        print(f"  Structure R²:   {r2_struct:.4f}")
        print(f"\n[Layer 2 — 内部叙事修正 (Ridge α={self.model_residual.alpha_:.1f})]")
        print(f"  Residual R²:    {r2_residual:.4f}")
        print(f"  Two-layer R²:   {r2_combined:.4f}")
        print(f"  内部层增益:      {r2_combined - r2_chaos:+.4f}")

        return results

    # ── Prediction ──

    def predict(self, states: list[MonthlyState]) -> np.ndarray:
        """预测下个月的 [chaos, constraint(5), cat_dist(5), hhi, entropy] = 13-dim."""
        if len(states) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} states")

        all_ext = np.array([s.ext_field for s in states])
        ext_pcs = self.pca_ext.transform(all_ext)

        feats = []
        for t in range(-self.lookback, 0):
            idx = len(ext_pcs) + t
            if idx >= 0:
                feats.extend(ext_pcs[idx].tolist())
            else:
                feats.extend([0.0] * self.pca_ext.n_components_)
            feats.append(states[t].chaos_axis)
            feats.extend(states[t].constraint.tolist())
            feats.extend(states[t].cat_dist.tolist())
            feats.append(states[t].attention_hhi)
            feats.append(states[t].cat_entropy)
            feats.append(np.log1p(states[t].total_attention))

        X = np.array(feats).reshape(1, -1)
        X_s = self.scaler.transform(X)

        # Layer 1
        chaos_base = self.model_ext.predict(X_s)[0]
        constraint_pred = self.model_constraint.predict(X_s)[0]
        cat_pred = self.model_cat.predict(X_s)[0]
        struct_pred = self.model_struct.predict(X_s)[0]

        # Layer 2: constraint history → residual
        X_res = np.array([
            np.concatenate([states[t].constraint.tolist() + [states[t].chaos_axis]
                          for t in range(-self.lookback, 0)])
        ])
        res_scaler = StandardScaler()
        all_res = []
        for i in range(self.lookback, len(states) - 1):
            all_res.append(np.concatenate(
                [states[i+t].constraint.tolist() + [states[i+t].chaos_axis]
                 for t in range(-self.lookback, 0)]))
        if all_res:
            res_scaler.fit(np.array(all_res))
            X_res_s = res_scaler.transform(X_res)
            residual = self.model_residual.predict(X_res_s)[0]
        else:
            residual = 0.0

        chaos_final = chaos_base + residual
        cat_pred = np.clip(cat_pred, 0, None)
        if cat_pred.sum() > 0:
            cat_pred /= cat_pred.sum()

        return np.concatenate([[chaos_final], constraint_pred, cat_pred, struct_pred])

    def forecast(self, states: list[MonthlyState], n_months: int = 6) -> list[dict]:
        """预测未来 N 个月."""
        forecasts = []
        extended = list(states)
        residual_std = getattr(self, '_last_residual_std', 0.10)

        for step in range(n_months):
            try:
                pred_vec = self.predict(extended)
            except Exception:
                break

            chaos = float(pred_vec[0])
            constraint = pred_vec[1:6]
            cat_dist = pred_vec[6:11]
            hhi = float(pred_vec[11])
            entropy = float(pred_vec[12])

            order_form = self._classify_order_form(chaos, constraint, cat_dist, hhi, entropy)

            last_month = extended[-1].month
            year, mon = int(last_month[:4]), int(last_month[5:7])
            new_mon = mon + 1
            new_year = year + (new_mon - 1) // 12
            new_mon = ((new_mon - 1) % 12) + 1
            forecast_month = f"{new_year}-{new_mon:02d}"

            forecasts.append({
                "month": forecast_month, "step": step + 1,
                "chaos_axis": chaos,
                "chaos_range": [chaos - residual_std, chaos + residual_std],
                "constraint": {CONSTRAINT_LABELS[i]: float(constraint[i]) for i in range(5)},
                "category_distribution": {CATEGORY_NAMES[i]: float(cat_dist[i]) for i in range(5)},
                "attention_hhi": hhi, "cat_entropy": entropy,
                "order_form": order_form,
            })

            extended.append(MonthlyState(
                month=forecast_month,
                ext_field=extended[-1].ext_field,
                chaos_axis=chaos,
                constraint=constraint,
                cat_dist=cat_dist,
                total_attention=extended[-1].total_attention,
                active_meme_count=extended[-1].active_meme_count,
                attention_hhi=hhi, cat_entropy=entropy,
                dominant_cat=max(CATEGORY_NAMES, key=lambda c: cat_dist[CATEGORY_NAMES.index(c)]),
            ))

        return forecasts

    # ── Order Form Classification ──

    def fit_order_forms(self, states: list[MonthlyState], n_forms: int = 8):
        """在历史状态上聚类. State: [chaos, constraint(5), cat_dist(5), hhi, entropy]."""
        X = np.array([
            np.concatenate([[s.chaos_axis], s.constraint, s.cat_dist,
                           [s.attention_hhi, s.cat_entropy]])
            for s in states
        ])

        self._kmeans = KMeans(n_clusters=n_forms, random_state=42, n_init=10)
        labels = self._kmeans.fit_predict(X)
        centers = self._kmeans.cluster_centers_

        form_names = []
        for center in centers:
            chaos = center[0]
            constraint = center[1:6]
            cat_dist = center[6:11]
            hhi = center[11]
            entropy = center[12]

            dom_cat = CATEGORY_NAMES[int(np.argmax(cat_dist))]
            dom_constraint = CONSTRAINT_LABELS[int(np.argmax(constraint))]

            if chaos > 0.10:
                chaos_dir = "秩序建构"
            elif chaos < -0.15:
                chaos_dir = "混沌释放"
            else:
                chaos_dir = "边界振荡"

            concentration = "集中" if hhi > 0.3 else "分散" if hhi < 0.15 else "中等"
            diversity = "多元" if entropy > 1.3 else "单一" if entropy < 0.8 else "中等"

            name = f"{chaos_dir}·{dom_cat}·{dom_constraint}·{concentration}"
            form_names.append(name)

        self._order_forms = {
            "n_forms": n_forms,
            "names": form_names,
            "centers": centers.tolist(),
            "labels": labels.tolist(),
        }

        label_counts = defaultdict(int)
        for l in labels:
            label_counts[int(l)] += 1

        print(f"\n[秩序形态聚类] {n_forms} 种形态:")
        for i, name in enumerate(form_names):
            print(f"  Form {i}: {name}  ({label_counts[i]} 个月)")

        return self._order_forms

    def _classify_order_form(self, chaos: float, constraint: np.ndarray,
                             cat_dist: np.ndarray, hhi: float, entropy: float) -> dict:
        """将预测状态归类到最近秩序形态."""
        if self._kmeans is None or self._order_forms is None:
            return {"name": "未定义", "confidence": 0.0}

        vec = np.concatenate([[chaos], constraint, cat_dist, [hhi, entropy]]).reshape(1, -1)
        label = int(self._kmeans.predict(vec)[0])
        center = np.array(self._order_forms["centers"][label])
        dist = np.linalg.norm(vec - center)
        confidence = max(0.0, 1.0 - dist / 1.0)

        return {
            "form_id": label,
            "name": self._order_forms["names"][label],
            "confidence": float(confidence),
        }

    # ── Backtest ──

    def backtest(self, states: list[MonthlyState], n_splits: int = 5, ext_pc: int = 8) -> dict:
        """时序交叉验证回测."""
        X, y = self._build_features(states, ext_pc=ext_pc)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        scores = defaultdict(list)
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            m_ext = RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0], cv=3)
            m_ext.fit(X_tr_s, y_tr[:, 0])
            y_pred_ext = m_ext.predict(X_te_s)

            # Layer 2 residual
            chaos_res_tr = y_tr[:, 0] - m_ext.predict(X_tr_s)
            n = len(states)
            X_res_all = []
            for i in range(self.lookback, n - self.horizon):
                X_res_all.append(np.concatenate(
                    [states[i+t].cat_dist.tolist() + [states[i+t].chaos_axis]
                     for t in range(-self.lookback, 0)]))
            X_res_all = np.array(X_res_all)
            X_res_tr = X_res_all[train_idx]
            X_res_te = X_res_all[test_idx]

            res_scaler = StandardScaler()
            m_res = RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0], cv=3)
            m_res.fit(res_scaler.fit_transform(X_res_tr), chaos_res_tr)
            y_pred_res = m_res.predict(res_scaler.transform(X_res_te))
            y_pred_2l = y_pred_ext + y_pred_res

            scores["mae_ext"].append(mean_absolute_error(y_te[:, 0], y_pred_ext))
            scores["mae_2layer"].append(mean_absolute_error(y_te[:, 0], y_pred_2l))
            scores["r2_ext"].append(r2_score(y_te[:, 0], y_pred_ext))
            scores["r2_2layer"].append(r2_score(y_te[:, 0], y_pred_2l))

        naive_mae = np.mean([abs(y[i, 0] - y[i-1, 0]) for i in range(1, len(y))])

        results = {
            "n_splits": n_splits,
            "mae_external": float(np.mean(scores["mae_ext"])),
            "mae_external_std": float(np.std(scores["mae_ext"])),
            "mae_two_layer": float(np.mean(scores["mae_2layer"])),
            "mae_two_layer_std": float(np.std(scores["mae_2layer"])),
            "r2_external": float(np.mean(scores["r2_ext"])),
            "r2_two_layer": float(np.mean(scores["r2_2layer"])),
            "naive_mae": float(naive_mae),
            "improvement_over_naive": float(1.0 - np.mean(scores["mae_2layer"]) / naive_mae),
            "internal_layer_gain": float(np.mean(scores["mae_ext"]) - np.mean(scores["mae_2layer"])),
        }

        print(f"\n[回测 {n_splits}-fold TimeSeriesSplit]")
        print(f"  Naive (lag-1) MAE:    {naive_mae:.4f}")
        print(f"  Layer-1 MAE:          {results['mae_external']:.4f} ± {results['mae_external_std']:.4f}")
        print(f"  Two-layer MAE:        {results['mae_two_layer']:.4f} ± {results['mae_two_layer_std']:.4f}")
        print(f"  Internal layer gain:  {results['internal_layer_gain']:+.4f}")
        print(f"  vs Naive:             {results['improvement_over_naive']:+.1%}")

        return results


# ═══════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════

def generate_report(
    states: list[MonthlyState],
    predictor: OrderFormPredictor,
    backtest: dict,
    forecasts: list[dict],
) -> str:
    """生成可读的集体情感系统报告."""
    latest = states[-1]
    lines = []
    lines.append("=" * 75)
    lines.append("  中国互联网集体情感系统 — 秩序形态预测报告")
    lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 75)

    # ── 当前状态 ──
    lines.append(f"\n  📍 当前状态 ({latest.month})")
    chaos_dir = "偏秩序" if latest.chaos_axis > 0 else "偏混沌"
    lines.append(f"     混沌轴: {latest.chaos_axis:+.3f} ({chaos_dir})")
    lines.append(f"     总注意力: {latest.total_attention:.0f}")
    lines.append(f"     活跃梗数: {latest.active_meme_count}")

    lines.append(f"\n  📊 类别分布 & 注意力结构:")
    dom_cat_idx = int(np.argmax(latest.cat_dist))
    lines.append(f"     主导类别: {CATEGORY_NAMES[dom_cat_idx]}")
    lines.append(f"     注意力集中度(HHI): {latest.attention_hhi:.3f} ({'集中' if latest.attention_hhi > 0.3 else '分散' if latest.attention_hhi < 0.15 else '中等'})")
    lines.append(f"     类别熵(多样性): {latest.cat_entropy:.3f} ({'多元' if latest.cat_entropy > 1.3 else '单一' if latest.cat_entropy < 0.8 else '中等'})")
    for i, cat in enumerate(CATEGORY_NAMES):
        lines.append(f"     {cat}: {latest.cat_dist[i]:.1%}")

    # ── 模型性能 ──
    lines.append(f"\n{'─'*50}")
    lines.append(f"  📈 预测模型性能")
    lines.append(f"     外部场 R²: {backtest['r2_external']:.4f}")
    lines.append(f"     双层 R²:   {backtest['r2_two_layer']:.4f}")
    lines.append(f"     内部层增益: {backtest['internal_layer_gain']:+.4f} MAE")
    lines.append(f"     vs 朴素基线: {backtest['improvement_over_naive']:+.1%}")

    # ── 秩序形态 ──
    if predictor._order_forms:
        lines.append(f"\n{'─'*50}")
        lines.append(f"  🗂️ 历史秩序形态分布 ({predictor._order_forms['n_forms']} 种)")
        label_counts = defaultdict(int)
        for l in predictor._order_forms["labels"]:
            label_counts[int(l)] += 1
        for i, name in enumerate(predictor._order_forms["names"]):
            pct = label_counts.get(i, 0) / len(predictor._order_forms["labels"]) * 100
            lines.append(f"     Form {i}: {name}  ({pct:.0f}%)")

    # ── 未来预测 ──
    lines.append(f"\n{'─'*50}")
    lines.append(f"  🔮 未来 {len(forecasts)} 个月预测")
    for fc in forecasts:
        chaos = fc["chaos_axis"]
        order = fc["order_form"]
        lines.append(f"\n     {fc['month']}:")
        lines.append(f"       混沌轴: {chaos:+.3f} (区间 [{fc['chaos_range'][0]:+.3f}, {fc['chaos_range'][1]:+.3f}])")
        lines.append(f"       秩序形态: {order['name']} (置信度 {order['confidence']:.0%})")
        lines.append(f"       HHI: {fc['attention_hhi']:.3f}  类别熵: {fc['cat_entropy']:.3f}")
        dom_cat = max(fc["category_distribution"], key=fc["category_distribution"].get)
        lines.append(f"       主导类别: {dom_cat} ({fc['category_distribution'][dom_cat]:.1%})")

    # ── LLM 月度叙事摘要 ──
    narrative_path = ROOT / "data/processed/monthly_narratives.jsonl"
    if narrative_path.exists():
        lines.append(f"\n{'─'*50}")
        lines.append(f"  🤖 LLM 月度集体叙事摘要")
        try:
            narratives = []
            with open(narrative_path, "r", encoding="utf-8") as f:
                for line in f:
                    narratives.append(json.loads(line.strip()))
            if narratives:
                latest_nar = narratives[-1]
                lines.append(f"     {latest_nar.get('summary', '无')}")
        except Exception:
            pass

    lines.append(f"\n{'─'*50}")
    lines.append(f"  外部场: 51 关键词 (Google Trends 2015-2025)")
    lines.append(f"  内部层: 43 梗叙事约束场 (LLM 概念打分 + 注意力加权聚合)")
    lines.append(f"  报告依据: 127 月历史数据 + 时序交叉验证")
    lines.append(f"{'='*75}\n")

    report = "\n".join(lines)
    return report


# ═══════════════════════════════════════════════
# Main Entry
# ═══════════════════════════════════════════════

def main():
    import argparse
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="秩序形态预测模型")
    parser.add_argument("--forecast", type=int, default=0, help="预测未来N个月")
    parser.add_argument("--report", action="store_true", help="仅生成当前报告")
    parser.add_argument("--save-model", type=str, default="", help="保存模型路径")
    args = parser.parse_args()

    print("=" * 75)
    print("MemeticChaos — 秩序形态预测模型 (Two-Layer)")
    print("=" * 75)

    # ── Load data ──
    ext_field = load_external_field()
    meme_trends = load_meme_trends()
    narratives = load_all_narratives()

    # Common months
    all_months = set()
    for d in ext_field.values():
        all_months.update(d.keys())
    for d in meme_trends.values():
        all_months.update(d.keys())
    months = sorted(m for m in all_months if "2015" <= m[:4] <= "2025")

    print(f"\n[数据]")
    print(f"  外部场: {len(ext_field)} 关键词")
    print(f"  梗趋势: {len(meme_trends)} 关键词")
    print(f"  叙事:   {len(narratives)} 条")
    print(f"  月度:   {len(months)} 个月 ({months[0]} → {months[-1]})")

    # ── Build monthly states ──
    print(f"\n[构建月度状态向量]...")
    states = build_monthly_states(ext_field, meme_trends, narratives, months)
    print(f"  构建了 {len(states)} 个月度状态")

    # Quick sanity checks
    chaos_vals = [s.chaos_axis for s in states]
    print(f"  混沌轴范围: [{min(chaos_vals):+.3f}, {max(chaos_vals):+.3f}]")
    non_zero_months = sum(1 for s in states if s.active_meme_count > 0)
    print(f"  有活跃梗的月份: {non_zero_months}/{len(states)}")

    # ── Train ──
    print(f"\n{'='*50}")
    print(f"训练两层预测模型")
    print(f"{'='*50}")
    predictor = OrderFormPredictor(lookback=6, horizon=1)
    train_results = predictor.fit(states)
    predictor._last_residual_std = train_results["residual_std"]

    # ── Order forms ──
    predictor.fit_order_forms(states, n_forms=8)

    # ── Backtest ──
    backtest_results = predictor.backtest(states)

    # ── Forecast ──
    forecasts = []
    if args.forecast > 0:
        forecasts = predictor.forecast(states, n_months=args.forecast)

    # ── Report ──
    report = generate_report(states, predictor, backtest_results, forecasts)
    print(report)

    # Save report
    report_path = PROCESSED_DIR / "order_form_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[报告] 已保存 → {report_path}")

    # Save clean JSON state for dashboard
    latest = states[-1]
    state_json = {
        "month": latest.month,
        "chaos_axis": float(latest.chaos_axis),
        "total_attention": float(latest.total_attention),
        "active_meme_count": latest.active_meme_count,
        "attention_hhi": float(latest.attention_hhi),
        "cat_entropy": float(latest.cat_entropy),
        "dominant_category": latest.dominant_cat,
        "constraint": {CONSTRAINT_LABELS[i]: float(latest.constraint[i]) for i in range(5)},
        "cat_dist": {CATEGORY_NAMES[i]: float(latest.cat_dist[i]) for i in range(5)},
        "forecasts": forecasts,
        "order_forms": predictor._order_forms["names"] if predictor._order_forms else [],
        "backtest": backtest_results,
        "generated_at": datetime.now().isoformat(),
    }
    state_path = PROCESSED_DIR / "dashboard_state.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state_json, f, ensure_ascii=False, indent=2)
    print(f"[Dashboard状态] 已保存 → {state_path}")

    # Save model if requested
    if args.save_model:
        import pickle
        model_path = Path(args.save_model)
        with open(model_path, "wb") as f:
            pickle.dump({
                "predictor": predictor,
                "states": states,
                "backtest": backtest_results,
                "months": months,
            }, f)
        print(f"[模型] 已保存 → {model_path}")

    return predictor, states, backtest_results, forecasts


if __name__ == "__main__":
    main()
