"""
Concept Bottleneck Model — 可观察概念 → 5D 约束场

GPT/Gemini 共识设计。不直接从 Narrative 跳到 5D 约束分数，
而是经过一层稳定的可观察概念矩阵。

Stage 1: LLM 从叙事中抽取 20-40 个可观察属性 (binary/continuous)
Stage 2: 规则 + 线性映射 → 5D Constraint (Identity, Humor, Conflict, Novelty, Accessibility)

为什么两阶段：Stage 1 是长期数据资产（以后 5000 梗也能复用），
Stage 2 以后可以重新定义映射规则，不会废数据。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ═══════════════════════════════════════════════
# Stage 1: Observable Concept Schema (稳定层)
# ═══════════════════════════════════════════════

# 20-40 个可观察叙事概念，LLM 从 narrative JSON 中抽取
# 每个概念是 binary 或 [0,1] continuous
OBSERVABLE_CONCEPTS = {
    # ── Origin type ──
    "official_release":    "官方发布 (如B站宣传片、政府报告)",
    "grassroots":           "草根自发 (如贴吧帖子、社区段子)",
    "celebrity_trigger":    "明星/名人触发",
    "accident_trigger":     "偶然事件触发",
    "policy_trigger":       "政策/制度触发",
    "platform_event":       "平台活动触发",

    # ── Spread mechanism ──
    "KOL_amplification":    "关键意见领袖放大",
    "algorithm_push":       "算法推荐推动",
    "cross_platform":       "跨平台迁移",
    "mainstream_media":     "主流媒体介入",
    "brand_hijack":         "品牌/营销蹭热点",

    # ── Conflict type ──
    "class_conflict":       "阶层冲突 (贫富/城乡/学历)",
    "gender_conflict":      "性别冲突",
    "generation_conflict":  "代际冲突",
    "political_conflict":   "政治敏感冲突",
    "value_conflict":       "价值观冲突 (传统 vs 现代)",

    # ── Mutation type ──
    "parody":               "戏仿/恶搞",
    "irony":                "反讽/解构",
    "semantic_drift":       "语义漂移 (原义→新义)",
    "remix":                "二次创作/模板化",
    "institutionalization": "制度化收编",

    # ── Emotion ──
    "anger":                "愤怒",
    "humor_laugh":          "幽默/好笑",
    "schadenfreude":        "幸灾乐祸",
    "identity_belonging":   "身份认同/归属",
    "hope":                 "希望/向往",
    "nihilism":             "虚无/无力",
    "anxiety":              "焦虑",
    "nostalgia":            "怀旧",

    # ── Audience ──
    "youth_dominant":       "年轻人主导",
    "white_collar":         "白领/职场人群",
    "student":              "学生群体",
    "rural":                "下沉/农村",
    "elite":                "精英/知识分子",
}

CONCEPT_NAMES = list(OBSERVABLE_CONCEPTS.keys())
N_CONCEPTS = len(CONCEPT_NAMES)  # 35


@dataclass
class ConceptMatrix:
    """一个 Trajectory 节点的可观察概念向量。"""
    values: np.ndarray  # shape (35,) — 每个概念 ∈ [0, 1]
    source: str = "llm"  # "llm" | "rule" | "manual"

    @classmethod
    def from_narrative(cls, narrative: dict) -> "ConceptMatrix":
        """从 LLM 抽取的 narrative JSON 构建概念向量。

        使用软匹配（字符 bigram Jaccard 相似度）替代硬正则。
        LLM 措辞变化不会导致概念归零坍缩。
        """
        v = np.zeros(N_CONCEPTS)

        # 构建叙事全文（所有可用文本字段拼接）
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
        ])

        # 字符 bigram 集合
        def bigrams(text: str) -> set:
            return set(text[i:i+2] for i in range(len(text)-1))

        def soft_score(text: str, anchor_keywords: str) -> float:
            """计算叙事文本与概念锚点关键词的软匹配分数。"""
            if not text.strip():
                return 0.0
            # Jaccard on character bigrams
            t_bg = bigrams(text)
            a_bg = bigrams(anchor_keywords)
            if not t_bg or not a_bg:
                return 0.0
            jaccard = len(t_bg & a_bg) / max(1, len(t_bg | a_bg))
            # Also check direct substring match for key terms
            kw_terms = anchor_keywords.replace(" ", "").replace(",", " ").split()
            direct_hits = sum(1 for kw in kw_terms if kw in text)
            direct_bonus = min(0.3, direct_hits * 0.1)
            return min(1.0, jaccard * 4.0 + direct_bonus)  # amplify Jaccard

        # ── 对每个概念计算软匹配分数 ──
        concept_anchors = {
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

        for name, anchor in concept_anchors.items():
            if name in CONCEPT_NAMES:
                idx = CONCEPT_NAMES.index(name)
                v[idx] = soft_score(narrative_full, anchor)

        return cls(values=v, source="soft_match")

    def to_dict(self) -> dict:
        return {CONCEPT_NAMES[i]: float(self.values[i]) for i in range(N_CONCEPTS)}

    @property
    def active_concepts(self) -> list[str]:
        """返回被激活的概念（值 > 0.3）。"""
        return [CONCEPT_NAMES[i] for i in range(N_CONCEPTS) if self.values[i] > 0.3]


# ═══════════════════════════════════════════════
# Stage 2: Concept → 5D Constraint Mapping
# ═══════════════════════════════════════════════

class ConstraintMapper:
    """将 Concept Matrix 映射到 5D 约束场。

    当前使用基于传播学物理先验的加权映射（不是学习出来的）。
    以后有了更多数据，这层可以换成线性回归或其他学习模型。
    """

    # 5 个约束维度的概念权重
    # 每行对应一个约束维度，每列对应一个可观察概念的贡献权重
    IDENTITY_WEIGHTS = {
        "identity_belonging": 0.4, "youth_dominant": 0.15, "white_collar": 0.1,
        "student": 0.1, "official_release": -0.2, "grassroots": 0.2,
        "hope": 0.15, "mainstream_media": 0.1, "KOL_amplification": 0.15,
    }

    HUMOR_WEIGHTS = {
        "humor_laugh": 0.5, "irony": 0.3, "parody": 0.25, "remix": 0.15,
        "anger": -0.2, "nihilism": -0.1, "semantic_drift": 0.1, "cross_platform": 0.1,
    }

    CONFLICT_WEIGHTS = {
        "class_conflict": 0.35, "gender_conflict": 0.3, "political_conflict": 0.3,
        "generation_conflict": 0.25, "value_conflict": 0.25, "anger": 0.2,
        "backlash": 0.2, "algorithm_push": -0.1, "identity_belonging": -0.15,
    }

    NOVELTY_WEIGHTS = {
        "parody": 0.3, "remix": 0.3, "semantic_drift": 0.25, "irony": 0.2,
        "accident_trigger": 0.2, "grassroots": 0.2, "official_release": -0.15,
        "cross_platform": 0.15, "KOL_amplification": 0.1,
    }

    ACCESSIBILITY_WEIGHTS = {
        "humor_laugh": 0.3, "remix": 0.25, "youth_dominant": 0.2,
        "algorithm_push": 0.25, "cross_platform": 0.2, "brand_hijack": 0.15,
        "political_conflict": -0.3, "class_conflict": -0.1, "elite": -0.15,
    }

    @staticmethod
    def _weighted_sum(concept_vec: np.ndarray, weights: dict) -> float:
        total = 0.0
        for name, w in weights.items():
            if name in CONCEPT_NAMES:
                idx = CONCEPT_NAMES.index(name)
                total += w * concept_vec[idx]
        # Sigmoid 压到 [0.05, 0.95]
        raw = np.clip(total + 0.5, 0.0, 1.0)
        return float(0.05 + 0.9 / (1.0 + np.exp(-6.0 * (raw - 0.5))))

    @classmethod
    def map(cls, concept: ConceptMatrix) -> np.ndarray:
        """Concept → 5D Constraint。"""
        v = concept.values
        p = np.zeros(5)
        p[0] = cls._weighted_sum(v, cls.IDENTITY_WEIGHTS)
        p[1] = cls._weighted_sum(v, cls.HUMOR_WEIGHTS)
        p[2] = cls._weighted_sum(v, cls.CONFLICT_WEIGHTS)
        p[3] = cls._weighted_sum(v, cls.NOVELTY_WEIGHTS)
        p[4] = cls._weighted_sum(v, cls.ACCESSIBILITY_WEIGHTS)
        return p


# ═══════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys, json
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 50)
    print("Concept Bottleneck — Demo")
    print("=" * 50)

    # 用后浪 narrative 演示
    import os
    nar_dir = "data/processed/narratives"
    for fn in os.listdir(nar_dir):
        if "后浪" in fn and fn.endswith(".json"):
            with open(os.path.join(nar_dir, fn), "r", encoding="utf-8") as f:
                nar = json.load(f)
            break

    cm = ConceptMatrix.from_narrative(nar)
    p = ConstraintMapper.map(cm)

    print(f"\nNarrative: {nar.get('meme_name', '?')}")
    print(f"\nActive concepts ({len(cm.active_concepts)}):")
    for c in cm.active_concepts:
        idx = CONCEPT_NAMES.index(c)
        print(f"  {c}: {cm.values[idx]:.2f}  ({OBSERVABLE_CONCEPTS[c][:50]})")

    print(f"\n5D Constraint:")
    labels = ["Identity", "Humor/Decon", "Conflict", "Novelty", "Accessibility"]
    for i, (label, val) in enumerate(zip(labels, p)):
        bar = "█" * int(val * 20)
        print(f"  p{i+1} {label:<16s}: {val:.3f} {bar}")
