"""
个体画像模型 — FR31 Layer 2

每个人有一个 Persona, 存储:
- 已知特质、偏好、行为模式
- 互动时间线 (关键事件 + 信号密度变化)
- 沟通风格指纹
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json
from pathlib import Path


@dataclass
class Persona:
    """一个人的完整画像."""
    name: str
    # 基础信息
    age: int = 0
    gender: str = ""
    occupation: str = ""          # 学生/专业
    mbti: str = ""                # 如有

    # 人格特质 (0-1)
    traits: dict = field(default_factory=lambda: {
        "openness": 0.5,          # 对新鲜体验的开放度
        "emotional_expressiveness": 0.5,  # 情感外露程度
        "conflict_avoidance": 0.5,       # 回避冲突倾向
        "initiative": 0.5,               # 主动发起互动的倾向
        "self_esteem_stability": 0.5,    # 自我价值感稳定性
        "rebellion_restrained": 0.5,     # 被压抑的反叛冲动
        "depth_processing": 0.5,         # 深度自我觉察倾向
        "social_energy": 0.5,            # 社交精力
    })

    # 沟通风格
    communication: dict = field(default_factory=lambda: {
        "response_delay_typical": "几分钟到几小时",
        "response_delay_when_unsure": "几小时到一天",
        "preferred_platforms": [],
        "emoji_density": "medium",
        "topic_initiation": "passive",       # passive / active / mixed
        "emotional_directness": "avoidant",  # avoidant / open / mixed
        "small_talk_style": "casual",
        "deep_talk_triggers": [],       # 什么话题能引发她说更多
        "retreat_triggers": [],         # 什么话题会让她退回去
        "humor_type": "self_deprecating",
    })

    # 已知偏好
    preferences: dict = field(default_factory=lambda: {
        "music": [],
        "games": [],
        "anime": [],
        "activities": [],
        "food": [],
        "dislikes": [],
    })

    # 约束条件 (她觉得什么情况下必须保持距离)
    constraints: dict = field(default_factory=lambda: {
        "family_pressure": 0.0,       # 家庭对她的行为约束强度
        "social_image_concern": 0.0,  # 在意社交形象
        "emotional_safety_need": 0.0, # 需要情感安全感才敢靠近
        "avoidance_threshold": 0.0,   # 什么程度的暧昧会触发回避
        "commitment_readiness": 0.0,  # 当前对确定关系的心理准备
    })

    # 关键事件时间线
    timeline: list[dict] = field(default_factory=list)
    # [{"date": "", "event": "", "significance": "", "signal_change": ""}]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "age": self.age,
            "occupation": self.occupation,
            "mbti": self.mbti,
            "traits": self.traits,
            "communication": self.communication,
            "preferences": self.preferences,
            "constraints": self.constraints,
            "timeline": self.timeline,
        }

    def save(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Persona":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        p = cls(name=d["name"])
        for k in ["age", "occupation", "mbti"]:
            if k in d: setattr(p, k, d[k])
        for k in ["traits", "communication", "preferences", "constraints"]:
            if k in d: setattr(p, k, {**getattr(p, k), **d[k]})
        if "timeline" in d: p.timeline = d["timeline"]
        return p


# ── 预置画像 ──

def ycs_profile() -> Persona:
    """杨铖爽 的画像 (基于聊天记录和互动历史提取)."""
    return Persona(
        name="杨铖爽",
        age=19,
        occupation="中国美术学院 大二 设计类",
        mbti="INFP",
        traits={
            "openness": 0.7,            # 愿意见网友、尝试新游戏
            "emotional_expressiveness": 0.3,  # 外表平静，内心强度高但不外露
            "conflict_avoidance": 0.8,   # 不喜欢冲突，父母面前顺从
            "initiative": 0.4,           # 偶尔主动，但更多是被动等待
            "self_esteem_stability": 0.3, # "觉得自己不够好"，羡慕他人
            "rebellion_restrained": 0.8,  # 通过虚拟世界释放反叛冲动
            "depth_processing": 0.3,      # 不习惯深层自我剖析
            "social_energy": 0.5,         # 可社交但需要独处恢复
        },
        communication={
            "response_delay_typical": "几分钟",
            "response_delay_when_unsure": "3小时到24小时",
            "preferred_platforms": ["微信大号(正式)", "微信小号(放飞)"],
            "emoji_density": "high",
            "topic_initiation": "passive_with_occasional_active",
            "emotional_directness": "avoidant",
            "small_talk_style": "轻松分享日常 + 表情包",
            "deep_talk_triggers": [
                "音乐/游戏/动漫的深度讨论",
                "童年回忆",
                "关于'被看见'和'被理解'的话题(间接)",
            ],
            "retreat_triggers": [
                "直接的情感确认",
                "被追问关系定义",
                "对方的情感强度突然升高",
                "感觉被'索取'回应",
            ],
            "humor_type": "自我调侃 + 表情包接梗",
        },
        preferences={
            "music": ["韩语歌", "郑润泽", "王菲", "蛋堡", "Porter Robinson", "XG", "PLAVE", "宇多田光", "Epik High"],
            "games": ["星露谷物语", "主播女孩重度依赖", "饥荒", "胡闹厨房"],
            "anime": ["mygo", "avemujica", "NANA", "电锯人"],
            "activities": ["旅游", "喝糖水/咖啡", "逛商场", "KTV", "看演唱会", "画画(有时)"],
            "food": ["糖水", "烧烤", "奶茶", "东北菜"],
            "dislikes": ["密室逃脱", "台球", "运动", "被班主任压力"],
        },
        constraints={
            "family_pressure": 0.6,       # 父母管得多，9点就睡
            "social_image_concern": 0.5,  # 在意但不算过度
            "emotional_safety_need": 0.75, # 需要感觉"安全"才会靠近
            "avoidance_threshold": 0.5,    # 中等程度的暧昧就会触发回避
            "commitment_readiness": 0.25,   # 当前对确定关系的心理准备很低
        },
        timeline=[
            {"date": "2026-01-13", "event": "初次线上认识(Soul)", "significance": "起点"},
            {"date": "2026-01-14", "event": "第一次线下见面(糖水+公园)", "significance": "她主动提前半小时到，主动请糖水、邀公园散步、追问理想型"},
            {"date": "2026-01-19", "event": "第二次线下见面(咖啡+牵手)", "significance": "牵手20分钟无抗拒，临走反复回头挥手。当晚线上互动仍积极"},
            {"date": "2026-01-21", "event": "你表白→她无法回应→你删除大小号", "significance": "情感过热导致系统被劫持，断联。她小号未删你，第二天大号发了见面照片"},
            {"date": "2026-01-24", "event": "第一次尝试加回大号(未通过)", "significance": "她大号拒绝，但小号未删=保留观察窗口"},
            {"date": "2026-02-09", "event": "用小号'金善旴'加她大号(通过)", "significance": "迂回重建联系"},
            {"date": "2026-02-19", "event": "恢复对话(演唱会话题)", "significance": "断联后第一次实质性对话，她隔24.5h回复(纠结)"},
            {"date": "2026-03-02", "event": "她主动搜索微信ID加回你大号", "significance": "关系从观察期转入正式对话期"},
            {"date": "2026-04-29", "event": "她生日, 你送超天酱+手绘贺卡", "significance": "她收到后主动拍照感谢，互动升温"},
            {"date": "2026-05-20", "event": "520 她主动找你聊健美操考试，发语音", "significance": "频率明显增加，她开始更主动"},
            {"date": "2026-06-29", "event": "她问'你什么时候放假呀'", "significance": "暑假将至，可能打开线下见面窗口。明天她飞首尔+出分"},
        ],
    )
