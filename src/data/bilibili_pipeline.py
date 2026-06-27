"""
B站字幕 → SIRS-M 真实参数拟合管道 (v2.0)

Gemini 贡献原始版本。2026-06-26 整合 GPT/Gemini 审计反馈重写：
- 拼音拓扑空间对齐匹配 (PhoneticMemeMatcher) — 替代全文纠错
- 已策展/未策展分流管道
- 稳健 SIRS-M 拟合 2.0 (L2正则 + S₀注入 + 活跃区间屏蔽 + 尺度参数)

对齐「微尘哲学」：
- ASR 噪声 = 信息传输中的熵增 → 在拼音空间与之共存，而非试图消灭
- 拟合 = 从混沌的经验数据中提取确定性结构（局部秩序建立）

用法：
    python src/data/bilibili_pipeline.py
"""

import os
import json
import re
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ── 拼音匹配依赖 (pip install pypinyin rapidfuzz) ──
try:
    import pypinyin
    HAS_PINYIN = True
except ImportError:
    HAS_PINYIN = False
    print("[警告] pypinyin 未安装，拼音匹配不可用。pip install pypinyin")

try:
    from rapidfuzz import fuzz as rf_fuzz
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False
    print("[警告] rapidfuzz 未安装，模糊匹配不可用。pip install rapidfuzz")


# ═══════════════════════════════════════════════
# 0. 视频 → 热梗映射表
# ═══════════════════════════════════════════════

# 已策展热梗 (12个): 视频目录名关键词 → (meme_id, 搜索关键词列表)
# 关键词包含 meme name + aliases
KNOWN_MEME_MAP = {
    "打工人": {
        "meme_id": "meme_001",
        "keywords": ["打工人", "打工魂", "打工都是人上人", "打工"],
        "category": "解构自嘲",
    },
    "内卷": {
        "meme_id": "meme_002",
        "keywords": ["内卷", "卷王", "卷起来", "太卷了", "反内卷", "involution",
                      "內卷", "內捷", "内捷", "内部竞争"],  # ASR常见误识别
        "category": "身份认同",
    },
    "躺平": {
        "meme_id": "meme_003",
        "keywords": ["躺平", "躺平主义", "三和大神", "lying flat"],
        "category": "虚无退却",
    },
    "普通却自信": {
        "meme_id": "meme_004",
        "keywords": ["普信男", "普信女", "普信", "那么普通那么自信", "普通却自信"],
        "category": "攻击发泄",
    },
    "小镇做题家": {
        "meme_id": "meme_005",
        "keywords": ["小镇做题家", "做题家", "985废物"],
        "category": "身份认同",
    },
    "摆烂": {
        "meme_id": "meme_006",
        "keywords": ["摆烂", "开摆", "摆烂了", "彻底疯狂"],
        "category": "虚无退却",
    },
    "润": {
        "meme_id": "meme_007",
        "keywords": ["润", "润学", "提桶跑路", "run", "潤", "潤學", "认"],  # ASR: 润→认(run→ren)
        "category": "虚无退却",
    },
    "孔乙己的长衫": {
        "meme_id": "meme_013",
        "keywords": ["孔乙己的长衫", "脱不掉的长衫", "学历是长衫", "孔乙己", "长衫"],
        "category": "身份认同",
    },
    "当代大学生的精神状态": {
        "meme_id": "meme_014",
        "keywords": ["精神状态良好", "精神状态稳定", "大学生的精神状态", "精神状态", "发疯文学"],
        "category": "解构自嘲",
    },
    "科目三": {
        "meme_id": "meme_017",
        "keywords": ["科目三", "广西科目三", "丝滑摇摆", "科末三", "科末3"],
        "category": "纯粹娱乐",
    },
    "后浪": {
        "meme_id": "meme_021",
        "keywords": ["后浪", "奔涌吧后浪", "前浪", "韭浪"],
        "category": "身份认同",
    },
    "鸡你胎没": {
        "meme_id": "meme_020",
        "keywords": ["鸡你太美", "只因你太美", "蔡徐坤", "小黑子", "鸡你胎没"],
        "category": "纯粹娱乐",
    },
}

# 未策展热梗 (10个): 视频目录名关键词 → 探索标记
UNKNOWN_MEME_DIRS = [
    "GO学长瓦学弟",
    "†升天†",
    "月薪喵",
    "刘慈欣还是留了一手",
    "嘉豪",
    "头顶尖尖",
    "我的刀盾",
    "曾经的王",
    "这是一场试炼",
    "熊出没",
]


# ═══════════════════════════════════════════════
# 1. 拼音拓扑对齐匹配器 (PhoneticMemeMatcher)
# ═══════════════════════════════════════════════

class PhoneticMemeMatcher:
    """在 ASR 转录文本中通过拼音空间对齐检测目标热梗。

    Gemini 设计：ASR 错误多为同音/近音字混淆，汉字编辑距离大但拼音距离≈0。
    将 ASR 文本与目标词库同时投射到拼音空间进行滑动窗口匹配，
    实现高召回率的容错检索。

    三层匹配策略 (GPT 建议):
    Layer 1: 精确拼音匹配 (拼音距离 = 0)
    Layer 2: 拼音模糊匹配 (拼音 Levenshtein ≤ 1)
    Layer 3: 部分匹配 (字符级 fuzzy ratio ≥ 85)
    """

    def __init__(self, target_dict: dict[str, list[str]]):
        """
        Args:
            target_dict: {meme_id: [keyword1, keyword2, ...]}
        """
        self.targets = target_dict
        # 预缓存所有关键词的拼音
        self.phonetic_lib = {}
        for meme_id, keywords in target_dict.items():
            self.phonetic_lib[meme_id] = [
                (kw, self._to_pinyin(kw)) for kw in keywords
            ]

    @staticmethod
    def _to_pinyin(text: str) -> str:
        """中文文本 → 无声调拼音串，数字→中文数字规范化。"""
        if not HAS_PINYIN:
            return text.lower()
        # Normalize digits to Chinese (ASR common: 科目3 vs 科目三)
        digit_map = str.maketrans("0123456789", "零一二三四五六七八九")
        normalized = text.translate(digit_map)
        py_list = pypinyin.pinyin(normalized, style=pypinyin.Style.NORMAL)
        return "".join([c[0] for c in py_list]).lower()

    def scan_segment(self, asr_text: str, start_time: float,
                     window_size: int = 8) -> list[dict]:
        """在单个 ASR 文本段落中滑动窗口扫描目标热梗。

        Args:
            asr_text: ASR 转录文本段落
            start_time: 该段落的起始时间 (秒)
            window_size: 滑动窗口最大宽度 (字符)

        Returns:
            [{"meme_id", "matched_text", "time", "confidence", "layer"}, ...]
        """
        hits = []
        if len(asr_text) < 2:
            return hits

        asr_py = self._to_pinyin(asr_text)
        asr_len = len(asr_text)

        for meme_id, kw_pinyin_pairs in self.phonetic_lib.items():
            for kw_text, kw_py in kw_pinyin_pairs:
                kw_len = len(kw_text)
                if kw_len > asr_len:
                    continue

                matched = False
                # ── Layer 1: exact pinyin match ──
                for i in range(asr_len - kw_len + 1):
                    sub_py = self._to_pinyin(asr_text[i:i + kw_len])
                    if sub_py == kw_py:
                        hits.append({
                            "meme_id": meme_id,
                            "matched_text": asr_text[i:i + kw_len],
                            "time": start_time,
                            "confidence": 1.0,
                            "layer": "exact_pinyin",
                        })
                        matched = True
                        break
                if matched:
                    continue

                # ── Layer 2: fuzzy pinyin (Levenshtein ≤ 1) ──
                for i in range(asr_len - kw_len + 1):
                    sub_text = asr_text[i:i + kw_len]
                    sub_py = self._to_pinyin(sub_text)
                    try:
                        from Levenshtein import distance as lev_dist
                        dist = lev_dist(sub_py, kw_py)
                    except ImportError:
                        dist = sum(1 for a, b in zip(sub_py, kw_py) if a != b) + abs(len(sub_py) - len(kw_py))
                    if dist <= 1 and len(sub_py) >= len(kw_py) * 0.7:
                        hits.append({
                            "meme_id": meme_id,
                            "matched_text": sub_text,
                            "time": start_time,
                            "confidence": 0.7 - 0.3 * dist,
                            "layer": "fuzzy_pinyin",
                        })
                        matched = True
                        break
                if matched:
                    continue

                # ── Layer 3: character-level fuzzy (rapidfuzz) ──
                if HAS_FUZZY:
                    for i in range(max(1, asr_len - window_size + 1)):
                        window = asr_text[i:i + window_size]
                        score = rf_fuzz.partial_ratio(kw_text, window) / 100.0
                        if score > 0.85:
                            hits.append({
                                "meme_id": meme_id,
                                "matched_text": window[:kw_len],
                                "time": start_time,
                                "confidence": score * 0.6,
                                "layer": "char_fuzzy",
                            })
                            break

        return hits

    def scan_transcript(self, transcript: list[dict]) -> list[dict]:
        """扫描整个视频的字幕文本。

        Args:
            transcript: [{"text": str, "start": float}, ...]

        Returns:
            所有匹配项的列表，按时间排序
        """
        all_hits = []
        for seg in transcript:
            text = seg.get("text", "")
            start = seg.get("start", seg.get("start_sec", 0))
            hits = self.scan_segment(text, start)
            all_hits.extend(hits)
        all_hits.sort(key=lambda h: h["time"])
        return all_hits


# ═══════════════════════════════════════════════
# 2. 视频加载 + 时间线抽取
# ═══════════════════════════════════════════════

def load_video_to_text_transcripts(video_dir: str) -> list[dict]:
    """加载 Video_to_Text 输出的单个视频字幕。

    Args:
        video_dir: 视频输出目录路径 (如 .../打工人(Av...)/)

    Returns:
        [{"text": "...", "start": 12.5, "duration": 2.1}, ...]
    """
    rp = os.path.join(video_dir, "report.json")
    if not os.path.exists(rp):
        print(f"  [跳过] 未找到 report.json: {video_dir}")
        return []

    with open(rp, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = []
    for seg in data.get("transcript", []):
        start = seg.get("start_sec", 0)
        end = seg.get("end_sec", start + 5)
        text = seg.get("text", "")
        segments.append({
            "text": text,
            "start": float(start),
            "duration": float(end - start),
        })
    return segments


def extract_meme_timeline(hits: list[dict], bin_width_sec: float = 15.0,
                          total_duration: float = None) -> tuple[np.ndarray, np.ndarray]:
    """从拼音匹配命中列表中构建 I(t) 时间序列。

    GPT 建议: 将关键词频率视为 I(t) 的带噪观测 (freq ≈ c × I(t))，
    不做过度归一化。

    Args:
        hits: PhoneticMemeMatcher 的输出
        bin_width_sec: 时间分箱宽度（秒）
        total_duration: 视频总时长 (用于确定分箱范围)

    Returns:
        t_data: 时间轴 (单位: bin index)
        I_data: 未归一化的命中密度序列 (后续拟合中由 scale 参数处理)
    """
    if not hits:
        return np.array([]), np.array([])

    if total_duration is None:
        total_duration = max(h["time"] for h in hits) + bin_width_sec

    num_bins = int(np.ceil(total_duration / bin_width_sec)) + 1
    counts = np.zeros(num_bins)
    conf_weighted = np.zeros(num_bins)

    for h in hits:
        bin_idx = int(h["time"] // bin_width_sec)
        if bin_idx < num_bins:
            counts[bin_idx] += 1
            conf_weighted[bin_idx] += h.get("confidence", 0.5)

    # 用置信度加权替代简单计数
    signal = conf_weighted

    # Rolling window smoothing (3-bin, center-aligned)
    smoothed = (
        pd.Series(signal)
        .rolling(window=3, min_periods=1, center=True)
        .mean()
        .values
    )

    # 不做 max-normalize (GPT 建议: 保留原始尺度供拟合)
    t_data = np.arange(len(smoothed))
    return t_data, smoothed


# ═══════════════════════════════════════════════
# 3. SIRS-M ODE 系统
# ═══════════════════════════════════════════════

def sirs_m_ode(t, y, beta, sigma, gamma, mu, N=1.0):
    """SIRS-M ODE 系统。

    dS/dt = -beta*S*I/N + gamma*R
    dI/dt =  beta*S*I/N - sigma*I + mu*R
    dR/dt =  sigma*I - gamma*R - mu*R
    """
    S, I, R = y
    dS_dt = -beta * S * I / N + gamma * R
    dI_dt = beta * S * I / N - sigma * I + mu * R
    dR_dt = sigma * I - gamma * R - mu * R
    return [dS_dt, dI_dt, dR_dt]


# ═══════════════════════════════════════════════
# 4. 稳健 SIRS-M 拟合 2.0
# ═══════════════════════════════════════════════

def robust_fit_sirs_m(t_data: np.ndarray, I_data: np.ndarray,
                      category: str = None, S0: float = 0.7):
    """稳健 SIRS-M 非线性拟合 (v2.0)。

    Gemini P1 建议 + GPT 建议 整合:
    - β bound 拓宽至 (0.01, 5.0) 兼容超传播模因
    - L2 正则化惩罚超大 β 和 μ，防止拟合局部噪声尖峰
    - S₀ 动态注入: 默认 0.7 (B站圈层限制)
    - 活跃区间屏蔽: 只拟合 I > 0.02*max(I) 的区域，避免静默期稀释精度
    - 尺度参数 a: I_obs ≈ a × I_fitted(t) (GPT: 不同视频的频次尺度不同)

    Args:
        t_data: 时间轴
        I_data: 经验命中密度序列
        category: 热梗类别 (用于选择 β 先验)
        S0: 初始易感人群比例 (默认 0.7 = B站圈层限制)

    Returns:
        {"beta", "sigma", "gamma", "mu", "R0", "scale_a", "S0", "success", "fun"} 或 None
    """
    if len(t_data) < 5:
        return None

    # ── 活跃区间屏蔽 (Gemini 建议) ──
    max_I = np.max(I_data)
    if max_I < 1e-6:
        return None
    active_mask = I_data > 0.02 * max_I
    if np.sum(active_mask) < 5:
        active_mask = np.ones_like(I_data, dtype=bool)

    t_active = t_data[active_mask].astype(float)
    I_active = I_data[active_mask].astype(float)

    # ── 类别先验 β (Gemini 建议) ──
    category_priors = {
        "攻击发泄": 0.8,
        "虚无退却": 0.5,
        "解构自嘲": 0.35,
        "身份认同": 0.4,
        "纯粹娱乐": 0.6,
    }
    beta_prior = category_priors.get(category, 0.5) if category else 0.5

    # ── 初始条件 ──
    I0 = max(0.001, I_active[0] / max(1.0, max_I))
    y0 = [S0 - I0, I0, 1.0 - S0]  # 剩余 1-S0 为完全不触达群体

    # ── 参数边界 (β拓宽至 5.0, Gemini P1) ──
    bounds = [(0.01, 5.0), (0.01, 1.5), (0.0, 0.5), (0.0, 0.5)]

    # ── 含 L2 正则 + 尺度参数的损失函数 ──
    def regularized_loss(params):
        beta, sigma, gamma, mu = params
        sol = solve_ivp(
            fun=sirs_m_ode,
            t_span=(t_active[0], t_active[-1]),
            y0=y0,
            t_eval=t_active,
            args=(beta, sigma, gamma, mu, S0),
            method="RK45",
        )
        if not sol.success:
            return 1e6

        I_fit = sol.y[1]

        # GPT 建议: 观测尺度参数 a
        # I_obs ≈ a * I_fit → a = argmin ||I_obs - a*I_fit||²
        # 解析解: a = sum(I_obs * I_fit) / sum(I_fit²)
        a_num = np.sum(I_active * I_fit)
        a_den = np.sum(I_fit ** 2)
        if a_den < 1e-12:
            a = 1.0
        else:
            a = max(0.1, min(10.0, a_num / a_den))

        I_scaled = a * I_fit
        ssr = float(np.sum((I_active - I_scaled) ** 2))

        # Gemini: L2 正则化惩罚项
        reg = 0.1 * (beta - beta_prior) ** 2 + 0.2 * (mu ** 2)

        return ssr + reg

    # ── L-BFGS-B 优化 ──
    init = [beta_prior, 0.3, 0.1, 0.05]
    res = minimize(regularized_loss, init, method="L-BFGS-B", bounds=bounds)

    if res.success:
        beta_f, sigma_f, gamma_f, mu_f = res.x
        r0 = beta_f / sigma_f if sigma_f > 0 else 0.0

        # 重新计算最优 a
        sol_final = solve_ivp(
            fun=sirs_m_ode,
            t_span=(t_active[0], t_active[-1]),
            y0=y0,
            t_eval=t_active,
            args=(beta_f, sigma_f, gamma_f, mu_f, S0),
            method="RK45",
        )
        I_final = sol_final.y[1] if sol_final.success else np.ones_like(t_active)
        a_num = np.sum(I_active * I_final)
        a_den = np.sum(I_final ** 2)
        scale_a = float(max(0.1, min(10.0, a_num / a_den))) if a_den > 1e-12 else 1.0

        return {
            "beta": float(beta_f),
            "sigma": float(sigma_f),
            "gamma": float(gamma_f),
            "mu": float(mu_f),
            "R0": float(r0),
            "scale_a": scale_a,
            "S0": S0,
            "success": True,
            "fun": float(res.fun),
            "n_active_points": int(np.sum(active_mask)),
        }
    return {"success": False, "message": str(res.message)}


# ═══════════════════════════════════════════════
# 5. 已策展热梗管道
# ═══════════════════════════════════════════════

def run_known_meme_pipeline(video_base_dir: str,
                            video_dirs: list[str],
                            meme_map: dict) -> list[dict]:
    """对已策展热梗运行完整拼音匹配 → SIRS-M 拟合管道。

    Args:
        video_base_dir: Video_to_Text 输出根目录
        video_dirs: 子目录名列表
        meme_map: {video_keyword: {meme_id, keywords, category}}

    Returns:
        [{"meme_id", "name", "category", "fit", "n_hits", ...}, ...]
    """
    results = []

    for keyword, info in meme_map.items():
        print(f"\n{'='*50}")
        print(f"  {keyword} → {info['meme_id']} [{info['category']}]")
        print(f"{'='*50}")

        # 找到匹配的视频目录
        matched_dir = None
        for d in video_dirs:
            if keyword in d:
                matched_dir = d
                break

        if not matched_dir:
            print(f"  [跳过] 未找到含 '{keyword}' 的视频目录")
            continue

        video_path = os.path.join(video_base_dir, matched_dir)
        transcript = load_video_to_text_transcripts(video_path)
        if not transcript:
            print(f"  [跳过] 无转录数据")
            continue

        print(f"  转录: {len(transcript)} 段, "
              f"总时长: {transcript[-1]['start'] + transcript[-1].get('duration', 0):.0f}s")

        # ── 拼音匹配 ──
        matcher = PhoneticMemeMatcher({info["meme_id"]: info["keywords"]})
        hits = matcher.scan_transcript(transcript)

        print(f"  匹配命中: {len(hits)} 次")
        if hits:
            layer_counts = {}
            for h in hits:
                layer_counts[h["layer"]] = layer_counts.get(h["layer"], 0) + 1
            print(f"  层次分布: {layer_counts}")
            # 展示几个样例匹配
            for h in hits[:3]:
                print(f"    [{h['time']:.0f}s] '{h['matched_text']}' "
                      f"(conf={h['confidence']:.2f}, {h['layer']})")

        # ── 时间线抽取 ──
        total_dur = transcript[-1]["start"] + transcript[-1].get("duration", 0)
        t_data, I_data = extract_meme_timeline(hits, bin_width_sec=15.0,
                                                total_duration=total_dur)

        if len(t_data) < 5 or np.max(I_data) < 1e-6:
            print(f"  [跳过] 有效数据点不足")
            results.append({
                "meme_id": info["meme_id"],
                "name": keyword,
                "category": info["category"],
                "n_hits": len(hits),
                "fit": None,
            })
            continue

        print(f"  时间线: {len(t_data)} bins, max_signal={np.max(I_data):.3f}")

        # ── SIRS-M 拟合 ──
        fit = robust_fit_sirs_m(t_data, I_data, category=info["category"])
        if fit and fit.get("success"):
            phase = "[激活] 模因爆发" if fit["R0"] > 1.0 else "[衰减] 未建立秩序"
            print(f"  拟合结果:")
            print(f"    β={fit['beta']:.3f}  σ={fit['sigma']:.3f}  "
                  f"γ={fit['gamma']:.3f}  μ={fit['mu']:.3f}")
            print(f"    R₀={fit['R0']:.3f}  scale_a={fit['scale_a']:.2f}  "
                  f"S₀={fit['S0']:.1f}  {phase}")
            print(f"    活跃点: {fit['n_active_points']}/{len(t_data)}  "
                  f"SSR={fit['fun']:.4f}")
        else:
            msg = fit.get("message", "未知错误") if fit else "无有效数据"
            print(f"  [失败] {msg}")

        results.append({
            "meme_id": info["meme_id"],
            "name": keyword,
            "category": info["category"],
            "n_hits": len(hits),
            "fit": fit,
        })

    return results


# ═══════════════════════════════════════════════
# 6. 未策展热梗探索
# ═══════════════════════════════════════════════

def explore_unknown_memes(video_base_dir: str, video_dirs: list[str],
                          unknown_keywords: list[str]) -> list[dict]:
    """对未策展视频做 TF-IDF 风格的关键词发现。

    GPT 建议: 不设固定关键词，而是抽取高频词 + 高 PMI 搭配，
    标记为候选新梗供后续人工策展。

    Returns:
        [{"video", "top_terms", "candidate_meme_name", ...}, ...]
    """
    results = []
    # 简易中文停用词
    stop_words = {"的", "是", "了", "在", "我", "有", "和", "就", "不", "人", "都",
                  "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
                  "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
                  "们", "那", "什么", "怎么", "为什么", "因为", "所以", "但是",
                  "可以", "这个", "那个", "就是", "还是", "已经", "这个梗",
                  "梗指南", "梗百科", "是什么梗", "键政"}

    for keyword in unknown_keywords:
        print(f"\n{'='*50}")
        print(f"  [探索] {keyword}")
        print(f"{'='*50}")

        matched_dir = None
        for d in video_dirs:
            if keyword in d:
                matched_dir = d
                break
        if not matched_dir:
            continue

        video_path = os.path.join(video_base_dir, matched_dir)
        transcript = load_video_to_text_transcripts(video_path)
        if not transcript:
            continue

        # 简单 TF 统计 (按 2-gram + 3-gram)
        all_text = " ".join([s["text"] for s in transcript])
        tf = {}
        for n in [2, 3]:
            for i in range(len(all_text) - n):
                gram = all_text[i:i + n]
                if any(c in stop_words for c in gram) and n == 2:
                    continue
                if gram in stop_words:
                    continue
                tf[gram] = tf.get(gram, 0) + 1

        # 排序取 top N
        sorted_terms = sorted(tf.items(), key=lambda x: -x[1])[:20]
        top_terms = [(t, c) for t, c in sorted_terms if c >= 3]

        print(f"  转录: {len(transcript)} 段, "
              f"总时长: {transcript[-1]['start']:.0f}s")
        print(f"  Top n-grams (TF ≥ 3):")
        for term, count in top_terms[:10]:
            print(f"    {term}: {count}")

        # 候选梗名: 取最高频的 3-gram 作为候选
        candidate = top_terms[0][0] if top_terms else keyword

        results.append({
            "video_keyword": keyword,
            "candidate_meme_name": candidate,
            "top_terms": top_terms[:10],
            "n_segments": len(transcript),
        })

    return results


# ═══════════════════════════════════════════════
# 7. 对比验证
# ═══════════════════════════════════════════════

def compare_estimated_vs_real(known_results: list[dict],
                              curated_data_path: str = None):
    """对比 curator 估算 R₀ 与真实 SIRS-M 拟合 R₀。

    Returns:
        {"estimated": [...], "real": [...], "pearson_r": float, "mae": float}
    """
    from src.data.curator import MemeCurator
    from scipy.stats import pearsonr

    curator = MemeCurator()
    estimations = curator.to_sir_estimation()

    # Build lookup: meme_id → estimated R₀
    est_lookup = {}
    for e in estimations:
        est_lookup[e["id"]] = e

    pairs = []
    for r in known_results:
        if r["fit"] is None or not r["fit"].get("success"):
            continue
        mid = r["meme_id"]
        if mid in est_lookup:
            pairs.append({
                "meme_id": mid,
                "name": r["name"],
                "category": r["category"],
                "R0_estimated": est_lookup[mid]["R0_estimated"],
                "R0_real": r["fit"]["R0"],
                "beta_real": r["fit"]["beta"],
                "n_hits": r["n_hits"],
            })

    if len(pairs) < 3:
        print("[警告] 有效对比数据不足")
        return {"pairs": pairs, "pearson_r": None, "mae": None}

    est_vals = [p["R0_estimated"] for p in pairs]
    real_vals = [p["R0_real"] for p in pairs]
    r_val, p_val = pearsonr(est_vals, real_vals)
    mae = np.mean([abs(e - r) for e, r in zip(est_vals, real_vals)])

    print(f"\n{'='*60}")
    print(f"  估算 R₀ vs 真实 R₀ 对比")
    print(f"{'='*60}")
    print(f"  {'热梗':<16s} {'类别':<10s} {'估算R₀':<10s} {'真实R₀':<10s} {'Δ':<8s}")
    print(f"  {'-'*54}")
    for p in pairs:
        delta = p["R0_real"] - p["R0_estimated"]
        print(f"  {p['name']:<16s} {p['category']:<10s} "
              f"{p['R0_estimated']:<10.3f} {p['R0_real']:<10.3f} {delta:+.3f}")
    print(f"  {'-'*54}")
    print(f"  Pearson r = {r_val:.4f} (p={p_val:.4f})")
    print(f"  MAE = {mae:.4f}")
    print(f"  结论: {'方向一致 ✓' if r_val > 0.3 else '相关性弱 — 需要更多数据'}")

    return {
        "pairs": pairs,
        "pearson_r": float(r_val),
        "p_value": float(p_val),
        "mae": float(mae),
    }


# ═══════════════════════════════════════════════
# 8. 运行入口
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("MemeticChaos — B站 ASR 数据接入管道 v2.0")
    print("=" * 60)

    # ── 配置 ──
    VIDEO_BASE = r"E:\Desktop\Video_to_Text\outputs\2026-06-25"
    PROCESSED_DIR = "data/processed"
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not os.path.exists(VIDEO_BASE):
        print(f"\n[错误] 视频目录不存在: {VIDEO_BASE}")
        print("请确认 Video_to_Text 输出路径。")
        sys.exit(1)

    video_dirs = os.listdir(VIDEO_BASE)
    print(f"\n[发现] {len(video_dirs)} 个视频目录")

    # ── 已策展热梗: 拼音匹配 + SIRS-M 拟合 ──
    print(f"\n{'#'*60}")
    print(f"# 已策展热梗 ({len(KNOWN_MEME_MAP)}个): 拼音匹配 + 稳健拟合")
    print(f"{'#'*60}")

    known_results = run_known_meme_pipeline(VIDEO_BASE, video_dirs, KNOWN_MEME_MAP)

    # 统计成功率
    n_success = sum(1 for r in known_results if r["fit"] and r["fit"].get("success"))
    print(f"\n[完成] 已策展: {n_success}/{len(known_results)} 成功拟合")

    # ── 未策展热梗: TF-IDF 探索 ──
    print(f"\n{'#'*60}")
    print(f"# 未策展热梗 ({len(UNKNOWN_MEME_DIRS)}个): TF-IDF 探索")
    print(f"{'#'*60}")

    unknown_results = explore_unknown_memes(VIDEO_BASE, video_dirs, UNKNOWN_MEME_DIRS)

    # ── 对比验证 ──
    if n_success >= 3:
        comparison = compare_estimated_vs_real(known_results)
    else:
        print(f"\n[跳过] 对比验证需要 ≥3 个成功拟合 (当前 {n_success})")

    # ── 保存结果 ──
    output_csv = os.path.join(PROCESSED_DIR, "real_R0_fits.csv")
    rows = []
    for r in known_results:
        fit = r["fit"]
        rows.append({
            "meme_id": r["meme_id"],
            "name": r["name"],
            "category": r["category"],
            "n_hits": r["n_hits"],
            "R0_real": fit["R0"] if fit and fit.get("success") else None,
            "beta": fit["beta"] if fit and fit.get("success") else None,
            "sigma": fit["sigma"] if fit and fit.get("success") else None,
            "gamma": fit["gamma"] if fit and fit.get("success") else None,
            "mu": fit["mu"] if fit and fit.get("success") else None,
            "scale_a": fit["scale_a"] if fit and fit.get("success") else None,
            "fit_success": fit["success"] if fit else False,
        })
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[保存] 拟合结果 → {output_csv}")

    # 未策展探索结果
    if unknown_results:
        explore_csv = os.path.join(PROCESSED_DIR, "unknown_meme_candidates.csv")
        pd.DataFrame(unknown_results).to_csv(explore_csv, index=False, encoding="utf-8-sig")
        print(f"[保存] 候选新梗 → {explore_csv}")

    print(f"\n{'='*60}")
    print("Pipeline complete.")
    print(f"{'='*60}")
