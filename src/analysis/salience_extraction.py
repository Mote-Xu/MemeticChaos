"""
Salience Extraction — 跨框"注意力"共同可观测量的构造设计 (2026-07-13, DESIGN STUB, 未跑数据)

★这是设计稿, 供 auditor 复核; 不执行、不动服务器、不需 GitHub。★

目标: 为 cross_frame comparison_a 构造 s_frame(M, t) = "梗 M 在时间 t 受多少关注",
从三个**异质框**各构造成可**对齐**的共同可观测量, 喂 nyblom_stationarity.py 测跨框共振/脱耦。

═══════════════════════════════════════════════════════════════════════
★核心立场 (别偷渡"同一个注意力")★
  三个框测的**不是同一种注意力**, 是三种不同的注意力-算子:
    Trends       = Google-CN-可达人群的**搜索行为**
    PeoplesDaily = 机构编辑**选择写它**的行为 (党媒, 大多数梗根本不写 → 稀疏/零)
    Wayback      = 平台算法判"热"的**排名** (策展+审查, top-50 阈值截断, 仅 2019-2021)
  所以 comparison_a ★不是★ "三框是否对同一注意力达成一致" —— 它们量的是不同东西。
  它测的是: 这三种不同注意力-算子的**动力学 (漂移/自相关/变点) 是否共动 or 脱耦**。
  共动 → **动力学共动 (E1 描述, ★必要非充分★: 不升'共享驱动'(E4) —— 可能共同季节性 / 月聚合把两边同样
    平滑 / 51 梗巧合)**; 脱耦 → 不同框捕到不同过程 (如: 梗搜索热、机构不写/被scrub) —— 两者都是硬结果, 非噪声。

★对齐原则: 对齐**动力学**, 不对齐**绝对水平**★
  三框单位不同 (搜索指数 / 计数 / 排名分), 尺度不可比。→ 每框**框内标准化** = ★z-score per M per frame over time★
  (审计: 不是 rank-across-M-per-t, 那答别的问题),
  跨框比较只谈**形状/动力学** (Nyblom 测的就是参数恒定性, 尺度无关)。★不把三条拉到同一绝对轴 (那是假等价)★。

★共同 meme 索引 M★: 用现有 51 关键词集 (Trends 原生有), 三框对**同一 M 列表**抽 salience → 可配对比较。
  ★偷渡 (审计 Q4a): 51 词当年为 Trends curate → 用作三框共同集 = privilege Trends 词汇, 也正是 News 那么多
    零的原因 (51 个 Trends-梗不是党媒写的话题), 且排除 Wayback 自然冒出的别的热词。= curator 预设偷进跨框
    设计。标注; 以后可 per-frame 扩 M (但破坏配对)。★

★共同时间分辨率★: **月级** 作骨干 (Trends 月级处处可靠; News 计数月聚合; Wayback 2019-2021 稀疏, 月级才有量)。
  日级只在 Trends 的 daily_reliable 段作 bonus, 不作跨框骨干 (低音量日级 unusable, 见 backfill flag4)。
  ★偷渡 (审计 Q4b): 月级骨干**没逃出分辨率墙** —— 历史重建本就是要逃月级 (Nyblom 月级功率有限)。骨干
    Trends月 vs News月计数 ≈ 127 月点/框 = 仍月级 = 可能仍欠功率。'最好功率'是**相对** (选项里最好), 非
    '够功率'; 日级只救 Trends 峰期。★别指望骨干因序列长就有功率。★

═══════════════════════════════════════════════════════════════════════
逐框提取逻辑
─────────────────────────────────────────────────────────────────────
(1) Trends [GoogleTrends-CN]  s_trends(M, t)
    = 搜索兴趣指数 (月级直接取, 全程可靠; 日级仅 daily_reliable 段)。
    直接现成 (external_field 月级 + backfill 日级 reliable 段)。★只注意力, 非 x(t)★。
    支持: 2015-2025 全。

(2) PeoplesDaily [机构叙事]  s_news(M, t)
    = 时间 bin 内**提及计数** = 含 M 词面 (或语义匹配) 的标题/正文条数。
    ★稀疏性★: 党媒大多不写梗 → 多数 M 多数月 = 0。零 ≠ "无注意力", 是"机构未采纳" —— 本身是信号
      (搜索热+机构零 = 脱耦候选)。但**零方差段 Nyblom 无意义** → 只在有非零支撑的 M/段上跑。
    ★偷渡 (审计 Q2a, re-lexicalization): 固定词面的"零"混淆 "机构没写这现象" vs "机构换词写了" (党媒不写
      '躺平' 但可能写 '消极人生观'/委婉语)。→ "搜索热+机构零=脱耦" 可能其实是"机构改名写了=耦合"被误读。
      News-零 = "此词面缺席" = 非采纳 OR 改词, ★分不开★; 脱耦判定必须挂此 caveat, 不坐实脱耦。★
    ★偷渡 (审计 Q2b, selection): 只在非零支撑梗上跑 Nyblom = 选了"机构写过的"子集 = 偏向耦合正样本 →
      标注 selection bias。★
    ★模型误设 (审计 Q4B, "真正的一刀"): 提及计数是稀疏非负**计数**, ★不是高斯 AR★ —— Nyblom 假设高斯 AR
      残差。直接把计数喂 Nyblom = 误设。→ 用计数适配 (Poisson-AR / INGARCH) 或先方差稳定变换 (√/Anscombe),
      或显式标残差错配。别把计数当连续高斯序列喂 Nyblom。★
    ★功率 (审计 Q2/new): 党媒对多数梗常年 near-zero → "Trends vs News 骨干" 非**均匀**最好功率, 只对"机构真
      写过的小撮梗"(严肃/机构相关) 有功率; 纯娱乐梗 News 全零 → 跑不了 Nyblom。对多数梗, News 是稀疏"采纳
      指示" → 用"与 Trends spike 的共现/领先滞后"测脱耦, 非全序列 Nyblom。"骨干最好功率" 降为 "仅机构覆盖子集"。★
    ★词面漂移★: (同 Q2a) M 的词面跨时代可能变 (观测算子时不变, 接 obs-operator-stationarity) → 固定词表
      会漏; 语义匹配 (embedding) 又引入另一算子。设计选固定词面 + 标此偏差。
    支持: 2015-2025 (待 tier② crawler 验版式)。

(3) Wayback [平台排名]  s_wayback(M, t)
    = 排名分: 某存档日 M 出现在 top-50 则记 score(rank), 否则 0。月聚合。★rank→score 变换 (1/rank vs
      51-rank vs Zipf) 是建模选择、影响动力学 (审计 Q4C): 选一个 + 敏感性扫, 或标注★。
    ★左截断★: top-50 阈值 = 左截断观测; 阈下 = **censored**, 不是 true-zero → 标 censored, Nyblom 前考虑
      截断稳健处理 (或只在 M 稳定在榜的段比较)。
    ★archive-intensity 不能只"除存档天数"★ (审计 Q3): 除小数 (2019 才 59 天) 放大方差 + 假设热 ∝ 存档天数
      线性 (未验) = 一阶粗糙 IPW。改用外部 AI 已收敛的 Problem-B 栈: ① **trimming** (只在存档密度足够段推断;
      2020 单独/降权) ② archive-intensity 作 **协变量**, 测"控制它后动力学还在不在"(robustness) ③ **敏感性界**。
    支持: 仅 2019-2021 (2015-16 空、2022+ 登录墙, 见 recon)。

═══════════════════════════════════════════════════════════════════════
对齐 → 比较 (comparison_a)
  1. 各框抽 s_frame(M, t), 月级, 51 个 M。
  2. 框内标准化 = z-score per M per frame **over time** (审计: 删 "rank across M per t" —— 那是每时刻梗间排序, 答别的对象)。
  3. ★共振操作化 = per-meme 配对 (审计 Q1b)★: 对每个 M, 比它在 A 框的动力学签名 vs 它在 B 框的
     (如滚动自相关轨迹相关 / 变点时间对齐 + null), ★不用"两框 aggregate 都显著/都不显著"粗二元★
     —— 两框各因无关原因显著 = 假共振。
  4. 两层比较 (功率分层, 见 flag3):
     ★骨干 (相对最好功率, ★非"够功率"★, 见 Q4b)★: Trends vs PeoplesDaily, 2015-2025 全月级。
     ★薄补充 (UNDERPOWERED)★: 加 Wayback 三方, 仅 2019-2021 (~36月, 稀疏+2020主导) → 大概率无功率;
       "没测到共振"别读成脱耦 (flag3)。
  5. 输出: 每 M 每对框的**动力学共动/脱耦**判定 (E1 描述, ★不升'共享驱动'E4, Q1a★) + 诚实功率标注 + 上述 caveat。

★承重诚实 (写进任何结论)★:
  - 三框是不同注意力-算子, 结论是"动力学共动/脱耦", 不是"注意力一致"。
  - News 稀疏 (机构不写梗) + Wayback 截断/2019-2021 + Trends 早期低音量不可靠 → 每框各有缺口, 别互相冒充。
  - 历史段仍无"国内大众个体"框 (operator_ledger honesty_ceiling) —— 这三个都不是大众个体。
  - ★审计折进的偷渡/caveat★ —— 第一批 (7): (Q1a) co-move ≠ 共享驱动 (必要非充分, E1 不升 E4);
    (Q1b) 共振 = per-meme 配对签名对齐, 非"两框 aggregate 都显著"; (Q2a) News 零 = 非采纳 OR 改词
    (re-lexicalization) 分不开; (Q2b) 非零子集 = coupling-biased selection; (Q3) Wayback 用 Problem-B 栈
    (trimming+协变量 robustness+敏感性界), 非只除存档天数; (Q4a) 51 词 M 是 Trends-centric = curator 预设;
    (Q4b) 月级骨干没逃分辨率墙, "最好功率"是相对非够。
    第二批 (复核 refine, +4): (标准化) 只 z-score per-M-per-frame **over time**, 删 rank-across-M-per-t;
    (News 功率) 骨干仅对**机构覆盖子集**有功率, 娱乐梗 News 全零→走共现/领先滞后非全序列 Nyblom;
    ★(Q4B 一刀) News 计数非高斯 AR → 喂 Nyblom 前 Poisson-AR/INGARCH 或方差稳定变换, 否则模型误设★;
    (Q4C) Wayback rank→score 变换是建模选择→选一个+敏感性或标注。
═══════════════════════════════════════════════════════════════════════
"""

# ── 函数骨架 (DESIGN, 未实现; 待 auditor 复核设计 + 数据到位) ──
import numpy as np


def salience_trends(meme, monthly_field, daily_reliable=None):
    """s_trends(M,t): 月级搜索指数 (处处可靠) + 日级 reliable 段 (bonus)。现成数据, 直接取。"""
    raise NotImplementedError("DESIGN STUB — 待复核")


def salience_news_mentions(meme, news_corpus, freq="MS"):
    """s_news(M,t): 时间 bin 内含 M 词面的条数。稀疏; 零=机构未采纳(信号非缺失)。待 tier② crawler。"""
    raise NotImplementedError("DESIGN STUB — 待复核")


def salience_wayback_rank(meme, hotsearch_by_day, archive_days_per_month):
    """s_wayback(M,t): top-50 排名分月聚合, ★除以该月存档天数 (archive-intensity 归一)★; 阈下=censored。"""
    raise NotImplementedError("DESIGN STUB — 待复核")


def within_frame_standardize(series):
    """框内 z-score (尺度不可比 → 只留动力学/形状)。"""
    raise NotImplementedError("DESIGN STUB — 待复核")


def cross_frame_salience_nyblom(memes, frames):
    """comparison_a: 各框 salience 各跑 Nyblom, 比参数漂移共振/脱耦。
    骨干=Trends vs News (2015-2025); 薄补充=+Wayback (2019-2021, UNDERPOWERED, 别读 null)。"""
    raise NotImplementedError("DESIGN STUB — 待复核")


if __name__ == "__main__":
    print(__doc__)
    print("DESIGN STUB — 未跑数据。待 auditor 复核设计后 + tier①全量/tier②crawler 数据到位再实现。")
