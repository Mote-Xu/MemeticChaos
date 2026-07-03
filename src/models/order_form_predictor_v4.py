"""
FR19 v4.0 — 三层串联预测器

Model A: 外部注意场 → 预测叙事生态 (Stage Occupancy)
Model B: 外部场 + 预测的叙事生态 → 预测注意力结构 (HHI, Entropy, Cat_Dist)

输出: 秩序形态转移概率，不是混沌轴值

用法:
    python src/models/order_form_predictor_v4.py              # 训练+评估
    python src/models/order_form_predictor_v4.py --forecast 6 # 预测
"""

import json, sys, os, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")
np.random.seed(42)

from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data/collector"
PROCESSED_DIR = ROOT / "data/processed"
LOOKBACK = 6

STAGE_ORDER = ["origin", "emergence", "peak", "controversy", "fixation"]
CATEGORY_NAMES = ["解构自嘲", "攻击发泄", "虚无退却", "身份认同", "纯粹娱乐"]

# ═══════════ Data Loading ═══════════

def load_data():
    """加载所有数据并构建月度特征矩阵."""
    # External field
    with open(DATA_DIR / "external_field_2015_2025.json", "r", encoding="utf-8") as f:
        ext = json.load(f)["data"]
    ext_kw = sorted(ext.keys())

    # Stage Occupancy
    stage_path = PROCESSED_DIR / "stage_occupancy.json"
    if not stage_path.exists():
        raise FileNotFoundError("Run stage_occupancy.py first")
    with open(stage_path, "r", encoding="utf-8") as f:
        stage_data = json.load(f)
    stage_months = stage_data["months"]
    stage_matrix = np.array(stage_data["matrix"])

    # Meme trends (for category distribution and HHI/entropy)
    with open(DATA_DIR / "google_trends_2015_2025.json", "r", encoding="utf-8") as f:
        trends = json.load(f)["memes"]

    # Category mapping (inline)
    TREND_TO_MEME = {
        "打工人": "解构自嘲", "内卷": "身份认同", "躺平": "虚无退却",
        "普信男": "攻击发泄", "普信": "攻击发泄",
        "小镇做题家": "身份认同", "摆烂": "虚无退却", "润": "虚无退却",
        "吗喽": "解构自嘲", "鼠鼠": "解构自嘲", "牛马": "解构自嘲",
        "i人 e人": "身份认同", "遥遥领先": "纯粹娱乐", "遥遥领先 华为": "纯粹娱乐",
        "孔乙己的长衫": "身份认同", "孔乙己 长衫": "身份认同",
        "精神状态": "解构自嘲", "雪糕刺客": "攻击发泄",
        "科目三": "纯粹娱乐", "鸡你太美": "纯粹娱乐",
        "后浪": "身份认同", "情绪价值": "身份认同",
        "原生家庭": "身份认同", "专家建议": "攻击发泄",
        "建议专家不要建议": "攻击发泄", "不结婚": "虚无退却",
        "不婚不育": "虚无退却", "显眼": "纯粹娱乐", "显眼包": "纯粹娱乐",
        "凡尔赛": "纯粹娱乐", "元宇宙": "纯粹娱乐",
        "citywalk": "纯粹娱乐", "芭比Q": "纯粹娱乐", "栓Q": "纯粹娱乐",
        "美拉德": "纯粹娱乐", "南方小土豆": "纯粹娱乐",
        "破防": "解构自嘲", "社恐": "解构自嘲", "社死": "解构自嘲",
        "精神内耗": "虚无退却", "尊嘟假嘟": "纯粹娱乐",
        "发疯文学 梗": "解构自嘲", "多巴胺穿搭": "纯粹娱乐",
    }

    # Common months
    all_months = set()
    for d in ext.values():
        all_months.update(d.keys())
    all_months &= set(stage_months)
    months = sorted(m for m in all_months if "2015" <= m[:4] <= "2025")

    n_months = len(months)
    n_ext = len(ext_kw)

    # Build external matrix
    X_ext = np.zeros((n_months, n_ext))
    for j, kw in enumerate(ext_kw):
        for i, m in enumerate(months):
            X_ext[i, j] = ext[kw].get(m, 0.0)

    # Fit PCA on external
    pca = PCA(n_components=8)
    X_ext_pca = pca.fit_transform(X_ext)
    print(f"外部场 PCA(8): 解释 {pca.explained_variance_ratio_.sum():.1%} 方差")

    # Build Stage Occupancy matrix aligned to months
    stage_idx_map = {m: i for i, m in enumerate(stage_months)}
    X_stage = np.zeros((n_months, 5))
    for i, m in enumerate(months):
        if m in stage_idx_map:
            X_stage[i] = stage_matrix[stage_idx_map[m]]

    # Build attention structure targets (HHI, entropy, cat_dist)
    y_struct = np.zeros((n_months, 7))  # HHI + entropy + 5 cat dist

    # Pre-compute meme category mappings
    meme_cats = {}
    for tk in trends:
        if tk in TREND_TO_MEME:
            meme_cats[tk] = TREND_TO_MEME[tk][1]

    for i, month in enumerate(months):
        weights = {}
        for tk in trends:
            w = trends[tk].get(month, 0.0)
            if w > 0:
                weights[tk] = w

        if not weights:
            y_struct[i] = [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
            continue

        total = sum(weights.values())
        # HHI
        shares = [w / total for w in weights.values()]
        hhi = sum(s**2 for s in shares)

        # Category distribution
        cat_sum = np.zeros(5)
        for tk, w in weights.items():
            cat = meme_cats.get(tk)
            if cat and cat in CATEGORY_NAMES:
                cat_sum[CATEGORY_NAMES.index(cat)] += w
        if cat_sum.sum() > 0:
            cat_dist = cat_sum / cat_sum.sum()
        else:
            cat_dist = np.ones(5) / 5

        # Entropy
        p = cat_dist[cat_dist > 0]
        entropy = -np.sum(p * np.log(p)) if len(p) > 0 else 0.0

        y_struct[i] = np.concatenate([[hhi], [entropy], cat_dist])

    return {
        "months": months,
        "X_ext_pca": X_ext_pca,
        "X_stage": X_stage,
        "y_struct": y_struct,
        "pca": pca,
        "ext_kw": ext_kw,
    }


# ═══════════ Models ═══════════

def build_features(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建时序特征.

    Returns:
        X_A: 外部场特征 → 预测 Stage Occupancy
        X_B: 外部场 + Stage Occupancy → 预测注意力结构
        y_struct: 下月注意力结构
    """
    n = len(data["months"])
    ext = data["X_ext_pca"]
    stage = data["X_stage"]
    y = data["y_struct"]

    X_A_list, X_B_list, y_list = [], [], []

    for i in range(LOOKBACK, n - 1):
        # Model A features: past LOOKBACK months external field
        feats_a = ext[i - LOOKBACK:i].flatten()  # 8 * 6 = 48

        # Model A target: current month stage occupancy
        y_a = stage[i]  # 5

        # Model B features: same external + ACTUAL current stage
        feats_b = np.concatenate([feats_a, stage[i]])  # 48 + 5 = 53

        # Target: next month structure
        y_target = y[i + 1]  # 7

        X_A_list.append(feats_a)
        X_B_list.append(feats_b)
        y_list.append(y_target)

    return np.array(X_A_list), np.array(X_B_list), np.array(y_list), stage[LOOKBACK:n-1]


def train_and_eval(data: dict):
    """训练两层串联模型."""
    X_A, X_B, y_struct, y_stage = build_features(data)

    n = len(X_A)
    print(f"\n样本: {n}, 特征: A={X_A.shape[1]}, B={X_B.shape[1]}")

    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=5)
    alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    scores_a = defaultdict(list)
    scores_b = defaultdict(list)

    for train_idx, test_idx in tscv.split(X_A):
        X_A_tr, X_A_te = X_A[train_idx], X_A[test_idx]
        X_B_tr, X_B_te = X_B[train_idx], X_B[test_idx]
        y_tr, y_te = y_struct[train_idx], y_struct[test_idx]

        sc_a = StandardScaler()
        sc_b = StandardScaler()
        sc_y = StandardScaler()

        # ── Model A: External → Stage Occupancy ──
        X_A_tr_s = sc_a.fit_transform(X_A_tr)
        X_A_te_s = sc_a.transform(X_A_te)

        stage_r2 = []
        for dim in range(5):
            m = RidgeCV(alphas=alphas, cv=3)
            m.fit(X_A_tr_s, y_stage[train_idx, dim])
            pred = m.predict(X_A_te_s)
            stage_r2.append(r2_score(y_stage[test_idx, dim], pred))

        # ── Model B: External + Stage → Structure ──
        # Use actual stage (oracle) for evaluation
        X_B_tr_s = sc_b.fit_transform(X_B_tr)
        X_B_te_s = sc_b.transform(X_B_te)

        y_tr_s = sc_y.fit_transform(y_tr)
        y_te_s = sc_y.transform(y_te)

        m_b = RidgeCV(alphas=alphas, cv=3)
        m_b.fit(X_B_tr_s, y_tr_s)
        pred_b = m_b.predict(X_B_te_s)
        pred_b_inv = sc_y.inverse_transform(pred_b)

        # R² per target dimension
        for dim, label in enumerate(["HHI", "Entropy", *CATEGORY_NAMES]):
            r2 = r2_score(y_te[:, dim], pred_b_inv[:, dim])
            scores_b[label].append(r2)

        for dim, stage in enumerate(STAGE_ORDER):
            scores_a[stage].append(stage_r2[dim])

    # Results
    print("\n═══ Model A: 外部场 → 叙事生态 ═══")
    for stage in STAGE_ORDER:
        vals = scores_a[stage]
        print(f"  {stage:<15s}: R²={np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print("\n═══ Model B: 外部场 + 叙事生态 → 注意力结构 ═══")
    for label, vals in sorted(scores_b.items()):
        print(f"  {label:<15s}: R²={np.mean(vals):.4f} ± {np.std(vals):.4f}")

    hhi_r2 = np.mean(scores_b["HHI"])
    ent_r2 = np.mean(scores_b["Entropy"])
    cat_r2 = np.mean([np.mean(scores_b[c]) for c in CATEGORY_NAMES])
    struct_r2 = (hhi_r2 + ent_r2) / 2

    print(f"\n══════ 汇总 ══════")
    print(f"  叙事生态预测 (Model A, avg 5D): R²={np.mean([np.mean(scores_a[s]) for s in STAGE_ORDER]):.4f}")
    print(f"  HHI+Entropy (Model B): R²={struct_r2:.4f}")
    print(f"  类别分布 (Model B, avg 5D): R²={cat_r2:.4f}")

    return scores_a, scores_b


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast", type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print("FR19 v4.0 — 三层串联预测器")
    print("=" * 60)

    data = load_data()
    train_and_eval(data)

    if args.forecast:
        print(f"\n预测功能待完善 — 基建验证完成")


if __name__ == "__main__":
    main()
