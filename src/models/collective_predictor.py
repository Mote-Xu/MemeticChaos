"""
集体情感系统预测模型 — 两层架构

Layer 1 (外部场 → 趋势): 从 51 维外部注意力场预测集体混沌轴 + 类别分布
Layer 2 (叙事残差 → 混沌分量): 学习外部场无法解释的残差波动模式

预测目标: 未来 1-6 个月系统主导秩序形态

用法:
    python src/models/collective_predictor.py           # 训练+评估
    python src/models/collective_predictor.py --forecast 6  # 预测未来6个月
"""

import json, sys, os
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)
DATA_DIR = Path("data/collector")

# 五相区混沌轴位置
CATEGORY_CHAOS = {
    "解构自嘲": -0.33, "攻击发泄": -0.62, "虚无退却": -0.59,
    "身份认同": +0.34, "纯粹娱乐": +0.19,
}

# Google Trends keyword → (name, category)
TREND_TO_MEME = {
    "打工人": ("打工人", "解构自嘲"), "内卷": ("内卷 / 卷", "身份认同"),
    "躺平": ("躺平", "虚无退却"), "普信男": ("普信男", "攻击发泄"),
    "小镇做题家": ("小镇做题家", "身份认同"), "摆烂": ("摆烂", "虚无退却"),
    "润": ("润", "虚无退却"), "吗喽": ("吗喽", "解构自嘲"),
    "鼠鼠": ("鼠鼠", "解构自嘲"), "牛马": ("牛马", "解构自嘲"),
    "i人 e人": ("i人/e人", "身份认同"), "遥遥领先 华为": ("遥遥领先", "纯粹娱乐"),
    "孔乙己的长衫": ("孔乙己的长衫", "身份认同"),
    "精神状态": ("精神状态良好", "解构自嘲"),
    "雪糕刺客": ("XX刺客", "攻击发泄"), "科目三": ("科目三", "纯粹娱乐"),
    "鸡你太美": ("鸡你太美", "纯粹娱乐"), "后浪": ("后浪", "身份认同"),
    "情绪价值": ("情绪价值", "身份认同"), "原生家庭": ("原生家庭", "身份认同"),
    "专家建议": ("建议专家不要建议", "攻击发泄"),
    "不结婚": ("四不/不婚不育", "虚无退却"), "显眼": ("显眼包", "纯粹娱乐"),
    "遥遥领先": ("遥遥领先", "纯粹娱乐"), "普信": ("普信男", "攻击发泄"),
    "社恐": ("社恐", "解构自嘲"), "社死": ("社死", "解构自嘲"),
    "凡尔赛": ("凡尔赛", "纯粹娱乐"), "破防": ("破防", "解构自嘲"),
    "芭比Q": ("芭比Q", "纯粹娱乐"), "栓Q": ("栓Q", "纯粹娱乐"),
    "元宇宙": ("元宇宙", "纯粹娱乐"), "citywalk": ("citywalk", "纯粹娱乐"),
    "美拉德": ("美拉德", "纯粹娱乐"), "南方小土豆": ("南方小土豆", "纯粹娱乐"),
    "精神内耗": ("精神内耗", "虚无退却"),
}


def load_all_data():
    """加载外部场 + 内部梗数据，构建对齐的月度特征矩阵。"""
    with open(DATA_DIR / "external_field_2015_2025.json", "r", encoding="utf-8") as f:
        ext = json.load(f)["data"]
    with open(DATA_DIR / "google_trends_2015_2025.json", "r", encoding="utf-8") as f:
        memes = json.load(f)["memes"]

    # Find common month range
    all_months = set()
    for d in ext.values():
        all_months.update(d.keys())
    for d in memes.values():
        all_months.update(d.keys())
    months = sorted(all_months)
    # Filter to 2015-01 to 2025-12
    months = [m for m in months if "2015" <= m[:4] <= "2025"]

    # Build feature matrix: each row = 1 month
    ext_keywords = sorted(ext.keys())
    meme_keywords = sorted(memes.keys())

    X_ext = np.zeros((len(months), len(ext_keywords)))
    for j, kw in enumerate(ext_keywords):
        for i, m in enumerate(months):
            X_ext[i, j] = ext[kw].get(m, 0.0)

    # Compute collective chaos axis from meme attention
    chaos_axis = np.zeros(len(months))
    cat_weights = np.zeros((len(months), 5))
    cat_names = list(CATEGORY_CHAOS.keys())

    for i, month in enumerate(months):
        w_sum = 0.0
        for trend_kw, (meme_name, cat) in TREND_TO_MEME.items():
            if trend_kw in memes and month in memes[trend_kw]:
                w = memes[trend_kw][month]
                chaos_axis[i] += CATEGORY_CHAOS.get(cat, 0.0) * w
                w_sum += w
                if cat in cat_names:
                    cat_weights[i, cat_names.index(cat)] += w
        if w_sum > 0:
            chaos_axis[i] /= w_sum
            cat_weights[i] /= w_sum

    total_attention = np.sum([memes[kw].get(m, 0) for kw in meme_keywords], axis=0) if meme_keywords else np.zeros(len(months))
    # Actually compute total properly
    total_att = np.zeros(len(months))
    for i, m in enumerate(months):
        total_att[i] = sum(memes[kw].get(m, 0) for kw in meme_keywords)

    return {
        "months": months,
        "X_ext": X_ext,
        "ext_keywords": ext_keywords,
        "chaos_axis": chaos_axis,
        "cat_weights": cat_weights,
        "cat_names": cat_names,
        "total_attention": total_att,
    }


def build_training_data(data: dict, lookback: int = 6, horizon: int = 1):
    """构建监督学习数据集。

    X: 过去 lookback 个月的外部场 + chaos_axis
    y: 未来 horizon 个月的 chaos_axis
    """
    X, y = [], []
    n = len(data["months"])

    # Features: external field (51 dims) + chaos (1 dim) × lookback months
    for i in range(lookback, n - horizon):
        # Past window
        feat = []
        for t in range(i - lookback, i):
            feat.extend(data["X_ext"][t].tolist())
            feat.append(data["chaos_axis"][t])
        X.append(feat)
        # Future target
        y.append(data["chaos_axis"][i + horizon - 1])

    return np.array(X), np.array(y)


def evaluate_model(data: dict):
    """训练 + 回测评估。"""
    print("=" * 60)
    print("集体情感预测模型 — 训练与评估")
    print("=" * 60)

    X, y = build_training_data(data, lookback=6, horizon=1)
    print(f"\n训练数据: {X.shape[0]} 样本, {X.shape[1]} 特征")

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    models = {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    }

    results = {}
    for name, model in models.items():
        scores_mae, scores_r2 = [], []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            scores_mae.append(mean_absolute_error(y_test, y_pred))
            scores_r2.append(r2_score(y_test, y_pred))

        results[name] = {
            "MAE": np.mean(scores_mae), "MAE_std": np.std(scores_mae),
            "R²": np.mean(scores_r2), "R²_std": np.std(scores_r2),
        }
        print(f"  {name:<15s}: MAE={np.mean(scores_mae):.4f}±{np.std(scores_mae):.4f}  "
              f"R²={np.mean(scores_r2):.3f}±{np.std(scores_r2):.3f}")

    # Baseline: always predict last value
    baseline_mae = np.mean([abs(y[i] - y[i-1]) for i in range(1, len(y))])
    print(f"  {'Naive (lag-1)':<15s}: MAE={baseline_mae:.4f}")

    # Chaos residual (internal layer)
    best_model_name = max(results, key=lambda k: results[k]["R²"])
    print(f"\n最佳模型: {best_model_name}")

    # Full fit for residual analysis
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    best_model = models[best_model_name]
    best_model.fit(X_s, y)
    y_pred_all = best_model.predict(X_s)

    residuals = y - y_pred_all
    residual_std = np.std(residuals)
    print(f"\n=== 内部混沌层 (残差分析) ===")
    print(f"  外部场可解释方差: {best_model.score(X_s, y):.3f}")
    print(f"  残差标准差 (混沌分量): {residual_std:.4f}")
    print(f"  混沌/信号比: {residual_std / np.std(y):.2f}")

    return {
        "model": best_model,
        "scaler": scaler,
        "residual_std": residual_std,
        "baseline_mae": baseline_mae,
        "results": results,
    }


def forecast_future(data: dict, model_info: dict, horizon: int = 6):
    """预测未来 N 个月的集体情感状态。"""
    print(f"\n{'='*60}")
    print(f"预测未来 {horizon} 个月")
    print(f"{'='*60}")

    model = model_info["model"]
    scaler = model_info["scaler"]
    residual_std = model_info["residual_std"]

    # Start from last known state
    n = len(data["months"])
    last_features = []
    for t in range(n - 6, n):
        last_features.extend(data["X_ext"][t].tolist())
        last_features.append(data["chaos_axis"][t])

    current = np.array(last_features).reshape(1, -1)
    current_s = scaler.transform(current)

    predictions = []
    for h in range(horizon):
        pred = model.predict(current_s)[0]
        # Add chaos uncertainty
        pred_with_noise = pred + np.random.normal(0, residual_std * 0.3)
        predictions.append({
            "month": f"2026-{h+1:02d}",
            "chaos_pred": float(pred),
            "chaos_range": [float(pred - residual_std), float(pred + residual_std)],
        })
        # Determine order form
        if pred > 0.15:
            order_form = "身份认同/纯粹娱乐 (秩序建构)"
        elif pred < -0.20:
            order_form = "虚无退却/攻击宣泄 (混沌释放)"
        else:
            order_form = "解构自嘲/中性过渡 (边界振荡)"
        predictions[-1]["order_form"] = order_form

        # Slide window
        new_feat = list(current_s.flatten())
        new_feat = new_feat[-(len(last_features) - 52):]  # drop oldest month
        new_feat.extend(data["X_ext"][-1].tolist())  # reuse last external (simplification)
        new_feat.append(pred)
        current_s = scaler.transform(np.array(new_feat).reshape(1, -1))

    print(f"\n  当前混沌轴: {data['chaos_axis'][-1]:+.3f}")
    for p in predictions:
        print(f"\n  {p['month']}:")
        print(f"    预测混沌轴: {p['chaos_pred']:+.3f} (区间 [{p['chaos_range'][0]:+.3f}, {p['chaos_range'][1]:+.3f}])")
        print(f"    秩序形态: {p['order_form']}")

    return predictions


if __name__ == "__main__":
    import argparse
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast", type=int, default=0, help="预测未来N个月")
    args = parser.parse_args()

    data = load_all_data()
    print(f"[数据] {len(data['months'])} 月, {len(data['ext_keywords'])} 外部特征")

    model_info = evaluate_model(data)

    if args.forecast > 0:
        forecast_future(data, model_info, args.forecast)
