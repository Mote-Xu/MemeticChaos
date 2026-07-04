"""
Representation Learning + H1 Verification — FR19 v4.0 Level 2

从 Level 1 硬事实 + 外部场 + 注意力结构组装特征矩阵,
用 PCA/UMAP 学习 Narrative State 的低维表示 x(t),
然后用 VARX/Ridge 检验 H1 假说链。

H1a: 降维后信息损失可控 (PCA reconstruction)
H1b: x(t) 对 x(t+1) 有预测力 (Ridge vs lag-1)
H1c: 未来演化主要由当前状态决定 (特征 ablation)

AlphaGo 原则: 不预设维度数/算法——由数据决定。
第一阶段以线性模型为基线; 若不能解释, 再上非线性。

用法:
    python src/models/representation_learning.py
    python src/models/representation_learning.py --ext-pc 8 --lookback 3
"""

import json, sys, os, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).parent.parent.parent

LEVEL1_PATH = ROOT / "data/processed/level1_hard_facts.json"
EXTERNAL_PATH = ROOT / "data/collector/external_field_2015_2025.json"
TRENDS_PATH = ROOT / "data/collector/google_trends_2015_2025.json"
OUTPUT_PATH = ROOT / "data/processed/representation_state.json"

# UMAP is optional — only for visualization
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ═══════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════

def load_level1() -> dict:
    with open(LEVEL1_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_external_field() -> tuple[list[str], np.ndarray]:
    """加载外部场 51 关键词 → (months, matrix)."""
    with open(EXTERNAL_PATH, "r", encoding="utf-8") as f:
        ef = json.load(f)

    data = ef["data"]
    # Collect all months
    all_months = set()
    for kw_series in data.values():
        all_months.update(kw_series.keys())
    months = sorted(m for m in all_months if "2015" <= m[:4] <= "2025")

    keywords = sorted(data.keys())
    matrix = np.zeros((len(months), len(keywords)))
    for j, kw in enumerate(keywords):
        for i, m in enumerate(months):
            matrix[i, j] = data[kw].get(m, 0.0)

    return months, matrix


def load_trends() -> dict[str, dict[str, float]]:
    with open(TRENDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["memes"]


# ═══════════════════════════════════════════════
# Feature assembly
# ═══════════════════════════════════════════════

def compute_attention_structure(trends: dict, months: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """计算每月的 HHI 和类别熵 (从 Trends 份额分布)."""
    n = len(months)
    hhi = np.zeros(n)
    entropy = np.zeros(n)
    total_attention = np.zeros(n)

    for i, m in enumerate(months):
        weights = []
        for kw, series in trends.items():
            w = series.get(m, 0.0)
            if w > 0:
                weights.append(w)

        total = sum(weights)
        total_attention[i] = total

        if total > 0 and len(weights) > 1:
            shares = [w / total for w in weights]
            hhi[i] = sum(s**2 for s in shares)
            p = np.array(shares)
            entropy[i] = float(-np.sum(p * np.log(p)))
        else:
            hhi[i] = 1.0 if len(weights) == 1 else 0.0
            entropy[i] = 0.0

    return hhi, entropy, total_attention


def assemble_features(
    l1: dict,
    ext_months: list[str],
    ext_matrix: np.ndarray,
    hhi: np.ndarray,
    entropy: np.ndarray,
    total_att: np.ndarray,
    ext_pc: int = 8,
) -> tuple[np.ndarray, list[str], dict]:
    """组装特征矩阵 X (127 月 × N 维).

    特征组成:
      Level 1 硬事实 (8): stage(5) + mutation_rate(1) + inst_rate(1) + drift(1)
      外部场 PCA (ext_pc): 从 51 关键词降维
      注意力 (2): HHI + entropy

    返回 (X, months, metadata).
    """
    months_l1 = l1["months"]
    n = len(months_l1)

    # ── Level 1 features ──
    stage = np.array(l1["stage_occupancy"])          # n × 5
    mut_rate = np.array(l1["mutation_rate"])          # n
    inst_rate = np.array(l1["institutionalization_rate"])  # n
    drift = np.array(l1["mean_semantic_drift"])       # n

    l1_features = np.column_stack([
        stage,
        mut_rate.reshape(-1, 1),
        inst_rate.reshape(-1, 1),
        drift.reshape(-1, 1),
    ])
    l1_dim = l1_features.shape[1]  # 8

    # ── External field PCA ──
    # Align months
    month_to_ext_idx = {m: i for i, m in enumerate(ext_months)}
    ext_aligned = np.zeros((n, ext_matrix.shape[1]))
    for i, m in enumerate(months_l1):
        if m in month_to_ext_idx:
            ext_aligned[i] = ext_matrix[month_to_ext_idx[m]]

    pca_ext = PCA(n_components=min(ext_pc, ext_matrix.shape[1]))
    ext_pcs = pca_ext.fit_transform(ext_aligned)
    ext_var = pca_ext.explained_variance_ratio_.sum()

    # ── Attention features ──
    # Align attention (computed from Trends, may have different months)
    att_months = months_l1  # use L1 months as canonical

    # HHI and entropy are already aligned to the Trends months
    # Recompute to align with L1 months
    # For now, assume they're aligned — we'll pass them in
    att_features = np.column_stack([
        hhi.reshape(-1, 1),
        entropy.reshape(-1, 1),
    ])

    # ── Assemble ──
    X = np.column_stack([l1_features, ext_pcs, att_features])
    feature_names = (
        [f"stage_{s}" for s in l1["stages"]] +
        ["mutation_rate", "inst_rate", "semantic_drift"] +
        [f"ext_pc_{i+1}" for i in range(ext_pcs.shape[1])] +
        ["hhi", "entropy"]
    )

    meta = {
        "n_months": n,
        "n_features": X.shape[1],
        "l1_dim": l1_dim,
        "ext_pc_dim": ext_pcs.shape[1],
        "ext_pca_variance": float(ext_var),
        "att_dim": 2,
        "feature_names": feature_names,
    }

    return X, months_l1, meta


# ═══════════════════════════════════════════════
# H1a: Intrinsic dimension
# ═══════════════════════════════════════════════

def verify_h1a(X: np.ndarray, feature_names: list[str]) -> dict:
    """PCA 降维分析: 找 intrinsic dimension.

    报告达到 90%/95%/99% 方差所需的维度数,
    以及各主成分的载荷 (用于事后解释).
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_s)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    dim_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    dim_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    dim_99 = int(np.searchsorted(cumvar, 0.99) + 1)

    # Top PC loadings
    top_loadings = []
    for pc_idx in range(min(5, len(feature_names))):
        loadings = pca.components_[pc_idx]
        top_idx = np.argsort(np.abs(loadings))[-5:][::-1]
        top_loadings.append([
            {"feature": feature_names[i], "loading": float(loadings[i])}
            for i in top_idx
        ])

    return {
        "n_components": len(feature_names),
        "dim_90": dim_90,
        "dim_95": dim_95,
        "dim_99": dim_99,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumvar.tolist(),
        "top_loadings": top_loadings,
        "recommendation": (
            f"Intrinsic dim ≈ {dim_90} (90% var), {dim_95} (95% var), {dim_99} (99% var). "
            f"Use {dim_90}–{dim_95} for VARX."
        ),
    }


# ═══════════════════════════════════════════════
# H1b + H1c: Dynamics verification
# ═══════════════════════════════════════════════

def verify_h1b_h1c(
    X: np.ndarray,
    months: list[str],
    feature_names: list[str],
    state_dim: int = 10,
    lookback: int = 3,
    ext_pc_dim: int = 8,
) -> dict:
    """H1b + H1c: 在降维后的状态表示上验证动力学.

    方法 (修正后):
    1. 先 PCA 降维到 state_dim 维 → x(t) 为低维状态表示
    2. 在低维状态上构建 VARX: x(t+1) = F(x(t), ..., x(t-lookback+1))
    3. 比较 Ridge vs lag-1
    4. Ablation: x(t) only vs x(t) + u(t) + y(t) (把 u 和 y 作为外生变量注入)

    这避免了原始方法中特征维度过高导致的过拟合。
    """
    n = X.shape[0]

    # ── Step 1: Learn state representation ──
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    pca = PCA(n_components=state_dim)
    x_state = pca.fit_transform(X_s)  # n × state_dim
    explained = pca.explained_variance_ratio_.sum()

    # ── Step 2: Separate u (external field PCs) and y (attention) from X ──
    # u: external field PCs (columns 8 to 8+ext_pc_dim-1)
    # y: hhi + entropy (last 2 columns)
    ext_start = 8
    ext_end = ext_start + ext_pc_dim
    u_raw = X[:, ext_start:ext_end]             # n × ext_pc_dim
    y_raw = X[:, -2:]                            # n × 2 (hhi, entropy)

    # ── Step 3: Lag-1 baseline on reduced state ──
    lag1_pred = x_state[:-1]                     # predict x(t+1) = x(t)
    lag1_target = x_state[1:]
    lag1_r2 = r2_score(lag1_target.ravel(), lag1_pred.ravel())
    lag1_mae = mean_absolute_error(lag1_target.ravel(), lag1_pred.ravel())

    # Per-dimension lag-1 for comparison
    lag1_per_dim = []
    for j in range(state_dim):
        lag1_per_dim.append(float(r2_score(lag1_target[:, j], lag1_pred[:, j])))

    # ── Step 4: Build VAR datasets ──
    def make_var_data(state, u, y, lookback):
        """构建: 用 t-lookback..t-1 的状态 + u(t-1) + y(t-1) 预测 x(t)."""
        X_list, y_list = [], []
        for i in range(lookback, len(state)):
            feats = []
            for t in range(i - lookback, i):
                feats.extend(state[t].tolist())
            # exogenous: u and y at most recent timestep
            feats.extend(u[i - 1].tolist())
            feats.extend(y[i - 1].tolist())
            X_list.append(feats)
            y_list.append(state[i].tolist())
        return np.array(X_list), np.array(y_list)

    def make_var_data_state_only(state, lookback):
        """仅用状态历史, 无外生变量."""
        X_list, y_list = [], []
        for i in range(lookback, len(state)):
            feats = []
            for t in range(i - lookback, i):
                feats.extend(state[t].tolist())
            X_list.append(feats)
            y_list.append(state[i].tolist())
        return np.array(X_list), np.array(y_list)

    # Full VARX: state + u + y
    X_var, y_var = make_var_data(x_state, u_raw, y_raw, lookback)

    # State-only: just state history
    X_state_only, y_state_only = make_var_data_state_only(x_state, lookback)

    # Time split
    split = int(len(X_var) * 0.8)
    X_tr, X_te = X_var[:split], X_var[split:]
    y_tr, y_te = y_var[:split], y_var[split:]
    Xs_tr, Xs_te = X_state_only[:split], X_state_only[split:]

    # ── Step 5: Train models ──
    alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]

    # Full VARX
    model_full = RidgeCV(alphas=alphas, cv=min(5, len(X_tr) - 1))
    model_full.fit(X_tr, y_tr)
    y_pred_full = model_full.predict(X_te)
    full_r2 = r2_score(y_te.ravel(), y_pred_full.ravel())
    full_mae = mean_absolute_error(y_te.ravel(), y_pred_full.ravel())
    full_train_r2 = r2_score(y_tr.ravel(), model_full.predict(X_tr).ravel())

    # State-only VAR
    model_state = RidgeCV(alphas=alphas, cv=min(5, len(Xs_tr) - 1))
    model_state.fit(Xs_tr, y_tr)
    y_pred_state = model_state.predict(Xs_te)
    state_r2 = r2_score(y_te.ravel(), y_pred_state.ravel())
    state_mae = mean_absolute_error(y_te.ravel(), y_pred_state.ravel())

    # ── Step 6: Predict Δx instead of x(t+1) ──
    # Δx(t) = x(t) - x(t-1), predict Δx(t+1) from state history
    dx = x_state[1:] - x_state[:-1]  # (n-1) × state_dim
    dx_state = x_state[:-1]           # state at time t (to predict Δ at t+1)

    X_dx_list, y_dx_list = [], []
    for i in range(lookback, len(dx)):
        feats = []
        for t in range(i - lookback, i):
            feats.extend(x_state[t].tolist())
        feats.extend(u_raw[i - 1].tolist())
        feats.extend(y_raw[i - 1].tolist())
        X_dx_list.append(feats)
        y_dx_list.append(dx[i].tolist())
    X_dx, y_dx = np.array(X_dx_list), np.array(y_dx_list)

    split_dx = int(len(X_dx) * 0.8)
    X_dx_tr, X_dx_te = X_dx[:split_dx], X_dx[split_dx:]
    y_dx_tr, y_dx_te = y_dx[:split_dx], y_dx[split_dx:]

    model_dx = RidgeCV(alphas=alphas, cv=min(5, len(X_dx_tr) - 1))
    model_dx.fit(X_dx_tr, y_dx_tr)
    y_dx_pred = model_dx.predict(X_dx_te)
    dx_r2 = r2_score(y_dx_te.ravel(), y_dx_pred.ravel())

    # Δx lag-1 baseline: Δx(t+1) = 0 (no change) → R² relative to zero-prediction
    dx_lag1_r2 = r2_score(y_dx_te.ravel(), np.zeros_like(y_dx_te.ravel()))

    # Per-dimension R² for full model
    per_dim_r2 = []
    for j in range(min(state_dim, y_te.shape[1])):
        r2j = r2_score(y_te[:, j], y_pred_full[:, j])
        per_dim_r2.append({
            "feature": f"PC{j+1}",
            "r2": float(r2j),
            "lag1_r2": lag1_per_dim[j] if j < len(lag1_per_dim) else 0.0,
        })

    # ── Ablation: test different lookback lengths ──
    ablation_results = {"full": float(full_r2), "state_only": float(state_r2)}

    for lb in range(1, lookback + 1):
        X_abl, y_abl = make_var_data(x_state, u_raw, y_raw, lb)
        split_abl = int(len(X_abl) * 0.8)
        Xa_tr, Xa_te = X_abl[:split_abl], X_abl[split_abl:]
        ya_tr, ya_te = y_abl[:split_abl], y_abl[split_abl:]
        # Need enough samples
        if len(Xa_tr) < 5 or len(Xa_te) < 2:
            ablation_results[f"lookback_{lb}"] = None
            continue
        model_ab = RidgeCV(alphas=alphas, cv=min(5, len(Xa_tr) - 1))
        model_ab.fit(Xa_tr, ya_tr)
        ablation_results[f"lookback_{lb}"] = float(
            r2_score(ya_te.ravel(), model_ab.predict(Xa_te).ravel()))

    return {
        "method": "PCA → VARX on reduced state",
        "state_dim": state_dim,
        "pca_variance_explained": float(explained),
        "n_samples": len(X_var),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "lag1_baseline": {
            "r2": float(lag1_r2),
            "mae": float(lag1_mae),
            "per_dim_r2": lag1_per_dim,
        },
        "full_varx": {
            "train_r2": float(full_train_r2),
            "test_r2": float(full_r2),
            "test_mae": float(full_mae),
            "alpha": float(model_full.alpha_),
        },
        "state_only_var": {
            "test_r2": float(state_r2),
            "test_mae": float(state_mae),
            "alpha": float(model_state.alpha_),
        },
        "delta_x_prediction": {
            "test_r2": float(dx_r2),
            "lag1_zero_r2": float(dx_lag1_r2),
            "alpha": float(model_dx.alpha_),
            "note": "R² for predicting Δx(t+1); lag-1 baseline = always predict 0",
        },
        "per_dimension_r2": per_dim_r2,
        "ablation": ablation_results,
        "h1b_verdict": (
            "SUPPORTED — VARX beats lag-1"
            if full_r2 > lag1_r2 else
            "PARTIAL — VARX ≈ lag-1 (within 5%)"
            if full_r2 >= lag1_r2 - 0.05 else
            "REJECTED — state is near-random-walk, lag-1 is optimal predictor"
        ),
        "h1c_verdict": (
            "SUPPORTED — state history alone drives prediction"
            if abs(state_r2 - full_r2) < 0.05 else
            "MIXED — exogenous variables (u, y) contribute meaningfully"
            if full_r2 > state_r2 + 0.02 else
            "EXOGENOUS DOMINATES — state alone insufficient"
        ),
    }


# ═══════════════════════════════════════════════
# UMAP Visualization (optional)
# ═══════════════════════════════════════════════

def umap_projection(X: np.ndarray, months: list[str]) -> dict | None:
    """UMAP 2D 投影用于可视化."""
    if not HAS_UMAP:
        return None

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_s = StandardScaler().fit_transform(X)
    embedding = reducer.fit_transform(X_s)

    return {
        "x": embedding[:, 0].tolist(),
        "y": embedding[:, 1].tolist(),
        "months": months,
    }


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Level 2: Representation Learning + H1 Verification")
    parser.add_argument("--ext-pc", type=int, default=8, help="External field PCA components")
    parser.add_argument("--lookback", type=int, default=3, help="Lookback months for VARX")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help="Output path for representation state")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 64)
    print("Level 2: Representation Learning + H1 Verification")
    print("=" * 64)

    # ── Load ──
    print("\n[1/5] 加载数据...")
    l1 = load_level1()
    ext_months, ext_matrix = load_external_field()
    trends = load_trends()

    l1_months = l1["months"]
    print(f"  Level 1: {len(l1_months)} 月 ({l1_months[0]} → {l1_months[-1]})")
    print(f"  External Field: {len(ext_months)} 月, {ext_matrix.shape[1]} 关键词")
    print(f"  Trends: {len(trends)} 关键词")

    # ── Compute attention structure ──
    hhi, entropy, total_att = compute_attention_structure(trends, l1_months)
    print(f"  Attention: HHI mean={hhi.mean():.3f}, entropy mean={entropy.mean():.3f}")

    # ── Assemble ──
    print(f"\n[2/5] 组装特征矩阵 (ext_pc={args.ext_pc})...")
    X, months, meta = assemble_features(
        l1, ext_months, ext_matrix, hhi, entropy, total_att, ext_pc=args.ext_pc)
    print(f"  特征矩阵: {X.shape[0]} 月 × {X.shape[1]} 维")
    print(f"  特征组成: L1({meta['l1_dim']}) + ExtPCA({meta['ext_pc_dim']}) + Att({meta['att_dim']})")
    print(f"  外部场 PCA 解释方差: {meta['ext_pca_variance']:.1%}")

    # ── H1a: Intrinsic dimension ──
    print(f"\n[3/5] H1a: 内在维度分析...")
    h1a = verify_h1a(X, meta["feature_names"])
    print(f"  PCA: {h1a['n_components']} 维 → "
          f"d90={h1a['dim_90']}, d95={h1a['dim_95']}, d99={h1a['dim_99']}")
    print(f"  累积方差: 90%={h1a['cumulative_variance'][h1a['dim_90']-1]:.1%}, "
          f"95%={h1a['cumulative_variance'][h1a['dim_95']-1]:.1%}, "
          f"99%={h1a['cumulative_variance'][h1a['dim_99']-1]:.1%}")
    print(f"  → {h1a['recommendation']}")

    # Top PC1 loadings
    print(f"\n  PC1 主要载荷:")
    for item in h1a["top_loadings"][0]:
        print(f"    {item['feature']:<20s} {item['loading']:+.4f}")

    # ── H1b + H1c: Dynamics ──
    state_dim = h1a["dim_90"]
    print(f"\n[4/5] H1b+H1c: 动力学验证 (state_dim={state_dim}, lookback={args.lookback})...")
    dynamics = verify_h1b_h1c(X, months, meta["feature_names"],
                              state_dim=state_dim, lookback=args.lookback,
                              ext_pc_dim=meta["ext_pc_dim"])

    print(f"  PCA 降维: {X.shape[1]} → {state_dim} 维 (保留 {dynamics['pca_variance_explained']:.1%} 方差)")
    print(f"  Lag-1 baseline: R²={dynamics['lag1_baseline']['r2']:.4f}, "
          f"MAE={dynamics['lag1_baseline']['mae']:.4f}")
    print(f"  Full VARX: train R²={dynamics['full_varx']['train_r2']:.4f}, "
          f"test R²={dynamics['full_varx']['test_r2']:.4f}, "
          f"MAE={dynamics['full_varx']['test_mae']:.4f}, α={dynamics['full_varx']['alpha']:.1f}")
    print(f"  State-only VAR: test R²={dynamics['state_only_var']['test_r2']:.4f}, "
          f"α={dynamics['state_only_var']['alpha']:.1f}")
    print(f"  Δx prediction: test R²={dynamics['delta_x_prediction']['test_r2']:.4f} "
          f"(baseline zero-R²={dynamics['delta_x_prediction']['lag1_zero_r2']:.4f})")

    print(f"\n  Ablation (test R²):")
    for k, v in dynamics["ablation"].items():
        delta = v - dynamics["ablation"]["full"]
        print(f"    {k:<20s} R²={v:+.4f} (Δ={delta:+.4f})")

    print(f"\n  Per-dimension (PC) R² vs lag-1:")
    for item in dynamics["per_dimension_r2"]:
        delta_vs_lag1 = item['r2'] - item['lag1_r2']
        bar = "█" * max(0, int((item['r2'] + 0.5) * 15))
        print(f"    {item['feature']:<10s} R²={item['r2']:+.4f} (lag1={item['lag1_r2']:+.4f}, Δ={delta_vs_lag1:+.4f}) {bar}")

    print(f"\n  H1b: {dynamics['h1b_verdict']}")
    print(f"  H1c: {dynamics['h1c_verdict']}")

    # ── UMAP ──
    print(f"\n[5/5] UMAP 可视化...")
    umap_result = umap_projection(X, months)
    if umap_result:
        print(f"  UMAP 2D: {len(umap_result['x'])} 点")
    else:
        print("  UMAP 未安装 (pip install umap-learn)")

    # ── Save ──
    output = {
        "source": "representation_learning.py — Level 2",
        "config": {
            "ext_pc": args.ext_pc,
            "lookback": args.lookback,
        },
        "feature_meta": meta,
        "h1a_intrinsic_dim": h1a,
        "h1b_h1c_dynamics": dynamics,
        "umap_2d": umap_result,
        "pca_transformed": None,  # populated below if needed
    }

    # Also save the reduced state for downstream use
    scaler_out = StandardScaler()
    X_s_out = scaler_out.fit_transform(X)
    pca_out = PCA(n_components=state_dim)
    x_reduced = pca_out.fit_transform(X_s_out)
    output["pca_transformed"] = {
        "n_components": state_dim,
        "x_reduced": x_reduced.tolist(),
        "months": months,
        "explained_variance": float(pca_out.explained_variance_ratio_.sum()),
    }

    print(f"\n  降维状态 x(t): {x_reduced.shape[1]} 维 (90% 方差)")
    print(f"  保存 → {args.output}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print("Level 2 完成。下一步: Level 3 后验解释 (特征载荷分析).")
    print("=" * 64)


if __name__ == "__main__":
    main()
