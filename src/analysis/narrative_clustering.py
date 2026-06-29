"""
叙事嵌入聚类 — 用 LLM 概念分数做无监督聚类, 发现 emergent 叙事类型

不预设类别, 让数据自己说话 (AlphaGo 范式).
输入: llm_concept_scores.json (48 条叙事, 35 维概念向量)
输出:
- 最优聚类数 (Silhouette/Elbow)
- 每个 cluster 的代表性梗 + 概念 profile
- 与人工 5 类别的一致性 (ARI)

用法:
    python src/analysis/narrative_clustering.py
    python src/analysis/narrative_clustering.py --plot
"""

import json, sys
import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT / "data/processed"
CONCEPT_PATH = PROCESSED_DIR / "llm_concept_scores.json"

CONSTRAINT_LABELS = ["Identity", "Humor/Decon", "Conflict", "Novelty", "Accessibility"]

# 人工标签 (用于对比, 不参与聚类)
CATEGORY_NAMES = ["解构自嘲", "攻击发泄", "虚无退却", "身份认同", "纯粹娱乐"]

CATEGORY_MAP = {
    # 解构自嘲
    "打工人": "解构自嘲", "吗喽": "解构自嘲", "鼠鼠": "解构自嘲",
    "牛马": "解构自嘲", "精神状态良好": "解构自嘲",
    "破防": "解构自嘲", "社恐": "解构自嘲", "社死": "解构自嘲",
    "发疯文学": "解构自嘲",
    # 攻击发泄
    "普信男": "攻击发泄", "XX刺客": "攻击发泄",
    "建议专家不要建议": "攻击发泄",
    # 虚无退却
    "躺平": "虚无退却", "摆烂": "虚无退却", "润": "虚无退却",
    "精神内耗": "虚无退却", "四不/不婚不育": "虚无退却",
    # 身份认同
    "内卷 / 卷": "身份认同", "小镇做题家": "身份认同",
    "i人/e人": "身份认同", "孔乙己的长衫": "身份认同",
    "后浪": "身份认同", "情绪价值": "身份认同",
    "原生家庭": "身份认同",
    # 纯粹娱乐
    "鸡你太美": "纯粹娱乐", "科目三": "纯粹娱乐",
    "遥遥领先": "纯粹娱乐", "尊嘟假嘟": "纯粹娱乐",
    "凡尔赛": "纯粹娱乐", "元宇宙": "纯粹娱乐",
    "citywalk": "纯粹娱乐", "芭比Q": "纯粹娱乐",
    "栓Q": "纯粹娱乐", "美拉德": "纯粹娱乐",
    "南方小土豆": "纯粹娱乐", "显眼包": "纯粹娱乐",
    "多巴胺穿搭": "纯粹娱乐",
}


def load_concept_vectors() -> tuple[np.ndarray, list[str], dict]:
    """加载概念分数矩阵."""
    with open(CONCEPT_PATH, "r", encoding="utf-8") as f:
        scores = json.load(f)

    meme_names = []
    vectors = []
    human_labels = []

    for name, entry in scores.items():
        if "concept_scores" not in entry:
            continue
        vec = [entry["concept_scores"].get(c, 0.0) for c in sorted(entry["concept_scores"].keys())]
        if sum(vec) == 0:
            continue
        meme_names.append(name)
        vectors.append(vec)
        human_labels.append(CATEGORY_MAP.get(name, "未标注"))

    X = np.array(vectors)
    return X, meme_names, human_labels


def find_optimal_clusters(X: np.ndarray, max_k: int = 12) -> dict:
    """Silhouette + Elbow 找最优聚类数."""
    X_s = StandardScaler().fit_transform(X)

    results = {}
    best_k, best_sil = 3, -1

    for k in range(2, min(max_k + 1, len(X))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_s)
        sil = silhouette_score(X_s, labels)
        inertia = km.inertia_

        results[k] = {"silhouette": round(sil, 4), "inertia": round(inertia, 1)}
        if sil > best_sil:
            best_sil = sil
            best_k = k

    return {"best_k": best_k, "best_silhouette": round(best_sil, 4), "all": results}


def cluster_narratives(X: np.ndarray, n_clusters: int = None) -> dict:
    """聚类叙事."""
    X_s = StandardScaler().fit_transform(X)

    # Find optimal k if not specified
    if n_clusters is None:
        opt = find_optimal_clusters(X)
        n_clusters = opt["best_k"]
        print(f"最优聚类数: {n_clusters} (Silhouette={opt['best_silhouette']})")

    # KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_s)

    # Agglomerative (hierarchical)
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg.fit_predict(X_s)

    return {
        "n_clusters": n_clusters,
        "kmeans_labels": km_labels.tolist(),
        "agg_labels": agg_labels.tolist(),
        "kmeans_centers": km.cluster_centers_.tolist(),
        "silhouette": round(silhouette_score(X_s, km_labels), 4),
    }


def describe_clusters(X: np.ndarray, labels: list[int], meme_names: list[str],
                      human_labels: list[str] = None) -> list[dict]:
    """描述每个 cluster 的特征."""
    n_clusters = max(labels) + 1
    X_s = StandardScaler().fit_transform(X)

    descriptions = []
    for c in range(n_clusters):
        idx = np.where(np.array(labels) == c)[0]
        members = [meme_names[i] for i in idx]

        # Mean concept vector
        mean_vec = X[idx].mean(axis=0)

        # Top distinguishing concepts
        overall_mean = X.mean(axis=0)
        diff = mean_vec - overall_mean
        top_concepts = []
        concept_names = sorted(CONCEPT_PATH and list(json.load(open(CONCEPT_PATH, "r", encoding="utf-8")).values())[0].get("concept_scores", {}).keys()) if False else []

        # Human label distribution (if available)
        label_dist = {}
        if human_labels:
            member_labels = [human_labels[i] for i in idx]
            label_dist = {l: member_labels.count(l) for l in set(member_labels)}
            dominant = max(label_dist, key=label_dist.get)
        else:
            dominant = "?"

        descriptions.append({
            "cluster_id": c,
            "size": len(idx),
            "members": members[:8],  # top 8
            "dominant_human_label": dominant,
            "label_distribution": label_dist,
            "mean_constraint": {
                "Identity": float(mean_vec[0]) if len(mean_vec) > 0 else 0,
                "Humor": float(mean_vec[1]) if len(mean_vec) > 1 else 0,
                "Conflict": float(mean_vec[2]) if len(mean_vec) > 2 else 0,
                "Novelty": float(mean_vec[3]) if len(mean_vec) > 3 else 0,
                "Accessibility": float(mean_vec[4]) if len(mean_vec) > 4 else 0,
            },
        })

    return descriptions


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("叙事嵌入聚类 — 无监督发现叙事类型")
    print("=" * 60)

    X, names, human_labels = load_concept_vectors()
    print(f"\n数据: {len(names)} 条叙事, {X.shape[1]} 维概念向量")

    # Optimal k
    opt = find_optimal_clusters(X)
    print(f"\n最优聚类数: {opt['best_k']} (Silhouette={opt['best_silhouette']})")
    print("K  Silhouette")
    for k, v in sorted(opt["all"].items()):
        print(f"  {k}: {v['silhouette']:.4f}")

    # Cluster
    result = cluster_narratives(X, n_clusters=opt["best_k"])
    print(f"\nKMeans Silhouette: {result['silhouette']:.4f}")

    # Compare with human labels
    if human_labels:
        label_nums = []
        label_to_num = {}
        next_num = 0
        for l in human_labels:
            if l not in label_to_num:
                label_to_num[l] = next_num
                next_num += 1
            label_nums.append(label_to_num[l])

        ari = adjusted_rand_score(label_nums, result["kmeans_labels"])
        print(f"\n与人工 5 类别一致性 (ARI): {ari:.4f}")
        print(f"(ARI=0 随机, ARI=1 完全一致)")

    # Describe clusters
    desc = describe_clusters(X, result["kmeans_labels"], names, human_labels)
    print(f"\n=== {result['n_clusters']} 个发现类型 ===")
    for d in desc:
        print(f"\n  Cluster {d['cluster_id']} ({d['size']} 条):")
        print(f"    主导人工标签: {d['dominant_human_label']}")
        print(f"    约束场: I={d['mean_constraint']['Identity']:.2f} "
              f"H={d['mean_constraint']['Humor']:.2f} "
              f"C={d['mean_constraint']['Conflict']:.2f} "
              f"N={d['mean_constraint']['Novelty']:.2f} "
              f"A={d['mean_constraint']['Accessibility']:.2f}")
        print(f"    代表梗: {', '.join(d['members'][:5])}")


if __name__ == "__main__":
    main()
