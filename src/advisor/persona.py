"""
FR31 Persona Encoder — Probabilistic Narrative Graph Projection

双层编码器: 用户文本 → sentence-transformers → 叙事图概率分布.
不是"你在 X 节点"的点定位 — 是 P(meme_i | user_text) 的完整分布.

核心安全机制 (epistemic guards):
- FREE_NOISE 熔断: max_sim < 0.65 → 用户文本未耦合到任何已知梗
- AMBIGUOUS 熔断: top2 gap < 0.05 → 文本跨节点, 拒绝强行归类
- WEAK_SIGNAL: max_sim > 0.65 但 gap/confidence 不足 → 输出但标注低置信
- 输出分布 + 熵 + 浓度, 不输出单点标签
- "不知道"是默认响应, "确定"需要多个信号同时通过

用法:
    python src/advisor/persona.py --text "老板又在群里发奋斗鸡汤..."
    python src/advisor/persona.py --text "..." --json
    python src/advisor/persona.py --benchmark  # 自检: 用已知梗文本测试召回

架构:
    Stage 1: 纯 embedding 余弦相似度 (LLM-free)
    Stage 2: 阈值熔断 + softmax 概率归一化 + 不确定性量化
"""

import json, sys, os, argparse, glob
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent.parent

NARRATIVE_DIRS = [
    ROOT / "data/processed/narratives",
    ROOT / "data/processed/narratives_from_trends",
]

# ── Epistemic guard thresholds ──
# 校准: paraphrase-multilingual-MiniLM-L12-v2 对中文跨域文本
# 余弦相似度范围: 自检(同域) mean=0.95, 跨域用户文本 0.42-0.57, 纯噪声 ~0.25
FREE_NOISE_THRESHOLD = 0.30    # max cosine sim below this → 真噪声, 无耦合 (Type C)
AMBIGUITY_MARGIN = 0.03        # top2 gap below this → 跨节点边界, 拒绝选边
PARTIAL_THRESHOLD = 0.40       # max_sim in [0.30, 0.40] → weak signal, could be Type B
# KNOWN/PARTIAL 不使用硬阈值分界 — 由 project() 的 confidence (HIGH/MEDIUM/LOW) 映射.
# 避免假装有精确分界线. 阈值需要外部验证数据 (虹姐语料) 才能校准.

# ── OOD detection thresholds ──
# 触发条件: Observation 层连续 N 月超出历史 3σ, 或数据源覆盖骤降
OOD_CONSECUTIVE_MONTHS = 3
OOD_SIGMA = 3.0


class PersonaEncoder:
    """个体→叙事图概率投影器."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.nodes: dict[str, str] = {}
        self.node_embeddings = None
        self.node_names: list[str] = []
        self._loaded = False

    def load_narrative_graph(self) -> int:
        """加载 57 条叙事 JSON, 为每个梗构建语义描述节点."""
        self.nodes = {}
        seen = set()

        for nar_dir in NARRATIVE_DIRS:
            if not nar_dir.exists():
                continue
            for fp in sorted(glob.glob(str(nar_dir / "*.json"))):
                if "_all" in fp:
                    continue
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                name = data.get("meme_name", Path(fp).stem)
                if name in seen:
                    continue
                seen.add(name)

                parts = []
                sd = data.get("semantic_drift")
                if isinstance(sd, dict):
                    curr = sd.get("current_meaning", "")
                    if curr:
                        parts.append(curr)

                summary = data.get("narrative_summary", "")
                if summary:
                    parts.append(summary)

                hint = data.get("social_context_hint", "")
                if hint:
                    parts.append(hint)

                descriptor = " ".join(parts).strip()
                if not descriptor:
                    descriptor = name

                self.nodes[name] = descriptor

        self.node_names = sorted(self.nodes.keys())
        node_texts = [self.nodes[n] for n in self.node_names]

        print(f"  编码 {len(node_texts)} 叙事节点...")
        self.node_embeddings = self.model.encode(
            node_texts, show_progress_bar=True, convert_to_numpy=True)
        self._loaded = True

        return len(self.node_names)

    def project(self, user_text: str) -> dict:
        """将用户文本投影到叙事图, 返回概率分布 + 不确定性."""
        if not self._loaded:
            return {"status": "ERROR", "reason": "NOT_LOADED"}

        user_text = user_text.strip()
        if not user_text:
            return {
                "status": "MELTDOWN",
                "reason": "EMPTY_INPUT",
                "interpretation": "输入为空.",
            }

        user_emb = self.model.encode([user_text], show_progress_bar=False,
                                      convert_to_numpy=True)[0]

        # Cosine similarities
        similarities = np.dot(self.node_embeddings, user_emb) / (
            np.linalg.norm(self.node_embeddings, axis=1) * np.linalg.norm(user_emb) + 1e-10)

        sorted_idx = np.argsort(-similarities)
        top_k = min(5, len(self.node_names))
        max_sim = float(similarities[sorted_idx[0]])

        # ── Guard 1: FREE_NOISE ──
        if max_sim < FREE_NOISE_THRESHOLD:
            return {
                "status": "MELTDOWN",
                "reason": "FREE_NOISE",
                "max_similarity": round(max_sim, 4),
                "threshold": FREE_NOISE_THRESHOLD,
                "interpretation": (
                    "用户文本未与大盘任何已知叙事节点产生有效耦合. "
                    "该情境处于叙事图的'自由噪声区', 宏观动力学在此处不施加约束. "
                    "FR31 不对此情境做结构化背书 — 请基于常识自行判断."
                ),
            }

        # ── Guard 2: AMBIGUOUS ──
        second_sim = float(similarities[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
        gap = max_sim - second_sim
        if gap < AMBIGUITY_MARGIN:
            return {
                "status": "MELTDOWN",
                "reason": "AMBIGUOUS_COMPETITION",
                "max_similarity": round(max_sim, 4),
                "top_nodes": [
                    {"node": self.node_names[sorted_idx[0]],
                     "similarity": round(max_sim, 4)},
                    {"node": self.node_names[sorted_idx[1]],
                     "similarity": round(second_sim, 4)},
                ],
                "gap": round(gap, 4),
                "margin": AMBIGUITY_MARGIN,
                "interpretation": (
                    f"文本同时高度亲和 [{self.node_names[sorted_idx[0]]}] 与 "
                    f"[{self.node_names[sorted_idx[1]]}], 处于叙事拓扑边界线. "
                    f"FR31 拒绝在竞争节点间强行选边 — 这不是测量精度的不足, "
                    f"而是系统的真实模糊性. 不假装读心."
                ),
            }

        # ── Normal projection ──
        sims_centered = similarities - np.max(similarities)
        probs = np.exp(sims_centered / 0.1)  # temperature=0.1, sharp but not binary
        probs = probs / probs.sum()

        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = np.log(len(self.node_names))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        top_k_probs = sorted(probs, reverse=True)[:5]
        concentration = float(sum(top_k_probs[:3]))

        if max_sim > 0.55 and gap > 0.06:
            confidence = "HIGH"
        elif max_sim > 0.40 and gap > 0.03:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        top_nodes = []
        for rank, idx in enumerate(sorted_idx[:top_k]):
            top_nodes.append({
                "rank": rank + 1,
                "node": self.node_names[idx],
                "probability": round(float(probs[idx]), 4),
                "cosine_similarity": round(float(similarities[idx]), 4),
            })

        return {
            "status": "SUPPORTED" if confidence == "HIGH" else ("WEAK_SIGNAL" if confidence == "MEDIUM" else "LOW_SIGNAL"),
            "confidence": confidence,
            "distribution": {
                "top_nodes": top_nodes,
                "entropy": round(entropy, 4),
                "normalized_entropy": round(normalized_entropy, 4),
                "top3_concentration": round(concentration, 4),
                "n_active_nodes": int(np.sum(probs > 0.01)),
            },
            "epistemic_note": (
                "本输出为 embedding 空间余弦相似度概率分布, "
                "非心理学推断. 不表征个体心理状态或人格特质. "
                "仅描述用户文本在当前互联网叙事图中的投影位置."
            ),
        }

    # ═══════════════════════════════════════════════
    # Five-state cognitive model (FORMALISM §11.6)
    # ═══════════════════════════════════════════════

    def assess(self, user_text: str, macro_state: dict = None,
               ood_check: bool = True) -> dict:
        """五态认知模型: KNOWN / PARTIAL / UNKNOWN / AMBIGUOUS / OOD.

        调用 project(), 然后根据相似度 + 宏观状态 + OOD 检测,
        将结果映射到五种认知状态之一.

        Args:
            user_text: 用户文本
            macro_state: 宏观指标 dict (可选, 用于 OOD 检测)
            ood_check: 是否执行 OOD 检测

        Returns:
            {cognitive_state, projection, macro_overlay, interpretation}
        """
        proj = self.project(user_text)

        # ── Map projection status to cognitive state ──
        if proj["status"] == "MELTDOWN":
            if proj.get("reason") == "FREE_NOISE":
                cognitive = "UNKNOWN"
                interp = (
                    "Type C — 你的情境在集体叙事数据中无对应结构. "
                    "FR31 对此不做推导. 请基于你自己的判断."
                )
            elif proj.get("reason") == "AMBIGUOUS_COMPETITION":
                cognitive = "AMBIGUOUS"
                interp = (
                    "你的情境同时触碰多个叙事节点, 无法区分. "
                    "这不是测量精度不足 — 是情境本身具有真实的多义性. "
                    "FR31 拒绝强行选边."
                )
            else:
                cognitive = "UNKNOWN"
                interp = "无法确定."
        else:
            conf = proj.get("confidence", "LOW")
            top_node = proj["distribution"]["top_nodes"][0]["node"]

            if conf == "HIGH":
                cognitive = "KNOWN"
                interp = (
                    f"Type A — 你的情境在叙事图上有清晰的投影 "
                    f"({top_node}). 宏观叙事系统对此有充分信息."
                )
            elif conf == "MEDIUM":
                cognitive = "PARTIAL"
                interp = (
                    f"Type B — 你的情境有宏观信号 ({top_node}), "
                    f"但个体层面的关键变量不在集体数据覆盖域内. "
                    f"FR31 可以描述宏观地形, 但不能替你回答'该不该'."
                )
            else:  # LOW
                cognitive = "UNKNOWN"
                interp = "信号过弱, 无法归类."

        # ── OOD check ──
        if ood_check and macro_state:
            ood = self._check_ood(macro_state)
            if ood["is_ood"]:
                cognitive = "OOD"
                interp = (
                    f"⚠ 模型失效域 — {ood['reason']}. "
                    f"当前媒介环境可能已发生结构性变化, "
                    f"历史统计关系的适用前提可能不成立."
                )

        result = {
            "cognitive_state": cognitive,
            "interpretation": interp,
            "projection": proj,
        }

        # ── Add macro overlay if available ──
        if macro_state:
            result["macro_overlay"] = {
                "inertia": macro_state.get("inertia"),
                "resilience": macro_state.get("resilience"),
                "sensitivity": macro_state.get("sensitivity"),
                "regime": macro_state.get("regime"),
            }

        return result

    def _check_ood(self, macro_state: dict) -> dict:
        """检测是否进入模型失效域 (OOD).

        检查宏观指标是否连续超出历史范围,
        以及数据源覆盖是否骤降.
        """
        # Simplified OOD check: if multiple indicators are at extremes
        alerts = []
        ood_score = 0

        inertia = macro_state.get("inertia", 0.5)
        resilience = macro_state.get("resilience", 0.5)
        sensitivity = macro_state.get("sensitivity", 0.5)

        # Extreme values
        if inertia > 0.85:
            alerts.append(f"Inertia={inertia:.2f} 极端高")
            ood_score += 1
        if resilience < 0.15:
            alerts.append(f"Resilience={resilience:.2f} 极端低")
            ood_score += 1
        if sensitivity > 0.70:
            alerts.append(f"Sensitivity={sensitivity:.2f} 极端高")
            ood_score += 1

        is_ood = ood_score >= 2

        return {
            "is_ood": is_ood,
            "ood_score": ood_score,
            "alerts": alerts,
            "reason": "; ".join(alerts) if alerts else "无异常",
        }

    def benchmark(self) -> dict:
        """自检: 每个梗的描述文本片段 → 能否在 top-3 中找回自身?"""
        if not self._loaded:
            self.load_narrative_graph()

        hits_top1, hits_top3, total = 0, 0, 0
        details = []

        for name, desc in self.nodes.items():
            query = desc[:80]
            result = self.project(query)
            total += 1

            if result["status"] != "MELTDOWN":
                top3 = [n["node"] for n in result["distribution"]["top_nodes"]]
                if top3 and top3[0] == name:
                    hits_top1 += 1
                if name in top3[:3]:
                    hits_top3 += 1

            details.append({
                "query_node": name,
                "status": result["status"],
                "top1": (result.get("distribution", {}).get("top_nodes", [{}])[0]
                         .get("node", "") if result["status"] != "MELTDOWN" else ""),
            })

        return {
            "total": total,
            "top1_accuracy": round(hits_top1 / max(total, 1), 4),
            "top3_accuracy": round(hits_top3 / max(total, 1), 4),
            "details": details,
        }


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FR31 Persona Encoder")
    parser.add_argument("--text", type=str, default=None, help="用户文本")
    parser.add_argument("--benchmark", action="store_true", help="自检召回率")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("FR31 Persona Encoder — 叙事图概率投影")
    print("=" * 60)

    encoder = PersonaEncoder()
    n_nodes = encoder.load_narrative_graph()
    print(f"  叙事图: {n_nodes} 节点\n")

    if args.benchmark:
        print("[Benchmark] 自检: 梗描述片段 → top-1/top-3 召回...")
        bench = encoder.benchmark()
        print(f"  Top-1: {bench['top1_accuracy']:.1%}  Top-3: {bench['top3_accuracy']:.1%}")
        if args.json:
            print(json.dumps(bench, ensure_ascii=False, indent=2))
        return

    if args.text:
        result = encoder.project(args.text)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        print(f"  状态: {result['status']}")
        if result["status"] == "MELTDOWN":
            print(f"  原因: {result['reason']}")
            print(f"\n  {result['interpretation']}")
        else:
            print(f"  置信度: {result.get('confidence', 'N/A')}")
            d = result["distribution"]
            print(f"\n  Top 节点:")
            for n in d["top_nodes"]:
                bar = "█" * int(n["probability"] * 50)
                print(f"    {n['rank']}. {n['node']:<20s} "
                      f"p={n['probability']:.4f} (sim={n['cosine_similarity']:.4f}) {bar}")
            print(f"\n  熵: {d['entropy']:.4f} (归一化={d['normalized_entropy']:.4f})")
            print(f"  Top-3 浓度: {d['top3_concentration']:.4f}")
            print(f"  活跃节点: {d['n_active_nodes']}")
            print(f"\n  {result['epistemic_note']}")
        return

    # Interactive
    print("输入用户文本 (Ctrl+C 退出):")
    try:
        while True:
            text = input("\n> ").strip()
            if not text:
                continue
            result = encoder.project(text)
            if result["status"] == "MELTDOWN":
                print(f"  [{result['reason']}] {result['interpretation'][:150]}...")
            else:
                top = result["distribution"]["top_nodes"][0]
                print(f"  [{result['confidence']}] → {top['node']} "
                      f"(p={top['probability']:.4f}, sim={top['cosine_similarity']:.4f})")
                print(f"  熵={result['distribution']['normalized_entropy']:.3f}")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
