"""
系统完整性验证 — 不测单个函数，测跨模块的端到端正确性。

运行: python tests/test_system_integrity.py
"""

import json, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = 0, 0

def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}  {detail}")

print("=" * 60)
print("MemeticChaos — System Integrity Verification")
print("=" * 60)

# ═══════════════════════════════════════════════
# 1. FNN on known dynamical systems
# ═══════════════════════════════════════════════
print("\n── 1. FNN (Kennel 1992) on known systems ──")
from src.models.attractor import estimate_embedding_dim, estimate_lyapunov
from scipy.integrate import solve_ivp

def lorenz(t,s): x,y,z=s; return [10*(y-x), x*(28-z)-y, x*y-8*z/3]
sol = solve_ivp(lorenz, [0,100], [1,1,1], max_step=0.05)
x_l = sol.y[0,-2000:]
check("Lorenz d=3", estimate_embedding_dim(x_l, max_dim=10, delay=10) == 3)
check("Lorenz λ>0 (chaotic)", estimate_lyapunov(x_l, delay=10) > 0)

def rossler(t,s): x,y,z=s; return [-y-z, x+0.2*y, 0.2+z*(x-5.7)]
sol2 = solve_ivp(rossler, [0,300], [1,1,1], max_step=0.1)
x_r = sol2.y[0,-3000:]
check("Rossler d=3", estimate_embedding_dim(x_r, max_dim=10, delay=15) == 3)

t = np.linspace(0,50,1000)
check("Sine d=2", estimate_embedding_dim(np.sin(2*np.pi*0.1*t)) == 2)
check("Constant d=1", estimate_embedding_dim(np.ones(500), max_dim=8) == 1)

# ═══════════════════════════════════════════════
# 2. Continuous R₀ — no clustering at 3 values
# ═══════════════════════════════════════════════
print("\n── 2. R₀ continuity ──")
from src.models.sir_meme import estimate_total_infected

r0s = []
for cc in range(3,8):
    for pi in [0.3, 0.5, 0.7, 0.9]:
        ti = estimate_total_infected(cc, pi, 12)
        from src.models.sir_meme import estimate_params_from_lifecycle
        r0 = estimate_params_from_lifecycle(30, ti, 90).R0
        r0s.append(round(r0, 4))
unique = len(set(r0s))
check("R₀ values all unique (no 3-value clustering)", unique == len(r0s),
      f"{unique}/{len(r0s)} unique")
check("R₀ in physical range [1.0, 3.0]", all(1.0 < r < 3.0 for r in r0s),
      f"range=[{min(r0s):.2f}, {max(r0s):.2f}]")

# ═══════════════════════════════════════════════
# 3. Trajectory completeness
# ═══════════════════════════════════════════════
print("\n── 3. Trajectory completeness ──")
with open("data/processed/trajectories.json", "r") as f:
    data = json.load(f)
trajs = data["trajectories"]

check("29 trajectories", len(trajs) == 29)
check("All have 3+ phases", all(len(t["nodes"]) >= 3 for t in trajs))
check("All have 4 state vectors per node",
      all(all(s in n for s in ["narrative_state","constraint_state","dynamic_state","social_context"])
          for t in trajs for n in t["nodes"]))
check("All constraint_state.pressures have 5 dims",
      all(len(n["constraint_state"]["pressures"]) == 5 for t in trajs for n in t["nodes"]))
check("All pressure values in [0,1]",
      all(0 <= p <= 1 for t in trajs for n in t["nodes"]
          for p in n["constraint_state"]["pressures"]))
check("Schema version 2.1", data.get("schema_version") == "2.1")

# Count phases
phases = {}
for t in trajs:
    for n in t["nodes"]:
        p = n["phase"]
        phases[p] = phases.get(p, 0) + 1
check("Has all expected phases", all(p in phases for p in ["origin","emergence","peak","fixation"]),
      f"found: {sorted(phases.keys())}")

# ═══════════════════════════════════════════════
# 4. Concept Bottleneck — known meme → expected constraint
# ═══════════════════════════════════════════════
print("\n── 4. Concept Bottleneck correctness ──")
from src.constraint.concept_bottleneck import ConceptMatrix, ConstraintMapper

# 后浪: official release + generation conflict → conflict dominant
for t in trajs:
    if t["name"] == "后浪":
        p = t["nodes"][0]["constraint_state"]["pressures"]
        check("后浪: conflict is dominant constraint",
              np.argmax(p) == 2, f"dominant={['id','humor','conflict','novelty','acc'][np.argmax(p)]}, p={[round(x,2) for x in p]}")
        break

# 普信男: conflict dominant (soft-match, moderate values)
for t in trajs:
    if t["name"] == "普信男":
        p = t["nodes"][0]["constraint_state"]["pressures"]
        check("普信男: conflict is dominant (soft-match)",
              np.argmax(p) == 2,
              f"dominant={['id','humor','conflict','novelty','acc'][np.argmax(p)]}, p={[round(x,2) for x in p]}")
        break

# ═══════════════════════════════════════════════
# 5. Delta Transition — physically sensible
# ═══════════════════════════════════════════════
print("\n── 5. Delta Transition correctness ──")
from src.constraint.delta_transition import DeltaTransitionModel

dtm = DeltaTransitionModel({"economic_stress": 0.7, "polarization": 0.6, "censorship": 0.3})
c0 = np.array([0.5, 0.5, 0.3, 0.5, 0.5])

# peak→controversy: conflict should rise
d1 = dtm.predict_delta(c0, "peak", "controversy")
check("peak→controversy: conflict rises", d1[2] > 0, f"Δconflict={d1[2]:+.3f}")

# controversy→fixation: conflict should fall
d2 = dtm.predict_delta(c0, "controversy", "fixation")
check("controversy→fixation: conflict falls", d2[2] < 0, f"Δconflict={d2[2]:+.3f}")

# Delta bounded
check("All deltas within [-0.4, 0.4]", all(-0.4 <= d <= 0.4 for d in d1) and all(-0.4 <= d <= 0.4 for d in d2))

# ═══════════════════════════════════════════════
# 6. Three Validators — catch violations, pass valid
# ═══════════════════════════════════════════════
print("\n── 6. Validator correctness ──")
from src.constraint.delta_transition import (NarrativeValidator, ConstraintValidator,
                                              DynamicsValidator, validate_trajectory)

# Narrative: reversal should fail
nv = NarrativeValidator()
r = nv.validate(["fixation", "peak", "origin"])
check("Narrative: catches phase reversal", not r.valid)

# Narrative: valid order should pass
r2 = nv.validate(["origin", "emergence", "peak", "fixation"])
check("Narrative: passes valid sequence", r2.valid)

# Constraint: huge jump should fail
cv = ConstraintValidator()
r3 = cv.validate([np.array([0.5]*5), np.array([0.9, 0.5, 0.5, 0.5, 0.5])])
check("Constraint: catches 0.4 single-dim jump", not r3.valid)

# Constraint: small jump should pass
r4 = cv.validate([np.array([0.5]*5), np.array([0.6, 0.5, 0.5, 0.5, 0.5])])
check("Constraint: passes small jump (0.1)", r4.valid)

# Dynamics: peak with R₀ < 1 should fail
dv = DynamicsValidator()
r5 = dv.validate([
    {"phase": "origin", "dynamic_state": {"R0": 0.1, "chaos_axis": 0}},
    {"phase": "peak", "dynamic_state": {"R0": 0.5, "chaos_axis": 0}},
])
check("Dynamics: catches peak R₀ < 1", not r5.valid)

# Dynamics: valid trajectory should pass
r6 = dv.validate([
    {"phase": "origin", "dynamic_state": {"R0": 0.1, "chaos_axis": 0}},
    {"phase": "peak", "dynamic_state": {"R0": 1.5, "chaos_axis": -0.3}},
    {"phase": "fixation", "dynamic_state": {"R0": 0.0, "chaos_axis": -0.5}},
])
check("Dynamics: passes valid trajectory", r6.valid, str(r6.violations))

# ═══════════════════════════════════════════════
# 7. All 29 trajectories pass validation
# ═══════════════════════════════════════════════
print("\n── 7. Full trajectory validation ──")
valid_count = 0
for t in trajs:
    phases = [n["phase"] for n in t["nodes"]]
    constraints = [np.array(n["constraint_state"]["pressures"]) for n in t["nodes"]]
    r = validate_trajectory(phases, constraints, t["nodes"])
    if r["valid"]:
        valid_count += 1
check("All 29 trajectories valid", valid_count == 29, f"{valid_count}/29")

# ═══════════════════════════════════════════════
# 8. Scraper signal detection logic
# ═══════════════════════════════════════════════
print("\n── 8. Scraper signal detection ──")
from src.data.scraper import detect_meme_signals

# Simulated hot items
fake_items = [
    {"rank": 1, "title": "躺平现象引发社会热议", "platform": "weibo", "hot_score": 9999},
    {"rank": 5, "title": "今日天气晴朗适合出游", "platform": "weibo", "hot_score": 100},
    {"rank": 2, "title": "后浪们开始反思消费主义", "platform": "weibo", "hot_score": 8888},
]
signals = detect_meme_signals(fake_items)
check("Detects 躺平", any(s["meme_name"] == "躺平" for s in signals))
check("Detects 后浪", any(s["meme_name"] == "后浪" for s in signals))
check("Ignores irrelevant items", not any(s["meme_name"] == "天气" for s in signals))
check("Correct count (2 signals)", len(signals) == 2, f"got {len(signals)}")

# ═══════════════════════════════════════════════
# 9. Data files exist and valid
# ═══════════════════════════════════════════════
print("\n── 9. Data file integrity ──")
files_to_check = [
    ("data/curated/memes_2020_2025.json", "curated dataset"),
    ("data/processed/trajectories.json", "trajectories"),
    ("data/processed/narratives/", "narrative dir"),
    ("data/scraped/", "scrape cache"),
]
for path, label in files_to_check:
    check(f"{label} exists", os.path.exists(path), path)

# Narrative JSON count
narr_dir = "data/processed/narratives"
narr_files = [f for f in os.listdir(narr_dir) if f.endswith(".json") and not f.startswith("_")]
check("22 narrative JSONs", len(narr_files) == 22, f"got {len(narr_files)}")

# ═══════════════════════════════════════════════
# 10. Constraint Field learns from data
# ═══════════════════════════════════════════════
print("\n── 10. Constraint Field learning ──")
try:
    dtm_learned = DeltaTransitionModel.learn_from_trajectories(trajs)
    check("Ridge model fitted", dtm_learned._ridge_coefs is not None)
    if dtm_learned._ridge_coefs is not None:
        # Test prediction works
        delta = dtm_learned.predict_delta(c0, "peak", "controversy")
        check("Learned delta prediction works", len(delta) == 5)
        check("Learned delta bounded", all(-0.4 <= d <= 0.4 for d in delta))
except Exception as e:
    check("Ridge learning runs without error", False, str(e))

# ═══════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Results: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
if FAIL == 0:
    print("VERDICT: All systems verified — correct and complete.")
else:
    print(f"VERDICT: {FAIL} checks failed — review above.")
print(f"{'='*60}")
