"""Table 5: robustness of the evidence gain to how the ground truth was drawn."""

CACHE, PLUS, SCAN_JSON = "outputs/test_cache.pkl", "outputs/tbv_cache.pkl", "data/scan_groups.json"
PX_TO_MM, CUT, MARGIN, N_BOOT = 0.15, 0.6, 0.1, 1000
PERIPH = "Peripheral distribution of follicles noted."; CENTRAL = "Follicles are centrally/scattered distributed."

import os, json, pickle
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0
def counts(P, R, thr=0.5):
    if not P or not R:
        return np.array([0, len(P), len(R)])
    M = np.array([[iou(p, r) for r in R] for p in P]); ri, ci = linear_sum_assignment(-M)
    return np.array([sum(M[i, j] >= thr for i, j in zip(ri, ci)), len(P), len(R)])
def f1(a):
    tp, n1, n2 = a; p, r = tp / max(n1, 1), tp / max(n2, 1); return 2 * p * r / (p + r) if p + r else 0.0
def rad(B, W, H): return float(np.mean([np.hypot((b[0] + b[2]) / 2 - W / 2, (b[1] + b[3]) / 2 - H / 2) / (min(W, H) / 2) for b in B]))
def decisive(B, W, H): return len(B) >= 2 and abs(rad(B, W, H) - CUT) >= MARGIN

cases = [c for c in pickle.load(open(CACHE, "rb")) if c.get("Hum")]
plus = {c["image"]: c for c in pickle.load(open(PLUS, "rb"))}
for c in cases:
    c["P_tbv"] = plus[c["image"]]["P_ea0_keep"]
scan_of = json.load(open(SCAN_JSON))["scan_of"] if os.path.exists(SCAN_JSON) else {}
groups = {}
for i, c in enumerate(cases):
    groups.setdefault(scan_of.get(c["image"], f"solo{i}"), []).append(i)
G = list(groups.values()); rng = np.random.default_rng(0)
BOOT = [np.concatenate([G[g] for g in rng.integers(0, len(G), len(G))]) for _ in range(N_BOOT)]

def row(R_of, thr=0.5):
    A = {k: [counts(c[k], R_of(c), thr) for c in cases] for k in ("P", "P_tbv")}
    f0, f1_ = f1(sum(A["P"])), f1(sum(A["P_tbv"]))
    lo, hi = np.percentile([f1(sum(A["P_tbv"][i] for i in ix)) - f1(sum(A["P"][i] for i in ix)) for ix in BOOT], [2.5, 97.5])
    return f0, f1_, lo, hi
def jitter(B, s, r):
    out = []
    for b in B:
        w, h = b[2] - b[0], b[3] - b[1]; cx, cy = (b[0] + b[2]) / 2 + r.uniform(-s, s) * w, (b[1] + b[3]) / 2 + r.uniform(-s, s) * h
        w *= 1 + r.uniform(-s, s); h *= 1 + r.uniform(-s, s); out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return out

print("=== Robustness of the evidence result (unverified MedGemma vs final TBV) ===")
rows = [("IoU $\\geq$ 0.3",) + row(lambda c: c["Hum"], 0.3)]
v = [row(lambda c, J={c["image"]: jitter(c["Hum"], 0.2, np.random.default_rng(100 + sd)) for c in cases}: J[c["image"]])
     for sd in range(5)]
rows.append(("Boxes perturbed by 20\\%",) + tuple(np.mean([x[k] for x in v]) for k in range(4)))
small = np.percentile([min(b[2] - b[0], b[3] - b[1]) for c in cases for b in c["Hum"]], 10)
rows.append(("Smallest 10\\% removed",) + row(lambda c: [b for b in c["Hum"] if min(b[2] - b[0], b[3] - b[1]) > small]))
for n, a, b, lo, hi in rows:
    print(f"{n:28s} unverified {a:.2f} | TBV {b:.2f} | gain 95% CI [{lo:+.2f}, {hi:+.2f}]")

print("\n=== Per near-duplicate group ===")
better = sum(f1(sum(counts(cases[i]["P_tbv"], cases[i]["Hum"]) for i in g)) > f1(sum(counts(cases[i]["P"], cases[i]["Hum"]) for i in g)) for g in G)
print(f"TBV raises F1 in {better} of {len(G)} groups")

print("\n=== Pattern claim (rewritten only on decisive evidence) ===")
elig = [c for c in cases if decisive(c["P_tbv"], c["W"], c["H"])]
changed = 0
for c in elig:
    new = "peripheral" if rad(c["P_tbv"], c["W"], c["H"]) > CUT else "central"
    old = "peripheral" if PERIPH in c["orig_report"] else "central" if CENTRAL in c["orig_report"] else None
    changed += old is not None and new != old
print(f"pattern rewritten in {changed} of {len(elig)} reports with decisive evidence")
print(f"follicles kept by TBV: {sum(len(c['P_tbv']) for c in cases)} of {sum(len(c['P']) for c in cases)} cited")

print("\n%%% LaTeX rows (robustness, bottom of the ablation table)")
for n, a, b, lo, hi in rows:
    print(f"{n} & \\multicolumn{{2}}{{c}}{{{a:.2f}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{{b:.2f}}}}} \\\\")
