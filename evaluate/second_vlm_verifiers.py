"""Table 2: PaliGemma 2 and LLaVA-Med as verifiers of MedGemma (keep/drop and refine)."""

CACHE = "outputs/test_cache.pkl"
PLUS = "outputs/tbv_cache.pkl"
VERIFIERS = {"PaliGemma 2": "outputs/second_vlms/pred_paligemma2.csv", "LLaVA-Med": "outputs/second_vlms/pred_llavamed.csv"}
SCAN_JSON = "data/scan_groups.json"
MATCH_IOU, GRID, PX_TO_MM, CUT, MARGIN, SIZE_TOL, N_BOOT = 0.3, 99, 0.15, 0.6, 0.1, 0.5, 1000

import os, re, json, pickle
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

EV = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*([\d.]+)\s*mm")
CR = re.compile(r"(\d+)\s+follicles?\s+identified\s+measuring\s+([\d.]+)-([\d.]+)\s*mm")
PERIPH = "Peripheral distribution of follicles noted."; CENTRAL = "Follicles are centrally/scattered distributed."

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0
def match(P, R, thr=0.5):
    if not P or not R:
        return np.array([0, len(P), len(R)]), []
    M = np.array([[iou(p, r) for r in R] for p in P]); ri, ci = linear_sum_assignment(-M)
    pr = [(i, j) for i, j in zip(ri, ci) if M[i, j] >= thr]; return np.array([len(pr), len(P), len(R)]), pr
def prf(a):
    tp, n1, n2 = a; p, r = tp / max(n1, 1), tp / max(n2, 1); return p, r, (2 * p * r / (p + r) if p + r else 0.0)
def diam(b): return ((b[2] - b[0]) + (b[3] - b[1])) / 2 * PX_TO_MM
def rad(B, W, H): return float(np.mean([np.hypot((b[0] + b[2]) / 2 - W / 2, (b[1] + b[3]) / 2 - H / 2) / (min(W, H) / 2) for b in B]))
def pat(B, W, H): return None if len(B) < 2 else ("peripheral" if rad(B, W, H) > CUT else "central")
def decisive(B, W, H): return len(B) >= 2 and abs(rad(B, W, H) - CUT) >= MARGIN
def rewrite(rep, B, W, H):  # size and pattern only
    m = CR.search(rep)
    if not B or m is None:
        return rep
    d = [diam(b) for b in B]; p = pat(B, W, H)
    if not decisive(B, W, H):
        p = "peripheral" if PERIPH in rep else "central" if CENTRAL in rep else p
    t = CR.sub(f"{m.group(1)} follicles identified measuring {min(d):.1f}-{max(d):.1f} mm", rep, count=1)
    if p and (PERIPH in t or CENTRAL in t):
        t = t.replace(PERIPH, "@@").replace(CENTRAL, "@@").replace("@@", PERIPH if p == "peripheral" else CENTRAL)
    return t

cases = [c for c in pickle.load(open(CACHE, "rb")) if c.get("Hum")]
plus = {c["image"]: c for c in pickle.load(open(PLUS, "rb"))}
for c in cases:
    c["P_tbv"] = plus[c["image"]]["P_ea0_keep"]
    c["P_ea0"] = plus[c["image"]]["P_ea0"]
    c["kept_tbv"] = {i for i, b in enumerate(c["P_ea0"]) if b in c["P_tbv"]}

# own citations, keep/drop and refine
ROWS = []
for vname, path in VERIFIERS.items():
    gen = pd.read_csv(path).drop_duplicates("image").set_index("image")["generated"].astype(str).to_dict()
    tag = vname.split()[0].lower().replace("-", "")
    for c in cases:
        W, H = c["W"], c["H"]
        V = [[int(a) / GRID * W, int(b) / GRID * H, int(cc) / GRID * W, int(d) / GRID * H]
             for a, b, cc, d, _ in EV.findall(gen.get(c["image"], "").partition("Report:")[0])]
        _, pairs = match(c["P"], V, MATCH_IOU); mp = dict(pairs)
        c[f"P_{tag}_keep"] = [b for i, b in enumerate(c["P"]) if i in mp]
        c[f"P_{tag}_ref"] = [V[mp[i]] if i in mp else list(b) for i, b in enumerate(c["P"])]
        c[f"keepidx_{tag}"] = set(mp)
    ROWS += [(f"{vname} keep/drop", f"P_{tag}_keep", "keep"), (f"{vname} refine", f"P_{tag}_ref", "ref")]
    print(f"{vname}: confirms {sum(len(c[f'keepidx_{tag}']) for c in cases)} of {sum(len(c['P']) for c in cases)} MedGemma follicles")

# metrics
FIX = []
for k, c in enumerate(cases):
    _, pairs = match(c["P"], c["Hum"], 0.1); FIX += [(k, i, j) for i, j in pairs]
def diam_err(key, mode):
    e, r = [], []
    for k, i, j in FIX:
        c = cases[k]; g = diam(c["Hum"][j])
        if mode == "tbv":  # only the follicles TBV keeps
            if i not in c["kept_tbv"]:
                continue
            d = diam(c["P_ea0"][i])
        elif mode == "keep":
            d = diam(c["P"][i])
        else:
            d = diam(c[key][i])
        e.append(abs(d - g)); r.append(abs(d - g) / g)
    return np.mean(e), 100 * np.mean(np.array(e) > 1), 100 * np.median(r)
def size_scores(T):
    ok, rel = [], []
    for t, c in zip(T, cases):
        m = CR.search(t); g = [diam(b) for b in c["Hum"]]
        if m is None:
            ok.append(False); continue
        lo, hi = float(m.group(2)), float(m.group(3))
        ok.append(abs(lo - min(g)) <= SIZE_TOL and abs(hi - max(g)) <= SIZE_TOL)
        rel.append(50 * (abs(lo - min(g)) / min(g) + abs(hi - max(g)) / max(g)))
    return 100 * np.mean(ok), np.mean(rel)
from nltk.translate.bleu_score import corpus_bleu
TOK = re.compile(r"\d+(?:\.\d+)?|\w+|[^\w\s]")  # numbers, words and punctuation as separate tokens
def tok(s): return TOK.findall(s)
from rouge_score import rouge_scorer
from bert_score import score as bertscore
RS = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True); refs = [c["ref_report"] for c in cases]
scan_of = json.load(open(SCAN_JSON))["scan_of"] if os.path.exists(SCAN_JSON) else {}
groups = {}
for k, c in enumerate(cases):
    groups.setdefault(scan_of.get(c["image"], f"solo{k}"), []).append(k)
G = list(groups.values()); rng = np.random.default_rng(0)
BOOT = [np.concatenate([G[g] for g in rng.integers(0, len(G), len(G))]) for _ in range(N_BOOT)]

Mtbv = [match(c["P_tbv"], c["Hum"])[0] for c in cases]
print(f"\n{'Method':28s} {'P':>5s} {'R':>5s} {'F1':>5s} {'MAE':>5s} {'>1mm':>5s} {'sz.5':>5s} {'rel':>5s} {'BLEU':>6s} {'R-L':>6s} {'BS':>6s}   TBV minus this (F1, 95% CI)")
latex = []
for name, key, mode in ROWS + [("TBV (ours)", "P_tbv", "tbv")]:
    B = [c[key] for c in cases]; T = [rewrite(c["orig_report"], b, c["W"], c["H"]) for c, b in zip(cases, B)]
    M = [match(b, c["Hum"])[0] for b, c in zip(B, cases)]; p, r, f = prf(sum(M))
    mae, gt1, _ = diam_err(key, mode)
    sz, rel = size_scores(T)
    bl = corpus_bleu([[tok(x)] for x in refs], [tok(t) for t in T])
    rl = np.mean([RS.score(a, b)["rougeL"].fmeasure for a, b in zip(refs, T)])
    bs = bertscore(T, refs, lang="en", verbose=False)[2].mean().item()
    other = mode != "tbv"
    lo, hi = np.percentile([prf(sum(Mtbv[i] for i in ix))[2] - prf(sum(M[i] for i in ix))[2] for ix in BOOT], [2.5, 97.5]) if other else (0, 0)
    print(f"{name:28s} {p:5.2f} {r:5.2f} {f:5.2f} {mae:5.2f} {gt1:4.0f}% {sz:4.0f}% {rel:4.0f}% {bl:6.3f} {rl:6.3f} {bs:6.3f}" +
          (f"   [{lo:+.2f}, {hi:+.2f}]" if other else ""))
    latex.append(f"{name} & {p:.2f} & {r:.2f} & {f:.2f} & {mae:.2f} & {gt1:.0f}\\% & {sz:.0f}\\% & {rel:.0f}\\% & {bl:.3f} & {rl:.3f} & {bs:.3f} \\\\")
print("\n%%% LaTeX rows for Table 2 (same columns)")
print("\n".join(latex))
