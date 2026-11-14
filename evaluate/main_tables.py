"""Tables 2 and 4: all verifiers on the same MedGemma outputs, scored against the blind annotations."""

PLUS = "outputs/tbv_cache.pkl"
CACHE = "outputs/test_cache.pkl"
EXTRA_CANDIDATES = ["outputs/baselines.pkl"]
SCAN_JSON = "data/scan_groups.json"
PX_TO_MM, PCOM, CUT, MARGIN, SIZE_TOL = 0.15, 12, 0.6, 0.1, 0.5
OTSU_EXPAND, P_YES, MIN_AREA, MIN_KEEP_IOU = 0.3, 0.5, 12, 0.1
N_BOOT, SEED = 1000, 0

import os, re, json, pickle
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

CR = re.compile(r"(\d+)\s+follicles?\s+identified\s+measuring\s+([\d.]+)-([\d.]+)\s*mm")
PERIPH = "Peripheral distribution of follicles noted."
CENTRAL = "Follicles are centrally/scattered distributed."
POS = "compatible with polycystic ovarian morphology"
PCOM_POS = ("Follicle number and size distribution are compatible with polycystic ovarian morphology (PCOM). "
            "Correlate with clinical and biochemical findings before diagnosing PCOS.")
PCOM_NEG = ("Findings do not meet follicle number criteria for polycystic ovarian morphology. "
            "If symptoms persist, correlate with endocrine evaluation and follow-up imaging as clinically indicated.")

# helpers
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0
def match(P, R, thr=0.5):
    if not P or not R:
        return np.array([0, len(P), len(R)]), []
    M = np.array([[iou(p, r) for r in R] for p in P]); ri, ci = linear_sum_assignment(-M)
    pairs = [(i, j) for i, j in zip(ri, ci) if M[i, j] >= thr]
    return np.array([len(pairs), len(P), len(R)]), pairs
def prf(a):
    tp, n1, n2 = a; p, r = tp / max(n1, 1), tp / max(n2, 1); return p, r, (2 * p * r / (p + r) if p + r else 0.0)
def diam(b): return ((b[2] - b[0]) + (b[3] - b[1])) / 2 * PX_TO_MM
def rad(B, W, H): return float(np.mean([np.hypot((b[0] + b[2]) / 2 - W / 2, (b[1] + b[3]) / 2 - H / 2) / (min(W, H) / 2) for b in B]))
def pat(B, W, H): return None if len(B) < 2 else ("peripheral" if rad(B, W, H) > CUT else "central")
def decisive(B, W, H): return len(B) >= 2 and abs(rad(B, W, H) - CUT) >= MARGIN
def apply_values(rep, n, lo, hi, p, pc):
    first = f"{n} follicles identified measuring {lo:.1f}-{hi:.1f} mm."
    if CR.search(rep) is None or (POS in rep) != pc:
        return " ".join([first, PERIPH if p == "peripheral" else CENTRAL, PCOM_POS]) if pc else " ".join([first, PCOM_NEG])
    t = CR.sub(first[:-1], rep, count=1)
    if p and (PERIPH in t or CENTRAL in t):
        t = t.replace(PERIPH, "@@").replace(CENTRAL, "@@").replace("@@", PERIPH if p == "peripheral" else CENTRAL)
    return t
def size_pattern(rep, lo, hi, p):  # size and pattern only
    m = CR.search(rep)
    if m is None:
        return rep
    t = CR.sub(f"{m.group(1)} follicles identified measuring {lo:.1f}-{hi:.1f} mm", rep, count=1)
    if p and (PERIPH in t or CENTRAL in t):
        t = t.replace(PERIPH, "@@").replace(CENTRAL, "@@").replace("@@", PERIPH if p == "peripheral" else CENTRAL)
    return t
def rewrite(rep, B, W, H, gate=True):  # gate=False rewrites every claim (ablation)
    if not B:
        return rep
    d = [diam(b) for b in B]; p = pat(B, W, H)
    if not gate:
        return apply_values(rep, len(B), min(d), max(d), p, len(B) >= PCOM)
    if not decisive(B, W, H):
        p = "peripheral" if PERIPH in rep else "central" if CENTRAL in rep else p
    return size_pattern(rep, min(d), max(d), p)
def otsu_thr(x):
    h, e = np.histogram(x, 256); h = h.astype(float); m = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * m) / np.maximum(w0, 1e-9); m1 = (np.sum(h * m) - np.cumsum(h * m)) / np.maximum(w1, 1e-9)
    return m[np.argmax(w0 * w1 * (m0 - m1) ** 2)]
def otsu_refine(gray, B, ex=OTSU_EXPAND):
    H, W = gray.shape; out = []
    for b in B:
        bw, bh = b[2] - b[0], b[3] - b[1]
        x1, y1 = int(max(0, b[0] - ex * bw)), int(max(0, b[1] - ex * bh))
        x2, y2 = int(min(W, b[2] + ex * bw + 1)), int(min(H, b[3] + ex * bh + 1))
        crop = ndimage.gaussian_filter(gray[y1:y2, x1:x2].astype(float), 1.5)
        if crop.size < 16 or crop.max() - crop.min() < 1e-3:
            out.append(list(b)); continue
        dark = crop < otsu_thr(crop); lab, n = ndimage.label(dark)
        if n == 0:
            out.append(list(b)); continue
        cy, cx = int((b[1] + b[3]) / 2) - y1, int((b[0] + b[2]) / 2) - x1
        k = lab[min(max(cy, 0), crop.shape[0] - 1), min(max(cx, 0), crop.shape[1] - 1)]
        if k == 0:
            k = int(np.argmax(ndimage.sum(dark, lab, range(1, n + 1)))) + 1
        ys, xs = np.where(lab == k)
        nb = [float(xs.min() + x1), float(ys.min() + y1), float(xs.max() + 1 + x1), float(ys.max() + 1 + y1)]
        out.append(nb if len(xs) >= MIN_AREA and iou(nb, b) >= MIN_KEEP_IOU else list(b))
    return out
def parse_judge(raw):
    s = str(raw); m = re.search(r"\{.*?\}", s, re.S) or re.search(r"\{.*", s, re.S)
    t = m.group(0) if m.group(0).rstrip().endswith("}") else m.group(0).rstrip().rstrip(",") + "}"
    j = json.loads(t.replace("True", "true").replace("False", "false"))
    lo, hi = sorted([float(j["min_mm"]), float(j["max_mm"])])
    pc = j["pcom"] if isinstance(j["pcom"], bool) else str(j["pcom"]).lower().startswith("t")
    return int(j["count"]), lo, hi, ("peripheral" if "periph" in str(j["pattern"]).lower() else "central"), pc

# load
cases = [c for c in pickle.load(open(CACHE, "rb")) if c.get("Hum")]
ex, used = {}, None
for f in EXTRA_CANDIDATES:
    if os.path.exists(f):
        e = pickle.load(open(f, "rb"))
        ok = [c["image"] in e and "P_sam2" in e[c["image"]] and len(e[c["image"]]["P_sam2"]) == len(c["P"]) for c in cases]
        if np.mean(ok) > 0.95:
            ex, used = e, f; break
print(f"{len(cases)} test images | Qwen/SAM2 outputs from: {used}")
assert ex, "outputs/baselines.pkl is missing or does not match these MedGemma outputs; run scripts/baselines/qwen_sam2.py"
for c in cases:
    e = ex[c["image"]]; g = np.asarray(Image.open(c["image"]).convert("L"), float)
    c["P_otsu"] = otsu_refine(g, c["P"]); c["P_sam2"] = e["P_sam2"]; c["P_qwen"] = e.get("P_qwen", c["P"])
    c["P_qkeep"] = [b for b, p in zip(c["P"], e.get("q_pyes", [1] * len(c["P"]))) if p >= P_YES]
    c["q_judge_raw"] = e.get("q_judge_raw")
assert os.path.exists(PLUS), "missing " + PLUS + ", run scripts/verify/tbv.py first"
plus = {c["image"]: c for c in pickle.load(open(PLUS, "rb"))}
for c in cases:
    c["P_tbv"] = plus[c["image"]]["P_ea0_keep"]
scan_of = json.load(open(SCAN_JSON))["scan_of"] if os.path.exists(SCAN_JSON) else {}
groups = {}
for i, c in enumerate(cases):
    groups.setdefault(scan_of.get(c["image"], f"solo{i}"), []).append(i)
G = list(groups.values()); rng = np.random.default_rng(SEED)
BOOT = [np.concatenate([G[g] for g in rng.integers(0, len(G), len(G))]) for _ in range(N_BOOT)]
def ci(fn): return np.percentile([fn(ix) for ix in BOOT], [2.5, 97.5])

# methods
def judge_text(c):
    try:
        n, lo, hi, p, pc = parse_judge(c["q_judge_raw"])
        return size_pattern(c["orig_report"], lo, hi, p)
    except Exception:
        return c["orig_report"]
BOX_METHODS = [
    ("Qwen2.5-VL check", "P_qkeep"), ("Qwen2.5-VL refine", "P_qwen"), ("Otsu", "P_otsu"),
    ("MedSAM", "P_off"), ("SAM2", "P_sam2"), ("TBV (ours)", "P_tbv")]
METHODS = [("No verification", lambda c: (c["P"], c["orig_report"])),
           ("Qwen2.5-VL judge", lambda c: (c["P"], judge_text(c)))] + \
          [(n, (lambda k: lambda c: (c[k], rewrite(c["orig_report"], c[k], c["W"], c["H"], True)))(k)) for n, k in BOX_METHODS]
ABLATION = [("MedGemma 1.5 (unverified)", lambda c: (c["P"], c["orig_report"])),
            ("+ correction from VLM boxes", lambda c: (c["P"], rewrite(c["orig_report"], c["P"], c["W"], c["H"], True))),
            ("+ correction, off-the-shelf MedSAM", lambda c: (c["P_off"], rewrite(c["orig_report"], c["P_off"], c["W"], c["H"], True))),
            ("+ correction, trained decoder (re-measure only)", lambda c: (c["P_fig0"], rewrite(c["orig_report"], c["P_fig0"], c["W"], c["H"], True))),
            ("TBV w/o claim-type rule (rewrite all claims)", lambda c: (c["P_tbv"], rewrite(c["orig_report"], c["P_tbv"], c["W"], c["H"], False))),
            ("TBV w/o correction (report unchanged)", lambda c: (c["P_tbv"], c["orig_report"])),
            ("TBV (full)", lambda c: (c["P_tbv"], rewrite(c["orig_report"], c["P_tbv"], c["W"], c["H"], True)))]

# metrics
def claims_of(t):
    m = CR.search(t)
    return (int(m.group(1)), float(m.group(2)), float(m.group(3)), "peripheral" if PERIPH in t else "central" if CENTRAL in t else None,
            POS in t) if m else None
GT = []
for c in cases:
    d = [diam(b) for b in c["Hum"]]
    GT.append((len(c["Hum"]), min(d), max(d), pat(c["Hum"], c["W"], c["H"]), len(c["Hum"]) >= PCOM, decisive(c["Hum"], c["W"], c["H"])))
def claim_scores(texts, ix):
    c1 = sz = pt = ptn = 0; rel = []; y, yh = [], []
    for i in ix:
        a, g = claims_of(texts[i]), GT[i]
        if a is None:
            continue
        c1 += abs(a[0] - g[0]) <= 1
        sz += abs(a[1] - g[1]) <= SIZE_TOL and abs(a[2] - g[2]) <= SIZE_TOL
        rel.append(50 * (abs(a[1] - g[1]) / g[1] + abs(a[2] - g[2]) / g[2]))
        if g[5] and a[3]:
            ptn += 1; pt += a[3] == g[3]
        y.append(a[4]); yh.append(g[4])
    y, yh = np.array(y), np.array(yh); n = len(ix)
    se = (y & yh).sum() / max(yh.sum(), 1); sp = (~y & ~yh).sum() / max((~yh).sum(), 1)
    return {"cnt": 100 * c1 / n, "sz": 100 * sz / n, "rel": np.mean(rel), "pat": 100 * pt / max(ptn, 1), "crit": (se + sp) / 2}

refs = [c["ref_report"] for c in cases]
try:
    from nltk.translate.bleu_score import corpus_bleu
except ImportError:
    corpus_bleu = None; print("pip install nltk for BLEU")
try:
    from rouge_score import rouge_scorer; RS = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
except ImportError:
    RS = None; print("pip install rouge-score for ROUGE-L")
try:
    from bert_score import score as bertscore
except ImportError:
    bertscore = None; print("pip install bert_score for BERTScore")
TOK = re.compile(r"\d+(?:\.\d+)?|\w+|[^\w\s]")  # numbers, words and punctuation as separate tokens
def tok(s): return TOK.findall(s)
def bleu(T, ix=None):
    ix = range(len(T)) if ix is None else ix
    return corpus_bleu([[tok(refs[i])] for i in ix], [tok(T[i]) for i in ix]) if corpus_bleu else np.nan

# diameters are compared on cited boxes that overlap a ground-truth follicle (IoU >= 0.1)
FIX = []
for ci_, c in enumerate(cases):
    _, pairs = match(c["P"], c["Hum"], 0.1)
    FIX += [(ci_, i, j) for i, j in pairs]
print(f"diameter set: {len(FIX)} cited follicles that correspond to a ground-truth follicle")
def diam_err(key):
    if key == "written":
        e = [abs(cases[ci_]["listed_d"][i] - diam(cases[ci_]["Hum"][j])) for ci_, i, j in FIX if i < len(cases[ci_].get("listed_d", []))]
        r = [abs(cases[ci_]["listed_d"][i] - diam(cases[ci_]["Hum"][j])) / diam(cases[ci_]["Hum"][j]) for ci_, i, j in FIX if i < len(cases[ci_].get("listed_d", []))]
    else:
        e, r = [], []
        for ci_, i, j in FIX:
            c = cases[ci_]
            if key == "P_qkeep":  # kept boxes keep the VLM diameter
                b = c["P"][i]
            else:
                b = c[key][i]
            g = diam(c["Hum"][j]); e.append(abs(diam(b) - g)); r.append(abs(diam(b) - g) / g)
    return np.mean(e), 100 * np.mean(np.array(e) > 1), 100 * np.median(r)

def run(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'Method':45s} {'F1':>5s} {'cnt+-1':>7s} {'sz.5mm':>7s} {'sz rel':>7s} {'pat':>5s} {'crit':>5s} {'BLEU':>6s} {'R-L':>6s} {'BERT':>6s}")
    out = {}
    for name, fn in rows:
        items = [fn(c) for c in cases]; B = [b for b, _ in items]; T = [t for _, t in items]
        F = prf(sum(match(b, c["Hum"])[0] for b, c in zip(B, cases)))[2]
        cs = claim_scores(T, range(len(cases)))
        bl = bleu(T); rl = np.mean([RS.score(r, t)["rougeL"].fmeasure for r, t in zip(refs, T)]) if RS else np.nan
        bs = bertscore(T, refs, lang="en", verbose=False)[2].mean().item() if bertscore else np.nan
        out[name] = dict(F=F, bl=bl, rl=rl, bs=bs, T=T, B=B, **cs)
        print(f"{name:45s} {F:5.2f} {cs['cnt']:6.0f}% {cs['sz']:6.0f}% {cs['rel']:6.0f}% {cs['pat']:4.0f}% {cs['crit']:5.2f} {bl:6.3f} {rl:6.3f} {bs:6.3f}")
    return out
A = run(METHODS, "TABLE A (main): every method + the same correction step, 100 test images vs ground truth")
C = run(ABLATION, "TABLE C (ablation)")

print("\n=== TABLE B (evidence): precision, recall, F1 (95% CI), diameter error on the fixed set ===")
DKEY = {"No verification": "written", "Qwen2.5-VL check": "P_qkeep", "Qwen2.5-VL refine": "P_qwen", "Otsu": "P_otsu",
        "MedSAM": "P_off", "SAM2": "P_sam2", "TBV (ours)": "P_tbv"}
Bt = {}
for name, key in DKEY.items():
    k = "P" if key == "written" else key
    M = [match(c[k], c["Hum"])[0] for c in cases]
    p, r, f = prf(sum(M)); lo, hi = ci(lambda ix: prf(sum(M[i] for i in ix))[2])
    # TBV drops boxes, so its diameter error is computed on its kept follicles in verify/tbv.py
    mae, gt1, rel = diam_err(key) if key != "P_tbv" else (np.nan, np.nan, np.nan); Bt[name] = (p, r, f, lo, hi, mae, gt1, rel)
    print(f"{name:20s} P {p:.2f} R {r:.2f} F1 {f:.2f} [{lo:.2f}, {hi:.2f}]  MAE {mae:.2f} mm  >1mm {gt1:.0f}%  rel {rel:.0f}%")
print(f"   (VLM's own boxes, for the text: MAE {diam_err('P')[0]:.2f} mm, >1mm {diam_err('P')[1]:.0f}%)")
Mtbv = [match(c["P_tbv"], c["Hum"])[0] for c in cases]
for name, key in DKEY.items():
    if name == "TBV (ours)":
        continue
    k = "P" if key == "written" else key; M = [match(c[k], c["Hum"])[0] for c in cases]
    lo, hi = ci(lambda ix: prf(sum(Mtbv[i] for i in ix))[2] - prf(sum(M[i] for i in ix))[2])
    print(f"   TBV minus {name:20s}: F1 diff 95% CI [{lo:.2f}, {hi:.2f}]")
if corpus_bleu:
    lo, hi = ci(lambda ix: bleu(A["TBV (ours)"]["T"], ix) - bleu(A["No verification"]["T"], ix))
    print(f"   BLEU TBV minus no verification: 95% CI [{lo:.3f}, {hi:.3f}]")

# LaTeX rows
def best(vals, hi=True):
    v = [x for x in vals if not np.isnan(x)]
    return (max(v) if hi else min(v)) if v else np.nan
print("\n%%% LaTeX rows, Table A (main)")
cols = [("F", True, "{:.2f}"), ("cnt", True, "{:.0f}\\%"), ("sz", True, "{:.0f}\\%"), ("rel", False, "{:.0f}\\%"),
        ("pat", True, "{:.0f}\\%"), ("crit", True, "{:.2f}"), ("bl", True, "{:.3f}"), ("rl", True, "{:.3f}"), ("bs", True, "{:.3f}")]
bests = {k: best([A[n][k] for n in A], h) for k, h, _ in cols}
for n in A:
    cells = []
    for k, h, fmt in cols:
        s = fmt.format(A[n][k]) if not np.isnan(A[n][k]) else "n/a"
        cells.append(f"\\textbf{{{s}}}" if not np.isnan(A[n][k]) and round(A[n][k], 3) == round(bests[k], 3) else s)
    print(f"{n} & " + " & ".join(cells) + " \\\\")
print("\n%%% LaTeX rows, Table B (evidence)")
for n, (p, r, f, lo, hi, mae, gt1, rel) in Bt.items():
    print(f"{n} & {p:.2f} & {r:.2f} & {f:.2f} & {mae:.2f} & {gt1:.0f}\\% & {rel:.0f}\\% \\\\")
print("\n%%% LaTeX rows, Table C (ablation)")
for n in C:
    print(f"{n} & {C[n]['F']:.2f} & {C[n]['sz']:.0f}\\% & {C[n]['rel']:.0f}\\% & {C[n]['pat']:.0f}\\% & {C[n]['bl']:.3f} \\\\")
