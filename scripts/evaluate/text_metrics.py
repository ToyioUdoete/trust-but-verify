"""Table 1: text metrics against the reference reports and the fact-corrected references."""

CACHE = "outputs/test_cache.pkl"
PLUS = "outputs/tbv_cache.pkl"
DETX = "outputs/counter_boxes.pkl"
EXTRA_CANDIDATES = ["outputs/baselines.pkl"]
VLM_CSV = {"PaliGemma 2": "outputs/second_vlms/pred_paligemma2.csv", "LLaVA-Med": "outputs/second_vlms/pred_llavamed.csv"}
SCAN_JSON, N_BOOT = "data/scan_groups.json", 1000
PX_TO_MM, PCOM, CUT, MARGIN, KAPPA, CONF = 0.15, 12, 0.6, 0.1, 3, 0.40
OTSU_EXPAND, MIN_AREA, MIN_KEEP_IOU, MATCH_IOU, GRID = 0.3, 12, 0.1, 0.3, 99

import os, re, json, pickle
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

EV = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*([\d.]+)\s*mm")
CR = re.compile(r"(\d+)\s+follicles?\s+identified\s+measuring\s+([\d.]+)-([\d.]+)\s*mm")
PERIPH = "Peripheral distribution of follicles noted."; CENTRAL = "Follicles are centrally/scattered distributed."
POS = "compatible with polycystic ovarian morphology"
PCOM_POS = ("Follicle number and size distribution are compatible with polycystic ovarian morphology (PCOM). "
            "Correlate with clinical and biochemical findings before diagnosing PCOS.")
PCOM_NEG = ("Findings do not meet follicle number criteria for polycystic ovarian morphology. "
            "If symptoms persist, correlate with endocrine evaluation and follow-up imaging as clinically indicated.")
TOK = re.compile(r"\d+(?:\.\d+)?|\w+|[^\w\s]")
def tok(s): return TOK.findall(s)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0
def match(P, R, thr):
    if not P or not R:
        return []
    M = np.array([[iou(p, r) for r in R] for p in P]); ri, ci = linear_sum_assignment(-M)
    return [(i, j) for i, j in zip(ri, ci) if M[i, j] >= thr]
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
    return apply_values(rep, int(m.group(1)), lo, hi, p, POS in rep)
def correct(rep, B, W, H):
    if not B or CR.search(rep) is None:
        return rep
    d = [diam(b) for b in B]; p = pat(B, W, H)
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
    return lo, hi, ("peripheral" if "periph" in str(j["pattern"]).lower() else "central")

# evidence of every method
cases = [c for c in pickle.load(open(CACHE, "rb")) if c.get("Hum")]
ex = None
for f in EXTRA_CANDIDATES:
    if os.path.exists(f):
        e = pickle.load(open(f, "rb"))
        if np.mean([c["image"] in e and len(e[c["image"]].get("P_sam2", [])) == len(c["P"]) for c in cases]) > 0.95:
            ex = e; break
assert ex is not None, "Qwen/SAM2 file not found"
plus = {c["image"]: c for c in pickle.load(open(PLUS, "rb"))}
det = pickle.load(open(DETX, "rb"))["boxes"]
gen = {n: pd.read_csv(p).drop_duplicates("image").set_index("image")["generated"].astype(str).to_dict() for n, p in VLM_CSV.items()}
for c in cases:
    e = ex[c["image"]]; W, H = c["W"], c["H"]
    c["P_otsu"] = otsu_refine(np.asarray(Image.open(c["image"]).convert("L"), float), c["P"])
    c["P_sam2"], c["P_qwen"], c["P_tbv"] = e["P_sam2"], e.get("P_qwen", c["P"]), plus[c["image"]]["P_ea0_keep"]
    for n in VLM_CSV:
        V = [[int(a) / GRID * W, int(b) / GRID * H, int(cc) / GRID * W, int(d) / GRID * H]
             for a, b, cc, d, _ in EV.findall(gen[n].get(c["image"], "").partition("Report:")[0])]
        mp = dict(match(c["P"], V, MATCH_IOU))
        c["P_" + n] = [V[mp[i]] if i in mp else list(b) for i, b in enumerate(c["P"])]
    c["Kd"] = int(sum(b[4] >= CONF for b in det[c["image"]]["yolo_fig"]))
def judge(c):
    try:
        return size_pattern(c["orig_report"], *parse_judge(ex[c["image"]].get("q_judge_raw")))
    except Exception:
        return c["orig_report"]
def tbv_count(c):
    t = correct(c["orig_report"], c["P_tbv"], c["W"], c["H"]); m = CR.search(t)
    if m is None or abs(int(m.group(1)) - c["Kd"]) <= KAPPA:
        return t
    p = PERIPH in t and "peripheral" or "central"
    return apply_values(t, c["Kd"], float(m.group(2)), float(m.group(3)), p, c["Kd"] >= PCOM)
box = lambda k: (lambda c: correct(c["orig_report"], c[k], c["W"], c["H"]))
METHODS = {"MedGemma 1.5 (unverified)": lambda c: c["orig_report"], "Qwen2.5-VL judge": judge,
           "Qwen2.5-VL refine": box("P_qwen"), "PaliGemma 2 refine": box("P_PaliGemma 2"), "LLaVA-Med refine": box("P_LLaVA-Med"),
           "Otsu": box("P_otsu"), "MedSAM": box("P_off"), "SAM2": box("P_sam2"), "TBV (ours)": box("P_tbv"),
           "TBV + count correction": tbv_count}
T = {n: [f(c) for c in cases] for n, f in METHODS.items()}

# references
REF = {"clinician-approved reference": [c["ref_report"] for c in cases]}
gt_ref = []
for c in cases:
    d = [diam(b) for b in c["Hum"]]; n = len(c["Hum"])
    gt_ref.append(apply_values(c["ref_report"], n, min(d), max(d), pat(c["Hum"], c["W"], c["H"]) or "central", n >= PCOM))
REF["ground-truth reference (facts from blind annotation)"] = gt_ref
print("example  clinician ref :", cases[0]["ref_report"][:160])
print("example  ground-truth  :", gt_ref[0][:160])

from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score as bertscore
RS = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
scan_of = json.load(open(SCAN_JSON))["scan_of"] if os.path.exists(SCAN_JSON) else {}
groups = {}
for i, c in enumerate(cases):
    groups.setdefault(scan_of.get(c["image"], f"solo{i}"), []).append(i)
G = list(groups.values()); rng = np.random.default_rng(0)
BOOT = [np.concatenate([G[g] for g in rng.integers(0, len(G), len(G))]) for _ in range(N_BOOT)]

for rname, refs in REF.items():
    print(f"\n=== Text metrics against the {rname} ===")
    S = {}
    for n, X in T.items():
        rl = np.array([RS.score(r, x)["rougeL"].fmeasure for r, x in zip(refs, X)])
        bs = bertscore(X, refs, lang="en", verbose=False)[2].numpy()
        S[n] = (rl, bs)
        print(f"{n:28s} BLEU {corpus_bleu([[tok(r)] for r in refs], [tok(x) for x in X]):.3f}  ROUGE-L {rl.mean():.3f}  BERTScore {bs.mean():.3f}")
    others = [n for n in T if n not in ("TBV (ours)", "TBV + count correction")]
    for metric, k in [("ROUGE-L", 0), ("BERTScore", 1)]:
        best = max(others, key=lambda n: S[n][k].mean())
        lo, hi = np.percentile([S["TBV (ours)"][k][ix].mean() - S[best][k][ix].mean() for ix in BOOT], [2.5, 97.5])
        print(f"  TBV minus best other on {metric} ({best}): 95% CI [{lo:+.3f}, {hi:+.3f}]")
    bl = lambda n, ix: corpus_bleu([[tok(refs[i])] for i in ix], [tok(T[n][i]) for i in ix])
    best = max(others, key=lambda n: bl(n, range(len(cases))))
    lo, hi = np.percentile([bl("TBV (ours)", ix) - bl(best, ix) for ix in BOOT[:300]], [2.5, 97.5])
    print(f"  TBV minus best other on BLEU ({best}): 95% CI [{lo:+.3f}, {hi:+.3f}]  (300 resamples)")
