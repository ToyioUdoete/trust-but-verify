"""Table 4, count correction: flagged reports take their count and conclusion from the counter."""

CACHE, PLUS, DETX = "outputs/test_cache.pkl", "outputs/tbv_cache.pkl", "outputs/counter_boxes.pkl"
SCAN_JSON, N_BOOT = "data/scan_groups.json", 1000
PX_TO_MM, PCOM, CUT, MARGIN, KAPPA, CONF, SIZE_TOL = 0.15, 12, 0.6, 0.1, 3, 0.40, 0.5

import os, re, json, pickle
import numpy as np

CR = re.compile(r"(\d+)\s+follicles?\s+identified\s+measuring\s+([\d.]+)-([\d.]+)\s*mm")
PERIPH = "Peripheral distribution of follicles noted."; CENTRAL = "Follicles are centrally/scattered distributed."
POS = "compatible with polycystic ovarian morphology"
PCOM_POS = ("Follicle number and size distribution are compatible with polycystic ovarian morphology (PCOM). "
            "Correlate with clinical and biochemical findings before diagnosing PCOS.")
PCOM_NEG = ("Findings do not meet follicle number criteria for polycystic ovarian morphology. "
            "If symptoms persist, correlate with endocrine evaluation and follow-up imaging as clinically indicated.")
TOK = re.compile(r"\d+(?:\.\d+)?|\w+|[^\w\s]")
def tok(s): return TOK.findall(s)
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
def tbv(rep, B, W, H, Kd=None):  # with Kd, flagged reports take the counter's count
    m = CR.search(rep)
    if not B or m is None:
        return rep
    d = [diam(b) for b in B]; p = pat(B, W, H)
    if not decisive(B, W, H):
        p = "peripheral" if PERIPH in rep else "central" if CENTRAL in rep else p
    K = int(m.group(1))
    n = Kd if (Kd is not None and abs(K - Kd) > KAPPA) else K
    return apply_values(rep, n, min(d), max(d), p, (n >= PCOM) if n != K else (POS in rep))

cases = [c for c in pickle.load(open(CACHE, "rb")) if c.get("Hum")]
plus = {c["image"]: c for c in pickle.load(open(PLUS, "rb"))}
det = pickle.load(open(DETX, "rb"))["boxes"]
for c in cases:
    c["P_tbv"] = plus[c["image"]]["P_ea0_keep"]
    c["Kd"] = int(sum(b[4] >= CONF for b in det[c["image"]]["yolo_fig"]))

V = {"MedGemma 1.5 (unverified)": [c["orig_report"] for c in cases],
     "TBV": [tbv(c["orig_report"], c["P_tbv"], c["W"], c["H"]) for c in cases],
     "TBV + count correction": [tbv(c["orig_report"], c["P_tbv"], c["W"], c["H"], c["Kd"]) for c in cases]}
flagged = sum(abs(int(CR.search(c["orig_report"]).group(1)) - c["Kd"]) > KAPPA for c in cases if CR.search(c["orig_report"]))
print(f"flagged reports (count taken from the counter): {flagged} of {len(cases)}")

# claims vs ground truth
def claims(T, ix):
    cnt, sz, crit_p, crit_t = [], [], [], []
    for i in ix:
        m = CR.search(T[i]); c = cases[i]; g = [diam(b) for b in c["Hum"]]
        if m is None:
            continue
        cnt.append(abs(int(m.group(1)) - len(c["Hum"])) <= 1)
        sz.append(abs(float(m.group(2)) - min(g)) <= SIZE_TOL and abs(float(m.group(3)) - max(g)) <= SIZE_TOL)
        crit_p.append(POS in T[i]); crit_t.append(len(c["Hum"]) >= PCOM)
    p, t = np.array(crit_p), np.array(crit_t)
    bal = ((p & t).sum() / max(t.sum(), 1) + (~p & ~t).sum() / max((~t).sum(), 1)) / 2
    return 100 * np.mean(cnt), 100 * np.mean(sz), bal

# text vs reference reports
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score as bertscore
RS = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True); refs = [c["ref_report"] for c in cases]
def text(T, ix=None):
    ix = range(len(T)) if ix is None else ix
    return (corpus_bleu([[tok(refs[i])] for i in ix], [tok(T[i]) for i in ix]),
            np.mean([RS.score(refs[i], T[i])["rougeL"].fmeasure for i in ix]))
BS = {n: bertscore(T, refs, lang="en", verbose=False)[2].numpy() for n, T in V.items()}

scan_of = json.load(open(SCAN_JSON))["scan_of"] if os.path.exists(SCAN_JSON) else {}
groups = {}
for i, c in enumerate(cases):
    groups.setdefault(scan_of.get(c["image"], f"solo{i}"), []).append(i)
G = list(groups.values()); rng = np.random.default_rng(0)
BOOT = [np.concatenate([G[g] for g in rng.integers(0, len(G), len(G))]) for _ in range(N_BOOT)]

print(f"\n{'Method':28s} {'count+-1':>8s} {'size.5':>7s} {'crit':>5s} | {'BLEU':>6s} {'R-L':>6s} {'BS':>6s}")
for n, T in V.items():
    c1, sz, cr = claims(T, range(len(T))); bl, rl = text(T)
    print(f"{n:28s} {c1:7.0f}% {sz:6.0f}% {cr:5.2f} | {bl:6.3f} {rl:6.3f} {BS[n].mean():6.3f}")
print("  (for reference, MedSAM in Table 1: BLEU 0.776, ROUGE-L 0.835, BERTScore 0.984; best BERTScore 0.985)")

print("\n95% CIs, TBV + count correction minus TBV:")
A, B = V["TBV + count correction"], V["TBV"]
for name, fn in [("BLEU", lambda ix: text(A, ix)[0] - text(B, ix)[0]), ("ROUGE-L", lambda ix: text(A, ix)[1] - text(B, ix)[1]),
                 ("BERTScore", lambda ix: BS["TBV + count correction"][ix].mean() - BS["TBV"][ix].mean()),
                 ("criterion bal.acc", lambda ix: claims(A, ix)[2] - claims(B, ix)[2]),
                 ("count within 1 (%)", lambda ix: claims(A, ix)[0] - claims(B, ix)[0])]:
    lo, hi = np.percentile([fn(ix) for ix in BOOT], [2.5, 97.5]); print(f"  {name:20s} [{lo:+.3f}, {hi:+.3f}]")
