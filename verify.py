"""TBV: re-measure MedGemma's citations, reject low-IoU ones and recompute the claims (Table 4)."""

TEST_PKL = "outputs/test_cache.pkl"
OUT_CACHE = "outputs/tbv_cache.pkl"
SCAN_JSON = "data/scan_groups.json"
MEDSAM_ID = "wanglab/medsam-vit-base"
OLD_DECODERS = "checkpoints/decoder_seed{}.pt"
EA_DECODERS = "checkpoints/decoder_error_aware_seed{}.pt"
THRESHOLDS = "checkpoints/rejection_thresholds.json"
SEEDS = [0, 1, 2]
N_BOOT = 1000
PX_TO_MM, CUT, MARGIN, MIN_AREA, MIN_KEEP_IOU = 0.15, 0.6, 0.1, 12, 0.1

import os, re, json, pickle
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from transformers import SamModel, SamProcessor

CR = re.compile(r"(\d+)\s+follicles?\s+identified\s+measuring\s+([\d.]+)-([\d.]+)\s*mm")
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = SamProcessor.from_pretrained(MEDSAM_ID); sam = SamModel.from_pretrained(MEDSAM_ID).to(dev).eval()


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

import torch.nn.functional as Fnn
PERIPH = "Peripheral distribution of follicles noted."; CENTRAL = "Follicles are centrally/scattered distributed."
POS = "compatible with polycystic ovarian morphology"
PCOM_POS = ("Follicle number and size distribution are compatible with polycystic ovarian morphology (PCOM). "
            "Correlate with clinical and biochemical findings before diagnosing PCOS.")
PCOM_NEG = ("Findings do not meet follicle number criteria for polycystic ovarian morphology. "
            "If symptoms persist, correlate with endocrine evaluation and follow-up imaging as clinically indicated.")
def rad(B, W, H): return float(np.mean([np.hypot((b[0] + b[2]) / 2 - W / 2, (b[1] + b[3]) / 2 - H / 2) / (min(W, H) / 2) for b in B]))
def pat(B, W, H): return None if len(B) < 2 else ("peripheral" if rad(B, W, H) > CUT else "central")
def decisive(B, W, H): return len(B) >= 2 and abs(rad(B, W, H) - CUT) >= MARGIN
def tbv_rewrite(rep, B, W, H):
    """SIZE from the kept re-measured follicles; PATTERN only on decisive evidence; COUNT and conclusion unchanged."""
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

THR = json.load(open(THRESHOLDS))
test = pickle.load(open(TEST_PKL, "rb"))
@torch.no_grad()
def run_dec(dec):
    sam.mask_decoder.load_state_dict(dec); sam.eval(); out = []
    for c in test:
        img = Image.open(c["image"]).convert("RGB"); W, H = img.size
        if not c["P"]:
            out.append(([], [])); continue
        emb = sam.get_image_embeddings(proc(img, return_tensors="pt")["pixel_values"].to(dev))
        o = sam(image_embeddings=emb, input_boxes=proc(img, input_boxes=[c["P"]], return_tensors="pt")["input_boxes"].to(dev), multimask_output=False)
        s = 1024 / max(H, W); rh, rw = int(round(H * s)), int(round(W * s))
        m = Fnn.interpolate(o.pred_masks[0, :, 0][:, None], size=(1024, 1024), mode="bilinear", align_corners=False)
        m = (torch.sigmoid(Fnn.interpolate(m[..., :rh, :rw], size=(H, W), mode="bilinear", align_corners=False)[:, 0]) > .5).cpu().numpy()
        res = []
        for orig, mask in zip(c["P"], m):
            lab, n = ndimage.label(mask)
            if n == 0:
                res.append(list(orig)); continue
            sz = ndimage.sum(mask, lab, range(1, n + 1)); k = int(np.argmax(sz)) + 1; ys, xs = np.where(lab == k)
            b = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            res.append(b if sz[k - 1] >= MIN_AREA and iou(b, orig) >= MIN_KEEP_IOU else list(orig))
        out.append((res, o.iou_scores[0, :, 0].cpu().numpy().tolist()))
    return out
RES = {}
for s_ in SEEDS:
    RES[("old", s_)] = run_dec(torch.load(OLD_DECODERS.format(s_), map_location=dev))
    RES[("ea", s_)] = run_dec(torch.load(EA_DECODERS.format(s_), map_location=dev))
print("re-measured the 100 test images with every decoder")

scan_of = json.load(open(SCAN_JSON))["scan_of"] if os.path.exists(SCAN_JSON) else {}
groups = {}
for i, c in enumerate(test):
    groups.setdefault(scan_of.get(c["image"], f"solo{i}"), []).append(i)
G_ = list(groups.values()); rngb = np.random.default_rng(0)
BOOT = [np.concatenate([G_[g] for g in rngb.integers(0, len(G_), len(G_))]) for _ in range(N_BOOT)]
# whitespace BLEU; Table 1 uses evaluate/text_metrics.py
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score as bertscore
RS = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True); refs = [c["ref_report"] for c in test]
FIX = []
for ci_, c in enumerate(test):
    _, pairs = match(c["P"], c["Hum"], 0.1); FIX += [(ci_, i, j) for i, j in pairs]

def evaluate(name, boxes_fn, text_fn, diam_fn):
    B = [boxes_fn(i) for i in range(len(test))]; T = [text_fn(i, B[i]) for i in range(len(test))]
    M = [match(B[i], test[i]["Hum"])[0] for i in range(len(test))]; p, r, f = prf(sum(M))
    e = [abs(d - diam(test[ci_]["Hum"][j])) for ci_, i, j in FIX for d in [diam_fn(ci_, i)] if d is not None]
    sz = []
    for i, c in enumerate(test):
        m = CR.search(T[i]); g = [diam(b) for b in c["Hum"]]
        sz.append(bool(m) and abs(float(m.group(2)) - min(g)) <= 0.5 and abs(float(m.group(3)) - max(g)) <= 0.5)
    bl = corpus_bleu([[x.split()] for x in refs], [t.split() for t in T])
    rl = np.mean([RS.score(a, b)["rougeL"].fmeasure for a, b in zip(refs, T)])
    bs = bertscore(T, refs, lang="en", verbose=False)[2].mean().item()
    return dict(name=name, M=M, P=p, R=r, F1=f, MAE=np.mean(e), n=len(e), gt1=100 * np.mean(np.array(e) > 1), size=100 * np.mean(sz),
                BLEU=bl, RL=rl, BS=bs, kept=sum(len(b) for b in B))

rows = [evaluate("MedGemma 1.5 (unverified)", lambda i: test[i]["P"], lambda i, b: test[i]["orig_report"],
                 lambda ci_, i: test[ci_]["listed_d"][i] if i < len(test[ci_].get("listed_d", [])) else None)]
for kind, label in [("old", "Re-measure"), ("ea", "Re-measure, error-aware decoder")]:
    for rej in (False, True):
        name = label + (" + rejection" if rej else "")
        t = THR[kind]
        per_seed = []
        for s_ in SEEDS:
            R = RES[(kind, s_)]
            keep = lambda i, R=R: [b for b, q in zip(*R[i]) if (q >= t or not rej)]
            def dfn(ci_, i, R=R):
                b, q = R[ci_][0][i], R[ci_][1][i]
                return diam(b) if (q >= t or not rej) else None
            per_seed.append(evaluate(f"{name} [seed {s_}]", keep, lambda i, b: tbv_rewrite(test[i]["orig_report"], b, test[i]["W"], test[i]["H"]), dfn))
        r0 = dict(per_seed[0]); r0["name"] = name; r0["F1sd"] = float(np.std([x["F1"] for x in per_seed])); r0["F1mean"] = float(np.mean([x["F1"] for x in per_seed]))
        rows.append(r0)

base_M = rows[0]["M"]; old_M = rows[1]["M"]
print(f"\n{'Method (seed 0; F1 mean+-sd over seeds)':44s} {'P':>5s} {'R':>5s} {'F1':>5s} {'F1 mean+-sd':>12s} {'MAE':>5s} {'>1mm':>5s} {'size':>5s} {'BLEU':>6s} {'R-L':>6s} {'BS':>6s} {'kept':>5s}")
for r in rows:
    ms = f"{r.get('F1mean', r['F1']):.3f}+-{r.get('F1sd', 0):.3f}"
    print(f"{r['name']:44s} {r['P']:5.2f} {r['R']:5.2f} {r['F1']:5.2f} {ms:>12s} {r['MAE']:5.2f} {r['gt1']:4.0f}% {r['size']:4.0f}% {r['BLEU']:6.3f} {r['RL']:6.3f} {r['BS']:6.3f} {r['kept']:5d}")
print("\n95% CIs of F1 differences (bootstrap over near-duplicate groups):")
for r in rows[1:]:
    for ref_name, ref_M in [("unverified", base_M), ("re-measure only", old_M)]:
        if r["M"] is ref_M:
            continue
        lo, hi = np.percentile([prf(sum(r["M"][i] for i in ix))[2] - prf(sum(ref_M[i] for i in ix))[2] for ix in BOOT], [2.5, 97.5])
        print(f"  {r['name']:40s} minus {ref_name:12s}: [{lo:+.3f}, {hi:+.3f}]")
print(f"\nrejection thresholds (chosen on Figshare): {THR}")
print("\n%%% LaTeX rows: Method & P & R & F1 & MAE & >1mm & size0.5 & BLEU & ROUGE-L & BERTScore")
for r in rows:
    print(f"{r['name']} & {r['P']:.2f} & {r['R']:.2f} & {r['F1']:.2f} & {r['MAE']:.2f} & {r['gt1']:.0f}\\% & {r['size']:.0f}\\% & {r['BLEU']:.3f} & {r['RL']:.3f} & {r['BS']:.3f} \\\\")
# kept boxes of TBV (seed 0)
for i, c in enumerate(test):
    c["P_ea0"] = RES[("ea", 0)][i][0]
    c["P_ea0_keep"] = [b for b, q in zip(*RES[("ea", 0)][i]) if q >= THR["ea"]]
    c["P_fig0_keep"] = [b for b, q in zip(*RES[("old", 0)][i]) if q >= THR["old"]]
pickle.dump(test, open(OUT_CACHE, "wb")); print(f"saved {OUT_CACHE}")


# diameter error on the same follicles
print("\nMAE and the number of follicles it is computed on:")
for r in rows:
    print(f"  {r['name']:40s} MAE {r['MAE']:.2f} mm on {r['n']:4d} follicles")

# every method is scored on the follicles TBV keeps
t_ea = THR["ea"]
R_ea, R_old = RES[("ea", 0)], RES[("old", 0)]
kept = [(ci, i, j) for ci, i, j in FIX if R_ea[ci][1][i] >= t_ea]

def gt_d(ci, j):
    return diam(test[ci]["Hum"][j])

def written_d(ci, i):
    L = test[ci].get("listed_d", [])
    return L[i] if i < len(L) else None

methods = {
    "MedGemma 1.5 (unverified, written mm)": written_d,
    "re-measure, decoder":                   lambda ci, i: diam(R_old[ci][0][i]),
    "re-measure, error-aware decoder":       lambda ci, i: diam(R_ea[ci][0][i]),
}

print(f"\nSame {len(kept)} follicles (those kept by full TBV) for every method:")
for name, fn in methods.items():
    e = [abs(d - gt_d(ci, j)) for ci, i, j in kept for d in [fn(ci, i)] if d is not None]
    print(f"  {name:40s} MAE {np.mean(e):.2f} mm  >1mm {100*np.mean(np.array(e) > 1):3.0f}%  (n={len(e)})")

print(f"\nAll {len(FIX)} follicles, no rejection:")
for name, fn in methods.items():
    e = [abs(d - gt_d(ci, j)) for ci, i, j in FIX for d in [fn(ci, i)] if d is not None]
    print(f"  {name:40s} MAE {np.mean(e):.2f} mm  >1mm {100*np.mean(np.array(e) > 1):3.0f}%  (n={len(e)})")
