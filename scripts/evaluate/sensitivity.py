"""Sensitivity to the rejection threshold, count tolerance and counter confidence; recall by follicle size."""

TEST_PKL, PLUS, DETX = "outputs/test_cache.pkl", "outputs/tbv_cache.pkl", "outputs/counter_boxes.pkl"
SCORES = "outputs/test_pred_iou.pkl"
MEDSAM_ID = "wanglab/medsam-vit-base"
EA_DECODERS = "checkpoints/decoder_error_aware_seed{}.pt"
SEEDS, MIN_AREA, MIN_KEEP_IOU = [0, 1, 2], 12, 0.1
TAU0, KAPPA0, CONF0 = 0.30, 3, 0.40
TAUS = [0.00, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
KAPPAS, CONFS = [1, 2, 3, 4, 5], [0.30, 0.35, 0.40, 0.45, 0.50]
PX_TO_MM, PCOM, T1, T2 = 0.15, 12, 3.2, 4.3  # size tertile cuts (mm)

import os, re, pickle
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

CR = re.compile(r"(\d+)\s+follicles?\s+identified\s+measuring\s+([\d.]+)-([\d.]+)\s*mm")
POS = "compatible with polycystic ovarian morphology"

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0
def tp(P, R, thr=0.5):
    if not P or not R:
        return np.array([0, len(P), len(R)])
    M = np.array([[iou(p, r) for r in R] for p in P]); ri, ci = linear_sum_assignment(-M)
    return np.array([sum(M[i, j] >= thr for i, j in zip(ri, ci)), len(P), len(R)])
def prf(a):
    t, n1, n2 = a; p, r = t / max(n1, 1), t / max(n2, 1); return p, r, (2 * p * r / (p + r) if p + r else 0.0)

test = pickle.load(open(TEST_PKL, "rb"))
cases = [c for c in test if c.get("Hum")]
det = pickle.load(open(DETX, "rb"))["boxes"]

# 1. rejection threshold (predicted IoU of every citation, cached)
if os.path.exists(SCORES):
    RES = pickle.load(open(SCORES, "rb")); print("loaded saved scores from", SCORES)
else:
    import torch, torch.nn.functional as Fnn
    from transformers import SamModel, SamProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = SamProcessor.from_pretrained(MEDSAM_ID); sam = SamModel.from_pretrained(MEDSAM_ID).to(dev)
    @torch.no_grad()
    def run_dec(dec):
        sam.mask_decoder.load_state_dict(dec); sam.eval(); out = []
        for c in test:
            img = Image.open(c["image"]).convert("RGB"); W, H = img.size
            if not c["P"]:
                out.append(([], [])); continue
            emb = sam.get_image_embeddings(proc(img, return_tensors="pt")["pixel_values"].to(dev))
            o = sam(image_embeddings=emb, input_boxes=proc(img, input_boxes=[c["P"]], return_tensors="pt")["input_boxes"].to(dev),
                    multimask_output=False)
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
        RES[s_] = run_dec(torch.load(EA_DECODERS.format(s_), map_location=dev))
        print(f"seed {s_}: re-measured {sum(len(b) for b, _ in RES[s_])} citations")
    pickle.dump(RES, open(SCORES, "wb")); print("saved", SCORES)

def at(tau, s_):
    S = np.zeros(3, int); kept = 0
    for c, (B, Q) in zip(test, RES[s_]):
        K = [b for b, q in zip(B, Q) if q >= tau]; kept += len(K); S += tp(K, c.get("Hum") or [])
    return (*prf(S), kept)
cited = sum(len(c["P"]) for c in test)
print(f"\n1) Rejection threshold (used: {TAU0}, chosen on Figshare) | {cited} cited follicles | P/R/F1 from seed 0, F1 mean +- sd over 3 seeds")
print(f"{'tau':>5s} {'kept':>6s} {'P':>5s} {'R':>5s} {'F1':>5s} {'F1 mean+-sd':>12s}")
for tau in TAUS:
    p, r, f, kept = at(tau, 0); fs = [at(tau, s_)[2] for s_ in SEEDS]
    mark = "  <- used" if abs(tau - TAU0) < 1e-9 else ("  (no rejection)" if tau == 0 else "")
    print(f"{tau:5.2f} {kept:6d} {p:5.2f} {r:5.2f} {f:5.2f} {np.mean(fs):6.3f}+-{np.std(fs):.3f}{mark}")
print("unverified MedGemma: F1 0.07")

# 2, 3. count tolerance and counter confidence
def claim_scores(kappa, conf):
    cnt, pc, gt = [], [], []
    for c in cases:
        m = CR.search(c["orig_report"]); n_gt = len(c["Hum"])
        if m is None:
            continue
        K = int(m.group(1)); Kd = sum(b[4] >= conf for b in det[c["image"]]["yolo_fig"])
        flag = abs(K - Kd) > kappa; n = Kd if flag else K
        cnt.append(abs(n - n_gt) <= 1); pc.append((n >= PCOM) if flag else (POS in c["orig_report"])); gt.append(n_gt >= PCOM)
    p, t = np.array(pc), np.array(gt)
    bal = ((p & t).sum() / max(t.sum(), 1) + (~p & ~t).sum() / max((~t).sum(), 1)) / 2
    return 100 * np.mean(cnt), bal, None
base_cnt = 100 * np.mean([abs(int(CR.search(c["orig_report"]).group(1)) - len(c["Hum"])) <= 1
                          for c in cases if CR.search(c["orig_report"])])
print(f"\n2) Count tolerance kappa (used: {KAPPA0}, counter conf {CONF0})   unverified: count+-1 {base_cnt:.0f}%")
print(f"{'kappa':>5s} {'count+-1':>9s} {'crit':>5s}")
for k in KAPPAS:
    cn, cr, _ = claim_scores(k, CONF0); mark = "  <- used" if k == KAPPA0 else ""
    print(f"{k:5d} {cn:8.0f}% {cr:5.2f}{mark}")
print(f"\n3) Counter confidence (used: {CONF0}, kappa {KAPPA0})")
print(f"{'conf':>5s} {'count+-1':>9s} {'crit':>5s}")
for cf in CONFS:
    cn, cr, _ = claim_scores(KAPPA0, cf); mark = "  <- used" if abs(cf - CONF0) < 1e-9 else ""
    print(f"{cf:5.2f} {cn:8.0f}% {cr:5.2f}{mark}")
print("\n(criterion accuracy of the unverified MedGemma reports is 0.59)")

# 4. recall by follicle size
def found(P, G):
    hit = np.zeros(len(G), bool)
    if P and G:
        M = np.array([[iou(p, g) for g in G] for p in P]); ri, ci = linear_sum_assignment(-M)
        for i, j in zip(ri, ci):
            hit[j] = M[i, j] >= 0.5
    return hit
d, before, after = [], [], []
for c, (B, Q) in zip(test, RES[0]):
    G = c.get("Hum") or []
    if not G:
        continue
    d += [((g[2] - g[0]) + (g[3] - g[1])) / 2 * PX_TO_MM for g in G]
    before += list(found(B, G)); after += list(found([b for b, q in zip(B, Q) if q >= TAU0], G))
d, before, after = map(np.array, (d, before, after))
t1, t2 = np.percentile(d, [33.3, 66.7])
print(f"\n4) Recall by size | {len(d)} ground-truth follicles | size tertiles at {t1:.1f} and {t2:.1f} mm")
print(f"{'size group':22s} {'n':>4s} {'recall before':>14s} {'after':>6s} {'lost':>5s}")
for name, m in [(f"small  (< {t1:.1f} mm)", d < t1), ("medium", (d >= t1) & (d < t2)), (f"large  (>= {t2:.1f} mm)", d >= t2), ("all", d > -1)]:
    lost = (before & ~after)[m].sum()
    print(f"{name:22s} {m.sum():4d} {100*before[m].mean():13.0f}% {100*after[m].mean():5.0f}% {lost:5d}")

d, hit = [], []
for c in cases:
    G = c["Hum"]; D = [list(b[:4]) for b in det[c["image"]]["yolo_fig"] if b[4] >= CONF0]
    d += [((g[2] - g[0]) + (g[3] - g[1])) / 2 * PX_TO_MM for g in G]; hit += list(found(D, G))
d, hit = np.array(d), np.array(hit)
print(f"\nCounter recall (conf {CONF0}), size cuts at {T1} and {T2} mm")
for name, m in [(f"small (< {T1} mm)", d < T1), ("medium", (d >= T1) & (d < T2)), (f"large (>= {T2} mm)", d >= T2), ("all", d > -1)]:
    print(f"{name:20s} n={m.sum():4d}  counter recall {100*hit[m].mean():.0f}%")
