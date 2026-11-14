"""Figures 1-3 from the evaluation outputs.

  teaser.pdf        Fig. 1: MedGemma's cited boxes (red) and the ground truth (green)
  qualitative.pdf   Fig. 3: three test scans (rows) x every method of Tables 1-2 (columns)
  method_*.png, method_claims.tex   panels of Fig. 2
  example_case.tex  the before/after report of Table 3
Panel label "x/y . n FP": x boxes match a ground-truth follicle (IoU >= 0.5) out of y; n unmatched boxes.

    python figures/make_figures.py
"""

CACHE = "outputs/test_cache.pkl"
DET = "outputs/counter_boxes.pkl"
PLUS = "outputs/tbv_cache.pkl"                    # final TBV boxes (verify.py)
EXTRA_CANDIDATES = ["outputs/baselines.pkl"]      # Qwen2.5-VL and SAM2 boxes (baselines/qwen_sam2.py)
VLM_CSV = {"PaliGemma 2": "outputs/second_vlms/pred_paligemma2.csv", "LLaVA-Med": "outputs/second_vlms/pred_llavamed.csv"}
MATCH_IOU, GRID = 0.3, 99                      # same as evaluate/second_vlm_verifiers.py
# ---- rows of Fig. 3 ----
BROWSE = False     # True: only write figures/candidates.png (all test images, numbered) and stop
PICK = [2, 5, 20]  # test images (numbers on candidates.png) used as the rows of Fig. 3
# ---- fix the crop of single images (by the same number as in PICK) ----
# CROP_FIX = {image number: (zoom, move_right, move_down)}
#   zoom > 1 cuts more away (1.15 = 15% closer); move_* shift the window as a fraction of its size (+ = right / down)
#   e.g. a white strip at the top of image 2:  CROP_FIX = {2: (1.15, 0.0, 0.08)}
CROP_FIX = {2: (1.15, 0.0, 0.08)}
N_SCANS = 3                                    # rows of Fig. 3 (each from a different near-duplicate group)
PANEL_PX = 512                                 # every panel is cropped to a square and resized to this size
SCAN_JSON = "data/scan_groups.json"           # near-duplicate groups
OUT = "figures"
PX_TO_MM, PCOM, CUT, MARGIN, COUNT_TOL = 0.15, 12, 0.6, 0.1, 3

import os, re, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy import ndimage

os.makedirs(OUT, exist_ok=True)
CR = re.compile(r"(\d+)\s+follicles?\s+identified\s+measuring\s+([\d.]+)-([\d.]+)\s*mm")
PERIPH = "Peripheral distribution of follicles noted."
CENTRAL = "Follicles are centrally/scattered distributed."
POS = "compatible with polycystic ovarian morphology"

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0
def matched(P, R):
    if not P or not R:
        return 0
    M = np.array([[iou(p, r) for r in R] for p in P]); ri, ci = linear_sum_assignment(-M)
    return int(sum(M[i, j] >= 0.5 for i, j in zip(ri, ci)))
def diam(b): return ((b[2] - b[0]) + (b[3] - b[1])) / 2 * PX_TO_MM
def radius(B, W, H): return float(np.mean([np.hypot((b[0] + b[2]) / 2 - W / 2, (b[1] + b[3]) / 2 - H / 2) / (min(W, H) / 2) for b in B]))

cases = [c for c in pickle.load(open(CACHE, "rb")) if c.get("Hum") and c["P"] and CR.search(c["orig_report"])]
for k, c in enumerate(cases):
    c["idx"] = k                                # the number shown on figures/candidates.png
det = pickle.load(open(DET, "rb")); thr = det["conf"]["yolo_fig"]

# ---- boxes of every compared method (as in evaluate/main_tables.py) ----
def otsu_thr(x):
    h, e = np.histogram(x, 256); h = h.astype(float); m = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * m) / np.maximum(w0, 1e-9); m1 = (np.sum(h * m) - np.cumsum(h * m)) / np.maximum(w1, 1e-9)
    return m[np.argmax(w0 * w1 * (m0 - m1) ** 2)]
def otsu_refine(gray, B, ex=0.3):
    H_, W_ = gray.shape; out = []
    for b in B:
        bw, bh = b[2] - b[0], b[3] - b[1]
        x1, y1 = int(max(0, b[0] - ex * bw)), int(max(0, b[1] - ex * bh))
        x2, y2 = int(min(W_, b[2] + ex * bw + 1)), int(min(H_, b[3] + ex * bh + 1))
        crop = ndimage.gaussian_filter(gray[y1:y2, x1:x2].astype(float), 1.5)
        if crop.size < 16 or crop.max() - crop.min() < 1e-3:
            out.append(list(b)); continue
        dark = crop < otsu_thr(crop); lb, n = ndimage.label(dark)
        if n == 0:
            out.append(list(b)); continue
        cy, cx = int((b[1] + b[3]) / 2) - y1, int((b[0] + b[2]) / 2) - x1
        k = lb[min(max(cy, 0), crop.shape[0] - 1), min(max(cx, 0), crop.shape[1] - 1)]
        if k == 0:
            k = int(np.argmax(ndimage.sum(dark, lb, range(1, n + 1)))) + 1
        ys, xs = np.where(lb == k)
        nb = [float(xs.min() + x1), float(ys.min() + y1), float(xs.max() + 1 + x1), float(ys.max() + 1 + y1)]
        out.append(nb if len(xs) >= 12 and iou(nb, b) >= 0.1 else list(b))
    return out
ex_ = {}
for f in EXTRA_CANDIDATES:
    if os.path.exists(f):
        e = pickle.load(open(f, "rb"))
        if np.mean([c["image"] in e and len(e[c["image"]].get("P_sam2", [])) == len(c["P"]) for c in cases]) > 0.95:
            ex_ = e; print("Qwen2.5-VL / SAM2 boxes from", f); break
assert ex_, "outputs/baselines.pkl not found, run baselines/qwen_sam2.py first"
for c in cases:
    e = ex_[c["image"]]
    c["P_otsu"] = otsu_refine(np.asarray(Image.open(c["image"]).convert("L"), float), c["P"])
    c["P_sam2"] = e["P_sam2"]; c["P_qwen"] = e.get("P_qwen", c["P"])
    c["P_qkeep"] = [b for b, p in zip(c["P"], e.get("q_pyes", [1] * len(c["P"]))) if p >= 0.5]
# PaliGemma 2 / LLaVA-Med refine: each MedGemma box replaced by the matching box the second VLM cites (IoU >= 0.3)
import pandas as pd
EVR = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*([\d.]+)\s*mm")
def match_pairs(P, R, thr):
    if not P or not R:
        return {}
    M = np.array([[iou(p, r) for r in R] for p in P]); ri, ci = linear_sum_assignment(-M)
    return {i: j for i, j in zip(ri, ci) if M[i, j] >= thr}
for vname, path in VLM_CSV.items():
    gen = pd.read_csv(path).drop_duplicates("image").set_index("image")["generated"].astype(str).to_dict()
    for c in cases:
        W_, H_ = c["W"], c["H"]
        V = [[int(a) / GRID * W_, int(b) / GRID * H_, int(cc) / GRID * W_, int(d) / GRID * H_]
             for a, b, cc, d, _ in EVR.findall(gen.get(c["image"], "").partition("Report:")[0])]
        mp = match_pairs(c["P"], V, MATCH_IOU)
        c["P_" + vname] = [V[mp[i]] if i in mp else list(b) for i, b in enumerate(c["P"])]
# TBV boxes: error-aware decoder + rejection (outputs/tbv_cache.pkl); without that file, re-measured boxes without rejection
plus = {c["image"]: c for c in pickle.load(open(PLUS, "rb"))} if os.path.exists(PLUS) else {}
for c in cases:
    c["P_tbv"] = plus[c["image"]]["P_ea0_keep"] if c["image"] in plus else c["P_fig0"]
print("TBV boxes:", "final (error-aware + rejection)" if plus else "re-measure only (outputs/tbv_cache.pkl not found)")
for c in cases:
    c["mV"], c["mO"], c["mF"] = matched(c["P"], c["Hum"]), matched(c["P_off"], c["Hum"]), matched(c["P_tbv"], c["Hum"])
    c["gain"] = c["mF"] - c["mV"]
# example scans: where TBV beats the BEST other method by the largest margin (fraction of ground-truth follicles matched)
OTHER_KEYS = ["P", "P_qwen", "P_PaliGemma 2", "P_LLaVA-Med", "P_otsu", "P_off", "P_sam2"]
for c in cases:
    nG = max(len(c["Hum"]), 1)
    c["margin"] = c["mF"] / nG - max(matched(c[k], c["Hum"]) for k in OTHER_KEYS) / nG
cases.sort(key=lambda c: (c["margin"], c["mF"] / max(len(c["Hum"]), 1)), reverse=True)
EX_KEY = "img3_jpg.rf.vMihMnsoFdCRWpU9uEEa"   # Fig. 2 uses the same test image as Table 3 / Figure 4

# ---- drawing ----
RED, GREEN = "#e53935", "#43d35d"   # every method's boxes in red, ground truth in green (all figures)

def crop_window(img):
    """Square window inside the ultrasound content. A row or column counts as content when most of its pixels are
    neither near-black nor near-white, so padding, white borders and the dark corners of the fan are cut away."""
    a = np.asarray(img, float); H_, W_ = a.shape
    ok = (a > 12) & (a < 243)
    rows = np.where(ok.mean(1) > 0.6)[0]; cols = np.where(ok.mean(0) > 0.6)[0]
    if not len(rows) or not len(cols):
        return 0, 0, W_, H_
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    side = min(y1 - y0, x1 - x0)
    x0 += (x1 - x0 - side) // 2; y0 += (y1 - y0 - side) // 2
    return x0, y0, x0 + side, y0 + side

def panel(ax, c, layers, label=None, crop=True):
    img = Image.open(c["image"]).convert("L")
    if crop:   # square crop of the ultrasound content, resized so EVERY panel has exactly the same size
        x0, y0, x1, y1 = crop_window(img)
        if c.get("idx") in CROP_FIX:          # manual adjustment of this one image
            zoom, mx, my = CROP_FIX[c["idx"]]; side = (x1 - x0) / zoom
            cx = (x0 + x1) / 2 + mx * (x1 - x0); cy = (y0 + y1) / 2 + my * (y1 - y0)
            W_, H_ = img.size; side = min(side, W_, H_)
            x0 = int(min(max(cx - side / 2, 0), W_ - side)); y0 = int(min(max(cy - side / 2, 0), H_ - side))
            x1, y1 = x0 + int(side), y0 + int(side)
        s = PANEL_PX / (x1 - x0)
        img = img.crop((x0, y0, x1, y1)).resize((PANEL_PX, PANEL_PX), Image.BILINEAR)
        tf = lambda b: [(b[0] - x0) * s, (b[1] - y0) * s, (b[2] - x0) * s, (b[3] - y0) * s]
    else:
        tf = lambda b: b
    ax.imshow(img, cmap="gray"); ax.axis("off")
    for boxes, col, lw in layers:
        for b in boxes:
            b = tf(b)
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col, lw=lw))
    if crop:
        ax.set_xlim(0, PANEL_PX); ax.set_ylim(PANEL_PX, 0)
    if label:   # inside the panel, bottom-left, black on white (no space needed between rows)
        ax.text(0.03, 0.03, label, transform=ax.transAxes, va="bottom", ha="left", fontsize=7, fontweight="bold",
                color="black", bbox=dict(facecolor="white", alpha=0.9, pad=1.5, lw=0))

def lab(B, c):
    m_ = matched(B, c["Hum"]); fp = len(B) - m_
    return f"{m_}/{len(c['Hum'])}" + (f" · {fp} FP" if fp else "")

# ---------------- Fig. 1 teaser (one column): VLM-cited (red) vs ground truth (green) ----------------
pick = [c for c in cases if c["mV"] <= 2][:2] or cases[:2]
fig, axs = plt.subplots(1, 2, figsize=(3.3, 1.65))
for ax, c in zip(axs, pick):
    panel(ax, c, [(c["Hum"], GREEN, 0.6), (c["P"], RED, 0.9)], lab(c["P"], c))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
fig.savefig(f"{OUT}/teaser.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(f"{OUT}/teaser.png", dpi=600, bbox_inches="tight", pad_inches=0.01); plt.close(fig)

# ---------------- Fig. 3 qualitative (full width): rows = 3 test scans, columns = ALL methods of Table 2 ----------------
# Same methods and names as Tables 1-2. The Qwen2.5-VL judge only rewrites text (no boxes), so it has no column.
N_ROWS = N_SCANS
exv = ex_
def thumb(c):
    a = np.asarray(Image.open(c["image"]).convert("L").resize((32, 32)), float).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)
import json
scan_of = json.load(open(SCAN_JSON))["scan_of"] if os.path.exists(SCAN_JSON) else {}
def thumb_c(c):                        # thumbnail of the CROPPED panel, so similar-looking ovaries are caught
    im = Image.open(c["image"]).convert("L"); x0, y0, x1, y1 = crop_window(im)
    a = np.asarray(im.crop((x0, y0, x1, y1)).resize((32, 32)), float).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)
def distinct(c, chosen):
    g = scan_of.get(c["image"], c["image"]); t = thumb_c(c)
    return all(scan_of.get(o["image"], o["image"]) != g and float(np.mean(t * thumb_c(o))) < 0.5 for o in chosen)
by_match = sorted(cases, key=lambda c: (c["mF"] / max(len(c["Hum"]), 1), c["gain"]), reverse=True)
examples = []
for c in by_match:                    # rows 1-2: TBV's best matches
    if distinct(c, examples):
        examples.append(c)
    if len(examples) == 2:
        break
for c in cases:                       # row 3: largest lead of TBV over the best other method, a different-looking scan
    if distinct(c, examples):
        examples.append(c); break
for c in by_match:                    # safety net: never fewer rows than N_ROWS
    if len(examples) >= N_ROWS:
        break
    if c not in examples and all(scan_of.get(o["image"], o["image"]) != scan_of.get(c["image"], c["image"]) for o in examples):
        examples.append(c)
if BROWSE:
    order = sorted(cases, key=lambda c: c["idx"])
    ncol = 10; nrow = int(np.ceil(len(order) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 1.6, nrow * 1.75))
    for ax in axs.ravel():
        ax.axis("off")
    for ax, c in zip(axs.ravel(), order):
        nG = len(c["Hum"]); best = max(matched(c[k], c["Hum"]) for k in OTHER_KEYS)
        panel(ax, c, [(c["Hum"], GREEN, 0.4), (c["P_tbv"], RED, 0.6)])
        ax.set_title(f"#{c['idx']}  TBV {c['mF']}/{nG}  best other {best}/{nG}\ngroup {scan_of.get(c['image'], '?')}",
                     fontsize=6.5, pad=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT}/candidates.png", dpi=110); plt.close(fig)
    print(f"saved {OUT}/candidates.png: green = ground truth, red = TBV. Put 3 numbers in PICK and set BROWSE = False.")
    print("\nTop 15 by TBV's lead over the best other method:")
    for c in cases[:15]:
        nG = len(c["Hum"]); best = max(matched(c[k], c["Hum"]) for k in OTHER_KEYS)
        print(f"  #{c['idx']:3d}  TBV {c['mF']:2d}/{nG:2d}  best other {best:2d}/{nG:2d}  MedGemma {c['mV']:2d}/{nG:2d}  "
              f"group {scan_of.get(c['image'], '?')}  {os.path.basename(c['image'])}")
    raise SystemExit
if PICK:
    examples = [next(c for c in cases if c["idx"] == k) for k in PICK]
    N_ROWS = len(examples)
COLS = [("MedGemma 1.5", "P", RED), ("Qwen2.5-VL", "P_qwen", RED), ("PaliGemma 2", "P_PaliGemma 2", RED),
        ("LLaVA-Med", "P_LLaVA-Med", RED), ("Otsu", "P_otsu", RED), ("MedSAM", "P_off", RED),
        ("SAM2", "P_sam2", RED), ("TBV (ours)", "P_tbv", RED)]
BOT = 0.07                                     # space for the method names under the last row
fig, axs = plt.subplots(N_ROWS, len(COLS), figsize=(6.9, 6.9 / len(COLS) * N_ROWS / (1 - BOT)))   # square cells
axs = np.atleast_2d(axs)
for r, c in enumerate(examples):
    for j, (name, key, col) in enumerate(COLS):
        panel(axs[r, j], c, [(c["Hum"], GREEN, 0.5), (c[key], col, 0.8)], lab(c[key], c))
        if r == N_ROWS - 1:
            axs[r, j].text(0.5, -0.04, name, transform=axs[r, j].transAxes, ha="center", va="top",
                           fontsize=7, fontweight="bold", color="black")
for ax in axs.ravel():
    for t in ax.texts:
        if t.get_position() == (0.03, 0.03):
            t.set_fontsize(5)
fig.subplots_adjust(left=0, right=1, bottom=BOT, top=1, wspace=0.015, hspace=0.015)   # no gaps
fig.savefig(f"{OUT}/qualitative.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(f"{OUT}/qualitative.png", dpi=600, bbox_inches="tight", pad_inches=0.01); plt.close(fig)
print("Fig. 3 columns:", [n.replace(chr(10), " ") for n, _, _ in COLS])
for r, c in enumerate(examples):      # check that the rows really are different images / scans
    print(f"  row {r+1}: {os.path.basename(c['image'])} | scan group {scan_of.get(c['image'], '?')} | "
          f"{len(c['Hum'])} ground-truth follicles | size {c['W']}x{c['H']}")

# ---------------- Fig. 2 panels + claims, Table 3 example ----------------
ex = next((c for c in cases if EX_KEY in c["image"]), examples[0]); W, H = ex["W"], ex["H"]
for name, layers in [("inputimg", []), ("method_vlm (1)", [(ex["P"], RED, 1.6)]),
                     ("method_remeasured", [(ex["P_tbv"], RED, 1.6)])]:   # file names used by the Fig. 2 TikZ
    f = plt.figure(figsize=(3, 3 * H / W)); ax = f.add_axes([0, 0, 1, 1]); panel(ax, ex, layers, crop=False)
    f.savefig(f"{OUT}/{name}.png", dpi=200); plt.close(f)

m = CR.search(ex["orig_report"]); n = int(m.group(1))
d1 = [diam(b) for b in ex["P_tbv"]]
r0, r1 = radius(ex["P"], W, H), radius(ex["P_tbv"], W, H)
pat0 = "peripheral" if PERIPH in ex["orig_report"] else "central" if CENTRAL in ex["orig_report"] else "not stated"
pat1 = ("peripheral" if r1 > CUT else "central") if abs(r1 - CUT) >= MARGIN else pat0
ndet = sum(b[4] >= thr for b in det["boxes"][ex["image"]]["yolo_fig"])
flag = abs(n - ndet) > COUNT_TOL
hd = [diam(b) for b in ex["Hum"]]
if pat0 == "not stated" and abs(r1 - CUT) < MARGIN:
    pat1 = "central" if r1 <= CUT else "peripheral"
critB = POS in ex["orig_report"]; critA = (ndet >= PCOM) if flag else critB
mac = {"CountB": str(n), "CountA": str(ndet if flag else n), "CountS": "corrected" if flag else "checked",
       "SizeB": f"{m.group(2)}--{m.group(3)} mm", "SizeA": f"{min(d1):.1f}--{max(d1):.1f} mm",
       "SizeS": "corrected", "PatB": pat0, "PatA": pat1, "PatS": "corrected" if pat1 != pat0 else "checked",
       "Crit": "meets" if critB else "below", "CritA": "meets" if critA else "below",
       "CritS": "corrected" if critA != critB else "checked", "Kt": str(ndet),
       "MatchB": str(ex["mV"]), "MatchA": str(ex["mF"]), "NRef": str(len(ex["Hum"]))}
open(f"{OUT}/method_claims.tex", "w").write("".join(f"\\def\\{k}{{{v}}}\n" for k, v in mac.items()))

def tex(t):
    return re.sub(r"(\d)-(\d)", r"\1--\2", t).replace(" mm", "\\,mm")
d0 = (float(m.group(2)), float(m.group(3)))
verified = ex["orig_report"]
verified = CR.sub(f"{n} follicles identified measuring {min(d1):.1f}-{max(d1):.1f} mm", verified, count=1)
if pat1 != pat0:
    verified = verified.replace(PERIPH, "@@").replace(CENTRAL, "@@").replace("@@", PERIPH if pat1 == "peripheral" else CENTRAL)
vt = tex(verified)
vt = vt.replace(f"{min(d1):.1f}--{max(d1):.1f}\\,mm", f"\\textbf{{{min(d1):.1f}--{max(d1):.1f}\\,mm}}")
if pat1 != pat0:
    sent = PERIPH if pat1 == "peripheral" else CENTRAL
    vt = vt.replace(sent, f"\\textbf{{{sent}}}")
if flag:      # TBV takes the count (and the conclusion) of a flagged report from the independent counter
    PCOM_POS = ("Follicle number and size distribution are compatible with polycystic ovarian morphology (PCOM). "
                "Correlate with clinical and biochemical findings before diagnosing PCOS.")
    PCOM_NEG = ("Findings do not meet follicle number criteria for polycystic ovarian morphology. "
                "If symptoms persist, correlate with endocrine evaluation and follow-up imaging as clinically indicated.")
    vt = vt.replace(f"{n} follicles identified", f"\\textbf{{{ndet}}} follicles identified", 1)
    if (POS in ex["orig_report"]) != (ndet >= PCOM):
        old_c = PCOM_POS if POS in ex["orig_report"] else PCOM_NEG
        new_c = PCOM_POS if ndet >= PCOM else PCOM_NEG
        vt = vt.replace(old_c, f"\\textbf{{{new_c}}}")
    vt += f" \\textit{{Flagged for review.}}"
evid = [f"[{a},{b},{c_},{d}] {mm}\\,mm" for a, b, c_, d, mm in
        re.findall(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*([\d.]+)\s*mm", str(ex["generated"]).split("Report:")[0])[:3]]
hd = [diam(b) for b in ex["Hum"]]
gt_pat = "peripheral" if len(ex["Hum"]) > 1 and radius(ex["Hum"], W, H) > CUT else "central/scattered"
gt = (f"{len(ex['Hum'])} follicles measuring {min(hd):.1f}--{max(hd):.1f}\\,mm, {gt_pat} distribution; "
      f"{'meets' if len(ex['Hum']) >= PCOM else 'below'} the follicle-number criterion.")
open(f"{OUT}/example_case.tex", "w").write(
    "\\begin{tabular}{@{}p{0.21\\linewidth}p{0.74\\linewidth}@{}}\n\\toprule\n"
    f"VLM output & \\textit{{Evidence:}} {'; '.join(evid)}; \\ldots\\ \\textit{{Report:}} {tex(ex['orig_report'])} \\\\\n\\midrule\n"
    f"Verified & {vt} \\\\\n\\midrule\n"
    f"Ground truth & {gt} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
print("Saved to figures/: teaser.pdf, qualitative.pdf, method_*.png, method_claims.tex, example_case.tex")
print("Example:", ex["image"], "| VLM", ex["mV"], "-> ours", ex["mF"], "of", len(ex["Hum"]), "| flagged:", flag)
