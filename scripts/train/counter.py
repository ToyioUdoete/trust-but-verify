"""Train the YOLO11s follicle counter on Figshare and pick its confidence threshold."""

FIG_ROOTS = [
    "data/figshare/figshare_part1",
    "data/figshare/figshare_part2",
]
KAGGLE_CSV = "data/kaggle_references.csv"
BASE_MODEL = "yolo11s.pt"
EPOCHS, IMGSZ, BATCH, NMS_IOU, PATIENCE = 150, 640, 4, 0.45, 40
WORKERS = 0
VAL_FRAC, DHASH_MAX, SEED = 0.15, 6, 0
WORK_DIR = "outputs/counter_data"
RUN_DIR = "outputs/counter_runs"
OUT_WEIGHTS = "checkpoints/counter_yolo11s.pt"
OUT_JSON = "checkpoints/counter.json"

import os, glob, json, random, hashlib, shutil
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def dhash(path, n=8):
    a = np.asarray(Image.open(path).convert("L").resize((n + 1, n)), dtype=float)
    return (a[:, 1:] > a[:, :-1]).ravel()


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


# 1. merged Figshare dataset
yaml = os.path.abspath(os.path.join(WORK_DIR, "data.yaml"))
if not os.path.exists(yaml):
    kaggle_hashes = [dhash(p) for p in pd.read_csv(KAGGLE_CSV)["image_path"]]
    pairs = []
    for root in FIG_ROOTS:
        if not os.path.isdir(root):
            print(f"  missing, skipped: {root}"); continue
        for p in sorted(glob.glob(f"{root}/**/images/*", recursive=True)):
            if p.lower().endswith(IMG_EXT):
                lbl = os.path.join(os.path.dirname(os.path.dirname(p)), "labels", os.path.splitext(os.path.basename(p))[0] + ".txt")
                if os.path.exists(lbl):
                    pairs.append((p, lbl))
    seen, keep, n_dup, n_kag = set(), [], 0, 0
    for p, l in pairs:
        h = hashlib.md5(Image.open(p).convert("L").tobytes()).hexdigest()
        if h in seen:
            n_dup += 1; continue
        seen.add(h)
        dh = dhash(p)
        if min(int((dh != t).sum()) for t in kaggle_hashes) <= DHASH_MAX:
            n_kag += 1; continue
        keep.append((p, l))
    if not keep:
        raise SystemExit("No Figshare images found, check FIG_ROOTS.")
    random.Random(SEED).shuffle(keep)
    n_val = int(VAL_FRAC * len(keep))
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    for split, items in [("val", keep[:n_val]), ("train", keep[n_val:])]:
        os.makedirs(f"{WORK_DIR}/{split}/images"); os.makedirs(f"{WORK_DIR}/{split}/labels")
        for k, (p, l) in enumerate(items):
            stem = f"fig_{k:04d}"
            Image.open(p).convert("RGB").save(f"{WORK_DIR}/{split}/images/{stem}.png")
            lines = []
            for line in open(l).read().strip().splitlines():
                v = [float(x) for x in line.split()[1:]]
                if len(v) == 4:
                    lines.append("0 " + " ".join(f"{x:.6f}" for x in v))
                elif len(v) >= 6:  # polygon -> box
                    xs, ys = np.array(v[0::2]), np.array(v[1::2])
                    lines.append(f"0 {(xs.min()+xs.max())/2:.6f} {(ys.min()+ys.max())/2:.6f} {xs.max()-xs.min():.6f} {ys.max()-ys.min():.6f}")
            open(f"{WORK_DIR}/{split}/labels/{stem}.txt", "w").write("\n".join(lines))
    open(yaml, "w").write(f"path: {os.path.abspath(WORK_DIR)}\ntrain: train/images\nval: val/images\nnames:\n  0: follicle\n")
    print(f"Figshare: {len(keep)} images ({n_dup} duplicates, {n_kag} near-copies of Kaggle images dropped) "
          f"-> train {len(keep) - n_val} / val {n_val}", flush=True)

# 2. train
best = os.path.join(RUN_DIR, "yolo_figshare", "weights", "best.pt")
if not os.path.exists(best):
    YOLO(BASE_MODEL).train(data=yaml, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, workers=WORKERS, cache=False, plots=False,
                           seed=SEED, patience=PATIENCE, project=os.path.abspath(RUN_DIR), name="yolo_figshare",
                           exist_ok=True, verbose=True)
model = YOLO(best)
m = model.val(data=yaml, imgsz=IMGSZ, conf=0.001, workers=WORKERS, plots=False, verbose=False)
print(f"Figshare val: mAP50 {m.box.map50:.3f}, mAP50-95 {m.box.map:.3f}", flush=True)

# 3. confidence threshold on the Figshare val split
val_preds, val_gts = [], []
for p in sorted(glob.glob(f"{WORK_DIR}/val/images/*.png")):
    r = model.predict(source=p, conf=0.05, iou=NMS_IOU, imgsz=IMGSZ, verbose=False)[0]
    H, W = r.orig_shape[:2]
    val_preds.append([b.xyxy[0].tolist() + [float(b.conf[0])] for b in r.boxes])
    g = []
    for line in open(p.replace("/images/", "/labels/").replace(".png", ".txt")).read().strip().splitlines():
        cx, cy, w, h = [float(x) for x in line.split()[1:5]]
        g.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    val_gts.append(g)
best_f1, conf = -1, 0.25
for t in np.arange(0.05, 0.80, 0.025):
    tp = npd = ngt = 0
    for P, G in zip(val_preds, val_gts):
        P = [b[:4] for b in P if b[4] >= t]; npd += len(P); ngt += len(G)
        if P and G:
            M = np.array([[box_iou(a, b) for b in G] for a in P]); ri, ci = linear_sum_assignment(-M)
            tp += sum(M[i, j] >= 0.5 for i, j in zip(ri, ci))
    pr, rc = tp / max(npd, 1), tp / max(ngt, 1); f = 2 * pr * rc / (pr + rc) if pr + rc else 0
    if f > best_f1:
        best_f1, conf = f, float(t)
print(f"Confidence threshold (Figshare val): {conf:.3f}, val F1@0.5 {best_f1:.3f}", flush=True)

os.makedirs(os.path.dirname(OUT_WEIGHTS), exist_ok=True)
shutil.copy(best, OUT_WEIGHTS)
json.dump({"conf": round(conf, 3), "nms_iou": NMS_IOU, "imgsz": IMGSZ}, open(OUT_JSON, "w"), indent=2)
print(f"Saved {OUT_WEIGHTS} and {OUT_JSON}")
