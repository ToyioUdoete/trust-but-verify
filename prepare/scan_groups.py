"""Group the Kaggle images into near-duplicate groups (copies of one source scan).

Every pair of images is matched on its central region (overlay text and borders cut off) with
ORB keypoints and a RANSAC similarity fit (|rotation| <= 45 deg, scale 0.7-1.4); the inlier
count is the edge weight. An edge is kept if it is strong, or if the two images share enough
neighbours, which stops a single false match from chaining two scans together. Groups are
the connected components. They are used for every bootstrap CI (images of one group are
resampled together).

    python prepare/scan_groups.py        # CPU, about 5-15 min with 8 workers

data/scan_groups.json is released with the code; this script documents how it was made.
"""

# ----------------------------- settings -----------------------------
GROUNDED_CSV = "data/kaggle_references.csv"
OUT_JSON = "data/scan_groups.json"
SEED = 42
VAL_FRAC, TEST_FRAC = 0.15, 0.15
BORDER_CROP = 0.12                  # cut this fraction off every side before matching
IMG_SIZE = 320
MAX_ROT_DEG, SCALE_RANGE = 45, (0.7, 1.4)
THRESHOLDS = [20, 30, 40, 60, 80, 120]
STRONG_EDGE = 120                   # edges this strong never need neighbour support
MIN_SHARED_NEIGHBOURS = 2           # weaker edges need this many common neighbours
MAX_GROUP_FRAC = 0.10               # auto-pick: largest group must be <= this share of images
CHOSEN_THRESHOLD = None             # None = auto-pick; the released scan_groups.json stores the value used ("threshold")
WORKERS = 8
PCOM_THRESHOLD = 12
# --------------------------------------------------------------------

import os, json, random, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("outputs", exist_ok=True)

g = pd.read_csv(GROUNDED_CSV)
paths = g["image_path"].tolist()
label = dict(zip(g["image_path"], (g["follicle_count"] >= PCOM_THRESHOLD).astype(int)))
N = len(paths)


def load(path):
    im = Image.open(path).convert("L")
    W, H = im.size
    bx, by = int(BORDER_CROP * W), int(BORDER_CROP * H)
    return np.array(im.crop((bx, by, W - bx, H - by)).resize((IMG_SIZE, IMG_SIZE)))


def features(path):
    kps, des = cv2.ORB_create(nfeatures=800).detectAndCompute(load(path), None)
    pts = np.array([k.pt for k in kps], dtype=np.float32) if kps else np.zeros((0, 2), np.float32)
    return pts, des


F = None
def _init(feats):
    global F, MATCHER
    F = feats
    MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING)


def inliers(ij):
    i, j = ij
    (pa, da), (pb, db) = F[i], F[j]
    if da is None or db is None or len(da) < 10 or len(db) < 10:
        return 0
    good = [m[0] for m in MATCHER.knnMatch(da, db, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 10:
        return 0
    M, mask = cv2.estimateAffinePartial2D(pa[[x.queryIdx for x in good]], pb[[x.trainIdx for x in good]],
                                          method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None or mask is None:
        return 0
    s = float(np.hypot(M[0, 0], M[0, 1])); r = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    if abs(r) > MAX_ROT_DEG or not (SCALE_RANGE[0] <= s <= SCALE_RANGE[1]):
        return 0
    return int(mask.sum())


if __name__ == "__main__":
    print(f"Images: {N}")
    t0 = time.time()
    feats = [features(p) for p in paths]
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    print(f"Scoring {len(pairs):,} pairs with {WORKERS} workers...")
    with Pool(WORKERS, initializer=_init, initargs=(feats,)) as pool:
        S = np.array(pool.map(inliers, pairs, chunksize=2000), dtype=np.int32)
    print(f"  done in {time.time() - t0:.0f}s")
    I, J = np.array(pairs).T
    np.savez_compressed("outputs/scan_edges.npz", i=I, j=J, inliers=S, paths=np.array(paths))

    def groups_at(t, supported):
        keep = S >= t
        ei, ej, es = I[keep], J[keep], S[keep]
        if supported:
            nb = defaultdict(set)
            for a, b in zip(ei, ej):
                nb[a].add(b); nb[b].add(a)
            ok = [(s >= STRONG_EDGE) or len(nb[a] & nb[b]) >= MIN_SHARED_NEIGHBOURS for a, b, s in zip(ei, ej, es)]
            ei, ej = ei[ok], ej[ok]
        parent = list(range(N))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for a, b in zip(ei, ej):
            parent[find(a)] = find(b)
        gr = defaultdict(list)
        for k in range(N):
            gr[find(k)].append(k)
        return sorted(gr.values(), key=len, reverse=True)

    print("\nInlier distribution of non-zero pairs:", np.percentile(S[S > 0], [50, 90, 99, 99.9]).round(0) if (S > 0).any() else "none")
    print(f"\n{'thr':>5} {'mode':>10} {'groups':>7} {'largest':>8} {'top-8 sizes'}")
    table = {}
    for t in THRESHOLDS:
        for sup in (False, True):
            gr = groups_at(t, sup)
            table[(t, sup)] = gr
            print(f"{t:>5} {'supported' if sup else 'plain':>10} {len(gr):>7} {len(gr[0]):>8} {[len(x) for x in gr[:8]]}")

    # --- example pairs per inlier band, to check by eye where "same scan" stops ---
    bands = [(20, 30), (30, 40), (40, 60), (60, 120), (120, 10 ** 6)]
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(bands), 6, figsize=(18, 3.2 * len(bands)))
    for r, (lo, hi) in enumerate(bands):
        idx = np.where((S >= lo) & (S < hi))[0]
        pick = rng.choice(idx, size=min(3, len(idx)), replace=False) if len(idx) else []
        for c in range(3):
            for k in range(2):
                ax = axes[r, 2 * c + k]; ax.axis("off")
                if c < len(pick):
                    ax.imshow(Image.open(paths[(I if k == 0 else J)[pick[c]]]).convert("L"), cmap="gray")
                    if k == 0:
                        ax.set_title(f"{S[pick[c]]} inliers", fontsize=9)
        axes[r, 0].text(-0.15, 0.5, f"{lo}-{hi if hi < 10 ** 6 else '+'}", transform=axes[r, 0].transAxes,
                        rotation=90, va="center", fontsize=11)
    plt.tight_layout(); plt.savefig("outputs/scan_pairs_by_inliers.png", dpi=70); plt.close(fig)
    print("\nSaved outputs/scan_pairs_by_inliers.png: each row shows 3 pairs from one inlier band.")

    # --- choose grouping ---
    if CHOSEN_THRESHOLD is not None:
        t, groups = CHOSEN_THRESHOLD, table.get((CHOSEN_THRESHOLD, True)) or groups_at(CHOSEN_THRESHOLD, True)
    else:
        ok = [t for t in THRESHOLDS if len(table[(t, True)][0]) <= MAX_GROUP_FRAC * N]
        t = ok[0] if ok else THRESHOLDS[-1]
        groups = table[(t, True)]
    print(f"\nUsing threshold {t} (neighbour-supported): {len(groups)} scan groups, largest {len(groups[0])}")

    # --- whole groups -> train/val/test, stratified by majority PCOM label ---
    rng2 = random.Random(SEED)
    split = {"train": [], "val": [], "test": []}
    for lab in (0, 1):
        gl = [x for x in groups if round(np.mean([label[paths[k]] for k in x])) == lab]
        rng2.shuffle(gl)
        n_lab = sum(len(x) for x in gl); filled = {"test": 0, "val": 0}
        for x in gl:
            if filled["test"] + len(x) <= TEST_FRAC * n_lab * 1.3 and filled["test"] < TEST_FRAC * n_lab:
                split["test"] += x; filled["test"] += len(x)
            elif filled["val"] + len(x) <= VAL_FRAC * n_lab * 1.3 and filled["val"] < VAL_FRAC * n_lab:
                split["val"] += x; filled["val"] += len(x)
            else:
                split["train"] += x
    scan_of = {paths[k]: gi for gi, x in enumerate(groups) for k in x}
    out = {s: [paths[k] for k in v] for s, v in split.items()}
    for s in out:
        print(f"  {s:5s}: {len(out[s]):4d} images from {len({scan_of[p] for p in out[s]}):3d} scans | "
              f"PCOM-positive {sum(label[p] for p in out[s])}")
    json.dump({**out, "scan_of": scan_of, "n_scans": len(groups), "threshold": int(t)}, open(OUT_JSON, "w"), indent=1)
    print(f"Saved {OUT_JSON}")
