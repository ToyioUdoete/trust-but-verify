"""Train the MedSAM mask decoder on Figshare pseudo-masks (frozen encoder, 3 seeds)."""

FIG_ROOTS = [
    "data/figshare/doctor_confirmed_136",
    "data/figshare/annotations_605",
]
KAGGLE_CSV = "data/kaggle_references.csv"  # used to drop near-copies of Kaggle images
SEEDS = [0, 1, 2]
DHASH_MAX = 6
MEDSAM_ID = "wanglab/medsam-vit-base"
OUT_DECODER = "checkpoints/decoder_seed{}.pt"
EMB_DIR = "outputs/emb_figshare"
EPOCHS = 30
LR = 1e-4
JITTER = 0.3
MAX_PROMPTS = 16
VAL_FRAC = 0.1
MAX_FILL, MAX_ASPECT, MIN_CONTRAST = 0.90, 2.5, 0.08

import os, glob, random, hashlib, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu
from transformers import SamModel, SamProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(os.path.dirname(OUT_DECODER), exist_ok=True)
SAM_SIZE, LOWRES = 1024, 256
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
print("Device:", device)


# pseudo-masks
def follicle_mask(gray, box):
    H, W = gray.shape
    x1, y1 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x2, y2 = min(W, int(np.ceil(box[2]))), min(H, int(np.ceil(box[3])))
    crop = gray[y1:y2, x1:x2]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    smooth = ndimage.gaussian_filter(crop, 1.5)
    if smooth.max() - smooth.min() < 1e-3:
        return None
    dark = smooth < threshold_otsu(smooth)
    lab, n = ndimage.label(dark)
    if n == 0:
        return None
    cy, cx = crop.shape[0] // 2, crop.shape[1] // 2
    k = lab[cy, cx] if lab[cy, cx] > 0 else int(np.argmax(ndimage.sum(dark, lab, range(1, n + 1)))) + 1
    comp = ndimage.binary_fill_holes(lab == k)
    if comp.mean() < 0.10 or comp.mean() > MAX_FILL:
        return None
    if smooth[~comp].mean() - smooth[comp].mean() < MIN_CONTRAST:
        return None
    bw, bh = x2 - x1, y2 - y1
    if max(bw, bh) / max(1, min(bw, bh)) > MAX_ASPECT:
        return None
    mx, my = int(0.3 * bw) + 2, int(0.3 * bh) + 2
    ox1, oy1, ox2, oy2 = max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx), min(H, y2 + my)
    ring = np.ones((oy2 - oy1, ox2 - ox1), bool)
    ring[y1 - oy1:y2 - oy1, x1 - ox1:x2 - ox1] = False
    if ring.any() and gray[oy1:oy2, ox1:ox2][ring].mean() - crop[comp].mean() < MIN_CONTRAST:
        return None
    full = np.zeros((H, W), bool); full[y1:y2, x1:x2] = comp
    return full

def to_lowres(mask, H, W):
    s = SAM_SIZE / max(H, W); rh, rw = int(round(H * s)), int(round(W * s))
    t = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None], size=(rh, rw), mode="bilinear", align_corners=False)
    t = F.pad(t, (0, SAM_SIZE - rw, 0, SAM_SIZE - rh))
    return (F.interpolate(t, size=(LOWRES, LOWRES), mode="area")[0, 0] > 0.5).to(torch.uint8)

def read_yolo(txt, W, H):
    boxes = []
    for line in open(txt).read().strip().splitlines():
        v = [float(x) for x in line.split()[1:]]
        if len(v) == 4:
            cx, cy, w, h = v; boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        elif len(v) >= 6:
            xs, ys = np.array(v[0::2]) * W, np.array(v[1::2]) * H; boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    return boxes


# Figshare boxes to pseudo-masks
def dhash(path, n=8):
    a = np.asarray(Image.open(path).convert("L").resize((n + 1, n)), dtype=float)
    return (a[:, 1:] > a[:, :-1]).ravel()

kaggle_hashes = [dhash(p) for p in pd.read_csv(KAGGLE_CSV)["image_path"]]
imgs = []
for root in FIG_ROOTS:
    if not os.path.isdir(root):
        print(f"  (skipping missing folder: {root})"); continue
    found = sorted(p for p in glob.glob(f"{root}/**/images/*", recursive=True) if p.lower().endswith(IMG_EXT))
    print(f"  {len(found)} images in {root}"); imgs += found
if not imgs:
    raise SystemExit("No Figshare images found, check FIG_ROOTS.")
items, n_fol, n_ok, seen, n_dup, n_kag = [], 0, 0, set(), 0, 0
for p in imgs:
    lbl = os.path.join(os.path.dirname(os.path.dirname(p)), "labels", os.path.splitext(os.path.basename(p))[0] + ".txt")
    if not os.path.exists(lbl):
        continue
    im = Image.open(p).convert("L")
    h = hashlib.md5(im.tobytes()).hexdigest()
    if h in seen:
        n_dup += 1; continue
    seen.add(h)
    dh = dhash(p)
    if min(int((dh != t).sum()) for t in kaggle_hashes) <= DHASH_MAX:
        n_kag += 1; continue
    gray = np.asarray(im, dtype=np.float32) / 255.0
    H, W = gray.shape
    boxes, targets = [], []
    for b in read_yolo(lbl, W, H):
        n_fol += 1
        m = follicle_mask(gray, b)
        if m is None:
            continue
        boxes.append(b); targets.append(np.packbits(to_lowres(m, H, W).numpy().ravel())); n_ok += 1
    if boxes:
        items.append({"path": p, "H": H, "W": W, "boxes": boxes, "targets": targets})
random.Random(0).shuffle(items)
n_val = max(1, int(VAL_FRAC * len(items)))
val_items, train_items = items[:n_val], items[n_val:]
print(f"Figshare: {len(items)} images used ({n_dup} duplicates across exports dropped, "
      f"{n_kag} near-copies of Kaggle images dropped), {n_fol} human boxes, {n_ok} usable pseudo-masks "
      f"-> train {len(train_items)} / val {len(val_items)}")


# training
proc = SamProcessor.from_pretrained(MEDSAM_ID)
sam = SamModel.from_pretrained(MEDSAM_ID).to(device)
DEC_OFF = {k: v.detach().clone() for k, v in sam.mask_decoder.state_dict().items()}
for p in sam.parameters():
    p.requires_grad = False
for p in sam.mask_decoder.parameters():
    p.requires_grad = True

os.makedirs(EMB_DIR, exist_ok=True)
@torch.no_grad()
def emb_path(path):
    f = os.path.join(EMB_DIR, hashlib.md5(path.encode()).hexdigest() + ".pt")
    if not os.path.exists(f):
        pv = proc(Image.open(path).convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
        torch.save(sam.get_image_embeddings(pv).half().cpu(), f)
    return f

t0 = time.time()
for k, it in enumerate(train_items + val_items):
    it["emb"] = emb_path(it["path"])
    if (k + 1) % 50 == 0:
        print(f"  embeddings {k + 1}/{len(items)}", flush=True)
print(f"Embeddings cached ({time.time() - t0:.0f}s)")

def unpack(t):
    return torch.from_numpy(np.unpackbits(t)[: LOWRES * LOWRES].reshape(LOWRES, LOWRES))

def jitter(box, W, H, rng):
    x1, y1, x2, y2 = box; w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2 + rng.uniform(-JITTER, JITTER) * w; cy = (y1 + y2) / 2 + rng.uniform(-JITTER, JITTER) * h
    nw, nh = w * np.exp(rng.uniform(-JITTER, JITTER)), h * np.exp(rng.uniform(-JITTER, JITTER))
    return [max(0, cx - nw / 2), max(0, cy - nh / 2), min(W, cx + nw / 2), min(H, cy + nh / 2)]

def forward(it, idx, rng):
    s = SAM_SIZE / max(it["H"], it["W"])
    boxes = [[v * s for v in jitter(it["boxes"][i], it["W"], it["H"], rng)] for i in idx]
    emb = torch.load(it["emb"], map_location=device, weights_only=True).float()
    out = sam(image_embeddings=emb, input_boxes=torch.tensor([boxes], dtype=torch.float32, device=device), multimask_output=False)
    return out.pred_masks[0, :, 0], torch.stack([unpack(it["targets"][i]) for i in idx]).float().to(device)

def dice_loss(lg, tg):
    p = torch.sigmoid(lg); inter = (p * tg).sum((1, 2))
    return 1 - ((2 * inter + 1) / (p.sum((1, 2)) + tg.sum((1, 2)) + 1)).mean()

@torch.no_grad()
def val_dice(its):
    sam.eval(); rng = np.random.default_rng(0); sc = []
    for it in its:
        idx = list(range(len(it["boxes"])))
        for c in range(0, len(idx), MAX_PROMPTS):
            lg, tg = forward(it, idx[c:c + MAX_PROMPTS], rng)
            pr = (torch.sigmoid(lg) > 0.5).float(); inter = (pr * tg).sum((1, 2))
            sc += ((2 * inter + 1e-6) / (pr.sum((1, 2)) + tg.sum((1, 2)) + 1e-6)).cpu().tolist()
    return float(np.mean(sc))

print(f"Val Dice before training (off-the-shelf): {val_dice(val_items):.4f}", flush=True)
for seed in SEEDS:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sam.mask_decoder.load_state_dict(DEC_OFF)  # every seed starts from MedSAM
    opt = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(seed); best, best_state = -1, None
    tr = list(train_items)
    for ep in range(EPOCHS):
        sam.train(); random.shuffle(tr); tot = 0.0
        for it in tr:
            idx = random.sample(range(len(it["boxes"])), min(MAX_PROMPTS, len(it["boxes"])))
            lg, tg = forward(it, idx, rng)
            loss = F.binary_cross_entropy_with_logits(lg, tg) + dice_loss(lg, tg)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        v = val_dice(val_items)
        if v > best:
            best, best_state = v, {k: x.detach().cpu().clone() for k, x in sam.mask_decoder.state_dict().items()}
        print(f"[seed {seed}] epoch {ep + 1:2d}/{EPOCHS} | loss {tot / len(tr):.4f} | val Dice {v:.4f}{'  <- best' if v == best else ''}", flush=True)
    torch.save(best_state, OUT_DECODER.format(seed))
    print(f"Saved {OUT_DECODER.format(seed)} (best val Dice {best:.4f})")
