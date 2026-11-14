"""Train the error-aware MedSAM decoder on MedGemma's out-of-fold citations and pick the rejection threshold."""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

OUT = "outputs/error_aware"
FIG_CSV = "outputs/figshare_targets.csv"
MODEL_ID = "google/medgemma-1.5-4b-it"
MEDSAM_ID = "wanglab/medsam-vit-base"
OLD_DECODERS = "checkpoints/decoder_seed{}.pt"
OUT_DECODER = "checkpoints/decoder_error_aware_seed{}.pt"
OUT_THRESHOLDS = "checkpoints/rejection_thresholds.json"
SEEDS = [0, 1, 2]
EPOCHS_VLM, MAX_NEW_TOKENS = 20, 512
EPOCHS_DEC, LR_DEC, JITTER, MAX_PROMPTS, VAL_FRAC = 30, 1e-4, 0.3, 16, 0.1
PX_TO_MM, GRID, PCOM, CUT, MARGIN, MIN_AREA, MIN_KEEP_IOU = 0.15, 99, 12, 0.6, 0.1, 12, 0.1
MAX_FILL, MAX_ASPECT, MIN_CONTRAST = 0.90, 2.5, 0.08
PROMPT = ("Generate a grounded ovarian ultrasound report. First list each follicle as "
          "F# [x1,y1,x2,y2] diameter, with box coordinates on a 0-99 grid, then write the report.")

import gc, glob, math, re, json, random, hashlib
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
os.makedirs(OUT, exist_ok=True)
fig = pd.read_csv(FIG_CSV)
EV = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*([\d.]+)\s*mm")
CR = re.compile(r"(\d+)\s+follicles?\s+identified\s+measuring\s+([\d.]+)-([\d.]+)\s*mm")

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
def read_yolo(txt, W, H):
    out = []
    for line in open(txt).read().strip().splitlines():
        v = [float(x) for x in line.split()[1:]]
        if len(v) == 4:
            out.append([(v[0] - v[2] / 2) * W, (v[1] - v[3] / 2) * H, (v[0] + v[2] / 2) * W, (v[1] + v[3] / 2) * H])
        elif len(v) >= 6:
            xs, ys = np.array(v[0::2]) * W, np.array(v[1::2]) * H; out.append([xs.min(), ys.min(), xs.max(), ys.max()])
    return out
def label_of(p):
    return os.path.join(os.path.dirname(os.path.dirname(p)), "labels", os.path.splitext(os.path.basename(p))[0] + ".txt")
def parse_boxes(text, W, H):
    ev = str(text).partition("Report:")[0]
    return [[int(a) / GRID * W, int(b) / GRID * H, int(c) / GRID * W, int(d) / GRID * H] for a, b, c, d, _ in EV.findall(ev)]
# two folds by image; 10% held out for model selection and the threshold
rng0 = random.Random(0); idx = list(range(len(fig))); rng0.shuffle(idx)
fig["fold"] = 0; fig.loc[idx[len(idx) // 2:], "fold"] = 1
fig["dec_val"] = False; fig.loc[idx[: max(1, int(VAL_FRAC * len(idx)))], "dec_val"] = True
print(f"Figshare: {len(fig)} images | fold sizes {fig['fold'].value_counts().to_dict()} | decoder-val images {fig['dec_val'].sum()}")


# 1. out-of-fold MedGemma citations on Figshare
# train on one half, cite on the other, then swap
import torch
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

OOF = f"{OUT}/oof_pred_figshare.csv"
processor = AutoProcessor.from_pretrained(MODEL_ID); eos = processor.tokenizer.eos_token or "<eos>"
chat = lambda: processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
                                             add_generation_prompt=True)
class DS(Dataset):
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]; img = Image.open(r["image"]).convert("RGB"); pt = chat()
        plen = processor(text=pt, images=img, return_tensors="pt")["input_ids"].shape[1]
        full = processor(text=pt + r["target"] + eos, images=img, return_tensors="pt")
        lab = full["input_ids"].squeeze(0).clone(); lab[:plen] = -100
        e = {k: v.squeeze(0) for k, v in full.items()}; e["labels"] = lab; return e
class Coll:
    def __call__(self, b):
        pad = processor.tokenizer.pad({"input_ids": [x["input_ids"] for x in b]}, padding=True, return_tensors="pt"); L = pad["input_ids"].shape[1]
        o = {"input_ids": pad["input_ids"], "attention_mask": pad["attention_mask"], "pixel_values": torch.stack([x["pixel_values"] for x in b]),
             "labels": torch.stack([torch.nn.functional.pad(x["labels"], (0, L - x["labels"].shape[0]), value=-100) for x in b])}
        if "pixel_attention_mask" in b[0]:
            o["pixel_attention_mask"] = torch.stack([x["pixel_attention_mask"] for x in b])
        return o

for fold in (0, 1):
    held = fig[fig["fold"] == fold]; train = fig[fig["fold"] != fold]
    done = set(pd.read_csv(OOF)["image"]) if os.path.exists(OOF) else set()
    if all(p in done for p in held["image"]):
        print(f"fold {fold}: already generated"); continue
    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0})
    adir, ck = f"{OUT}/adapter_fold{fold}", f"{OUT}/ckpt_fold{fold}"
    if os.path.exists(f"{adir}/adapter_config.json"):
        model = PeftModel.from_pretrained(base, adir); print(f"fold {fold}: adapter found")
    else:
        print(f"fold {fold}: training MedGemma on {len(train)} Figshare images, citing on {len(held)} held-out images", flush=True)
        base.config.use_cache = False
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        model.enable_input_require_grads()
        steps = math.ceil(len(train) / 16) * EPOCHS_VLM
        args = TrainingArguments(output_dir=ck, num_train_epochs=EPOCHS_VLM, learning_rate=2e-4, warmup_steps=int(0.1 * steps),
                                 per_device_train_batch_size=1, gradient_accumulation_steps=16, eval_strategy="no",
                                 save_strategy="epoch", save_total_limit=1, logging_steps=10, bf16=True,
                                 remove_unused_columns=False, report_to="none", dataloader_num_workers=0, seed=42)
        cks = sorted(glob.glob(f"{ck}/checkpoint-*"), key=lambda s: int(s.rsplit("-", 1)[1]))
        Trainer(model=model, args=args, train_dataset=DS(train), data_collator=Coll()).train(resume_from_checkpoint=cks[-1] if cks else None)
        model.save_pretrained(adir)
    model.eval(); model.config.use_cache = True; buf = []
    todo = [p for p in held["image"] if p not in done]
    for i, p in enumerate(todo):
        inp = processor(images=Image.open(p).convert("RGB"), text=chat(), return_tensors="pt")
        inp = {k: (v.to(model.device, torch.bfloat16) if v.is_floating_point() else v.to(model.device)) for k, v in inp.items()}
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        buf.append({"image": p, "fold": fold, "generated": processor.decode(o[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)})
        if len(buf) == 10 or i == len(todo) - 1:
            pd.DataFrame(buf).to_csv(OOF, mode="a", header=not os.path.exists(OOF), index=False); buf = []
            print(f"  fold {fold}: cited on {i + 1}/{len(todo)} held-out images", flush=True)
    del model, base; gc.collect(); torch.cuda.empty_cache()

oof = pd.read_csv(OOF).drop_duplicates("image")
F = []
for _, r in oof.iterrows():
    im = Image.open(r["image"]); W, H = im.size
    F.append(match(parse_boxes(r["generated"], W, H), read_yolo(label_of(r["image"]), W, H))[0])
print(f"\nMedGemma on held-out Figshare images: evidence F1 = {prf(sum(F))[2]:.3f}")


# 2. error-aware decoder (3 seeds)
# half the prompts are MedGemma's citations, half are jittered annotated boxes
import torch, torch.nn.functional as Fnn
from skimage.filters import threshold_otsu
from transformers import SamModel, SamProcessor
dev = "cuda"; SAM_SIZE, LOWRES = 1024, 256

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
    dark = smooth < threshold_otsu(smooth); lab, n = ndimage.label(dark)
    if n == 0:
        return None
    cy, cx = crop.shape[0] // 2, crop.shape[1] // 2
    k = lab[cy, cx] if lab[cy, cx] > 0 else int(np.argmax(ndimage.sum(dark, lab, range(1, n + 1)))) + 1
    comp = ndimage.binary_fill_holes(lab == k)
    if comp.mean() < 0.10 or comp.mean() > MAX_FILL or smooth[~comp].mean() - smooth[comp].mean() < MIN_CONTRAST:
        return None
    bw, bh = x2 - x1, y2 - y1
    if max(bw, bh) / max(1, min(bw, bh)) > MAX_ASPECT:
        return None
    full = np.zeros((H, W), bool); full[y1:y2, x1:x2] = comp; return full
def to_lowres(mask, H, W):
    s = SAM_SIZE / max(H, W); rh, rw = int(round(H * s)), int(round(W * s))
    t = Fnn.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None], size=(rh, rw), mode="bilinear", align_corners=False)
    t = Fnn.pad(t, (0, SAM_SIZE - rw, 0, SAM_SIZE - rh))
    return (Fnn.interpolate(t, size=(LOWRES, LOWRES), mode="area")[0, 0] > 0.5).to(torch.uint8)

proc = SamProcessor.from_pretrained(MEDSAM_ID); sam = SamModel.from_pretrained(MEDSAM_ID).to(dev)
DEC_OFF = {k: v.detach().clone() for k, v in sam.mask_decoder.state_dict().items()}
for p in sam.parameters():
    p.requires_grad = False
EMB = f"{OUT}/emb"; os.makedirs(EMB, exist_ok=True)
@torch.no_grad()
def emb_file(path):
    f = f"{EMB}/{hashlib.md5(path.encode()).hexdigest()}.pt"
    if not os.path.exists(f):
        torch.save(sam.get_image_embeddings(proc(Image.open(path).convert("RGB"), return_tensors="pt")["pixel_values"].to(dev)).half().cpu(), f)
    return f

oof = pd.read_csv(f"{OUT}/oof_pred_figshare.csv").drop_duplicates("image").set_index("image")["generated"].to_dict()
items = []
for k, r in fig.iterrows():
    im = Image.open(r["image"]).convert("L"); W, H = im.size; gray = np.asarray(im, np.float32) / 255.0
    G = read_yolo(label_of(r["image"]), W, H)
    gmask = [follicle_mask(gray, b) for b in G]
    gpack = [np.packbits(to_lowres(m, H, W).numpy().ravel()) if m is not None else None for m in gmask]
    P = parse_boxes(oof.get(r["image"], ""), W, H)
    _, pairs = match(P, G, 0.1); j_of = {i: j for i, j in pairs}
    items.append(dict(path=r["image"], W=W, H=H, G=G, gpack=gpack, P=P, pj=[j_of.get(i, -1) for i in range(len(P))],
                      val=bool(r["dec_val"]), emb=emb_file(r["image"])))
    if (k + 1) % 100 == 0:
        print(f"  prepared {k + 1}/{len(fig)}", flush=True)
tr = [it for it in items if not it["val"]]; va = [it for it in items if it["val"]]
nP = sum(len(it["P"]) for it in tr); nM = sum(sum(j >= 0 for j in it["pj"]) for it in tr)
print(f"train {len(tr)} / val {len(va)} images | MedGemma-cited prompts {nP} ({nM} point near a real follicle, {nP - nM} at none)")

def unpack(t): return torch.from_numpy(np.unpackbits(t)[: LOWRES * LOWRES].reshape(LOWRES, LOWRES)).float()
def jit(b, W, H, rng):
    x1, y1, x2, y2 = b; w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2 + rng.uniform(-JITTER, JITTER) * w; cy = (y1 + y2) / 2 + rng.uniform(-JITTER, JITTER) * h
    nw, nh = w * np.exp(rng.uniform(-JITTER, JITTER)), h * np.exp(rng.uniform(-JITTER, JITTER))
    return [max(0, cx - nw / 2), max(0, cy - nh / 2), min(W, cx + nw / 2), min(H, cy + nh / 2)]
def prompts_for(it, rng, n=MAX_PROMPTS):
    """half MedGemma's real cited boxes (matched or not), half jittered human boxes; each with (gt index or -1)"""
    out = []
    real = list(range(len(it["P"]))); rng.shuffle(real)
    for i in real[: n // 2]:
        out.append((it["P"][i], it["pj"][i]))
    good = [j for j, m in enumerate(it["gpack"]) if m is not None]; rng.shuffle(good)
    for j in good[: n - len(out)]:
        out.append((jit(it["G"][j], it["W"], it["H"], rng), j))
    return out
def box_iou_t(a, b):
    ix = (torch.minimum(a[:, 2], b[:, 2]) - torch.maximum(a[:, 0], b[:, 0])).clamp(min=0)
    iy = (torch.minimum(a[:, 3], b[:, 3]) - torch.maximum(a[:, 1], b[:, 1])).clamp(min=0)
    inter = ix * iy; ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / ua.clamp(min=1e-6)
def mask_boxes(prob):  # box of each low-res mask (1024 coords), zeros if empty
    out = []
    for m in (prob > 0.5):
        ys, xs = torch.where(m)
        out.append(torch.stack([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]).float() * 4 if len(xs) else torch.zeros(4, device=prob.device))
    return torch.stack(out)
def forward(it, pr):
    s = SAM_SIZE / max(it["H"], it["W"])
    boxes = torch.tensor([[[v * s for v in b] for b, _ in pr]], dtype=torch.float32, device=dev)
    o = sam(image_embeddings=torch.load(it["emb"], map_location=dev, weights_only=True).float(), input_boxes=boxes, multimask_output=False)
    return o.pred_masks[0, :, 0], o.iou_scores[0, :, 0], s
def losses(it, pr):
    lg, sc, s = forward(it, pr)
    js = [j for _, j in pr]
    has = [k for k, j in enumerate(js) if j >= 0 and it["gpack"][j] is not None]
    loss = torch.zeros((), device=dev)
    if has:
        tg = torch.stack([unpack(it["gpack"][js[k]]) for k in has]).to(dev); l = lg[has]; p = torch.sigmoid(l)
        dice = 1 - ((2 * (p * tg).sum((1, 2)) + 1) / (p.sum((1, 2)) + tg.sum((1, 2)) + 1)).mean()
        loss = loss + Fnn.binary_cross_entropy_with_logits(l, tg) + dice
    with torch.no_grad():  # IoU target, 0 if the prompt has no follicle
        mb = mask_boxes(torch.sigmoid(lg))
        gt = torch.tensor([[v * s for v in it["G"][j]] if j >= 0 else [0, 0, 0, 0] for j in js], dtype=torch.float32, device=dev)
        tgt = box_iou_t(mb, gt) * torch.tensor([float(j >= 0) for j in js], device=dev)
    return loss + Fnn.mse_loss(sc, tgt)

@torch.no_grad()
def val_score(its):
    sam.eval(); C = np.zeros(3)
    for it in its:
        if not it["P"]:
            C += [0, 0, len(it["G"])]; continue
        lg, _, s = forward(it, [(b, -1) for b in it["P"]])
        mb = (mask_boxes(torch.sigmoid(lg)) / s).cpu().numpy().tolist()
        R = [m if (m[2] > m[0] and iou(m, b) >= MIN_KEEP_IOU) else b for m, b in zip(mb, it["P"])]
        C += match(R, it["G"])[0]
    return prf(C)[2]

for p in sam.mask_decoder.parameters():
    p.requires_grad = True
for seed in SEEDS:
    path = OUT_DECODER.format(seed)
    if os.path.exists(path):
        print(f"seed {seed}: {path} exists, skipping"); continue
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    sam.mask_decoder.load_state_dict(DEC_OFF)
    opt = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=LR_DEC, weight_decay=1e-4)
    best, best_state = -1, None
    for ep in range(EPOCHS_DEC):
        sam.train(); order = list(tr); random.shuffle(order); tot = 0.0
        for it in order:
            pr = prompts_for(it, rng)
            if not pr:
                continue
            l = losses(it, pr); opt.zero_grad(); l.backward(); opt.step(); tot += l.item()
        v = val_score(va)
        if v > best:
            best, best_state = v, {k: x.detach().cpu().clone() for k, x in sam.mask_decoder.state_dict().items()}
        print(f"[seed {seed}] epoch {ep + 1:2d}/{EPOCHS_DEC} | loss {tot / len(order):.4f} | val F1 of re-measured MedGemma boxes {v:.3f}"
              f"{'  <- best' if v == best else ''}", flush=True)
    torch.save(best_state, path); print(f"saved {path} (val F1 {best:.3f})")
# reference: the Figshare decoder on the same images
sam.mask_decoder.load_state_dict(torch.load(OLD_DECODERS.format(0), map_location=dev))
print(f"decoder from scripts/train/decoder.py on the same images: val F1 {val_score(va):.3f}")


# 3. rejection threshold: maximises evidence F1 on held-out Figshare
@torch.no_grad()
def scored(it, dec):
    sam.mask_decoder.load_state_dict(dec); sam.eval()
    if not it["P"]:
        return [], []
    lg, sc, s = forward(it, [(b, -1) for b in it["P"]])
    mb = (mask_boxes(torch.sigmoid(lg)) / s).cpu().numpy().tolist()
    R = [m if (m[2] > m[0] and iou(m, b) >= MIN_KEEP_IOU) else b for m, b in zip(mb, it["P"])]
    return R, sc.cpu().numpy().tolist()
GRID_T = np.round(np.arange(0.0, 0.951, 0.05), 2)
THR = {}
for name, dec_path in [("old", OLD_DECODERS.format(0)), ("ea", OUT_DECODER.format(0))]:
    dec = torch.load(dec_path, map_location=dev)
    S = [(scored(it, dec), it["G"]) for it in va]
    f1s = [prf(sum(match([b for b, q in zip(R, sc) if q >= t], G)[0] for (R, sc), G in S))[2] for t in GRID_T]
    THR[name] = float(GRID_T[int(np.argmax(f1s))])
    print(f"{name:4s} decoder: best rejection threshold on Figshare val = {THR[name]:.2f} (val F1 {max(f1s):.3f}; no rejection {f1s[0]:.3f})")
json.dump(THR, open(OUT_THRESHOLDS, "w")); print(f"saved {OUT_THRESHOLDS}")
