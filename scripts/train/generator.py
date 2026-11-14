"""Fine-tune MedGemma 1.5 on Figshare, cite follicles on Kaggle, and re-measure the citations."""

FIG_SOURCES = [
    "data/figshare/figshare_part2",
    "data/figshare/figshare_part1",
]
KAGGLE_CSV = "data/kaggle_references.csv"
MAP_CSV, LABEL_DIR = "data/test_annotation_map.csv", "data/test_labels"
MODEL_ID = "google/medgemma-1.5-4b-it"
MEDSAM_ID = "wanglab/medsam-vit-base"
DECODERS = "checkpoints/decoder_seed*.pt"
COUNTER_WEIGHTS, COUNTER_JSON = "checkpoints/counter_yolo11s.pt", "checkpoints/counter.json"
ADAPTER = "checkpoints/medgemma_lora"
OUT = "outputs"
EPOCHS, MAX_NEW_TOKENS = 20, 512
PX_TO_MM, GRID = 0.15, 99
PCOM_THRESHOLD = 12
PATTERN_CUTOFF, PATTERN_MARGIN = 0.6, 0.1
MIN_AREA, MIN_KEEP_IOU = 12, 0.1
DHASH_MAX = 6
PROMPT = ("Generate a grounded ovarian ultrasound report. First list each follicle as "
          "F# [x1,y1,x2,y2] diameter, with box coordinates on a 0-99 grid, then write the report.")

import os, sys, gc, glob, json, math, pickle, re, hashlib, zipfile
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

os.makedirs(OUT, exist_ok=True)
def log(s):
    print(s, flush=True); open(f"{OUT}/generator_log.txt", "a").write(str(s) + "\n")

PERIPH = "Peripheral distribution of follicles noted."
CENTRAL = "Follicles are centrally/scattered distributed."
PCOM_POS = ("Follicle number and size distribution are compatible with polycystic ovarian morphology (PCOM). "
            "Correlate with clinical and biochemical findings before diagnosing PCOS.")
PCOM_NEG = ("Findings do not meet follicle number criteria for polycystic ovarian morphology. "
            "If symptoms persist, correlate with endocrine evaluation and follow-up imaging as clinically indicated.")
POS = "compatible with polycystic ovarian morphology"
COUNT_RANGE = re.compile(r"(\d+)\s+follicles?\s+identified\s+measuring\s+([\d.]+)-([\d.]+)\s*mm")
EV = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*([\d.]+)\s*mm")

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0
def diam(b): return ((b[2] - b[0]) + (b[3] - b[1])) / 2 * PX_TO_MM
def rad(B, W, H): return float(np.mean([np.hypot((b[0] + b[2]) / 2 - W / 2, (b[1] + b[3]) / 2 - H / 2) / (min(W, H) / 2) for b in B]))
def pat(B, W, H): return None if len(B) < 2 else ("peripheral" if rad(B, W, H) > PATTERN_CUTOFF else "central")
def decisive(B, W, H): return len(B) >= 2 and abs(rad(B, W, H) - PATTERN_CUTOFF) >= PATTERN_MARGIN
def report_from_boxes(B, W, H):
    """Rule engine and template used for the Kaggle reference reports."""
    d = [diam(b) for b in B]
    first = f"{len(B)} follicles identified measuring {min(d):.1f}-{max(d):.1f} mm."
    if len(B) >= PCOM_THRESHOLD:
        return " ".join([first, PERIPH if pat(B, W, H) == "peripheral" else CENTRAL, PCOM_POS])
    return " ".join([first, PCOM_NEG])
def target_from_boxes(B, W, H):
    order = sorted(range(len(B)), key=lambda i: (round(B[i][0] / W * 10), B[i][1]))
    ev = [f"F{k} [{round(B[i][0]/W*GRID)},{round(B[i][1]/H*GRID)},{round(B[i][2]/W*GRID)},{round(B[i][3]/H*GRID)}] {diam(B[i]):.1f}mm"
          for k, i in enumerate(order, 1)]
    return "Evidence: " + "; ".join(ev) + ". Report: " + report_from_boxes(B, W, H)
def read_yolo(txt, W, H):
    out = []
    for line in open(txt).read().strip().splitlines():
        v = [float(x) for x in line.split()[1:]]
        if len(v) == 4:
            out.append([(v[0] - v[2] / 2) * W, (v[1] - v[3] / 2) * H, (v[0] + v[2] / 2) * W, (v[1] + v[3] / 2) * H])
        elif len(v) >= 6:
            xs, ys = np.array(v[0::2]) * W, np.array(v[1::2]) * H; out.append([xs.min(), ys.min(), xs.max(), ys.max()])
    return out
def dhash(p, n=8):
    a = np.asarray(Image.open(p).convert("L").resize((n + 1, n)), dtype=float); return (a[:, 1:] > a[:, :-1]).ravel()


# 1. Kaggle images and reference reports
g = pd.read_csv(KAGGLE_CSV)
kag = []
for _, r in g.iterrows():
    W, H = int(r["img_w"]), int(r["img_h"])
    kag.append({"image": r["image_path"], "W": W, "H": H, "ref_report": str(r["report"]),
                "R": [[b["xyxy_norm"][0] * W, b["xyxy_norm"][1] * H, b["xyxy_norm"][2] * W, b["xyxy_norm"][3] * H] for b in json.loads(r["boxes_json"])]})
# the rule engine should reproduce the reference reports
same = np.mean([report_from_boxes(k["R"], k["W"], k["H"]).split(".")[0] == k["ref_report"].split(".")[0] for k in kag if k["R"]])
log(f"Kaggle: {len(kag)} images. First sentence reproduced from the reference boxes in {100*same:.0f}% of reports")
log(f"  example reference: {kag[0]['ref_report'][:160]}")
log(f"  rebuilt from boxes: {report_from_boxes(kag[0]['R'], kag[0]['W'], kag[0]['H'])[:160]}")
idmap = pd.read_csv(MAP_CSV).set_index("id")["image"].to_dict() if os.path.exists(MAP_CSV) else {}
labelled = set(idmap.values())
kag.sort(key=lambda k: k["image"] not in labelled)  # the 100 annotated test images first


# 2. Figshare training targets
FIG_CSV = f"{OUT}/figshare_targets.csv"
if not os.path.exists(FIG_CSV):
    roots = []
    for src in FIG_SOURCES:
        if src.endswith(".zip") and os.path.exists(src):
            dst = os.path.join(os.path.dirname(src), os.path.splitext(os.path.basename(src))[0].replace(" ", "_"))
            if not os.path.isdir(dst):
                zipfile.ZipFile(src).extractall(dst); log(f"unzipped {src} -> {dst}")
            roots.append(dst)
        elif os.path.isdir(src):
            roots.append(src)
        else:
            log(f"  (missing, skipped: {src})")
    kh = [dhash(k["image"]) for k in kag]
    rows, seen, ndup, nkag = [], set(), 0, 0
    for root in roots:
        for p in sorted(glob.glob(f"{root}/**/images/*", recursive=True)):
            if not p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            lbl = os.path.join(os.path.dirname(os.path.dirname(p)), "labels", os.path.splitext(os.path.basename(p))[0] + ".txt")
            if not os.path.exists(lbl):
                continue
            im = Image.open(p).convert("L"); h = hashlib.md5(im.tobytes()).hexdigest()
            if h in seen:
                ndup += 1; continue
            seen.add(h)
            if (lambda hp: min(int((hp != t).sum()) for t in kh))(dhash(p)) <= DHASH_MAX:
                nkag += 1; continue
            W, H = im.size; B = read_yolo(lbl, W, H)
            if B:
                rows.append({"image": p, "source": os.path.basename(root), "W": W, "H": H, "n": len(B), "target": target_from_boxes(B, W, H)})
    pd.DataFrame(rows).to_csv(FIG_CSV, index=False)
    log(f"Figshare training set: {len(rows)} images ({ndup} duplicates, {nkag} near-copies of Kaggle images removed) "
        f"from {[os.path.basename(r) for r in roots]}")
fig = pd.read_csv(FIG_CSV)
log(f"Figshare: {len(fig)} training images, {fig['n'].sum()} follicles, PCOM+ {100*(fig['n'] >= PCOM_THRESHOLD).mean():.0f}% "
    f"| example target: {fig['target'].iloc[0][:150]}...")


# 3. train MedGemma, generate on Kaggle
PRED = f"{OUT}/kaggle_predictions.csv"
done = set(pd.read_csv(PRED)["image"]) if os.path.exists(PRED) else set()
if len(done) < len(kag):
    import torch
    from torch.utils.data import Dataset
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments
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
    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0})
    adir = ADAPTER
    if os.path.exists(f"{adir}/adapter_config.json"):
        log(f"Loading adapter from {adir}"); model = PeftModel.from_pretrained(base, adir)
    else:
        log(f"Training MedGemma on {len(fig)} Figshare images ({EPOCHS} epochs)")
        base.config.use_cache = False
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        model.enable_input_require_grads()
        steps = math.ceil(len(fig) / 16) * EPOCHS
        args = TrainingArguments(output_dir=f"{OUT}/medgemma_ckpt", num_train_epochs=EPOCHS, learning_rate=2e-4, warmup_steps=int(0.1 * steps),
                                 per_device_train_batch_size=1, gradient_accumulation_steps=16, eval_strategy="no",
                                 save_strategy="epoch", save_total_limit=1, logging_steps=10, bf16=True,
                                 remove_unused_columns=False, report_to="none", dataloader_num_workers=0, seed=42)
        ck = sorted(glob.glob(f"{OUT}/medgemma_ckpt/checkpoint-*"), key=lambda s: int(s.rsplit("-", 1)[1]))
        Trainer(model=model, args=args, train_dataset=DS(fig), data_collator=Coll()).train(resume_from_checkpoint=ck[-1] if ck else None)
        model.save_pretrained(adir); log("Training done")
    model.eval(); model.config.use_cache = True
    buf = []
    for i, k in enumerate(kag):
        if k["image"] in done:
            continue
        inp = processor(images=Image.open(k["image"]).convert("RGB"), text=chat(), return_tensors="pt")
        inp = {kk: (v.to(model.device, torch.bfloat16) if v.is_floating_point() else v.to(model.device)) for kk, v in inp.items()}
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        buf.append({"image": k["image"], "generated": processor.decode(o[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)})
        if len(buf) == 10 or i == len(kag) - 1:
            pd.DataFrame(buf).to_csv(PRED, mode="a", header=not os.path.exists(PRED), index=False); buf = []
            log(f"generated {len(set(pd.read_csv(PRED)['image']))}/{len(kag)} Kaggle images")
    del model, base; gc.collect(); torch.cuda.empty_cache()
pred = pd.read_csv(PRED).drop_duplicates("image").set_index("image")["generated"].to_dict()


# 4. re-measure citations, run the counter
CACHE = f"{OUT}/kaggle_cache.pkl"
cache = {c["image"]: c for c in pickle.load(open(CACHE, "rb"))} if os.path.exists(CACHE) else {}
todo = [k for k in kag if k["image"] in pred and k["image"] not in cache]
decs = sorted(glob.glob(DECODERS))
if not decs:
    log(f"Generation finished, but no decoders match {DECODERS}. Run scripts/train/decoder.py, then run this again (it resumes here).")
    sys.exit(0)
if todo:
    import torch, torch.nn.functional as F
    from transformers import SamModel, SamProcessor
    dev = "cuda"
    proc = SamProcessor.from_pretrained(MEDSAM_ID); sam = SamModel.from_pretrained(MEDSAM_ID).to(dev).eval()
    DEC = {"off": {a: b.clone() for a, b in sam.mask_decoder.state_dict().items()}}
    for j, d in enumerate(decs):
        DEC[f"fig{j}"] = torch.load(d, map_location=dev)
    @torch.no_grad()
    def refine(img, emb, B, dec):
        if not B:
            return []
        sam.mask_decoder.load_state_dict(dec); W, H = img.size
        o = sam(image_embeddings=emb, input_boxes=proc(img, input_boxes=[B], return_tensors="pt")["input_boxes"].to(dev), multimask_output=False)
        s = 1024 / max(H, W); rh, rw = int(round(H * s)), int(round(W * s))
        m = F.interpolate(o.pred_masks[0, :, 0][:, None], size=(1024, 1024), mode="bilinear", align_corners=False)
        m = (torch.sigmoid(F.interpolate(m[..., :rh, :rw], size=(H, W), mode="bilinear", align_corners=False)[:, 0]) > .5).cpu().numpy()
        res = []
        for orig, mask in zip(B, m):
            lab, n = ndimage.label(mask)
            if n == 0:
                res.append(list(orig)); continue
            sz = ndimage.sum(mask, lab, range(1, n + 1)); kk = int(np.argmax(sz)) + 1; ys, xs = np.where(lab == kk)
            b = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            res.append(b if sz[kk - 1] >= MIN_AREA and iou(b, orig) >= MIN_KEEP_IOU else list(orig))
        return res
    for i, k in enumerate(todo):
        img = Image.open(k["image"]).convert("RGB"); W, H = img.size
        ev, _, rep = str(pred[k["image"]]).partition("Report:")
        items = EV.findall(ev)
        P = [[int(a) / GRID * W, int(b) / GRID * H, int(c) / GRID * W, int(d) / GRID * H] for a, b, c, d, _ in items]
        emb = sam.get_image_embeddings(proc(img, return_tensors="pt")["pixel_values"].to(dev))
        e = {**k, "W": W, "H": H, "generated": pred[k["image"]], "orig_report": rep.strip(), "P": P,
             "listed_d": [float(x[4]) for x in items], "P_off": refine(img, emb, P, DEC["off"])}
        for j in range(len(decs)):
            e[f"P_fig{j}"] = refine(img, emb, P, DEC[f"fig{j}"])
        e["P_fig"] = e.get("P_fig0", P)
        cache[k["image"]] = e
        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            pickle.dump(list(cache.values()), open(CACHE, "wb")); log(f"re-measured {i + 1}/{len(todo)}")
    del sam; gc.collect(); torch.cuda.empty_cache()
cases = [cache[k["image"]] for k in kag if k["image"] in cache]

DETX = f"{OUT}/counter_boxes.pkl"
if not os.path.exists(DETX):
    from ultralytics import YOLO
    cfg = json.load(open(COUNTER_JSON))
    counter = YOLO(COUNTER_WEIGHTS); boxes = {}
    for c in cases:
        res = counter.predict(source=c["image"], conf=0.05, iou=cfg["nms_iou"], imgsz=cfg["imgsz"], verbose=False)[0]
        boxes[c["image"]] = {"yolo_fig": [b.xyxy[0].tolist() + [float(b.conf[0])] for b in res.boxes]}
    pickle.dump({"boxes": boxes, "conf": {"yolo_fig": cfg["conf"]}}, open(DETX, "wb")); log(f"Saved {DETX}")
det = pickle.load(open(DETX, "rb"))
for c in cases:
    c["D_fig"] = [b[:4] for b in det["boxes"][c["image"]]["yolo_fig"] if b[4] >= det["conf"]["yolo_fig"]]

# blind annotations of the 100 test images
for f in glob.glob(f"{LABEL_DIR}/**/*.txt", recursive=True):
    m = re.match(r"(test_\d{3})", os.path.basename(f))
    if m and idmap.get(m.group(1)) in cache:
        c = cache[idmap[m.group(1)]]; c["Hum"] = read_yolo(f, c["W"], c["H"])
lab100 = [c for c in cases if "Hum" in c]
pickle.dump(lab100, open(f"{OUT}/test_cache.pkl", "wb"))
log(f"{len(cases)} Kaggle images processed, {len(lab100)} with blind annotations -> {OUT}/test_cache.pkl")
