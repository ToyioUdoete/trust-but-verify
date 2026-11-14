"""Zero-shot verifiers of MedGemma's citations: SAM2.1 and Qwen2.5-VL 32B (check, refine, judge)."""

CACHE_PKL = "outputs/test_cache.pkl"
OUT_PKL = "outputs/baselines.pkl"
RUN_SAM2 = True
SAM2_ID = "facebook/sam2.1-hiera-large"
RUN_QWEN = True
QWEN_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
QWEN_MAX_PIXELS = 1024 * 28 * 28
MIN_AREA, MIN_KEEP_IOU = 12, 0.1
GRID = 99

import os
import gc, re, pickle
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

device = "cuda" if torch.cuda.is_available() else "cpu"
cache = pickle.load(open(CACHE_PKL, "rb"))
out = pickle.load(open(OUT_PKL, "rb")) if os.path.exists(OUT_PKL) else {}
for c in cache:
    out.setdefault(c["image"], {})
def save():
    pickle.dump(out, open(OUT_PKL, "wb"))

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0

def masks_to_boxes(masks, boxes):
    """Largest component of each mask -> tight box; keeps the original box when the mask is empty or drifts away."""
    res = []
    for orig, m in zip(boxes, masks):
        m = np.asarray(m).astype(bool).squeeze()
        lab, n = ndimage.label(m)
        if n == 0:
            res.append(list(orig)); continue
        sizes = ndimage.sum(m, lab, range(1, n + 1)); k = int(np.argmax(sizes)) + 1
        ys, xs = np.where(lab == k)
        b = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        res.append(b if sizes[k - 1] >= MIN_AREA and iou(b, orig) >= MIN_KEEP_IOU else list(orig))
    return res


# SAM2
if RUN_SAM2 and any("P_sam2" not in out[c["image"]] for c in cache):
    try:
        from transformers import Sam2Model, Sam2Processor
        proc = Sam2Processor.from_pretrained(SAM2_ID)
        sam2 = Sam2Model.from_pretrained(SAM2_ID).to(device).eval()
        for k, c in enumerate(cache):
            if "P_sam2" in out[c["image"]] or not c["P"]:
                out[c["image"]].setdefault("P_sam2", list(c["P"])); continue
            img = Image.open(c["image"]).convert("RGB")
            inp = proc(images=img, input_boxes=[[list(map(float, b)) for b in c["P"]]], return_tensors="pt").to(device)
            with torch.no_grad():
                o = sam2(**inp, multimask_output=False)
            masks = proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])[0]
            out[c["image"]]["P_sam2"] = masks_to_boxes(masks.numpy(), c["P"])
            if (k + 1) % 20 == 0:
                save(); print(f"  SAM2 {k + 1}/{len(cache)}", flush=True)
        save(); print("SAM2 done")
        del sam2; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"SAM2 skipped ({type(e).__name__}: {e}). It needs transformers >= 4.56.")


# Qwen2.5-VL (32B, zero-shot)
if RUN_QWEN and any(not {"q_pyes", "q_refine_raw", "q_judge_raw"} <= set(out[c["image"]]) for c in cache):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    qp = AutoProcessor.from_pretrained(QWEN_ID, min_pixels=64 * 28 * 28, max_pixels=QWEN_MAX_PIXELS)
    qm = AutoModelForImageTextToText.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map={"": 0}).eval()
    tok = qp.tokenizer
    CONTEXT = ("You are assisting a radiologist in a research study that checks AI-drafted ovarian ultrasound reports. "
               "All outputs are reviewed by a clinician; you are not making a diagnosis. ")
    BOX_Q = CONTEXT + ("Does the region inside the red box contain an ovarian follicle (a round or oval, dark, "
                       "fluid-filled structure)? Answer yes or no.")
    REFINE_Q = CONTEXT + (
        "This image is {W}x{H} pixels. An AI listed the following follicle boxes as [x1, y1, x2, y2] in pixel "
        "coordinates of this image:\n{lst}\nFor EACH numbered box, return a corrected bounding box that tightly "
        "encloses the follicle (round or oval, dark, fluid-filled structure) that the box points to. If there is no "
        "follicle there, return the box unchanged. Answer only with JSON: a list with one object per box, in the same "
        "order, like {{\"id\": 1, \"bbox_2d\": [x1, y1, x2, y2]}}.")
    JUDGE_Q = CONTEXT + (
        "Below is an AI draft for this image: a follicle list (boxes x1,y1,x2,y2 on a 0-99 grid, diameter in mm) and a "
        "report. Look at the image carefully, count the follicles yourself, and give the values you believe are "
        "correct, as JSON with keys count (int), min_mm (float), max_mm (float), pattern (\"peripheral\" or "
        "\"central\"), pcom (true or false; true requires at least 12 follicles).\n\nDraft:\n{draft}")
    YES = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["yes", "Yes", " yes", " Yes", "YES"]})
    NO = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["no", "No", " no", " No", "NO"]})

    def fit28(img):
        """Resize so both sides are multiples of 28 and the pixel count is at most QWEN_MAX_PIXELS; the model then
        sees exactly this image, so box coordinates in the prompt and in the answer use the same frame."""
        W, H = img.size; s = min(1.0, (QWEN_MAX_PIXELS / (W * H)) ** 0.5)
        W2, H2 = max(28, int(W * s) // 28 * 28), max(28, int(H * s) // 28 * 28)
        return img.resize((W2, H2)), W2 / W, H2 / H

    def build(img, text, prefill=""):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = qp.apply_chat_template(msgs, add_generation_prompt=True) + prefill
        inp = qp(text=[prompt], images=[img], return_tensors="pt")
        return {k: (v.to(qm.device, torch.bfloat16) if v.is_floating_point() else v.to(qm.device)) for k, v in inp.items()}

    @torch.no_grad()
    def gen(img, text, prefill, n):
        inp = build(img, text, prefill); L = inp["input_ids"].shape[-1]
        o = qm.generate(**inp, max_new_tokens=n, do_sample=False)
        return prefill + qp.decode(o[0][L:], skip_special_tokens=True)

    @torch.no_grad()
    def p_yes(img, text):
        lg = qm(**build(img, text)).logits[0, -1].float()
        return torch.sigmoid(torch.logsumexp(lg[YES], 0) - torch.logsumexp(lg[NO], 0)).item()

    for k, c in enumerate(cache):
        rec = out[c["image"]]
        if {"q_pyes", "q_refine_raw", "q_judge_raw"} <= set(rec):
            continue
        img0 = Image.open(c["image"]).convert("RGB")
        img, sx, sy = fit28(img0)
        P = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in c["P"]]
        if "q_pyes" not in rec:
            pys = []
            for b in P:
                m = img.copy(); ImageDraw.Draw(m).rectangle([int(v) for v in b], outline=(255, 0, 0), width=max(2, img.size[0] // 200))
                pys.append(p_yes(m, BOX_Q))
            rec["q_pyes"] = pys
        if "q_refine_raw" not in rec:
            lst = "\n".join(f"{i + 1}: [{int(b[0])}, {int(b[1])}, {int(b[2])}, {int(b[3])}]" for i, b in enumerate(P))
            rec["q_refine_raw"] = gen(img, REFINE_Q.format(W=img.size[0], H=img.size[1], lst=lst), "[", 40 * len(P) + 40) if P else "[]"
            rec["q_scale"] = (sx, sy)
        if "q_judge_raw" not in rec:
            rec["q_judge_raw"] = gen(img, JUDGE_Q.format(draft=str(c["generated"]).strip()), '{"count": ', 80)
        save(); torch.cuda.empty_cache()
        print(f"  Qwen {k + 1}/{len(cache)} | box check kept {sum(p >= 0.5 for p in rec['q_pyes'])}/{len(P)}", flush=True)
    print("Qwen done")

# boxes that cannot be parsed are left unchanged
for c in cache:
    rec = out[c["image"]]
    if "q_refine_raw" in rec:
        sx, sy = rec.get("q_scale", (1.0, 1.0))
        got = {int(m.group(1)): [float(m.group(i)) for i in range(2, 6)]
               for m in re.finditer(r'"id"\s*:\s*(\d+)\s*,\s*"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', rec["q_refine_raw"])}
        boxes, n_ok = [], 0
        for i, b in enumerate(c["P"]):
            g = got.get(i + 1)
            if g and g[2] > g[0] and g[3] > g[1]:
                nb = [g[0] / sx, g[1] / sy, g[2] / sx, g[3] / sy]
                boxes.append(nb); n_ok += 1
            else:
                boxes.append(list(b))
        rec["P_qwen"] = boxes; rec["q_refine_parsed"] = n_ok
save()
n_par = sum(out[c["image"]].get("q_refine_parsed", 0) for c in cache); n_box = sum(len(c["P"]) for c in cache)
print(f"Saved {OUT_PKL}. Qwen refinement parsed for {n_par}/{n_box} boxes; "
      f"keys per image: {sorted(out[cache[0]['image']].keys())}")
