"""Fine-tune LLaVA-Med v1.5 and PaliGemma 2 on Figshare like MedGemma and cite follicles on the test images."""

import os, argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
OUT = "outputs/second_vlms"
FIG_CSV = "outputs/figshare_targets.csv"
TEST_PKL = "outputs/test_cache.pkl"
EPOCHS, SEED, MAX_NEW_TOKENS = 20, 42, 512
PROMPT = ("Generate a grounded ovarian ultrasound report. First list each follicle as "
          "F# [x1,y1,x2,y2] diameter, with box coordinates on a 0-99 grid, then write the report.")

import gc, glob, math, pickle
import pandas as pd
from PIL import Image
os.makedirs(OUT, exist_ok=True)
fig = pd.read_csv(FIG_CSV)
test = pickle.load(open(TEST_PKL, "rb"))
TEST_IMAGES = [c["image"] for c in test]
print(f"Training: {len(fig)} Figshare images | Test: {len(TEST_IMAGES)} Kaggle images with ground truth")
print("Example target:", fig["target"].iloc[0][:200], "...")

def free_gpu():
    import torch
    gc.collect(); torch.cuda.empty_cache()

def generate_all(name, gen_fn):
    """gen_fn(PIL image) -> text. Resumable: images already in the csv are skipped."""
    path = f"{OUT}/pred_{name}.csv"
    done = set(pd.read_csv(path)["image"]) if os.path.exists(path) else set()
    buf = []
    for i, p in enumerate(TEST_IMAGES):
        if p in done:
            continue
        buf.append({"image": p, "generated": gen_fn(Image.open(p).convert("RGB"))})
        if len(buf) == 10 or i == len(TEST_IMAGES) - 1:
            pd.DataFrame(buf).to_csv(path, mode="a", header=not os.path.exists(path), index=False); buf = []
            print(f"  {name}: generated {len(set(pd.read_csv(path)['image']))}/{len(TEST_IMAGES)}", flush=True)
    print(f"{name}: done -> {path}")
    print("example output:", pd.read_csv(path)["generated"].iloc[0][:300])

def train_args(outdir):
    from transformers import TrainingArguments
    steps = math.ceil(len(fig) / 16) * EPOCHS
    return TrainingArguments(output_dir=outdir, num_train_epochs=EPOCHS, learning_rate=2e-4, warmup_steps=int(0.1 * steps),
                             per_device_train_batch_size=1, gradient_accumulation_steps=16, eval_strategy="no",
                             save_strategy="epoch", save_total_limit=1, logging_steps=10, bf16=True,
                             gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                             remove_unused_columns=False, report_to="none", dataloader_num_workers=0, seed=SEED)
def last_ckpt(outdir):
    ck = sorted(glob.glob(f"{outdir}/checkpoint-*"), key=lambda s: int(s.rsplit("-", 1)[1]))
    return ck[-1] if ck else None
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

def run_llavamed():
    import torch
    from torch.utils.data import Dataset
    from transformers import LlavaForConditionalGeneration, AutoProcessor, Trainer
    from peft import LoraConfig, get_peft_model, PeftModel

    NAME, MODEL_ID = "llavamed", "chaoyinshe/llava-med-v1.5-mistral-7b-hf"  # LLaVA-Med v1.5 in HF format
    ADIR, CK = f"{OUT}/adapter_{NAME}", f"{OUT}/ckpt_{NAME}"
    processor = AutoProcessor.from_pretrained(MODEL_ID); eos = processor.tokenizer.eos_token
    prompt_text = lambda: processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
                                                        add_generation_prompt=True)
    class DS(Dataset):
        def __init__(self, df): self.df = df.reset_index(drop=True)
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            r = self.df.iloc[i]; img = Image.open(r["image"]).convert("RGB"); pt = prompt_text()
            enc = processor(text=pt + r["target"] + eos, images=img, return_tensors="pt")
            plen = processor(text=pt, images=img, return_tensors="pt")["input_ids"].shape[1]
            enc = {k: v.squeeze(0) for k, v in enc.items()}; lab = enc["input_ids"].clone(); lab[:plen] = -100; enc["labels"] = lab
            return enc
    def collate(b):
        out = {}
        for k in b[0]:
            if k == "pixel_values":
                out[k] = torch.stack([x[k] for x in b])
            else:
                L = max(x[k].shape[0] for x in b)
                out[k] = torch.stack([torch.nn.functional.pad(x[k], (0, L - x[k].shape[0]), value=-100 if k == "labels" else 0) for x in b])
        return out

    base = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0})
    if os.path.exists(f"{ADIR}/adapter_config.json"):
        print(f"loading adapter from {ADIR}"); model = PeftModel.from_pretrained(base, ADIR)
    else:
        print(f"Training {NAME} on {len(fig)} Figshare images ({EPOCHS} epochs)")
        model = get_peft_model(base, LoraConfig(**LORA)); model.enable_input_require_grads()
        Trainer(model=model, args=train_args(CK), train_dataset=DS(fig), data_collator=collate).train(resume_from_checkpoint=last_ckpt(CK))
        model.save_pretrained(ADIR); print("training done")
    model.eval(); model.config.use_cache = True
    @torch.no_grad()
    def gen(img):
        inp = processor(images=img, text=prompt_text(), return_tensors="pt")
        inp = {k: (v.to(model.device, torch.bfloat16) if v.is_floating_point() else v.to(model.device)) for k, v in inp.items()}
        o = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return processor.decode(o[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    generate_all(NAME, gen)
    del model, base
    free_gpu()


def run_paligemma2():
    import torch
    from torch.utils.data import Dataset
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, Trainer
    from peft import LoraConfig, get_peft_model, PeftModel

    NAME, MODEL_ID = "paligemma2", "google/paligemma2-3b-mix-448"
    ADIR, CK = f"{OUT}/adapter_{NAME}", f"{OUT}/ckpt_{NAME}"
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
    class DS(Dataset):
        def __init__(self, df): self.df = df.reset_index(drop=True)
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            r = self.df.iloc[i]
            enc = processor(images=Image.open(r["image"]).convert("RGB"), text=PROMPT, suffix=r["target"], return_tensors="pt", padding=False)
            return {k: v.squeeze(0) for k, v in enc.items()}
    def collate(b):
        out = {}
        for k in b[0]:
            if k == "pixel_values":
                out[k] = torch.stack([x[k] for x in b])
            else:
                L = max(x[k].shape[0] for x in b)
                pv = -100 if k == "labels" else 0
                out[k] = torch.stack([torch.nn.functional.pad(x[k], (0, L - x[k].shape[0]), value=pv) for x in b])
        return out

    base = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0})
    if os.path.exists(f"{ADIR}/adapter_config.json"):
        print(f"loading adapter from {ADIR}"); model = PeftModel.from_pretrained(base, ADIR)
    else:
        print(f"Training {NAME} on {len(fig)} Figshare images ({EPOCHS} epochs)")
        model = get_peft_model(base, LoraConfig(**LORA)); model.enable_input_require_grads()
        Trainer(model=model, args=train_args(CK), train_dataset=DS(fig), data_collator=collate).train(resume_from_checkpoint=last_ckpt(CK))
        model.save_pretrained(ADIR); print("training done")
    model.eval()
    @torch.no_grad()
    def gen(img):
        inp = processor(images=img, text=PROMPT, return_tensors="pt")
        inp = {k: (v.to(model.device, torch.bfloat16) if v.is_floating_point() else v.to(model.device)) for k, v in inp.items()}
        o = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return processor.decode(o[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    generate_all(NAME, gen)
    del model, base
    free_gpu()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llavamed", "paligemma2", "all"], default="all")
    which = ap.parse_args().model
    if which in ("llavamed", "all"):
        run_llavamed()
    if which in ("paligemma2", "all"):
        run_paligemma2()
