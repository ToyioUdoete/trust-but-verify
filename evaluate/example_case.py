"""Table 3: one test report from MedGemma, off-the-shelf MedSAM and TBV, with the ground truth."""

IMAGE_KEY = "img3_jpg.rf.vMihMnsoFdCRWpU9uEEa"
CACHE, PLUS, DETX = "outputs/test_cache.pkl", "outputs/tbv_cache.pkl", "outputs/counter_boxes.pkl"
PX_TO_MM, PCOM, CUT, MARGIN, KAPPA, CONF = 0.15, 12, 0.6, 0.1, 3, 0.40

import re, pickle
import numpy as np
CR = re.compile(r"(\d+)\s+follicles?\s+identified\s+measuring\s+([\d.]+)-([\d.]+)\s*mm")
PERIPH = "Peripheral distribution of follicles noted."; CENTRAL = "Follicles are centrally/scattered distributed."
POS = "compatible with polycystic ovarian morphology"
PCOM_POS = ("Follicle number and size distribution are compatible with polycystic ovarian morphology (PCOM). "
            "Correlate with clinical and biochemical findings before diagnosing PCOS.")
PCOM_NEG = ("Findings do not meet follicle number criteria for polycystic ovarian morphology. "
            "If symptoms persist, correlate with endocrine evaluation and follow-up imaging as clinically indicated.")
def diam(b): return ((b[2] - b[0]) + (b[3] - b[1])) / 2 * PX_TO_MM
def rad(B, W, H): return float(np.mean([np.hypot((b[0] + b[2]) / 2 - W / 2, (b[1] + b[3]) / 2 - H / 2) / (min(W, H) / 2) for b in B]))
def pat(B, W, H): return None if len(B) < 2 else ("peripheral" if rad(B, W, H) > CUT else "central")
def decisive(B, W, H): return len(B) >= 2 and abs(rad(B, W, H) - CUT) >= MARGIN
def apply_values(rep, n, lo, hi, p, pc):
    first = f"{n} follicles identified measuring {lo:.1f}-{hi:.1f} mm."
    if CR.search(rep) is None or (POS in rep) != pc:
        return " ".join([first, PERIPH if p == "peripheral" else CENTRAL, PCOM_POS]) if pc else " ".join([first, PCOM_NEG])
    t = CR.sub(first[:-1], rep, count=1)
    if p and (PERIPH in t or CENTRAL in t):
        t = t.replace(PERIPH, "@@").replace(CENTRAL, "@@").replace("@@", PERIPH if p == "peripheral" else CENTRAL)
    return t
def correct(rep, B, W, H, Kd=None):
    m = CR.search(rep)
    if not B or m is None:
        return rep
    d = [diam(b) for b in B]; p = pat(B, W, H)
    if not decisive(B, W, H):
        p = "peripheral" if PERIPH in rep else "central" if CENTRAL in rep else p
    K = int(m.group(1)); flag = Kd is not None and abs(K - Kd) > KAPPA
    n = Kd if flag else K
    return apply_values(rep, n, min(d), max(d), p, (n >= PCOM) if flag else (POS in rep))

c = [x for x in pickle.load(open(CACHE, "rb")) if IMAGE_KEY in x["image"]][0]
tb = [x for x in pickle.load(open(PLUS, "rb")) if x["image"] == c["image"]][0]["P_ea0_keep"]
Kd = int(sum(b[4] >= CONF for b in pickle.load(open(DETX, "rb"))["boxes"][c["image"]]["yolo_fig"]))
W, H = c["W"], c["H"]
g = [diam(b) for b in c["Hum"]]
rows = {"MedGemma": c["orig_report"],
        "MedSAM": correct(c["orig_report"], c["P_off"], W, H),
        "TBV": correct(c["orig_report"], tb, W, H, Kd)}
for k, v in rows.items():
    print(f"{k:9s}: {v}")
print(f"Truth    : {len(g)} follicles, {min(g):.1f}-{max(g):.1f} mm, {pat(c['Hum'], W, H)}; "
      f"{'meets' if len(g) >= PCOM else 'below'} the follicle-number criterion.")

def short(t):  # drop the last template sentence
    for s in ("Correlate with clinical and biochemical findings before diagnosing PCOS.",
              "If symptoms persist, correlate with endocrine evaluation and follow-up imaging as clinically indicated."):
        t = t.replace(" " + s, "")
    return re.sub(r"(\d)-(\d)", r"\1--\2", t).replace(" mm", "\\,mm")
print("\n%%% LaTeX rows (bold the parts that differ from MedGemma by hand if needed)")
for k, v in rows.items():
    print(f"{k} & {short(v)} \\\\")
print(f"Truth & {len(g)} follicles, {min(g):.1f}--{max(g):.1f}\\,mm, {pat(c['Hum'], W, H)}; "
      f"{'meets' if len(g) >= PCOM else 'below'} the follicle-number criterion. \\\\")
