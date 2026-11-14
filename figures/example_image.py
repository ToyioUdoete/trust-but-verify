"""Figure 4: the test image of Table 3 with its ground-truth follicles (green).

    python figures/example_image.py
"""

IMAGE_KEY = "img3_jpg.rf.vMihMnsoFdCRWpU9uEEa"
CACHE, OUT, GREEN = "outputs/test_cache.pkl", "figures", "#43d35d"
import os, pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
os.makedirs(OUT, exist_ok=True)
c = [x for x in pickle.load(open(CACHE, "rb")) if IMAGE_KEY in x["image"]][0]
img = Image.open(c["image"]).convert("L"); a = np.asarray(img, float)
rows, cols = np.where(a.std(1) > 4)[0], np.where(a.std(0) > 4)[0]
y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
fig, ax = plt.subplots(figsize=(3.3, 3.3 * (y1 - y0) / (x1 - x0)))
ax.imshow(img, cmap="gray"); ax.axis("off"); ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
for b in c["Hum"]:
    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=GREEN, lw=1.0))
ax.text(0.02, 0.03, f"{len(c['Hum'])} follicles", transform=ax.transAxes, fontsize=8, fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.9, pad=1.5, lw=0))
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(f"{OUT}/example_gt.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(f"{OUT}/example_gt.png", dpi=400, bbox_inches="tight", pad_inches=0.01)
print("saved figures/example_gt.pdf")
