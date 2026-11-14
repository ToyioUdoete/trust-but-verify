"""Point the released CSV/JSON files at your local copy of the Kaggle images.

The reference CSV, the test-annotation map and the near-duplicate groups store the image path
of every Kaggle image. This rewrites those paths to IMAGE_DIR, matching by file name, so the
rest of the code runs unchanged. Run it once after downloading the data (see DATASETS.md).

    python prepare/remap_paths.py --image_dir /path/to/kaggle_images
"""

import os, json, argparse, glob
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--image_dir", required=True, help="folder that contains the Kaggle images (searched recursively)")
ap.add_argument("--data_dir", default="data")
args = ap.parse_args()

local = {}
for p in glob.glob(os.path.join(args.image_dir, "**", "*"), recursive=True):
    if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        local.setdefault(os.path.basename(p), os.path.abspath(p))
print(f"{len(local)} images found under {args.image_dir}")


def remap(p):
    name = os.path.basename(str(p))
    if name not in local:
        raise SystemExit(f"not found under --image_dir: {name}")
    return local[name]


refs = os.path.join(args.data_dir, "kaggle_references.csv")
df = pd.read_csv(refs); df["image_path"] = df["image_path"].map(remap); df.to_csv(refs, index=False)
print(f"updated {refs} ({len(df)} images)")

amap = os.path.join(args.data_dir, "test_annotation_map.csv")
df = pd.read_csv(amap); df["image"] = df["image"].map(remap); df.to_csv(amap, index=False)
print(f"updated {amap} ({len(df)} images)")

groups = os.path.join(args.data_dir, "scan_groups.json")
g = json.load(open(groups))
g["scan_of"] = {remap(k): v for k, v in g["scan_of"].items()}
for split in ("train", "val", "test"):
    if split in g:
        g[split] = [remap(p) for p in g[split]]
json.dump(g, open(groups, "w"), indent=1)
print(f"updated {groups}")
