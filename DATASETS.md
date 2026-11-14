# Datasets

TBV is trained on one public dataset (Figshare) and tested on another (Kaggle). No Kaggle image or label is used for training.

## Layout

```
data/
├── kaggle/                        # Kaggle images (see below)
├── figshare/
│   ├── doctor_confirmed_136/      # Roboflow export: train|valid|test/images, train|valid|test/labels
│   └── annotations_605/           # Roboflow export of the remaining Figshare images
├── kaggle_references.csv          # 664 Kaggle images: reference report + reference boxes
├── test_annotation_map.csv        # test_000 ... test_099 -> Kaggle image
├── test_labels/                   # blind follicle annotations of the 100 test images (YOLO format)
└── scan_groups.json               # near-duplicate groups of the Kaggle images (used for all CIs)
```

## Figshare (training)

- Images: PCOS Dataset, Indirani (2024), [doi:10.6084/m9.figshare.27682557.v1](https://doi.org/10.6084/m9.figshare.27682557.v1).
- Follicle boxes: drawn by the first author in Roboflow for every visible follicle; a clinician reviewed a subset. After removing duplicates, 556 images contain 4,377 annotated follicles.
- Both Roboflow exports are on the [release page](https://github.com/ToyioUdoete/trust-but-verify/releases) (`tbv_figshare.zip`); unzip into `data/figshare/`. YOLO format (one `class cx cy w h` line per follicle, normalised).

## Kaggle (test)

- Images: Polycystic Ovary Syndrome (PCOS) ultrasound dataset, Anagha Choudhari, [Kaggle](https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images).
- Reference reports: drafted from YOLO11 detections for 781 images and rewritten/approved by a clinician; 664 are kept (`kaggle_references.csv`). Columns: `image_path`, `img_w`, `img_h`, `follicle_count`, `boxes_json` (reference boxes, `xyxy_norm`), `report`.
- Test set: 100 images (14 near-duplicate groups). They were renamed, shuffled and annotated without any model output, boxes or reports visible (1,403 follicles, `test_labels/`). `test_annotation_map.csv` links each `test_XXX` file to its Kaggle image.
- Put the images in `data/kaggle/` with the file names used in `kaggle_references.csv`.
- Near-duplicate groups (`scan_groups.json`, key `scan_of`): the Kaggle images come from a small number of source scans; bootstrap CIs resample whole groups.

Pixel spacing is not available for either dataset, so all millimetre values use a nominal 0.15 mm per pixel.
