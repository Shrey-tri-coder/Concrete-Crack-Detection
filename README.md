# Automated Concrete Crack Detection on SDNET2018
**CE 235 – Artificial Intelligence & Data Science · Course Project**
Shreyansh Tripathi (24B0726), Department of Civil Engineering, IIT Bombay

Binary classification of 256×256 concrete surface tiles (bridge decks, walls, pavements) as
*cracked* or *uncracked*, comparing three methods:

1. **Logistic Regression** on 49 hand-crafted image features
2. **Random Forest** on the same features
3. **Convolutional Neural Network** trained from scratch on raw pixels

## Results (held-out test set, 8,781 tiles)

| Method | F1 | PR-AUC | ROC-AUC | MCC | Precision | Recall | Specificity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression (49 features) | 0.450 | 0.474 | 0.783 | 0.359 | 0.477 | 0.425 | 0.917 |
| Random Forest (49 features) | 0.487 | 0.526 | 0.801 | 0.396 | 0.488 | 0.486 | 0.909 |
| **CNN (536K params, from scratch)** | **0.649** | **0.715** | **0.883** | **0.597** | 0.717 | 0.593 | 0.958 |

Only 15.1% of tiles are cracked, so predicting "no crack" everywhere already scores 84.9% accuracy; models are compared on PR-AUC, F1 and MCC. Full metrics with 95% bootstrap CIs: `results/metrics_test.csv`.

> The SDNET2018 images (56,092 tiles) and the trained weights are not included in this repo. Download the dataset from the link below and rerun the pipeline.

## Folder layout
```
CE235_Project_24B0726/
├── DATA_Maguire_20180517_ALL/   # raw SDNET2018 dataset (56,092 .jpg tiles), as downloaded
│   └── DATA_Maguire_20180517_ALL/
│       ├── ReadMe_SDNET2018 .txt
│       └── SDNET2018/{D,P,W}/{CD,UD,CP,UP,CW,UW}/*.jpg
├── requirements.txt
├── code/                    # all source code (run in numeric order)
│   ├── config.py            # paths & constants
│   ├── 01_preprocess.py     # cleaning, grayscale 128x128 conversion, group-aware split
│   ├── 02_features.py       # 49 hand-crafted features -> data_processed/features.csv
│   ├── 03_eda.py            # exploratory analysis figures
│   ├── 04_classical_models.py  # Logistic Regression + Random Forest (tuning, CV, importance)
│   ├── 05_cnn.py            # CNN training (PyTorch, CPU)
│   ├── 06_evaluate.py       # test-set comparison, bootstrap CIs, McNemar, Grad-CAM
│   ├── 07_build_report.py   # fills the report template with results and prints the PDF
│   ├── cnn_model.py         # CNN architecture
│   └── plot_style.py        # figure styling
├── data_processed/
│   ├── metadata.csv         # one row per cleaned image: file, surface, label, parent photo, split, QC stats
│   ├── features.csv         # processed dataset: metadata + 49 features per image
│   └── cleaning_report.csv  # data-quality checks (13 duplicate pairs -> 15 files, 29 blank; 44 removed)
├── results/                 # metrics tables, predictions, trained CNN weights, logs
├── figures/                 # all figures used in the report
├── presentation/            # 8-slide deck (PDF)
└── report/                  # project report (PDF)
```

## How to reproduce
1. Download
   SDNET2018 from https://doi.org/10.15142/T3TD19 and unzip it into the project folder so that
   `DATA_Maguire_20180517_ALL/DATA_Maguire_20180517_ALL/SDNET2018/{D,P,W}` exists
   (`code/config.py` also accepts it one level up, next to the project folder).
2. `pip install -r requirements.txt` (Python 3.11)
3. From `code/`, run:
   ```
   python 01_preprocess.py   # ~3 min, writes cache/images_128.npy (~0.9 GB, not submitted)
   python 02_features.py     # ~8 min
   python 03_eda.py
   python 04_classical_models.py   # ~10 min
   python 05_cnn.py          # ~40 min on a laptop CPU
   python 06_evaluate.py
   python 07_build_report.py # optional; needs report/report_template.html, report/results_section.html
                             # and Microsoft Edge for HTML-to-PDF
   ```
Steps 1-6 regenerate everything in `data_processed/`, `results/` and `figures/`.
All randomness is seeded (`SEED = 42`).

## Data source & licence
Maguire, M., Dorafshan, S., & Thomas, R. J. (2018). *SDNET2018: A concrete crack image dataset for
machine learning applications.* Utah State University. https://doi.org/10.15142/T3TD19 — CC BY 4.0.

## Libraries
NumPy, pandas, SciPy, scikit-learn, scikit-image, OpenCV, PyTorch, Matplotlib.
