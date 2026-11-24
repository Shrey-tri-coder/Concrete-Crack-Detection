"""Step 1 - Data cleaning and preprocessing.

* Builds a metadata table (file, surface, label, parent image id) for all 56,092 images.
* Validates every file (readable, 256x256, RGB) and removes exact duplicates (MD5: one copy of each
  pair is kept; pairs filed under both C and U are dropped entirely) and near-blank images (grey std < 2).
  Every count is written to data_processed/cleaning_report.csv.
* Converts each image to grayscale and resizes it to 128x128 (INTER_AREA) - this array is
  cached for the feature extractor and the CNN.
* Creates a GROUP-AWARE train/val/test split: all 256x256 sub-images cut from the same
  parent photograph go to the same split, so the models are tested on unseen photographs
  (a random split would leak near-identical neighbouring tiles between train and test).
"""
import hashlib
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from config import RAW_DIR, PROC_DIR, CACHE_DIR, IMG_SIZE, SEED, SPLIT_FRAC


def scan():
    rows = []
    for surf in ("D", "P", "W"):
        for cls in ("C", "U"):
            folder = RAW_DIR / surf / f"{cls}{surf}"
            for f in sorted(folder.glob("*.jpg")):
                parent = f.stem.split("-")[0]
                rows.append({"file": f"{surf}/{cls}{surf}/{f.name}", "surface": surf,
                             "label": int(cls == "C"), "parent_id": f"{surf}_{parent}"})
    return pd.DataFrame(rows)


def load_one(rel):
    path = RAW_DIR / rel
    raw = path.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return rel, md5, False, 0, 0, 0, np.nan, np.nan, None
    h, w, c = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return rel, md5, True, h, w, c, float(gray.mean()), float(gray.std()), small


def group_split(df):
    """Assign parent images to splits per surface; choose the permutation whose
    image counts and crack ratios best match the target fractions."""
    rng = np.random.default_rng(SEED)
    split = pd.Series(index=df.index, dtype=object)
    for surf, sdf in df.groupby("surface"):
        g = sdf.groupby("parent_id")["label"].agg(["size", "sum"])
        total, pos_rate = g["size"].sum(), g["sum"].sum() / g["size"].sum()
        best, best_cost = None, np.inf
        for _ in range(3000):
            order = rng.permutation(g.index.values)
            cum = g.loc[order, "size"].cumsum().values / total
            assign = np.where(cum <= SPLIT_FRAC["train"], "train",
                              np.where(cum <= SPLIT_FRAC["train"] + SPLIT_FRAC["val"], "val", "test"))
            a = pd.Series(assign, index=order)
            cost = 0.0
            for s, frac in SPLIT_FRAC.items():
                sel = g.loc[a.index[a == s]]
                if len(sel) == 0:
                    cost += 10
                    continue
                cost += abs(sel["size"].sum() / total - frac) + 2 * abs(sel["sum"].sum() / sel["size"].sum() - pos_rate)
            if cost < best_cost:
                best, best_cost = a, cost
        split.loc[sdf.index] = sdf["parent_id"].map(best).values
    return split


def main():
    df = scan()
    print(f"Scanned {len(df)} images, {df.parent_id.nunique()} parent photographs")

    with ProcessPoolExecutor() as ex:
        results = list(ex.map(load_one, df["file"], chunksize=256))

    imgs = np.zeros((len(df), IMG_SIZE, IMG_SIZE), np.uint8)
    for i, (rel, md5, ok, h, w, c, m, s, small) in enumerate(results):
        df.loc[i, ["md5", "readable", "height", "width", "channels", "gray_mean", "gray_std"]] = \
            [md5, ok, h, w, c, m, s]
        if small is not None:
            imgs[i] = small

    # ---- cleaning checks -------------------------------------------------
    report = {
        "total_images": len(df),
        "unreadable": int((~df.readable.astype(bool)).sum()),
        "not_256x256": int(((df.height != 256) | (df.width != 256)).sum()),
        "not_rgb": int((df.channels != 3).sum()),
    }
    # exact duplicates: identical files (same MD5) stored twice, e.g. "7053-36.jpg" and "7053-36_2.jpg"
    dup_labels = df[df.md5.duplicated(keep=False)].groupby("md5")["label"].nunique()
    conflict_md5 = dup_labels.index[dup_labels > 1]          # same image filed under both C and U
    redundant = df.duplicated("md5", keep="first") & ~df.md5.isin(conflict_md5)
    dup_mask = redundant | df.md5.isin(conflict_md5)         # ambiguous label -> drop every copy
    report["duplicate_groups_found"] = len(dup_labels)
    report["duplicate_groups_conflicting_labels"] = len(conflict_md5)
    report["redundant_copies_removed"] = int(redundant.sum())
    report["conflicting_label_files_removed"] = int(df.md5.isin(conflict_md5).sum())
    report["duplicate_files_removed_total"] = int(dup_mask.sum())
    blank = df.gray_std < 2.0
    report["near_blank_removed"] = int((blank & ~dup_mask).sum())

    keep = df.readable.astype(bool) & ~dup_mask & ~blank & (df.height == 256) & (df.width == 256)
    report["total_removed"] = int((~keep).sum())
    df, imgs = df[keep].reset_index(drop=True), imgs[keep.values]
    report["images_after_cleaning"] = len(df)
    report["cracked_after_cleaning"] = int(df.label.sum())
    report["uncracked_after_cleaning"] = int((1 - df.label).sum())
    assert report["total_images"] - report["total_removed"] == report["images_after_cleaning"]

    df["split"] = group_split(df).values
    for k, v in report.items():
        print(f"  {k:38s} {v}")
    print(df.groupby(["split", "surface"])["label"].agg(["size", "mean"]))

    pd.Series(report).to_csv(PROC_DIR / "cleaning_report.csv", header=["value"])
    df.drop(columns=["readable", "channels"]).to_csv(PROC_DIR / "metadata.csv", index=False)
    np.save(CACHE_DIR / f"images_{IMG_SIZE}.npy", imgs)
    print("Saved metadata.csv and image cache", imgs.shape)


if __name__ == "__main__":
    main()
