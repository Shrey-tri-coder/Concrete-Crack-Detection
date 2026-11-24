"""Step 2 - Hand-crafted feature extraction (input for Logistic Regression & Random Forest).

Each 128x128 grayscale image is summarised by 49 interpretable features in 6 groups:
  1. Intensity statistics      - brightness / contrast of the surface
  2. Gradient & edge features  - Sobel magnitude, Laplacian variance, Canny edge density
  3. Morphological (black-hat) - highlights thin DARK structures, i.e. crack candidates
  4. Crack-shape descriptors   - size / elongation / span of the largest dark connected component
  5. Ridge (Frangi) filter     - Hessian-based line detector, responds to crack-like curvilinear features
  6. Texture                   - GLCM (Haralick) statistics and uniform-LBP histogram
Output: data_processed/features.csv (metadata + features), the processed dataset used downstream.
"""
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import frangi
from skimage.measure import label, regionprops

from config import PROC_DIR, CACHE_DIR, IMG_SIZE

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


def features(img):
    f = {}
    x = img.astype(np.float32)
    # 1. intensity
    f["int_mean"], f["int_std"] = x.mean(), x.std()
    f["int_skew"], f["int_kurt"] = skew(x.ravel()), kurtosis(x.ravel())
    p1, p5, p50, p95 = np.percentile(x, [1, 5, 50, 95])
    f["int_p1"], f["int_p5"], f["int_p50"], f["int_p95"] = p1, p5, p50, p95
    f["int_range_5_95"] = p95 - p5
    f["dark_frac_2sd"] = (x < x.mean() - 2 * x.std()).mean()

    # 2. gradients / edges (light blur first to suppress JPEG and surface grain noise)
    b = cv2.GaussianBlur(img, (3, 3), 0)
    gx, gy = cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    f["grad_mean"], f["grad_std"], f["grad_p90"], f["grad_p99"] = \
        mag.mean(), mag.std(), np.percentile(mag, 90), np.percentile(mag, 99)
    f["lap_var"] = cv2.Laplacian(b, cv2.CV_32F).var()
    med = np.median(b)
    f["canny_auto_density"] = (cv2.Canny(b, int(max(0, 0.67 * med)), int(min(255, 1.33 * med))) > 0).mean()
    f["canny_50_150_density"] = (cv2.Canny(b, 50, 150) > 0).mean()

    # 3. black-hat morphology (closing - image): thin dark lines become bright
    bh = cv2.morphologyEx(b, cv2.MORPH_BLACKHAT, KERNEL).astype(np.float32)
    f["bh_mean"], f["bh_std"] = bh.mean(), bh.std()
    f["bh_p95"], f["bh_p99"], f["bh_max"] = np.percentile(bh, 95), np.percentile(bh, 99), bh.max()
    f["bh_row_max"] = bh.mean(axis=1).max() / (bh.mean() + 1e-6)   # straight horizontal crack
    f["bh_col_max"] = bh.mean(axis=0).max() / (bh.mean() + 1e-6)   # straight vertical crack

    # 4. crack-shape descriptors of the thresholded black-hat map
    thr = max(bh.mean() + 3 * bh.std(), 10)
    mask = (bh > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    f["bh_bin_frac"] = mask.mean()
    lab = label(mask, connectivity=2)
    props = regionprops(lab)
    f["n_components"] = len(props)
    if props:
        big = max(props, key=lambda p: p.area)
        f["largest_area"] = big.area / mask.size
        f["largest_elong"] = big.axis_major_length / (big.axis_minor_length + 1)
        r0, c0, r1, c1 = big.bbox
        f["largest_span"] = max(r1 - r0, c1 - c0) / IMG_SIZE
        f["largest_ecc"] = big.eccentricity
        f["sum_top3_area"] = sum(sorted((p.area for p in props), reverse=True)[:3]) / mask.size
    else:
        f.update(largest_area=0, largest_elong=0, largest_span=0, largest_ecc=0, sum_top3_area=0)

    # 5. Frangi ridge filter (dark ridges)
    fr = frangi(img / 255.0, sigmas=(1, 2, 3), black_ridges=True)
    fr = fr / (fr.max() + 1e-12) if fr.max() > 0 else fr
    f["frangi_mean"], f["frangi_p99"] = fr.mean(), np.percentile(fr, 99)
    f["frangi_frac_0.2"] = (fr > 0.2).mean()

    # 6a. GLCM texture (32 grey levels, distance 1 & 3, 4 directions averaged)
    q = (img // 8).astype(np.uint8)
    glcm = graycomatrix(q, [1, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32,
                        symmetric=True, normed=True)
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        f[f"glcm_{prop}"] = graycoprops(glcm, prop).mean()
    # 6b. uniform LBP histogram (P=8, R=1 -> 10 bins)
    lbp = local_binary_pattern(img, 8, 1, method="uniform")
    hist = np.bincount(lbp.astype(int).ravel(), minlength=10) / lbp.size
    for i, h in enumerate(hist):
        f[f"lbp_{i}"] = h
    return f


def chunk_features(chunk):
    return [features(im) for im in chunk]


def main():
    meta = pd.read_csv(PROC_DIR / "metadata.csv")
    imgs = np.load(CACHE_DIR / f"images_{IMG_SIZE}.npy")
    chunks = np.array_split(imgs, 400)
    with ProcessPoolExecutor() as ex:
        rows = [r for part in ex.map(chunk_features, chunks) for r in part]
    feats = pd.DataFrame(rows).astype(np.float32)
    out = pd.concat([meta[["file", "surface", "parent_id", "split", "label"]], feats], axis=1)
    out.to_csv(PROC_DIR / "features.csv", index=False, float_format="%.6g")
    print("features.csv:", out.shape, "| feature columns:", feats.shape[1])
    print(out.groupby("label")[["bh_p99", "frangi_p99", "largest_span", "canny_auto_density"]].mean())


if __name__ == "__main__":
    main()
