"""Step 3 - Exploratory data analysis and visualisation."""
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import PROC_DIR, RAW_DIR, FIG_DIR, RES_DIR, SURFACES, SEED
from plot_style import plt, save, BLUE, ORANGE, INK2, CLASS_COL, CLASS_NAME

df = pd.read_csv(PROC_DIR / "features.csv")
feat_cols = df.columns[5:]
rng = np.random.default_rng(SEED)

# ---- 1. class distribution per surface --------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 2.8))
g = df.groupby(["surface", "label"]).size().unstack()
x = np.arange(3)
order = ["D", "W", "P"]
for k, off in ((0, -0.2), (1, 0.2)):
    vals = g.loc[order, k].values
    ax.bar(x + off, vals, 0.38, color=CLASS_COL[k], label=CLASS_NAME[k])
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 300, f"{v:,}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(x, [f"{SURFACES[s]}\n({g.loc[s, 1] / g.loc[s].sum():.1%} cracked)" for s in order])
ax.set_ylabel("Number of images")
ax.set_title("Class distribution by surface type")
ax.legend(loc="upper left")
ax.grid(axis="x", visible=False)
save(fig, FIG_DIR / "eda_class_distribution.png")

# ---- 2. sample images --------------------------------------------------------
fig, axes = plt.subplots(3, 8, figsize=(8, 3.3))
for r, s in enumerate(order):
    for c in range(8):
        lab = 1 if c < 4 else 0
        f = df[(df.surface == s) & (df.label == lab)].sample(1, random_state=int(rng.integers(1e6))).file.iloc[0]
        img = cv2.cvtColor(cv2.imread(str(RAW_DIR / f)), cv2.COLOR_BGR2RGB)
        ax = axes[r, c]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(CLASS_COL[lab]); sp.set_linewidth(2)
        if c == 0:
            ax.set_ylabel(SURFACES[s], fontsize=8, color=INK2)
axes[0, 1].set_title("Cracked (C)", loc="center", color=ORANGE, x=1.05)
axes[0, 5].set_title("Uncracked (U)", loc="center", color=BLUE, x=1.05)
fig.subplots_adjust(wspace=0.05, hspace=0.08)
save(fig, FIG_DIR / "eda_samples.png")

# ---- 3. feature separability: Cohen's d -------------------------------------
c, u = df[df.label == 1][feat_cols], df[df.label == 0][feat_cols]
d = (c.mean() - u.mean()) / np.sqrt((c.var() + u.var()) / 2)
d.sort_values(key=abs, ascending=False).to_csv(RES_DIR / "feature_cohens_d.csv", header=["cohens_d"])
top = d.reindex(d.abs().sort_values(ascending=False).index)[:15][::-1]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.barh(top.index, top.values, color=[ORANGE if v > 0 else BLUE for v in top.values], height=0.65)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("Cohen's d  (cracked − uncracked, in pooled SDs)")
ax.set_title("Top-15 most discriminative hand-crafted features")
ax.grid(axis="y", visible=False)
save(fig, FIG_DIR / "eda_feature_effect.png")

# ---- 4. distributions of key features ----------------------------------------
keys = list(d.abs().sort_values(ascending=False).index[:6])
fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.8))
for ax, k in zip(axes.ravel(), keys):
    lo, hi = np.percentile(df[k], [0.5, 99.5])
    bins = np.linspace(lo, hi, 50)
    for lab in (0, 1):
        ax.hist(df.loc[df.label == lab, k].clip(lo, hi), bins, density=True, alpha=0.55,
                color=CLASS_COL[lab], label=CLASS_NAME[lab])
    ax.set_title(k, fontsize=9)
    ax.set_yticks([])
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
save(fig, FIG_DIR / "eda_feature_distributions.png")

# ---- 5. correlation heatmap ---------------------------------------------------
corr = df[feat_cols].corr()
fig, ax = plt.subplots(figsize=(6.6, 5.8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(feat_cols)), feat_cols, rotation=90, fontsize=5)
ax.set_yticks(range(len(feat_cols)), feat_cols, fontsize=5)
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
ax.set_title("Correlation between the 49 features")
save(fig, FIG_DIR / "eda_correlation.png")
high = (corr.abs().where(np.triu(np.ones(corr.shape, bool), 1)) > 0.9).sum().sum()
print("feature pairs with |r|>0.9:", high)

# ---- 6. PCA projection ---------------------------------------------------------
Z = StandardScaler().fit_transform(df[feat_cols])
pca = PCA(n_components=10, random_state=SEED).fit(Z)
P = pca.transform(Z)[:, :2]
sub = rng.choice(len(df), 6000, replace=False)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.4, 1]})
for lab in (0, 1):
    m = sub[df.label.values[sub] == lab]
    axes[0].scatter(P[m, 0], P[m, 1], s=3, alpha=0.4, color=CLASS_COL[lab], label=CLASS_NAME[lab], linewidths=0)
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
axes[0].set_title("PCA of features (6,000-image sample)")
axes[0].legend(markerscale=4, fontsize=7)
axes[1].bar(range(1, 11), pca.explained_variance_ratio_ * 100, color=BLUE, width=0.7)
axes[1].plot(range(1, 11), np.cumsum(pca.explained_variance_ratio_) * 100, color=ORANGE, marker="o", ms=4)
axes[1].set_xlabel("Principal component"); axes[1].set_ylabel("% variance")
axes[1].set_title("Explained variance")
fig.tight_layout()
save(fig, FIG_DIR / "eda_pca.png")
print("PC1+PC2 variance: %.1f%%" % (pca.explained_variance_ratio_[:2].sum() * 100))

# ---- summary tables ---------------------------------------------------------------
split_tab = df.groupby(["split", "surface"]).agg(images=("label", "size"), cracked=("label", "sum"),
                                                 parents=("parent_id", "nunique"))
split_tab.to_csv(RES_DIR / "split_summary.csv")
print(split_tab)
print(df.groupby("surface").agg(parents=("parent_id", "nunique"), images=("label", "size"), crack_rate=("label", "mean")))
