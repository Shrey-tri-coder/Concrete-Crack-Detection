"""Step 6 - Results & comparison of the three methods on the held-out TEST split.

* Threshold metrics (accuracy, balanced accuracy, precision, recall, specificity, F1, MCC)
  using each model's validation-tuned threshold, plus threshold-free ROC-AUC and PR-AUC.
* 95% confidence intervals by cluster bootstrap over parent photographs (1,000 resamples).
* Per-surface breakdown, McNemar tests between methods, inference-time benchmark.
* Figures: ROC/PR curves, confusion matrices, metric bars, per-surface bars, CNN learning
  curves, feature importance, Grad-CAM explanations of CNN predictions.
"""
import importlib
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from cnn_model import CrackCNN, to_tensor
from config import PROC_DIR, CACHE_DIR, RAW_DIR, RES_DIR, FIG_DIR, IMG_SIZE, SURFACES, SEED
from plot_style import plt, save, METHOD_COL, BLUE, ORANGE, AQUA, INK2, CLASS_COL

meta = pd.read_csv(PROC_DIR / "metadata.csv")
classical = json.load(open(RES_DIR / "classical_summary.json"))
cnn_info = json.load(open(RES_DIR / "cnn_history.json"))
METHODS = {"Logistic Regression": ("lr", classical["Logistic Regression"]["threshold"]),
           "Random Forest": ("rf", classical["Random Forest"]["threshold"]),
           "CNN": ("cnn", cnn_info["threshold"])}

test = meta[meta.split == "test"].copy()
for name, (key, thr) in METHODS.items():
    p = pd.read_csv(RES_DIR / f"{key}_test_probs.csv").set_index("idx").prob
    test[f"p_{key}"] = p.reindex(test.index).values
    test[f"y_{key}"] = (test[f"p_{key}"] >= thr).astype(int)
y = test.label.values


def metrics(yt, prob, pred):
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(yt, pred), "Balanced acc.": balanced_accuracy_score(yt, pred),
            "Precision": precision_score(yt, pred, zero_division=0), "Recall": recall_score(yt, pred),
            "Specificity": tn / (tn + fp), "F1": f1_score(yt, pred), "MCC": matthews_corrcoef(yt, pred),
            "ROC-AUC": roc_auc_score(yt, prob), "PR-AUC": average_precision_score(yt, prob)}


# ---- overall metrics with cluster-bootstrap CIs --------------------------------------
rng = np.random.default_rng(SEED)
parents = test.parent_id.unique()
groups = {g: np.where(test.parent_id.values == g)[0] for g in parents}
boot = {n: {"F1": [], "PR-AUC": [], "ROC-AUC": []} for n in METHODS}
for _ in range(1000):
    ids = np.concatenate([groups[g] for g in rng.choice(parents, len(parents))])
    if y[ids].sum() == 0:
        continue
    for n, (k, _) in METHODS.items():
        boot[n]["F1"].append(f1_score(y[ids], test[f"y_{k}"].values[ids]))
        boot[n]["PR-AUC"].append(average_precision_score(y[ids], test[f"p_{k}"].values[ids]))
        boot[n]["ROC-AUC"].append(roc_auc_score(y[ids], test[f"p_{k}"].values[ids]))

rows = []
for n, (k, thr) in METHODS.items():
    m = metrics(y, test[f"p_{k}"], test[f"y_{k}"])
    for s in ("F1", "PR-AUC", "ROC-AUC"):
        lo, hi = np.percentile(boot[n][s], [2.5, 97.5])
        m[f"{s} 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
    rows.append({"Method": n, "Threshold": thr, **m})
overall = pd.DataFrame(rows).set_index("Method")
overall.to_csv(RES_DIR / "metrics_test.csv", float_format="%.4f")
print(overall.T)

# ---- per-surface --------------------------------------------------------------------------
ps = []
for s in ("D", "W", "P"):
    sel = test.surface.values == s
    for n, (k, _) in METHODS.items():
        m = metrics(y[sel], test[f"p_{k}"].values[sel], test[f"y_{k}"].values[sel])
        ps.append({"Surface": SURFACES[s], "Method": n, **m})
per_surface = pd.DataFrame(ps)
per_surface.to_csv(RES_DIR / "metrics_per_surface.csv", index=False, float_format="%.4f")
print(per_surface.pivot(index="Surface", columns="Method", values=["F1", "PR-AUC"]).round(3))

# ---- McNemar tests ----------------------------------------------------------------------
mc = []
names = list(METHODS)
for i in range(3):
    for j in range(i + 1, 3):
        a = test[f"y_{METHODS[names[i]][0]}"].values == y
        b = test[f"y_{METHODS[names[j]][0]}"].values == y
        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        mc.append({"A": names[i], "B": names[j], "A right, B wrong": n01, "A wrong, B right": n10,
                   "p-value": binomtest(n01, n01 + n10, 0.5).pvalue})
pd.DataFrame(mc).to_csv(RES_DIR / "mcnemar.csv", index=False)
print(pd.DataFrame(mc))

# ---- inference-time benchmark (single image, single process) -----------------------------
feat_mod = importlib.import_module("02_features")
imgs = np.load(CACHE_DIR / f"images_{IMG_SIZE}.npy", mmap_mode="r")
sample = np.array(imgs[test.index.values[:300]])
t0 = time.perf_counter(); [feat_mod.features(im) for im in sample]; t_feat = (time.perf_counter() - t0) / 300
model = CrackCNN(); model.load_state_dict(torch.load(RES_DIR / "cnn_best.pt")); model.eval()
torch.set_num_threads(1)
with torch.no_grad():
    t0 = time.perf_counter(); [model(to_tensor(sample[i:i + 1])) for i in range(300)]
    t_cnn = (time.perf_counter() - t0) / 300
torch.set_num_threads(8)
timing = {"feature_extraction_ms": t_feat * 1000, "cnn_forward_ms": t_cnn * 1000}
json.dump(timing, open(RES_DIR / "timing.json", "w"), indent=2)
print(timing)

# =========================== FIGURES ===========================================
# ROC & PR curves
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
for n, (k, _) in METHODS.items():
    fpr, tpr, _ = roc_curve(y, test[f"p_{k}"])
    axes[0].plot(fpr, tpr, color=METHOD_COL[n], label=f"{n} (AUC {overall.loc[n, 'ROC-AUC']:.3f})")
    p, r, _ = precision_recall_curve(y, test[f"p_{k}"])
    axes[1].plot(r, p, color=METHOD_COL[n], label=f"{n} (AP {overall.loc[n, 'PR-AUC']:.3f})")
axes[0].plot([0, 1], [0, 1], ls="--", lw=1, color=INK2)
axes[1].axhline(y.mean(), ls="--", lw=1, color=INK2)
axes[1].text(0.6, y.mean() + 0.02, f"chance = {y.mean():.3f}", fontsize=7, color=INK2)
axes[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curves (test)")
axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curves (test)", ylim=(0, 1.02))
for ax in axes:
    ax.legend(fontsize=7, loc="lower right" if ax is axes[0] else "upper right")
fig.tight_layout()
save(fig, FIG_DIR / "res_roc_pr.png")

# confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
for ax, (n, (k, _)) in zip(axes, METHODS.items()):
    cm = confusion_matrix(y, test[f"y_{k}"])
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        frac = v / cm[i].sum()
        ax.text(j, i, f"{v:,}\n({frac:.0%})", ha="center", va="center", fontsize=8,
                color="white" if frac > 0.55 else "#0b0b0b")
    ax.set_xticks([0, 1], ["Pred U", "Pred C"]); ax.set_yticks([0, 1], ["True U", "True C"])
    ax.grid(False); ax.set_title(n, fontsize=9)
fig.tight_layout()
save(fig, FIG_DIR / "res_confusion.png")

# metric comparison bars
mets = ["Balanced acc.", "Precision", "Recall", "F1", "MCC", "ROC-AUC", "PR-AUC"]
fig, ax = plt.subplots(figsize=(7.2, 2.8))
x = np.arange(len(mets)); w = 0.26
for i, n in enumerate(METHODS):
    vals = overall.loc[n, mets].astype(float).values
    ax.bar(x + (i - 1) * w, vals, w - 0.02, color=METHOD_COL[n], label=n)
    for xi, v in zip(x + (i - 1) * w, vals):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6, color=INK2)
ax.set_xticks(x, mets); ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
ax.set_title("Test-set performance of the three methods"); ax.legend(ncol=3, fontsize=7, loc="upper left")
save(fig, FIG_DIR / "res_metric_bars.png")

# per-surface F1 & PR-AUC
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
for ax, met in zip(axes, ("F1", "PR-AUC")):
    piv = per_surface.pivot(index="Surface", columns="Method", values=met).loc[[SURFACES[s] for s in "DWP"]]
    x = np.arange(3)
    for i, n in enumerate(METHODS):
        ax.bar(x + (i - 1) * w, piv[n], w - 0.02, color=METHOD_COL[n], label=n)
        for xi, v in zip(x + (i - 1) * w, piv[n]):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6, color=INK2)
    ax.set_xticks(x, piv.index); ax.set_title(f"{met} by surface type"); ax.grid(axis="x", visible=False)
axes[0].set_ylim(0, 1); axes[0].legend(fontsize=7, loc="upper left")
fig.tight_layout()
save(fig, FIG_DIR / "res_per_surface.png")

# CNN learning curves
h = pd.DataFrame(cnn_info["history"])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
axes[0].plot(h.epoch, h.train_loss, color=AQUA, marker="o", ms=4)
axes[0].set(xlabel="Epoch", ylabel="Weighted BCE loss", title="CNN training loss")
axes[1].plot(h.epoch, h.val_pr_auc, color=AQUA, marker="o", ms=4, label="CNN val PR-AUC")
axes[1].plot(h.epoch, h.val_roc_auc, color=AQUA, ls="--", marker="s", ms=4, alpha=0.6, label="CNN val ROC-AUC")
axes[1].axhline(classical["Random Forest"]["cv_pr_auc"][0], color=ORANGE, lw=1, ls=":", label="RF CV PR-AUC")
axes[1].set(xlabel="Epoch", ylabel="Score", title="CNN validation performance", ylim=(0, 1))
axes[1].legend(fontsize=7, loc="lower right")
fig.tight_layout()
save(fig, FIG_DIR / "res_cnn_curves.png")

# feature importance (RF permutation) and LR coefficients
perm = pd.read_csv(RES_DIR / "rf_permutation_importance.csv", index_col=0).importance[:12][::-1]
coef = pd.read_csv(RES_DIR / "lr_coefficients.csv", index_col=0).coef[:12][::-1]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
axes[0].barh(perm.index, perm.values, color=ORANGE, height=0.65)
axes[0].set(title="RF permutation importance", xlabel="Drop in val PR-AUC")
axes[1].barh(coef.index, coef.values, color=[ORANGE if v > 0 else BLUE for v in coef.values], height=0.65)
axes[1].axvline(0, color=INK2, lw=0.8)
axes[1].set(title="LR standardised coefficients", xlabel="Coefficient (+ → crack)")
for ax in axes:
    ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=7)
fig.tight_layout()
save(fig, FIG_DIR / "res_feature_importance.png")


# Grad-CAM on the last conv block
class GradCAM:
    def __init__(self, model, layer):
        self.model, self.act, self.grad = model, None, None
        layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.act = out
        out.register_hook(lambda g: setattr(self, "grad", g))

    def __call__(self, x):
        self.model.zero_grad()
        out = self.model(x)
        out.sum().backward()
        w = self.grad.mean((2, 3), keepdim=True)
        cam = torch.relu((w * self.act).sum(1)).detach().numpy()
        return cam, torch.sigmoid(out).detach().numpy()


cam_model = CrackCNN(); cam_model.load_state_dict(torch.load(RES_DIR / "cnn_best.pt")); cam_model.eval()
for mod in cam_model.modules():
    if isinstance(mod, torch.nn.ReLU):
        mod.inplace = False
last_relu = [m for m in cam_model.features if isinstance(m, torch.nn.ReLU)][-1]
gc = GradCAM(cam_model, last_relu)
thr = METHODS["CNN"][1]
tp = test[(test.label == 1) & (test.y_cnn == 1)].sort_values("p_cnn", ascending=False)
picks = {"True positive": tp.iloc[::max(1, len(tp) // 40)].head(4),
         "False negative": test[(test.label == 1) & (test.y_cnn == 0)].sample(2, random_state=SEED),
         "False positive": test[(test.label == 0) & (test.y_cnn == 1)].sample(2, random_state=SEED)}
fig, axes = plt.subplots(2, 8, figsize=(8, 2.4))
col = 0
for kind, sub in picks.items():
    for idx, row in sub.iterrows():
        rgb = cv2.cvtColor(cv2.imread(str(RAW_DIR / row.file)), cv2.COLOR_BGR2RGB)
        cam, prob = gc(to_tensor(imgs[[idx]]))
        cam = cv2.resize(cam[0] / (cam[0].max() + 1e-9), (256, 256))
        axes[0, col].imshow(rgb); axes[1, col].imshow(rgb); axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
        axes[0, col].set_title(f"{kind}\np={prob[0]:.2f}", fontsize=6.5, loc="center",
                               color=CLASS_COL[int(row.label)])
        col += 1
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
axes[0, 0].set_ylabel("Image", fontsize=7); axes[1, 0].set_ylabel("Grad-CAM", fontsize=7)
fig.subplots_adjust(wspace=0.04, hspace=0.05)
save(fig, FIG_DIR / "res_gradcam.png")
