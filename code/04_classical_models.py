"""Step 4 - Methods 1 & 2: Logistic Regression and Random Forest on hand-crafted features.

* Hyper-parameters chosen on the validation split (metric: PR-AUC, robust to imbalance).
* class_weight='balanced' handles the ~15% crack prevalence.
* Decision threshold tuned on validation (max F1), applied unchanged to the test split.
* Robustness: 5-fold StratifiedGroupKFold CV (groups = parent photographs) on train+val.
"""
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import PROC_DIR, RES_DIR, SEED

df = pd.read_csv(PROC_DIR / "features.csv")
feat_cols = list(df.columns[5:])
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
Xtr, ytr, Xva, yva, Xte, yte = tr[feat_cols], tr.label, va[feat_cols], va.label, te[feat_cols], te.label


def make_lr(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def make_rf(depth, leaf, mf):
    return RandomForestClassifier(n_estimators=400, max_depth=depth, min_samples_leaf=leaf, max_features=mf,
                                  class_weight="balanced_subsample", n_jobs=10, random_state=SEED)


grids = {
    "Logistic Regression": (make_lr, [{"C": c} for c in (0.001, 0.01, 0.1, 1, 10)]),
    "Random Forest": (make_rf, [dict(depth=d, leaf=l, mf=m) for d, l, m in
                                itertools.product((12, 24, None), (1, 5), ("sqrt", 0.3))]),
}

summary = {}
for name, (builder, grid) in grids.items():
    tuning = []
    for params in grid:
        m = builder(**params).fit(Xtr, ytr)
        ap = average_precision_score(yva, m.predict_proba(Xva)[:, 1])
        tuning.append({**params, "val_pr_auc": ap})
        print(f"{name:20s} {params} val PR-AUC {ap:.4f}", flush=True)
    best = max(tuning, key=lambda r: r["val_pr_auc"])
    params = {k: v for k, v in best.items() if k != "val_pr_auc"}
    model = builder(**params).fit(Xtr, ytr)
    pv, pt = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
    p, r, t = precision_recall_curve(yva, pv)
    thr = float(t[np.argmax((2 * p * r / (p + r + 1e-12))[:-1])])

    # grouped 5-fold CV on train+val with the chosen hyper-parameters
    dv = pd.concat([tr, va])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_ap, cv_auc, cv_f1 = [], [], []
    for a, b in cv.split(dv[feat_cols], dv.label, dv.parent_id):
        m = builder(**params).fit(dv[feat_cols].iloc[a], dv.label.iloc[a])
        pb = m.predict_proba(dv[feat_cols].iloc[b])[:, 1]
        yb = dv.label.iloc[b]
        cv_ap.append(average_precision_score(yb, pb)); cv_auc.append(roc_auc_score(yb, pb))
        cv_f1.append(f1_score(yb, pb >= thr))

    key = "lr" if name.startswith("Logistic") else "rf"
    pd.DataFrame({"idx": va.index, "prob": pv}).to_csv(RES_DIR / f"{key}_val_probs.csv", index=False)
    pd.DataFrame({"idx": te.index, "prob": pt}).to_csv(RES_DIR / f"{key}_test_probs.csv", index=False)
    pd.DataFrame(tuning).to_csv(RES_DIR / f"{key}_tuning.csv", index=False)

    if key == "lr":
        coef = pd.Series(model[-1].coef_[0], index=feat_cols)
        coef.sort_values(key=abs, ascending=False).to_csv(RES_DIR / "lr_coefficients.csv", header=["coef"])
    else:
        imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
        imp.to_csv(RES_DIR / "rf_gini_importance.csv", header=["importance"])
        perm = permutation_importance(model, Xva, yva, scoring="average_precision", n_repeats=5,
                                      random_state=SEED, n_jobs=10)
        pd.Series(perm.importances_mean, index=feat_cols).sort_values(ascending=False) \
            .to_csv(RES_DIR / "rf_permutation_importance.csv", header=["importance"])

    summary[name] = {"best_params": params, "threshold": thr,
                     "cv_pr_auc": [float(np.mean(cv_ap)), float(np.std(cv_ap))],
                     "cv_roc_auc": [float(np.mean(cv_auc)), float(np.std(cv_auc))],
                     "cv_f1": [float(np.mean(cv_f1)), float(np.std(cv_f1))]}
    print(name, json.dumps(summary[name]), flush=True)

json.dump(summary, open(RES_DIR / "classical_summary.json", "w"), indent=2, default=str)
