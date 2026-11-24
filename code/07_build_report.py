"""Step 7 - Assemble the report: fill the HTML templates with numbers from results/ and print to PDF."""
import json
import subprocess
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cnn_model import CrackCNN
from config import ROOT, PROC_DIR, RES_DIR, SEED

REP = ROOT / "report"
meta = pd.read_csv(PROC_DIR / "metadata.csv")
m = pd.read_csv(RES_DIR / "metrics_test.csv", index_col=0)
ps = pd.read_csv(RES_DIR / "metrics_per_surface.csv")
mc = pd.read_csv(RES_DIR / "mcnemar.csv")
cls = json.load(open(RES_DIR / "classical_summary.json"))
cnn = json.load(open(RES_DIR / "cnn_history.json"))
timing = json.load(open(RES_DIR / "timing.json"))
feats = pd.read_csv(PROC_DIR / "features.csv").iloc[:, 5:]
f3 = lambda x: f"{x:.3f}"

best = m["PR-AUC"].idxmax()
v = {"n_clean": f"{len(meta):,}", "n_pos": f"{meta.label.sum():,}", "n_neg": f"{(1 - meta.label).sum():,}",
     "prev": f3(meta[meta.split == "test"].label.mean()),
     "acc_trivial": f"{100 * (1 - meta.label.mean()):.1f}",
     "best": best, "best_f1": f3(m.loc[best, "F1"]), "best_prauc": f3(m.loc[best, "PR-AUC"]),
     "best_rocauc": f3(m.loc[best, "ROC-AUC"]),
     "lr_C": cls["Logistic Regression"]["best_params"]["C"],
     "rf_params": ", ".join(f"{k}={v}" for k, v in cls["Random Forest"]["best_params"].items())
                     .replace("depth=None", "max depth = none").replace("depth=", "max depth = ")
                     .replace("leaf=", "min leaf = ").replace("mf=sqrt", "√p features"),
     "cnn_params": f"{sum(p.numel() for p in CrackCNN().parameters()):,}", "cnn_epochs": len(cnn["history"]),
     "high_corr": str(int((np.triu(feats.corr().abs().values, 1) > 0.9).sum())),
     "pca2": f"{100 * PCA(2, random_state=SEED).fit(StandardScaler().fit_transform(feats)).explained_variance_ratio_.sum():.1f}",
     "t_feat": f"{timing['feature_extraction_ms']:.0f}", "t_cnn": f"{timing['cnn_forward_ms']:.1f}"}
for key, name in (("lr", "Logistic Regression"), ("rf", "Random Forest"), ("cnn", "CNN")):
    for met in ("F1", "PR-AUC", "ROC-AUC", "Precision", "Recall", "Specificity", "MCC", "Balanced acc.", "Accuracy"):
        v[f"{key}_{met.lower().replace('-', '').replace('.', '').replace(' ', '_')}"] = f3(m.loc[name, met])
    for met in ("F1", "PR-AUC", "ROC-AUC"):
        v[f"{key}_{met.lower().replace('-', '')}_ci"] = m.loc[name, f"{met} 95% CI"]
    v[f"{key}_thr"] = f"{m.loc[name, 'Threshold']:.2f}"
for key, name in (("lr", "Logistic Regression"), ("rf", "Random Forest")):
    for met in ("pr_auc", "roc_auc", "f1"):
        mu, sd = cls[name][f"cv_{met}"]
        v[f"{key}_cv_{met}"] = f"{mu:.3f} ± {sd:.3f}"

# split table rows
rows = []
for s, use in (("train", "fit models"), ("val", "tune hyper-parameters &amp; threshold"), ("test", "final evaluation only")):
    d = meta[meta.split == s]
    photos = " / ".join(str(d[d.surface == x].parent_id.nunique()) for x in "DWP")
    rows.append(f"<tr><td>{s.capitalize()}</td><td class='n'>{photos}</td><td class='n'>{len(d):,}</td>"
                f"<td class='n'>{d.label.sum():,}</td><td class='n'>{100 * d.label.mean():.1f}</td><td>{use}</td></tr>")
v["split_rows"] = "\n  ".join(rows)

# main metrics table
cols = ["Precision", "Recall", "Specificity", "F1", "Balanced acc.", "MCC", "ROC-AUC", "PR-AUC"]
hdr = "".join(f"<th class='n'>{c}</th>" for c in cols)
body = []
for name in m.index:
    cells = "".join(f"<td class='n'>{m.loc[name, c]:.3f}</td>" for c in cols)
    body.append(f"<tr{' class=best' if name == best else ''}><td>{name}</td><td class='n'>{m.loc[name, 'Threshold']:.2f}</td>{cells}</tr>")
v["metrics_table"] = f"<tr><th>Method</th><th class='n'>Thr.</th>{hdr}</tr>" + "".join(body)

# per-surface table
body = []
for surf in ("Bridge deck", "Wall", "Pavement"):
    d = ps[ps.Surface == surf].set_index("Method")
    cells = "".join(f"<td class='n'>{d.loc[n, 'F1']:.3f}</td>" for n in m.index) + \
            "".join(f"<td class='n'>{d.loc[n, 'PR-AUC']:.3f}</td>" for n in m.index)
    body.append(f"<tr><td>{surf}</td>{cells}</tr>")
v["surface_table"] = ("<tr><th></th><th class='n' colspan=3 style='text-align:center'>F1</th>"
                      "<th class='n' colspan=3 style='text-align:center'>PR-AUC</th></tr>"
                      "<tr><th>Surface</th>" + "".join(f"<th class='n'>{n.replace('Logistic Regression', 'LR').replace('Random Forest', 'RF')}</th>" for n in m.index) * 2 +
                      "</tr>" + "".join(body))
for _, r in ps.iterrows():
    k = {"Logistic Regression": "lr", "Random Forest": "rf", "CNN": "cnn"}[r.Method]
    v[f"{k}_{r.Surface.split()[-1].lower()}_f1"] = f3(r.F1)
    v[f"{k}_{r.Surface.split()[-1].lower()}_prauc"] = f3(r["PR-AUC"])

# McNemar table
body = []
for _, r in mc.iterrows():
    p = "&lt;&nbsp;0.001" if r["p-value"] < 1e-3 else f"{r['p-value']:.3f}"
    ab = lambda x: {"Logistic Regression": "LR", "Random Forest": "RF"}.get(x, x)
    body.append(f"<tr><td>{ab(r.A)} vs {ab(r.B)}</td><td class='n'>{r['A right, B wrong']:,}</td>"
                f"<td class='n'>{r['A wrong, B right']:,}</td><td class='n'>{p}</td></tr>")
v["mcnemar_table"] = ("<tr><th>Comparison (A vs B)</th><th class='n'>A right, B wrong</th>"
                      "<th class='n'>A wrong, B right</th><th class='n'>p-value</th></tr>" + "".join(body))

# confusion counts
test = meta[meta.split == "test"]
v["n_test"], v["n_test_pos"] = f"{len(test):,}", f"{test.label.sum():,}"

res_html = Template((REP / "results_section.html").read_text(encoding="utf-8")).substitute(v)
v["results_section"] = res_html
html = Template((REP / "report_template.html").read_text(encoding="utf-8")).substitute(v)
out_html = REP / "report.html"
out_html.write_text(html, encoding="utf-8")

edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
pdf = REP / "CE235_Project_Report_Shreyansh_Tripathi_24B0726.pdf"
subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                f"--print-to-pdf={pdf}", out_html.as_uri()], check=True, capture_output=True, timeout=180)
print("wrote", pdf)
