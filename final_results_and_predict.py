"""
final_results_and_predict.py
============================
Three things in one script:

PART 1 — Confusion matrices for 6 models side by side:
  LightGBM, XGBoost, Random Forest, SVM, QSVM, V2 Stacking

PART 2 — User enters a peptide sequence → V2 Stacking predicts
  Hemolytic or Non-Hemolytic with probability and explanation

PART 3 — Peptide property plots:
  - Amino acid composition comparison (Hemolytic vs Non-Hemolytic)
  - GRAVY score distribution
  - Net charge distribution
  - Feature importance bar chart

Run: python final_results_and_predict.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc
)
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

HYDRO = {
    'A': 1.8,  'C': 2.5,  'D':-3.5,  'E':-3.5,  'F': 2.8,
    'G':-0.4,  'H':-3.2,  'I': 4.5,  'K':-3.9,  'L': 3.8,
    'M': 1.9,  'N':-3.5,  'P':-1.6,  'Q':-3.5,  'R':-4.5,
    'S':-0.8,  'T':-0.7,  'V': 4.2,  'W':-0.9,  'Y':-1.3
}

COLOR_HEMO    = "#C0392B"   # red for hemolytic
COLOR_NONHEMO = "#2980B9"   # blue for non-hemolytic
COLOR_PRED    = "#8E44AD"   # purple for prediction

# ════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION FROM RAW SEQUENCE
# ════════════════════════════════════════════════════════════════
def extract_features(sequence):
    """Extract 23 features from a raw peptide sequence string."""
    seq    = sequence.upper().strip()
    length = len(seq)
    if length == 0:
        raise ValueError("Empty sequence.")

    aac           = [seq.count(aa) / length for aa in AMINO_ACIDS]
    pos           = seq.count("K") + seq.count("R") + seq.count("H")
    neg           = seq.count("D") + seq.count("E")
    net_charge    = pos - neg
    pos_neg_ratio = pos / (neg + 1e-6)
    gravy         = sum(HYDRO.get(aa, 0) for aa in seq) / length
    return np.array(aac + [net_charge, pos_neg_ratio, gravy])

# ════════════════════════════════════════════════════════════════
# QFE CIRCUIT V2
# ════════════════════════════════════════════════════════════════
def build_circuit_v2(angles):
    """
    Improved QFE circuit:
    RY → CNOT ring → RX(2x) → CNOT ring → RY → Measure
    """
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(4)
    for i, a in enumerate(angles):  qc.ry(float(a), i)
    for i in range(4):              qc.cx(i, (i+1) % 4)
    for i, a in enumerate(angles):  qc.rx(2.0*float(a), i)
    for i in range(4):              qc.cx(i, (i+1) % 4)
    for i, a in enumerate(angles):  qc.ry(float(a), i)
    return qc

def extract_qfe_features(X_subset, label=""):
    """Extract 7 quantum observables per sample using V2 circuit."""
    from qiskit.quantum_info import Statevector, SparsePauliOp
    ops = [
        SparsePauliOp("IIIZ"), SparsePauliOp("IIZI"),
        SparsePauliOp("IZII"), SparsePauliOp("ZIII"),
        SparsePauliOp("IIZZ"), SparsePauliOp("IZZI"),
        SparsePauliOp("ZZII"),
    ]
    feats, n = [], len(X_subset)
    for idx, angles in enumerate(X_subset):
        if idx % 200 == 0 and label:
            print(f"    {label}: {idx}/{n}")
        sv = Statevector(build_circuit_v2(angles))
        feats.append([sv.expectation_value(op).real for op in ops])
    return np.array(feats)

# ════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  FINAL RESULTS + PREDICTION PIPELINE")
print("=" * 65)

print("\n[STEP 1]  Loading data")
print("-" * 50)

train_df     = pd.read_csv("data/processed/train_features.csv")
test_df      = pd.read_csv("data/processed/test_features.csv")
feature_cols = [c for c in train_df.columns if c != "label"]

X_train = train_df[feature_cols].values
y_train = train_df["label"].values
X_test  = test_df[feature_cols].values
y_test  = test_df["label"].values

scaler_c = MinMaxScaler()
X_tr_c   = scaler_c.fit_transform(X_train)
X_te_c   = scaler_c.transform(X_test)

scaler_q = MinMaxScaler(feature_range=(0, np.pi))
X_tr_q   = scaler_q.fit_transform(X_train)
X_te_q   = scaler_q.transform(X_test)

pos_weight = (y_train==0).sum() / (y_train==1).sum()
print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE SELECTION + QFE V2 FEATURES
# ════════════════════════════════════════════════════════════════
print("\n[STEP 2]  Loading QFE V2 features")
print("-" * 50)

lgb_sel = lgb.LGBMClassifier(n_estimators=300, random_state=42,
                              class_weight="balanced", verbose=-1)
lgb_sel.fit(X_tr_c, y_train)
imp      = lgb_sel.feature_importances_
top4_idx = np.argsort(imp)[::-1][:4]
top4     = [feature_cols[i] for i in top4_idx]
print(f"  Top-4 for QFE: {top4}")

# Load V2 features
if (os.path.exists("results/Q_tr_v2.npy") and
        os.path.exists("results/Q_te_v2.npy")):
    Q_tr_v2 = np.load("results/Q_tr_v2.npy")
    Q_te_v2 = np.load("results/Q_te_v2.npy")
    if (Q_tr_v2.shape == (X_train.shape[0], 7) and
            Q_te_v2.shape == (X_test.shape[0], 7)):
        print(f"  Loaded V2 features — Q_tr: {Q_tr_v2.shape}")
    else:
        Q_tr_v2, Q_te_v2 = None, None
else:
    Q_tr_v2, Q_te_v2 = None, None

if Q_tr_v2 is None:
    print("  Computing V2 features (~3 min)...")
    Q_tr_v2 = extract_qfe_features(X_tr_q[:, top4_idx], "Train")
    Q_te_v2 = extract_qfe_features(X_te_q[:, top4_idx], "Test")
    np.save("results/Q_tr_v2.npy", Q_tr_v2)
    np.save("results/Q_te_v2.npy", Q_te_v2)
    print("  Saved Q_tr_v2.npy and Q_te_v2.npy")

X_tr_30 = np.hstack([X_tr_c, Q_tr_v2])
X_te_30 = np.hstack([X_te_c, Q_te_v2])

# ════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN ALL MODELS
# ════════════════════════════════════════════════════════════════
print("\n[STEP 3]  Training all models")
print("-" * 50)

cms     = {}   # confusion matrices
metrics = {}   # accuracy, recall etc.
probas  = {}   # test probabilities for ROC

def record(name, y_pred, y_proba):
    cm = confusion_matrix(y_test, y_pred)
    cms[name] = cm
    tn, fp, fn, tp = cm.ravel()
    metrics[name] = {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred), 4),
        "AUC":       round(roc_auc_score(y_test, y_proba), 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }
    m = metrics[name]
    probas[name] = y_proba
    print(f"  {name:<38} Acc={m['Accuracy']:.4f}  "
          f"Rec={m['Recall']:.4f}  AUC={m['AUC']:.4f}  "
          f"TP={m['TP']} TN={m['TN']}")

# ── Classical models ─────────────────────────────────────────────
svm_clf = SVC(kernel="rbf", C=1.0, probability=True,
              class_weight="balanced", random_state=42)
svm_clf.fit(X_tr_c, y_train)
record("SVM (RBF, 23 feat)",
       svm_clf.predict(X_te_c),
       svm_clf.predict_proba(X_te_c)[:, 1])

rf_clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=42)
rf_clf.fit(X_tr_c, y_train)
record("Random Forest (23 feat)",
       rf_clf.predict(X_te_c),
       rf_clf.predict_proba(X_te_c)[:, 1])

xgb_clf = xgb.XGBClassifier(n_estimators=300, random_state=42,
                              eval_metric="logloss", verbosity=0,
                              scale_pos_weight=pos_weight)
xgb_clf.fit(X_tr_c, y_train)
record("XGBoost (23 feat)",
       xgb_clf.predict(X_te_c),
       xgb_clf.predict_proba(X_te_c)[:, 1])

lgb_clf = lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                              random_state=42, verbose=-1)
lgb_clf.fit(X_tr_c, y_train)
record("LightGBM (23 feat)",
       lgb_clf.predict(X_te_c),
       lgb_clf.predict_proba(X_te_c)[:, 1])

# ── QSVM ─────────────────────────────────────────────────────────
qsvm_pred = qsvm_proba = None
if (os.path.exists("results/K_tr_sharp.npy") and
        os.path.exists("results/K_te_sharp.npy")):
    K_tr = np.load("results/K_tr_sharp.npy")
    K_te = np.load("results/K_te_sharp.npy")
    if (K_tr.shape[0] == X_train.shape[0] and
            K_te.shape[0] == X_test.shape[0]):
        qsvm = SVC(kernel="precomputed", C=1.0, probability=True,
                   class_weight="balanced", random_state=42)
        qsvm.fit(K_tr, y_train)
        qsvm_pred  = qsvm.predict(K_te)
        qsvm_proba = qsvm.predict_proba(K_te)[:, 1]
        record("QSVM + Sharpening (8 feat)",
               qsvm_pred, qsvm_proba)
        qsvm_model = qsvm
    else:
        print("  QSVM kernel shape mismatch — skipping")
else:
    print("  QSVM kernels not found — skipping (run main_pipeline.py first)")

# ── V2 Stacking ensemble ─────────────────────────────────────────
print("\n  Building V2 Stacking ensemble (5-fold OOF)...")
skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb  = np.zeros(len(y_train))
oof_svm  = np.zeros(len(y_train))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_c, y_train)):
    m1 = lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                             random_state=42, verbose=-1)
    m1.fit(X_tr_c[tr_idx], y_train[tr_idx])
    oof_lgb[val_idx] = m1.predict_proba(X_tr_c[val_idx])[:, 1]

    m2 = SVC(kernel="rbf", C=1.0, probability=True,
             class_weight="balanced", random_state=42)
    m2.fit(X_tr_30[tr_idx], y_train[tr_idx])
    oof_svm[val_idx] = m2.predict_proba(X_tr_30[val_idx])[:, 1]
    print(f"    Fold {fold+1}/5 done")

meta_tr  = np.column_stack([oof_lgb, oof_svm])
meta_clf = LogisticRegression(random_state=42)
meta_clf.fit(meta_tr, y_train)

# Full models for test + prediction
lgb_full = lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                               random_state=42, verbose=-1)
lgb_full.fit(X_tr_c, y_train)

svm_full = SVC(kernel="rbf", C=1.0, probability=True,
               class_weight="balanced", random_state=42)
svm_full.fit(X_tr_30, y_train)

meta_te      = np.column_stack([lgb_full.predict_proba(X_te_c)[:, 1],
                                 svm_full.predict_proba(X_te_30)[:, 1]])
stack_pred   = meta_clf.predict(meta_te)
stack_proba  = meta_clf.predict_proba(meta_te)[:, 1]
record("V2 Stacking (LightGBM + QFE+SVM)",
       stack_pred, stack_proba)

print(f"\n  Meta-learner weights: "
      f"LightGBM={meta_clf.coef_[0][0]:.3f}, "
      f"V2 QFE+SVM={meta_clf.coef_[0][1]:.3f}")

# ════════════════════════════════════════════════════════════════
# PLOT 1 — CONFUSION MATRICES (all models, 2 rows)
# ════════════════════════════════════════════════════════════════
print("\n[PLOT 1]  Confusion matrices")

panel_cfg = [
    ("SVM (RBF, 23 feat)",              plt.cm.Oranges),
    ("Random Forest (23 feat)",         plt.cm.YlGn),
    ("XGBoost (23 feat)",               plt.cm.Greens),
    ("LightGBM (23 feat)",              plt.cm.Blues),
    ("QSVM + Sharpening (8 feat)",      plt.cm.Purples),
    ("V2 Stacking (LightGBM + QFE+SVM)", plt.cm.RdPu),
]
# Filter to only models we actually have
panel_cfg = [(n, c) for n, c in panel_cfg if n in cms]

n_panels = len(panel_cfg)
ncols    = 3
nrows    = (n_panels + ncols - 1) // ncols

fig1, axes1 = plt.subplots(nrows, ncols,
                            figsize=(6.5*ncols, 5.5*nrows))
axes1_flat  = axes1.flatten() if nrows > 1 else axes1

fig1.suptitle(
    "Confusion Matrices — All Models Comparison\n"
    "HemoPI-2 Hemolytic Peptide Classification (202 Test Peptides)",
    fontsize=15, fontweight="bold", color="#1F4E79", y=1.02
)

for idx, (name, cmap) in enumerate(panel_cfg):
    ax  = axes1_flat[idx]
    cm  = cms[name]
    m   = metrics[name]
    tn, fp, fn, tp = cm.ravel()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Hemolytic", "Hemolytic"]
    )
    disp.plot(ax=ax, cmap=cmap, colorbar=False, values_format="d",
              text_kw={"fontsize": 16, "fontweight": "bold"})

    is_best = name == "V2 Stacking (LightGBM + QFE+SVM)"
    title_color = "#8E44AD" if is_best else "#1F4E79"
    star = " ★ NOVEL" if is_best else ""

    ax.set_title(f"{name}{star}", fontsize=11, fontweight="bold",
                 color=title_color, pad=8)

    stats = (f"Acc={m['Accuracy']:.3f}  Recall={m['Recall']:.3f}  "
             f"AUC={m['AUC']:.4f}\n"
             f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    ax.set_xlabel(f"Predicted Label\n{stats}", fontsize=9)
    ax.set_ylabel("True Label", fontsize=10)

    if is_best:
        for spine in ax.spines.values():
            spine.set_edgecolor("#8E44AD")
            spine.set_linewidth(3)

# Hide unused panels
for idx in range(len(panel_cfg), len(axes1_flat)):
    axes1_flat[idx].set_visible(False)

plt.tight_layout()
plt.savefig("results/all_confusion_matrices.png",
            dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: results/all_confusion_matrices.png")

# ════════════════════════════════════════════════════════════════
# PLOT 2 — ROC CURVES (all models)
# ════════════════════════════════════════════════════════════════
print("\n[PLOT 2]  ROC curves")

roc_cfg = [
    ("SVM (RBF, 23 feat)",               "#FF7F00", "--",  1.6),
    ("Random Forest (23 feat)",           "#377EB8", "-.",  1.6),
    ("XGBoost (23 feat)",                 "#4DAF4A", ":",   1.6),
    ("LightGBM (23 feat)",                "#A65628", "--",  2.0),
    ("QSVM + Sharpening (8 feat)",        "#984EA3", "-.",  2.0),
    ("V2 Stacking (LightGBM + QFE+SVM)", "#8E44AD", "-",   3.2),
]

fig2, ax2 = plt.subplots(figsize=(9, 7))
ax2.plot([0,1],[0,1], "k--", lw=1.2,
         label="Random Baseline (AUC=0.50)", alpha=0.5)

for name, color, ls, lw in roc_cfg:
    if name not in probas:
        continue
    fpr_r, tpr_r, _ = roc_curve(y_test, probas[name])
    auc_r = metrics[name]["AUC"]
    star  = " ★" if name == "V2 Stacking (LightGBM + QFE+SVM)" else ""
    ax2.plot(fpr_r, tpr_r, color=color, linestyle=ls, linewidth=lw,
             label=f"{name}{star}  (AUC={auc_r:.4f})")

ax2.set_xlim([-0.01, 1.01])
ax2.set_ylim([-0.01, 1.05])
ax2.set_xlabel("False Positive Rate", fontsize=13)
ax2.set_ylabel("True Positive Rate",  fontsize=13)
ax2.set_title(
    "ROC Curves — All Models\n"
    "HemoPI-2 Hemolytic Peptide Classification",
    fontsize=13, fontweight="bold", color="#1F4E79"
)
ax2.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
ax2.grid(True, alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/all_roc_curves.png",
            dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: results/all_roc_curves.png")

# ════════════════════════════════════════════════════════════════
# PLOT 3 — PEPTIDE PROPERTY DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════
print("\n[PLOT 3]  Peptide property distributions")

# Compute properties for all training peptides
try:
    from feature_extraction import extract_features as fe_ext
    all_features = np.vstack([X_train, X_test])
    all_labels   = np.concatenate([y_train, y_test])
except Exception:
    all_features = np.vstack([X_train, X_test])
    all_labels   = np.concatenate([y_train, y_test])

feat_df = pd.DataFrame(all_features, columns=feature_cols)
feat_df["label"] = all_labels

hemo    = feat_df[feat_df["label"] == 1]
nonhemo = feat_df[feat_df["label"] == 0]

fig3 = plt.figure(figsize=(18, 14))
gs   = GridSpec(2, 3, figure=fig3, hspace=0.45, wspace=0.35)

fig3.suptitle(
    "Peptide Property Analysis — Hemolytic vs Non-Hemolytic\n"
    "HemoPI-2 Dataset (812 train + 202 test = 1014 peptides)",
    fontsize=15, fontweight="bold", color="#1F4E79", y=1.01
)

# ── Panel A: GRAVY score distribution ────────────────────────────
ax_a = fig3.add_subplot(gs[0, 0])
gravy_col = "hydrophobicity" if "hydrophobicity" in feature_cols else feature_cols[-1]

# Use last 3 features as PCP
gravy_idx  = len(AMINO_ACIDS) + 2
hemo_gravy    = hemo.iloc[:, gravy_idx]
nonhemo_gravy = nonhemo.iloc[:, gravy_idx]

ax_a.hist(nonhemo_gravy, bins=25, alpha=0.7, color=COLOR_NONHEMO,
          label=f"Non-Hemolytic (n={len(nonhemo)})", edgecolor="white")
ax_a.hist(hemo_gravy,    bins=25, alpha=0.7, color=COLOR_HEMO,
          label=f"Hemolytic (n={len(hemo)})",     edgecolor="white")
ax_a.axvline(hemo_gravy.mean(),    color=COLOR_HEMO,    ls="--", lw=2,
             label=f"Hemo mean={hemo_gravy.mean():.2f}")
ax_a.axvline(nonhemo_gravy.mean(), color=COLOR_NONHEMO, ls="--", lw=2,
             label=f"Non-Hemo mean={nonhemo_gravy.mean():.2f}")
ax_a.set_xlabel("GRAVY Score (Hydropathy)", fontsize=11)
ax_a.set_ylabel("Count", fontsize=11)
ax_a.set_title("GRAVY Score Distribution", fontsize=12, fontweight="bold")
ax_a.legend(fontsize=8)
ax_a.grid(True, alpha=0.3)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# ── Panel B: Net charge distribution ─────────────────────────────
ax_b = fig3.add_subplot(gs[0, 1])
charge_idx    = len(AMINO_ACIDS)
hemo_charge    = hemo.iloc[:, charge_idx]
nonhemo_charge = nonhemo.iloc[:, charge_idx]

charge_bins = np.arange(hemo_charge.min()-0.5,
                         max(hemo_charge.max(), nonhemo_charge.max())+1.5)
ax_b.hist(nonhemo_charge, bins=charge_bins, alpha=0.7,
          color=COLOR_NONHEMO,
          label=f"Non-Hemolytic mean={nonhemo_charge.mean():.1f}",
          edgecolor="white")
ax_b.hist(hemo_charge, bins=charge_bins, alpha=0.7,
          color=COLOR_HEMO,
          label=f"Hemolytic mean={hemo_charge.mean():.1f}",
          edgecolor="white")
ax_b.set_xlabel("Net Charge at pH 7.4", fontsize=11)
ax_b.set_ylabel("Count", fontsize=11)
ax_b.set_title("Net Charge Distribution", fontsize=12, fontweight="bold")
ax_b.legend(fontsize=9)
ax_b.grid(True, alpha=0.3)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

# ── Panel C: Top-10 amino acid frequency comparison ───────────────
ax_c = fig3.add_subplot(gs[0, 2])
lgb_imp  = lgb_sel.feature_importances_
top10idx = np.argsort(lgb_imp)[::-1][:10]
top10aa  = [feature_cols[i] for i in top10idx]

hemo_means    = [hemo.iloc[:, i].mean() for i in top10idx]
nonhemo_means = [nonhemo.iloc[:, i].mean() for i in top10idx]

x_pos = np.arange(len(top10aa))
w     = 0.38
ax_c.bar(x_pos - w/2, nonhemo_means, w, color=COLOR_NONHEMO,
         alpha=0.85, label="Non-Hemolytic", edgecolor="white")
ax_c.bar(x_pos + w/2, hemo_means,    w, color=COLOR_HEMO,
         alpha=0.85, label="Hemolytic",     edgecolor="white")
ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(top10aa, rotation=30, fontsize=10)
ax_c.set_xlabel("Feature (Top-10 by Importance)", fontsize=11)
ax_c.set_ylabel("Mean Frequency", fontsize=11)
ax_c.set_title("Top-10 Feature Comparison", fontsize=12, fontweight="bold")
ax_c.legend(fontsize=9)
ax_c.grid(True, alpha=0.3, axis="y")
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)

# ── Panel D: Accuracy bar chart ───────────────────────────────────
ax_d = fig3.add_subplot(gs[1, 0])
model_names = list(metrics.keys())
model_accs  = [metrics[m]["Accuracy"]*100 for m in model_names]
model_aucs  = [metrics[m]["AUC"] for m in model_names]

colors_bar = [COLOR_PRED if "Stacking" in m else
              COLOR_HEMO if "QSVM" in m else
              "#5D6D7E" for m in model_names]

short_names = [n.split("(")[0].strip()[:18] for n in model_names]
bars = ax_d.bar(short_names, model_accs, color=colors_bar,
                edgecolor="white", linewidth=0.8, alpha=0.9)

for bar, acc in zip(bars, model_accs):
    ax_d.text(bar.get_x() + bar.get_width()/2,
              bar.get_height() + 0.3,
              f"{acc:.1f}%", ha="center", va="bottom",
              fontsize=9, fontweight="bold")

ax_d.axhline(y=80.20, color="#E74C3C", ls="--", lw=2,
             label="LightGBM baseline 80.20%")
ax_d.set_ylim(40, 95)
ax_d.set_ylabel("Accuracy (%)", fontsize=11)
ax_d.set_title("Model Accuracy Comparison", fontsize=12, fontweight="bold")
ax_d.set_xticklabels(short_names, rotation=25, fontsize=8, ha="right")
ax_d.legend(fontsize=8)
ax_d.grid(True, alpha=0.3, axis="y")
ax_d.spines["top"].set_visible(False)
ax_d.spines["right"].set_visible(False)

# ── Panel E: AUC bar chart ────────────────────────────────────────
ax_e = fig3.add_subplot(gs[1, 1])
bars_auc = ax_e.bar(short_names, model_aucs, color=colors_bar,
                    edgecolor="white", linewidth=0.8, alpha=0.9)

for bar, a in zip(bars_auc, model_aucs):
    ax_e.text(bar.get_x() + bar.get_width()/2,
              bar.get_height() + 0.003,
              f"{a:.4f}", ha="center", va="bottom",
              fontsize=8, fontweight="bold")

ax_e.axhline(y=metrics["LightGBM (23 feat)"]["AUC"],
             color="#E74C3C", ls="--", lw=2,
             label=f"LightGBM AUC={metrics['LightGBM (23 feat)']['AUC']:.4f}")
ax_e.set_ylim(0.60, 0.95)
ax_e.set_ylabel("AUC", fontsize=11)
ax_e.set_title("Model AUC Comparison", fontsize=12, fontweight="bold")
ax_e.set_xticklabels(short_names, rotation=25, fontsize=8, ha="right")
ax_e.legend(fontsize=8)
ax_e.grid(True, alpha=0.3, axis="y")
ax_e.spines["top"].set_visible(False)
ax_e.spines["right"].set_visible(False)

# ── Panel F: Feature importance ───────────────────────────────────
ax_f = fig3.add_subplot(gs[1, 2])
top15idx  = np.argsort(lgb_imp)[::-1][:15]
top15     = [feature_cols[i] for i in top15idx]
top15_imp = [lgb_imp[i] for i in top15idx]
colors_fi = [COLOR_HEMO if f in top4 else "#5D6D7E" for f in top15]

ax_f.barh(range(len(top15)), top15_imp[::-1],
          color=colors_fi[::-1], edgecolor="white", alpha=0.9)
ax_f.set_yticks(range(len(top15)))
ax_f.set_yticklabels(top15[::-1], fontsize=9)
ax_f.set_xlabel("LightGBM Importance Score", fontsize=11)
ax_f.set_title("Feature Importance\n(Red = used in QFE circuit)",
               fontsize=12, fontweight="bold")
ax_f.grid(True, alpha=0.3, axis="x")
ax_f.spines["top"].set_visible(False)
ax_f.spines["right"].set_visible(False)

plt.savefig("results/peptide_analysis_plots.png",
            dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: results/peptide_analysis_plots.png")

# ════════════════════════════════════════════════════════════════
# STEP 4 — USER PREDICTION WITH V2 STACKING
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  USER PREDICTION — V2 STACKING MODEL")
print("=" * 65)
print("""
  This model combines:
    Branch 1 : LightGBM on 23 classical features
    Branch 2 : SVM on 30 features (23 classical + 7 V2 quantum)
    Fusion   : Logistic Regression (learned weights)
""")

sequence = input("  Enter peptide sequence: ").strip().upper()

if not sequence:
    print("  No sequence entered.")
else:
    valid_aa = set(AMINO_ACIDS)
    invalid  = [c for c in sequence if c not in valid_aa]
    if invalid:
        print(f"  WARNING: Invalid amino acids found: {set(invalid)}")
        print("  Valid amino acids: ACDEFGHIKLMNPQRSTVWY")
    else:
        # Extract 23 classical features
        raw_feat = extract_features(sequence)

        # Scale for classical branch
        feat_c = scaler_c.transform(raw_feat.reshape(1, -1))

        # Compute QFE V2 features for this single peptide
        feat_q_input = scaler_q.transform(raw_feat.reshape(1, -1))[:, top4_idx]
        feat_q = extract_qfe_features(feat_q_input)   # (1, 7)

        # 30-feature vector for SVM branch
        feat_30 = np.hstack([feat_c, feat_q])

        # Get branch probabilities
        p_lgb = lgb_full.predict_proba(feat_c)[0, 1]
        p_svm = svm_full.predict_proba(feat_30)[0, 1]

        # Fusion
        meta_input = np.array([[p_lgb, p_svm]])
        final_pred  = meta_clf.predict(meta_input)[0]
        final_proba = meta_clf.predict_proba(meta_input)[0, 1]

        # Compute peptide properties for display
        length       = len(sequence)
        pos          = sequence.count("K")+sequence.count("R")+sequence.count("H")
        neg          = sequence.count("D")+sequence.count("E")
        net_charge   = pos - neg
        gravy_score  = sum(HYDRO.get(aa, 0) for aa in sequence) / length
        aa_comp      = {aa: sequence.count(aa)/length for aa in AMINO_ACIDS
                        if sequence.count(aa) > 0}

        label     = "HEMOLYTIC" if final_pred == 1 else "NON-HEMOLYTIC"
        conf_pct  = final_proba*100 if final_pred == 1 else (1-final_proba)*100

        print("\n" + "=" * 55)
        print(f"  PREDICTION RESULT")
        print("=" * 55)
        print(f"  Sequence   : {sequence}")
        print(f"  Length     : {length} amino acids")
        print(f"  Result     : {'🔴 ' if final_pred==1 else '🟢 '}{label}")
        print(f"  Confidence : {conf_pct:.1f}%")
        print(f"  Hemolytic probability : {final_proba:.4f}")
        print("-" * 55)
        print(f"  Branch probabilities:")
        print(f"    LightGBM  (classical) : {p_lgb:.4f}")
        print(f"    V2 QFE+SVM (quantum)  : {p_svm:.4f}")
        print(f"    Fused (meta-learner)  : {final_proba:.4f}")
        print("-" * 55)
        print(f"  Peptide properties:")
        print(f"    Net Charge   : {net_charge:+.0f}")
        print(f"    GRAVY Score  : {gravy_score:.3f} "
              f"({'hydrophobic' if gravy_score>0 else 'hydrophilic'})")
        print(f"    Top AAs      : "
              f"{', '.join([f'{aa}={v:.2f}' for aa,v in sorted(aa_comp.items(), key=lambda x:-x[1])[:5]])}")

        if final_pred == 1:
            print("\n  ⚠  WARNING: This peptide is predicted HEMOLYTIC.")
            print("     It may destroy red blood cells.")
            print("     NOT recommended for drug development.")
        else:
            print("\n  ✓  This peptide is predicted NON-HEMOLYTIC.")
            print("     It appears safe with respect to red blood cell damage.")
        print("=" * 55)

        # ── PLOT 4: Prediction visualisation ─────────────────────
        print("\n[PLOT 4]  Prediction visualization")

        fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))
        fig4.suptitle(
            f"Prediction Analysis for: {sequence[:30]}{'...' if len(sequence)>30 else ''}\n"
            f"Result: {label}  |  Confidence: {conf_pct:.1f}%",
            fontsize=13, fontweight="bold",
            color=COLOR_HEMO if final_pred==1 else "#2980B9",
            y=1.04
        )

        # Panel A: Probability gauge
        ax4a = axes4[0]
        categories = ["Non-Hemolytic", "Hemolytic"]
        bar_probs  = [1 - final_proba, final_proba]
        bar_colors = [COLOR_NONHEMO, COLOR_HEMO]
        bars4 = ax4a.bar(categories, [p*100 for p in bar_probs],
                         color=bar_colors, edgecolor="white",
                         linewidth=2, alpha=0.9, width=0.5)
        for bar, prob in zip(bars4, bar_probs):
            ax4a.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 0.5,
                      f"{prob*100:.1f}%",
                      ha="center", va="bottom",
                      fontsize=14, fontweight="bold")

        ax4a.axhline(y=50, color="black", ls="--", lw=1.5,
                     alpha=0.5, label="50% threshold")
        ax4a.set_ylim(0, 110)
        ax4a.set_ylabel("Probability (%)", fontsize=12)
        ax4a.set_title("Prediction Probability", fontsize=12, fontweight="bold")
        ax4a.legend(fontsize=9)
        ax4a.grid(True, alpha=0.3, axis="y")
        ax4a.spines["top"].set_visible(False)
        ax4a.spines["right"].set_visible(False)

        # Panel B: Branch contributions
        ax4b = axes4[1]
        branch_names  = ["LightGBM\n(Classical)", "V2 QFE+SVM\n(Quantum)",
                          "Meta-Learner\n(Fused)"]
        branch_probs  = [p_lgb*100, p_svm*100, final_proba*100]
        branch_colors = ["#2E75B6", "#8E44AD",
                         COLOR_HEMO if final_pred==1 else COLOR_NONHEMO]

        bars_b = ax4b.bar(branch_names, branch_probs,
                          color=branch_colors, edgecolor="white",
                          linewidth=2, alpha=0.9, width=0.5)
        for bar, prob in zip(bars_b, branch_probs):
            ax4b.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 0.5,
                      f"{prob:.1f}%",
                      ha="center", va="bottom",
                      fontsize=12, fontweight="bold", color="black")

        ax4b.axhline(y=50, color="black", ls="--", lw=1.5,
                     alpha=0.5, label="Decision threshold 50%")
        ax4b.set_ylim(0, 115)
        ax4b.set_ylabel("Hemolytic Probability (%)", fontsize=12)
        ax4b.set_title("Branch Probabilities\n(How each model voted)",
                       fontsize=12, fontweight="bold")
        ax4b.legend(fontsize=9)
        ax4b.grid(True, alpha=0.3, axis="y")
        ax4b.spines["top"].set_visible(False)
        ax4b.spines["right"].set_visible(False)

        # Panel C: Amino acid composition vs dataset averages
        ax4c = axes4[2]
        top8_aa  = [feature_cols[i] for i in np.argsort(lgb_imp)[::-1][:8]]
        top8_idx2 = [feature_cols.index(aa) for aa in top8_aa]

        user_vals = [raw_feat[i] for i in top8_idx2]
        hemo_avg  = [hemo.iloc[:, i].mean() for i in top8_idx2]
        nh_avg    = [nonhemo.iloc[:, i].mean() for i in top8_idx2]

        x8   = np.arange(len(top8_aa))
        w8   = 0.28
        ax4c.bar(x8 - w8,   nh_avg,    w8, color=COLOR_NONHEMO,
                 alpha=0.7, label="Avg Non-Hemolytic", edgecolor="white")
        ax4c.bar(x8,        hemo_avg,  w8, color=COLOR_HEMO,
                 alpha=0.7, label="Avg Hemolytic",     edgecolor="white")
        ax4c.bar(x8 + w8,   user_vals, w8, color=COLOR_PRED,
                 alpha=0.9, label=f"Your peptide",     edgecolor="white")

        ax4c.set_xticks(x8)
        ax4c.set_xticklabels(top8_aa, fontsize=10)
        ax4c.set_ylabel("Feature Value", fontsize=12)
        ax4c.set_title("Your Peptide vs Dataset\n(Top-8 important features)",
                       fontsize=12, fontweight="bold")
        ax4c.legend(fontsize=9)
        ax4c.grid(True, alpha=0.3, axis="y")
        ax4c.spines["top"].set_visible(False)
        ax4c.spines["right"].set_visible(False)

        plt.tight_layout()
        out_name = f"results/prediction_{sequence[:10]}.png"
        plt.savefig(out_name, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {out_name}")

# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  OUTPUT FILES")
print("=" * 65)
print("""
  results/all_confusion_matrices.png  — 6 models side by side
  results/all_roc_curves.png          — ROC curves all models
  results/peptide_analysis_plots.png  — property distributions
  results/prediction_[seq].png        — your peptide analysis

  All plots are high resolution (220 DPI) — ready for PPT.
""")
print("  ALL DONE.")