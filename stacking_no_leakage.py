"""
stacking_no_leakage.py
======================
Re-runs the V2 stacking ensemble WITHOUT feature-selection leakage.

OLD (leaky) procedure:
  top-4 features chosen once using all 812 training labels
  -> QFE features computed once -> 5-fold stacking

CORRECTED procedure:
  inside EACH fold, using only that fold's training samples:
    1. fit MinMaxScaler
    2. LightGBM picks top-4 features
    3. fit [0, pi] scaler on those 4 features
    4. build V2 QFE features
    5. train LightGBM (23 feat) and SVM (30 feat)
  -> predict the held-out fold -> collect out-of-fold probabilities
  -> train Logistic Regression on them
  Final models: same steps on all 812 samples, evaluated on the 202 test set.

Both procedures are run with 3 different fold splits (seeds) so you can
see how stable the result is.

Needs only:
  data/processed/train_features.csv
  data/processed/test_features.csv

Outputs (results/):
  noleak_summary.csv
  noleak_confusion_matrices.png
  noleak_roc.png
"""

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
import lightgbm as lgb
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

N_FOLDS = 5
SEEDS   = [42, 7, 2024]

OPS = [SparsePauliOp(s) for s in
       ["IIIZ", "IIZI", "IZII", "ZIII", "IIZZ", "IZZI", "ZZII"]]


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════
def build_circuit_v2(angles):
    """V2 circuit: RY -> CNOT ring -> RX(2x) -> CNOT ring -> RY"""
    qc = QuantumCircuit(4)
    for i, a in enumerate(angles):
        qc.ry(float(a), i)
    for i in range(4):
        qc.cx(i, (i + 1) % 4)
    for i, a in enumerate(angles):
        qc.rx(2.0 * float(a), i)
    for i in range(4):
        qc.cx(i, (i + 1) % 4)
    for i, a in enumerate(angles):
        qc.ry(float(a), i)
    return qc


def qfe_features(angles_matrix):
    """4 angles per peptide -> 7 expectation values per peptide."""
    out = []
    for angles in angles_matrix:
        sv = Statevector(build_circuit_v2(angles))
        out.append([sv.expectation_value(op).real for op in OPS])
    return np.array(out)


def make_lgb():
    return lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                              random_state=42, verbose=-1)


def make_svm():
    return SVC(kernel="rbf", C=1.0, probability=True,
               class_weight="balanced", random_state=42)


def prepare_features(X_fit, y_fit, X_other):
    """
    Everything that learns from data is fitted on X_fit / y_fit ONLY.
    X_other (fold-validation or test) is only transformed, never used to fit.

    Returns: Xc_fit, Xc_other (23 scaled features),
             Q_fit,  Q_other  (7 quantum features),
             top4 feature indices (in qubit order)
    """
    sc_c   = MinMaxScaler().fit(X_fit)
    Xc_fit = sc_c.transform(X_fit)
    Xc_oth = sc_c.transform(X_other)

    selector = make_lgb().fit(Xc_fit, y_fit)
    top4     = np.argsort(selector.feature_importances_)[::-1][:4]

    sc_q  = MinMaxScaler(feature_range=(0, np.pi)).fit(X_fit[:, top4])
    A_fit = sc_q.transform(X_fit[:, top4])
    A_oth = sc_q.transform(X_other[:, top4])

    return Xc_fit, Xc_oth, qfe_features(A_fit), qfe_features(A_oth), top4


def get_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Recall":    recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred),
        "AUC":       roc_auc_score(y_true, y_proba),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }, cm


# ════════════════════════════════════════════════════════════════
# STEP 1 - LOAD DATA
# ════════════════════════════════════════════════════════════════
print("=" * 66)
print("  STACKING WITHOUT FEATURE-SELECTION LEAKAGE")
print("=" * 66)

train_df     = pd.read_csv("data/processed/train_features.csv")
test_df      = pd.read_csv("data/processed/test_features.csv")
feature_cols = [c for c in train_df.columns if c != "label"]

X_train = train_df[feature_cols].values
y_train = train_df["label"].values
X_test  = test_df[feature_cols].values
y_test  = test_df["label"].values

print(f"\n  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
se = np.sqrt(0.8 * 0.2 / len(y_test))
print(f"  NOTE: with {len(y_test)} test peptides, 1 peptide = "
      f"{100/len(y_test):.2f}% accuracy and the 95% CI at 80% is about "
      f"+/-{1.96*se*100:.1f}%.")

# ════════════════════════════════════════════════════════════════
# STEP 2 - FINAL BASE MODELS (trained on all 812, tested on 202)
#          These are the same for the old and corrected procedures.
# ════════════════════════════════════════════════════════════════
print("\n[STEP 2]  Final base models on all 812 training samples")
t0 = time.time()
Xc_tr, Xc_te, Q_tr, Q_te, top4_final = prepare_features(
    X_train, y_train, X_test)
print(f"  Final top-4 features (qubit order): "
      f"{[feature_cols[i] for i in top4_final]}")

lgb_full = make_lgb().fit(Xc_tr, y_train)
svm_full = make_svm().fit(np.hstack([Xc_tr, Q_tr]), y_train)

p_lgb_te = lgb_full.predict_proba(Xc_te)[:, 1]
p_svm_te = svm_full.predict_proba(np.hstack([Xc_te, Q_te]))[:, 1]
meta_te  = np.column_stack([p_lgb_te, p_svm_te])
print(f"  Done in {time.time() - t0:.1f}s")

# Baseline: LightGBM alone on 23 features
base_pred  = lgb_full.predict(Xc_te)
base_m, base_cm = get_metrics(y_test, base_pred, p_lgb_te)
print(f"  LightGBM baseline: Acc={base_m['Accuracy']:.4f}  "
      f"AUC={base_m['AUC']:.4f}")

# ════════════════════════════════════════════════════════════════
# STEP 3 - OUT-OF-FOLD PREDICTIONS (old vs corrected)
# ════════════════════════════════════════════════════════════════
def oof_predictions(per_fold_selection, seed):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros((len(y_train), 2))
    chosen = []
    for tr, va in skf.split(X_train, y_train):
        if per_fold_selection:
            # CORRECTED: everything fitted on this fold's training part only
            Xc_a, Xc_b, Q_a, Q_b, top4 = prepare_features(
                X_train[tr], y_train[tr], X_train[va])
            chosen.append([feature_cols[i] for i in top4])
        else:
            # OLD: features prepared once from all 812 labels (leaky)
            Xc_a, Xc_b = Xc_tr[tr], Xc_tr[va]
            Q_a,  Q_b  = Q_tr[tr],  Q_tr[va]

        m_lgb = make_lgb().fit(Xc_a, y_train[tr])
        m_svm = make_svm().fit(np.hstack([Xc_a, Q_a]), y_train[tr])
        oof[va, 0] = m_lgb.predict_proba(Xc_b)[:, 1]
        oof[va, 1] = m_svm.predict_proba(np.hstack([Xc_b, Q_b]))[:, 1]
    return oof, chosen


print("\n[STEP 3]  Stacking - old (leaky) vs corrected")
print("-" * 66)

MODES = [("OLD stacking (selection on all 812)", False),
         ("CORRECTED stacking (selection per fold)", True)]

rows  = [{"Method": "LightGBM baseline (23 feat)", "Seed": "-", **base_m}]
store = {}
fold_features_seed42 = []

for mode_name, per_fold in MODES:
    for seed in SEEDS:
        t0 = time.time()
        oof, chosen = oof_predictions(per_fold, seed)
        meta  = LogisticRegression(random_state=42).fit(oof, y_train)
        pred  = meta.predict(meta_te)
        proba = meta.predict_proba(meta_te)[:, 1]
        m, cm = get_metrics(y_test, pred, proba)
        rows.append({"Method": mode_name, "Seed": seed, **m})
        store[(mode_name, seed)] = (pred, proba, cm)
        if per_fold and seed == SEEDS[0]:
            fold_features_seed42 = chosen
        print(f"  {mode_name:<42} seed={seed:<5} "
              f"Acc={m['Accuracy']:.4f}  Rec={m['Recall']:.4f}  "
              f"AUC={m['AUC']:.4f}  "
              f"w=[{meta.coef_[0][0]:.2f}, {meta.coef_[0][1]:.2f}]  "
              f"({time.time() - t0:.0f}s)")

# Which features did each fold pick?
print("\n  Top-4 features chosen inside each fold (corrected, "
      f"seed {SEEDS[0]}):")
for i, names in enumerate(fold_features_seed42, 1):
    print(f"    Fold {i}: {names}")
same = len({frozenset(n) for n in fold_features_seed42}) == 1
print(f"  Same 4 features in every fold? {'YES' if same else 'NO'}"
      f"  -> {'leakage had little effect on selection' if same else 'selection varies between folds'}")

# ════════════════════════════════════════════════════════════════
# STEP 4 - SUMMARY TABLE
# ════════════════════════════════════════════════════════════════
df = pd.DataFrame(rows)
df.to_csv("results/noleak_summary.csv", index=False)

print("\n[STEP 4]  Summary (mean +/- std over 3 fold-split seeds)")
print("=" * 66)
print(f"  {'Method':<42} {'Accuracy':>16} {'AUC':>16}")
print("  " + "-" * 76)
print(f"  {'LightGBM baseline (23 feat)':<42} "
      f"{base_m['Accuracy']:>16.4f} {base_m['AUC']:>16.4f}")
for mode_name, _ in MODES:
    sub = df[df["Method"] == mode_name]
    print(f"  {mode_name:<42} "
          f"{sub['Accuracy'].mean():>9.4f} +/-{sub['Accuracy'].std():.3f} "
          f"{sub['AUC'].mean():>9.4f} +/-{sub['AUC'].std():.3f}")

# ════════════════════════════════════════════════════════════════
# STEP 5 - PLOTS (seed 42)
# ════════════════════════════════════════════════════════════════
old_name, new_name = MODES[0][0], MODES[1][0]
s = SEEDS[0]
old_pred, old_proba, old_cm = store[(old_name, s)]
new_pred, new_proba, new_cm = store[(new_name, s)]

panels = [
    ("LightGBM baseline\n(23 features)", base_cm, plt.cm.Blues),
    ("OLD stacking\n(leaky selection)", old_cm, plt.cm.Oranges),
    ("CORRECTED stacking\n(per-fold selection)", new_cm, plt.cm.Purples),
]
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.suptitle("Confusion Matrices - Old vs Corrected Stacking "
             "(202 test peptides)", fontsize=13, fontweight="bold", y=1.03)
for ax, (title, cm, cmap) in zip(axes, panels):
    ConfusionMatrixDisplay(cm, display_labels=["Non-Hemolytic", "Hemolytic"]
                           ).plot(ax=ax, cmap=cmap, colorbar=False,
                                  values_format="d",
                                  text_kw={"fontsize": 18, "fontweight": "bold"})
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Predicted\nAcc={(tp+tn)/cm.sum():.3f}  "
                  f"Recall={tp/(tp+fn):.3f}\n"
                  f"TP={tp} TN={tn} FP={fp} FN={fn}", fontsize=9)
plt.tight_layout()
plt.savefig("results/noleak_confusion_matrices.png", dpi=220,
            bbox_inches="tight", facecolor="white")
plt.close()

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.50)")
for name, proba, col, ls in [
        ("LightGBM baseline", p_lgb_te, "#2E75B6", "--"),
        ("OLD stacking (leaky)", old_proba, "#E67E22", "-."),
        ("CORRECTED stacking", new_proba, "#8E44AD", "-")]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, color=col, ls=ls, lw=2.4,
            label=f"{name} (AUC={roc_auc_score(y_test, proba):.4f})")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC - Old vs Corrected Stacking", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/noleak_roc.png", dpi=220,
            bbox_inches="tight", facecolor="white")
plt.close()

print("\n  Saved: results/noleak_summary.csv")
print("  Saved: results/noleak_confusion_matrices.png")
print("  Saved: results/noleak_roc.png")
print("\n  ALL DONE.")