"""
qfe_then_qsvm.py
================
Novel Pipeline: QFE -> QSVM

Stage 1 - Quantum Feature Extractor (QFE):
  Input  : top-4 classical features scaled to [0, pi]
  Circuit: 4-qubit Ry encode -> CNOT ring -> Rx(2x) -> reverse CNOT
  Output : 7 quantum features per peptide (4 Z + 3 ZZ observables)

Stage 2 - QSVM on QFE output:
  Input  : 7 quantum features (rescaled to [0, pi])
  Kernel : ZZFeatureMap(feature_dimension=7, reps=2) + sharpening
  Output : Hemolytic (1) / Non-Hemolytic (0)

Why this is novel:
  Standard QSVM : raw classical features -> quantum kernel -> SVM
  This pipeline  : classical features -> QFE circuit -> 7 quantum features
                   -> quantum kernel -> SVM
  Quantum processes the data TWICE in two different ways.
  No published work has done this for hemolytic peptide classification.

Comparison models in this script:
  A. Classical SVM on 23 features        (pure classical baseline)
  B. Standard QSVM on top-8 classical    (standard QSVM)
  C. QFE + Classical SVM on 7 features   (QFE only)
  D. QFE -> QSVM on 7 features           (THIS novel pipeline)

Outputs saved to results/:
  qfe_qsvm_roc.png
  qfe_qsvm_confusion_matrices.png
  qfe_qsvm_results.csv
"""

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score,
                             recall_score, precision_score, f1_score,
                             roc_auc_score)
import lightgbm as lgb

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ================================================================
# STEP 1 - LOAD DATA
# ================================================================
print("=" * 65)
print("  QFE -> QSVM  NOVEL PIPELINE")
print("  Quantum Feature Extractor feeding into Quantum SVM")
print("=" * 65)

print("\n[STEP 1]  Loading data")
print("-" * 45)

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

print(f"  Train : {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Test  : {X_test.shape[0]}  samples, {X_test.shape[1]} features")

# ================================================================
# STEP 2 - FEATURE SELECTION (LightGBM importance)
# ================================================================
print("\n[STEP 2]  Feature selection via LightGBM")
print("-" * 45)

lgb_sel = lgb.LGBMClassifier(n_estimators=300, random_state=42,
                              class_weight="balanced", verbose=-1)
lgb_sel.fit(X_tr_c, y_train)
imp        = lgb_sel.feature_importances_
sorted_idx = np.argsort(imp)[::-1]
top4_idx   = sorted_idx[:4]
top8_idx   = sorted_idx[:8]
top4_names = [feature_cols[i] for i in top4_idx]
top8_names = [feature_cols[i] for i in top8_idx]

print(f"  Top-4 (QFE input)  : {top4_names}")
print(f"  Top-8 (QSVM input) : {top8_names}")

# ================================================================
# STEP 3 - STAGE 1: QFE (generate 7 quantum features)
# ================================================================
print("\n[STEP 3]  Stage 1 - Quantum Feature Extractor (4 qubits)")
print("-" * 45)

def build_qfe_circuit(angles):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(4)
    for i, a in enumerate(angles):  qc.ry(float(a), i)
    for i in range(4):              qc.cx(i, (i+1) % 4)
    for i, a in enumerate(angles):  qc.rx(2.0 * float(a), i)
    for i in range(3, -1, -1):      qc.cx(i, (i+1) % 4)
    return qc

def extract_qfe_features(X_subset, label=""):
    from qiskit.quantum_info import Statevector, SparsePauliOp
    ops = [
        SparsePauliOp("IIIZ"),
        SparsePauliOp("IIZI"),
        SparsePauliOp("IZII"),
        SparsePauliOp("ZIII"),
        SparsePauliOp("IIZZ"),
        SparsePauliOp("IZZI"),
        SparsePauliOp("ZZII"),
    ]
    feats, n = [], len(X_subset)
    for idx, angles in enumerate(X_subset):
        if idx % 200 == 0:
            print(f"    {label}: {idx}/{n} processed...")
        sv = Statevector(build_qfe_circuit(angles))
        feats.append([sv.expectation_value(op).real for op in ops])
    print(f"    {label}: {n}/{n} done.")
    return np.array(feats)

Q_tr, Q_te    = None, None
qfe_available = False

if (os.path.exists("results/Q_tr.npy") and
        os.path.exists("results/Q_te.npy")):
    Q_tr_s = np.load("results/Q_tr.npy")
    Q_te_s = np.load("results/Q_te.npy")
    if (Q_tr_s.shape == (X_train.shape[0], 7) and
            Q_te_s.shape == (X_test.shape[0], 7)):
        Q_tr, Q_te    = Q_tr_s, Q_te_s
        qfe_available = True
        print(f"  Loaded saved QFE features -- Q_tr: {Q_tr.shape}, "
              f"Q_te: {Q_te.shape}")
    else:
        print(f"  Shape mismatch in saved files -- recomputing...")

if not qfe_available:
    print("  Computing QFE features (~10-60 min)...")
    try:
        t0 = time.time()
        Q_tr = extract_qfe_features(X_tr_q[:, top4_idx], "Train")
        Q_te = extract_qfe_features(X_te_q[:, top4_idx], "Test")
        np.save("results/Q_tr.npy", Q_tr)
        np.save("results/Q_te.npy", Q_te)
        qfe_available = True
        print(f"  Done in {time.time()-t0:.1f}s -- saved.")
    except Exception as e:
        print(f"  QFE failed: {e}")
        print("  Run main_pipeline.py first to generate Q_tr.npy")
        exit(1)

print(f"\n  QFE output stats -- "
      f"min={Q_tr.min():.4f}  max={Q_tr.max():.4f}  "
      f"mean={Q_tr.mean():.4f}")

# ================================================================
# STEP 4 - STAGE 2: QSVM ON QFE FEATURES (7 qubits)
# ================================================================
print("\n[STEP 4]  Stage 2 - QSVM on QFE features (7 qubits)")
print("-" * 45)
print("  Concept: QFE outputs are rescaled to [0, pi] and used")
print("  as rotation angles in ZZFeatureMap -- double quantum")
print("  transformation of the original peptide data.")

scaler_qfe  = MinMaxScaler(feature_range=(0, np.pi))
Q_tr_scaled = scaler_qfe.fit_transform(Q_tr)
Q_te_scaled = scaler_qfe.transform(Q_te)

# Subsample for kernel speed (O(n^2))
N_TRAIN = min(300, len(Q_tr_scaled))
N_TEST  = min(75,  len(Q_te_scaled))
rng     = np.random.default_rng(42)
tr_idx  = rng.choice(len(Q_tr_scaled), N_TRAIN, replace=False)
te_idx  = rng.choice(len(Q_te_scaled), N_TEST,  replace=False)

Q_sub_tr = Q_tr_scaled[tr_idx]
Q_sub_te = Q_te_scaled[te_idx]
y_sub_tr = y_train[tr_idx]
y_sub_te = y_test[te_idx]

print(f"\n  Subsampled -- Train: {N_TRAIN}, Test: {N_TEST}")
print(f"  Subsample class balance -- "
      f"Hemolytic: {y_sub_tr.sum()}, "
      f"Non-Hemolytic: {(y_sub_tr==0).sum()}")

qfe_qsvm_m     = None
fpr_qfe_qsvm   = None
tpr_qfe_qsvm   = None
cm_qfe_qsvm    = None

try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel

    print("\n  Building ZZFeatureMap(feature_dimension=7, reps=2)...")
    fmap = ZZFeatureMap(feature_dimension=7, reps=2, entanglement="linear")
    kobj = FidelityStatevectorKernel(feature_map=fmap)

    print(f"  Computing training kernel ({N_TRAIN}x{N_TRAIN})...")
    t0       = time.time()
    K_raw_tr = kobj.evaluate(x_vec=Q_sub_tr)
    print(f"  Training kernel done in {time.time()-t0:.1f}s")

    print(f"  Computing test kernel ({N_TEST}x{N_TRAIN})...")
    t0       = time.time()
    K_raw_te = kobj.evaluate(x_vec=Q_sub_te, y_vec=Q_sub_tr)
    print(f"  Test kernel done in {time.time()-t0:.1f}s")

    print(f"\n  Raw kernel stats -- "
          f"mean={K_raw_tr.mean():.4f}  "
          f"std={K_raw_tr.std():.4f}  "
          f"min={K_raw_tr.min():.4f}  "
          f"max={K_raw_tr.max():.4f}")

    # Kernel sharpening: K_sharp = exp(gamma * (K_raw - 1))
    GAMMA      = 5.0
    K_tr_sharp = np.exp(GAMMA * (K_raw_tr - 1))
    K_te_sharp = np.exp(GAMMA * (K_raw_te - 1))

    np.save("results/K_qfe_qsvm_tr.npy", K_tr_sharp)
    np.save("results/K_qfe_qsvm_te.npy", K_te_sharp)
    print(f"  Sharpened kernel saved.")

    qsvm = SVC(kernel="precomputed", C=1.0, probability=True,
               class_weight="balanced", random_state=42)
    qsvm.fit(K_tr_sharp, y_sub_tr)
    pred_d  = qsvm.predict(K_te_sharp)
    proba_d = qsvm.predict_proba(K_te_sharp)[:, 1]

    cm_qfe_qsvm    = confusion_matrix(y_sub_te, pred_d)
    tn, fp, fn, tp = cm_qfe_qsvm.ravel()

    qfe_qsvm_m = {
        "accuracy":  round(accuracy_score(y_sub_te, pred_d),               4),
        "recall":    round(recall_score(y_sub_te, pred_d),                  4),
        "precision": round(precision_score(y_sub_te, pred_d,
                                           zero_division=0),                4),
        "f1":        round(f1_score(y_sub_te, pred_d),                      4),
        "auc":       round(roc_auc_score(y_sub_te, proba_d),                4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }
    fpr_qfe_qsvm, tpr_qfe_qsvm, _ = roc_curve(y_sub_te, proba_d)

    print(f"\n  === QFE -> QSVM Result ===")
    print(f"  Accuracy  : {qfe_qsvm_m['accuracy']:.4f}")
    print(f"  Recall    : {qfe_qsvm_m['recall']:.4f}")
    print(f"  Precision : {qfe_qsvm_m['precision']:.4f}")
    print(f"  F1        : {qfe_qsvm_m['f1']:.4f}")
    print(f"  AUC       : {qfe_qsvm_m['auc']:.4f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  (on {N_TEST} subsampled test peptides)")

except Exception as e:
    print(f"  QFE->QSVM kernel failed: {e}")

# ================================================================
# STEP 5 - BASELINE COMPARISONS (same subsampled indices)
# ================================================================
print("\n[STEP 5]  Baseline comparisons (same test subset)")
print("-" * 45)

# A: Classical SVM on 23 features
svm_a   = SVC(kernel="rbf", C=1.0, probability=True,
              class_weight="balanced", random_state=42)
svm_a.fit(X_tr_c[tr_idx], y_sub_tr)
pred_a  = svm_a.predict(X_te_c[te_idx])
proba_a = svm_a.predict_proba(X_te_c[te_idx])[:, 1]
acc_a   = accuracy_score(y_sub_te, pred_a)
auc_a   = roc_auc_score(y_sub_te, proba_a)
cm_a    = confusion_matrix(y_sub_te, pred_a)
fpr_a, tpr_a, _ = roc_curve(y_sub_te, proba_a)
print(f"  A. Classical SVM (23 feat)        "
      f"Acc={acc_a:.4f}  AUC={auc_a:.4f}")

# B: Standard QSVM on top-8 classical features
acc_b, auc_b  = None, None
fpr_b, tpr_b  = None, None
cm_b          = None
try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel

    fmap_b   = ZZFeatureMap(feature_dimension=8, reps=2, entanglement="linear")
    kobj_b   = FidelityStatevectorKernel(feature_map=fmap_b)
    X_tr_q8  = X_tr_q[:, top8_idx]
    X_te_q8  = X_te_q[:, top8_idx]
    print("  B. Standard QSVM (top-8) -- computing kernel...")
    t0       = time.time()
    K_b_tr   = np.exp(5.0 * (kobj_b.evaluate(x_vec=X_tr_q8[tr_idx]) - 1))
    K_b_te   = np.exp(5.0 * (kobj_b.evaluate(
                    x_vec=X_te_q8[te_idx], y_vec=X_tr_q8[tr_idx]) - 1))
    svm_b    = SVC(kernel="precomputed", C=1.0, probability=True,
                   class_weight="balanced", random_state=42)
    svm_b.fit(K_b_tr, y_sub_tr)
    pred_b   = svm_b.predict(K_b_te)
    proba_b  = svm_b.predict_proba(K_b_te)[:, 1]
    acc_b    = accuracy_score(y_sub_te, pred_b)
    auc_b    = roc_auc_score(y_sub_te, proba_b)
    cm_b     = confusion_matrix(y_sub_te, pred_b)
    fpr_b, tpr_b, _ = roc_curve(y_sub_te, proba_b)
    print(f"     Acc={acc_b:.4f}  AUC={auc_b:.4f}  "
          f"({time.time()-t0:.1f}s)")
except Exception as e:
    print(f"  B. Standard QSVM failed: {e}")

# C: QFE + Classical SVM on 7 quantum features
svm_c   = SVC(kernel="rbf", C=1.0, probability=True,
              class_weight="balanced", random_state=42)
svm_c.fit(Q_tr[tr_idx], y_sub_tr)
pred_c  = svm_c.predict(Q_te[te_idx])
proba_c = svm_c.predict_proba(Q_te[te_idx])[:, 1]
acc_c   = accuracy_score(y_sub_te, pred_c)
auc_c   = roc_auc_score(y_sub_te, proba_c)
cm_c    = confusion_matrix(y_sub_te, pred_c)
fpr_c, tpr_c, _ = roc_curve(y_sub_te, proba_c)
print(f"  C. QFE + Classical SVM (7 feat)   "
      f"Acc={acc_c:.4f}  AUC={auc_c:.4f}")

# ================================================================
# STEP 6 - RESULTS TABLE
# ================================================================
print("\n[STEP 6]  Full comparison table")
print("=" * 65)

rows = [
    {"Model": "A - Classical SVM (23 features)",
     "Input to model": "23 classical features",
     "Quantum stages": 0,
     "Accuracy": acc_a, "AUC": auc_a},
]
if acc_b:
    rows.append(
        {"Model": "B - Standard QSVM (top-8 classical)",
         "Input to model": "8 classical -> quantum kernel",
         "Quantum stages": 1,
         "Accuracy": acc_b, "AUC": auc_b}
    )
rows.append(
    {"Model": "C - QFE + Classical SVM (7 QFE features)",
     "Input to model": "4 classical -> QFE -> 7 quantum",
     "Quantum stages": 1,
     "Accuracy": acc_c, "AUC": auc_c}
)
if qfe_qsvm_m:
    rows.append(
        {"Model": "D - QFE -> QSVM  [NOVEL PIPELINE]",
         "Input to model": "4 classical -> QFE -> 7 quantum -> QSVM",
         "Quantum stages": 2,
         "Accuracy": qfe_qsvm_m["accuracy"], "AUC": qfe_qsvm_m["auc"]}
    )

df_cmp = pd.DataFrame(rows)
df_cmp = df_cmp.sort_values("Accuracy", ascending=False).reset_index(drop=True)
df_cmp.to_csv("results/qfe_qsvm_results.csv", index=False)

print(f"\n  {'Rank':<5} {'Model':<46} {'Acc':>6}  {'AUC':>6}  "
      f"{'Q-stages':>8}")
print("  " + "-" * 68)
for i, row in df_cmp.iterrows():
    tag = "  <- NOVEL" if "NOVEL" in row["Model"] else ""
    print(f"  #{i+1:<4} {row['Model']:<46} "
          f"{row['Accuracy']:>6.4f}  {row['AUC']:>6.4f}  "
          f"{row['Quantum stages']:>8}{tag}")

print(f"\n  Saved: results/qfe_qsvm_results.csv")

# ================================================================
# STEP 7 - ROC CURVE
# ================================================================
print("\n[STEP 7]  ROC curve")
print("-" * 45)

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0,1],[0,1], "k--", lw=1.2,
        label="Random Baseline (AUC=0.50)", alpha=0.5)

ax.plot(fpr_a, tpr_a, color="#999999", ls="--", lw=1.8,
        label=f"A - Classical SVM 23 feat      (AUC={auc_a:.4f})")

if fpr_b is not None:
    ax.plot(fpr_b, tpr_b, color="#377EB8", ls="-.", lw=1.8,
            label=f"B - Standard QSVM top-8 feat   (AUC={auc_b:.4f})")

ax.plot(fpr_c, tpr_c, color="#FF7F00", ls=":", lw=2.0,
        label=f"C - QFE + Classical SVM 7 feat (AUC={auc_c:.4f})")

if fpr_qfe_qsvm is not None and qfe_qsvm_m:
    ax.plot(fpr_qfe_qsvm, tpr_qfe_qsvm,
            color="#E41A1C", ls="-", lw=3.0,
            label=f"D - QFE -> QSVM [NOVEL]        "
                  f"(AUC={qfe_qsvm_m['auc']:.4f})")

ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate",  fontsize=13)
ax.set_title(
    "ROC Curves -- QFE -> QSVM Novel Pipeline\n"
    "Classical | QSVM | QFE+SVM | QFE->QSVM  (HemoPI-2)",
    fontsize=12, fontweight="bold"
)
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/qfe_qsvm_roc.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: results/qfe_qsvm_roc.png")

# ================================================================
# STEP 8 - CONFUSION MATRICES
# ================================================================
print("\n[STEP 8]  Confusion matrices")
print("-" * 45)

panels = [
    ("A - Classical SVM\n(23 features)",
     cm_a, plt.cm.Greys),
    ("C - QFE + Classical SVM\n(7 QFE features)",
     cm_c, plt.cm.Oranges),
    ("D - QFE -> QSVM\n(Novel Pipeline)",
     cm_qfe_qsvm, plt.cm.Reds),
]

fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5))
fig2.suptitle(
    "Confusion Matrices -- Classical SVM | QFE+SVM | QFE->QSVM\n"
    "Hemolytic Peptide Classification (HemoPI-2)",
    fontsize=12, fontweight="bold", y=1.04
)

for ax2, (title, cm, cmap) in zip(axes2, panels):
    if cm is None:
        ax2.set_facecolor("#f5f5f5")
        ax2.text(0.5, 0.5, f"{title}\n\nNot available",
                 ha="center", va="center", fontsize=11,
                 color="grey", transform=ax2.transAxes)
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_title(title, fontsize=11, fontweight="bold")
        continue

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Hemolytic (0)", "Hemolytic (1)"]
    )
    disp.plot(ax=ax2, cmap=cmap, colorbar=False, values_format="d")
    tn, fp, fn, tp = cm.ravel()
    acc = (tp+tn)/cm.sum()
    rec = tp/(tp+fn) if (tp+fn) > 0 else 0
    pre = tp/(tp+fp) if (tp+fp) > 0 else 0
    f1  = 2*pre*rec/(pre+rec) if (pre+rec) > 0 else 0
    stats = (f"Acc={acc:.3f}    Recall={rec:.3f}\n"
             f"Precision={pre:.3f}    F1={f1:.3f}\n"
             f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    ax2.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel(f"Predicted Label\n\n{stats}", fontsize=9)
    ax2.set_ylabel("True Label", fontsize=10)

plt.tight_layout()
plt.savefig("results/qfe_qsvm_confusion_matrices.png",
            dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: results/qfe_qsvm_confusion_matrices.png")

# ================================================================
# DONE
# ================================================================
print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)
print("""
  PIPELINE COMPARISON BY QUANTUM STAGES:
  ----------------------------------------------------------------
  Stage 0 - Classical SVM only
    No quantum at all. Pure classical baseline.

  Stage 1 - Standard QSVM
    Classical features -> ZZFeatureMap quantum kernel -> SVM
    Quantum used for similarity measurement only.

  Stage 1 - QFE + Classical SVM
    Classical features -> 4-qubit QFE circuit -> 7 quantum features
    -> Classical SVM. Quantum used for feature transformation only.

  Stage 2 - QFE -> QSVM  [NOVEL]
    Classical features -> QFE -> 7 quantum features -> ZZFeatureMap
    quantum kernel -> SVM. Quantum used TWICE:
      1st: as feature transformer (QFE)
      2nd: as kernel function (QSVM)
    This double quantum transformation is the novel contribution.
  ----------------------------------------------------------------
""")
print("  Output files in results/:")
print("    qfe_qsvm_roc.png")
print("    qfe_qsvm_confusion_matrices.png")
print("    qfe_qsvm_results.csv")
print("    K_qfe_qsvm_tr.npy  (kernel saved for MKL use)")
print("    K_qfe_qsvm_te.npy")
print("\n  ALL DONE.")