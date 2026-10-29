"""
main_pipeline.py
================
Quantum-Classical Hybrid ML for Hemolytic Peptide Classification
Dataset : HemoPI-2

This single script replaces:
  - run_all_models.py
  - quantum_feature_extractor.py

What this script does (in order):
  SECTION 1  — Load data
  SECTION 2  — Train all 4 classical models (SVM, RF, XGBoost, LightGBM)
  SECTION 3  — Run QSVM with kernel sharpening
  SECTION 4  — Run Quantum Feature Extractor (QFE)
  SECTION 5  — Compare all results in one clean table
  SECTION 6  — Predict hemolytic activity for a user-entered peptide sequence

Feature sets used:
  23 features  = 20 AAC + 3 PCP  (classical baseline)
  30 features  = 23 + 7 quantum  (QFE hybrid)
"""

# ════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════
import os, time, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")
os.makedirs("results",     exist_ok=True)
os.makedirs("evaluation",  exist_ok=True)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             classification_report)
import xgboost as xgb
import lightgbm as lgb

# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

HYDRO = {
    'A': 1.8,  'C': 2.5,  'D':-3.5,  'E':-3.5,  'F': 2.8,
    'G':-0.4,  'H':-3.2,  'I': 4.5,  'K':-3.9,  'L': 3.8,
    'M': 1.9,  'N':-3.5,  'P':-1.6,  'Q':-3.5,  'R':-4.5,
    'S':-0.8,  'T':-0.7,  'V': 4.2,  'W':-0.9,  'Y':-1.3
}

def extract_features_from_sequence(sequence):
    """
    Extracts 23 features from a raw peptide sequence string.
    Used for predicting a user-entered sequence at the end.

    Features:
      20 AAC  — frequency of each amino acid
       3 PCP  — net charge, pos/neg ratio, GRAVY hydropathy score
    """
    seq    = sequence.upper().strip()
    length = len(seq)
    if length == 0:
        raise ValueError("Sequence cannot be empty.")

    # 20 Amino Acid Composition features
    aac = [seq.count(aa) / length for aa in AMINO_ACIDS]

    # 3 Physicochemical Property features
    pos       = seq.count("K") + seq.count("R") + seq.count("H")
    neg       = seq.count("D") + seq.count("E")
    net_charge        = pos - neg
    pos_neg_ratio     = pos / (neg + 1e-6)
    gravy             = sum(HYDRO.get(aa, 0) for aa in seq) / length

    return np.array(aac + [net_charge, pos_neg_ratio, gravy])

def print_metrics(name, metrics):
    """Prints a single model's metrics in a clean one-line format."""
    print(f"  {name:<38}  "
          f"Acc={metrics['accuracy']:.4f}  "
          f"Rec={metrics['recall']:.4f}  "
          f"AUC={metrics.get('auc', float('nan')):.4f}  "
          f"TP={metrics['TP']}  TN={metrics['TN']}  "
          f"FP={metrics['FP']}  FN={metrics['FN']}")

def compute_metrics(y_true, y_pred, y_proba=None):
    """Returns a dict of all evaluation metrics."""
    cm            = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else float("nan")
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred),                    4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0),  4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0),      4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0),          4),
        "auc":       round(auc,                                                 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }

# ════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  QUANTUM-CLASSICAL HYBRID ML — HemoPI-2")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65)

print("\n[SECTION 1]  Loading data")
print("-" * 45)

train_df     = pd.read_csv("data/processed/train_features.csv")
test_df      = pd.read_csv("data/processed/test_features.csv")
feature_cols = [c for c in train_df.columns if c != "label"]

X_train_raw  = train_df[feature_cols].values
y_train      = train_df["label"].values
X_test_raw   = test_df[feature_cols].values
y_test       = test_df["label"].values

# Scale to [0,1] for classical models
scaler_c     = MinMaxScaler()
X_tr_c       = scaler_c.fit_transform(X_train_raw)
X_te_c       = scaler_c.transform(X_test_raw)

# Scale to [0, π] for quantum angle encoding
scaler_q     = MinMaxScaler(feature_range=(0, np.pi))
X_tr_q       = scaler_q.fit_transform(X_train_raw)
X_te_q       = scaler_q.transform(X_test_raw)

pos_weight   = (y_train == 0).sum() / (y_train == 1).sum()   # for XGBoost

print(f"  Train : {X_train_raw.shape[0]} samples, {X_train_raw.shape[1]} features")
print(f"  Test  : {X_test_raw.shape[0]} samples,  {X_test_raw.shape[1]} features")
print(f"  Class ratio (neg/pos) for XGBoost weight: {pos_weight:.2f}")

# ════════════════════════════════════════════════════════════════════
# SECTION 2 — CLASSICAL MODELS (all use 23 features)
# ════════════════════════════════════════════════════════════════════
print("\n[SECTION 2]  Classical Models (23 features each)")
print("-" * 65)
print(f"  {'Model':<38}  {'Acc':>6}  {'Rec':>6}  {'AUC':>6}  "
      f"{'TP':>4}  {'TN':>4}  {'FP':>4}  {'FN':>4}")
print("  " + "-" * 61)

all_results    = []   # collects every model result for final table
trained_models = {}   # stores fitted model objects for prediction

# ── SVM ──────────────────────────────────────────────────────────
svm_clf = SVC(kernel="rbf", C=1.0, probability=True,
              class_weight="balanced", random_state=42)
svm_clf.fit(X_tr_c, y_train)
svm_pred  = svm_clf.predict(X_te_c)
svm_proba = svm_clf.predict_proba(X_te_c)[:, 1]
svm_m     = compute_metrics(y_test, svm_pred, svm_proba)
svm_m["model"] = "SVM (RBF)"
print_metrics("SVM (RBF)", svm_m)
all_results.append(svm_m)
trained_models["SVM (RBF)"] = svm_clf

# ── Random Forest ─────────────────────────────────────────────────
rf_clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=42)
rf_clf.fit(X_tr_c, y_train)
rf_pred  = rf_clf.predict(X_te_c)
rf_proba = rf_clf.predict_proba(X_te_c)[:, 1]
rf_m     = compute_metrics(y_test, rf_pred, rf_proba)
rf_m["model"] = "Random Forest"
print_metrics("Random Forest", rf_m)
all_results.append(rf_m)
trained_models["Random Forest"] = rf_clf

# ── XGBoost ───────────────────────────────────────────────────────
xgb_clf = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    random_state=42, verbosity=0, eval_metric="logloss"
)
xgb_clf.fit(X_tr_c, y_train)
xgb_pred  = xgb_clf.predict(X_te_c)
xgb_proba = xgb_clf.predict_proba(X_te_c)[:, 1]
xgb_m     = compute_metrics(y_test, xgb_pred, xgb_proba)
xgb_m["model"] = "XGBoost"
print_metrics("XGBoost", xgb_m)
all_results.append(xgb_m)
trained_models["XGBoost"] = xgb_clf

# ── LightGBM ──────────────────────────────────────────────────────
lgb_clf = lgb.LGBMClassifier(n_estimators=300, max_depth=4,
                              class_weight="balanced",
                              random_state=42, verbose=-1)
lgb_clf.fit(X_tr_c, y_train)
lgb_pred  = lgb_clf.predict(X_te_c)
lgb_proba = lgb_clf.predict_proba(X_te_c)[:, 1]
lgb_m     = compute_metrics(y_test, lgb_pred, lgb_proba)
lgb_m["model"] = "LightGBM"
print_metrics("LightGBM", lgb_m)
all_results.append(lgb_m)
trained_models["LightGBM"] = lgb_clf

# ── Feature importance from LightGBM (used for QSVM + QFE) ───────
importances = lgb_clf.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]
top4_idx    = sorted_idx[:4]
top8_idx    = sorted_idx[:8]
top4_names  = [feature_cols[i] for i in top4_idx]
top8_names  = [feature_cols[i] for i in top8_idx]

print(f"\n  Top-4 features (used for QFE)  : {top4_names}")
print(f"  Top-8 features (used for QSVM) : {top8_names}")

# ════════════════════════════════════════════════════════════════════
# SECTION 3 — QSVM WITH KERNEL SHARPENING
# ════════════════════════════════════════════════════════════════════
print("\n[SECTION 3]  QSVM with Kernel Sharpening")
print("-" * 65)

qsvm_m = None

try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel

    # Use top-8 features scaled to [0, π]
    X_tr_q8 = X_tr_q[:, top8_idx]
    X_te_q8 = X_te_q[:, top8_idx]

    # Subsample for speed (kernel is O(n^2))
    N_TRAIN = min(300, len(X_tr_q8))
    N_TEST  = min(75,  len(X_te_q8))
    rng     = np.random.default_rng(42)
    tr_idx  = rng.choice(len(X_tr_q8), N_TRAIN, replace=False)
    te_idx  = rng.choice(len(X_te_q8), N_TEST,  replace=False)

    X_sub_tr  = X_tr_q8[tr_idx]
    X_sub_te  = X_te_q8[te_idx]
    y_sub_tr  = y_train[tr_idx]
    y_sub_te  = y_test[te_idx]

    print(f"  Subsampled — Train: {N_TRAIN}, Test: {N_TEST}")

    # Build quantum kernel
    fmap   = ZZFeatureMap(feature_dimension=8, reps=2, entanglement="linear")
    kobj   = FidelityStatevectorKernel(feature_map=fmap)

    print("  Computing training kernel matrix…")
    t0     = time.time()
    K_raw_tr = kobj.evaluate(x_vec=X_sub_tr)
    K_raw_te = kobj.evaluate(x_vec=X_sub_te, y_vec=X_sub_tr)
    print(f"  Kernel computed in {time.time()-t0:.1f}s")
    print(f"  Raw kernel stats: mean={K_raw_tr.mean():.4f}, "
          f"std={K_raw_tr.std():.4f}")

    # Kernel sharpening: K_sharp = exp(gamma * (K_raw - 1))
    GAMMA_SHARP = 5.0
    K_tr_sharp  = np.exp(GAMMA_SHARP * (K_raw_tr - 1))
    K_te_sharp  = np.exp(GAMMA_SHARP * (K_raw_te - 1))

    # Save for combine_kernels.py
    np.save("results/K_tr_sharp.npy", K_tr_sharp)
    np.save("results/K_te_sharp.npy", K_te_sharp)

    # Train SVM on sharpened kernel
    qsvm = SVC(kernel="precomputed", C=1.0, probability=True,
               class_weight="balanced", random_state=42)
    qsvm.fit(K_tr_sharp, y_sub_tr)
    qsvm_pred  = qsvm.predict(K_te_sharp)
    qsvm_proba = qsvm.predict_proba(K_te_sharp)[:, 1]

    qsvm_m             = compute_metrics(y_sub_te, qsvm_pred, qsvm_proba)
    qsvm_m["model"]    = "QSVM + Sharpening"
    print_metrics("QSVM + Sharpening", qsvm_m)
    print(f"  (Note: evaluated on {N_TEST} test samples, not full 202)")
    all_results.append(qsvm_m)
    trained_models["QSVM + Sharpening"] = qsvm

except Exception as e:
    print(f"  QSVM failed: {e}")
    print("  Skipping QSVM section.")

# ════════════════════════════════════════════════════════════════════
# SECTION 4 — QUANTUM FEATURE EXTRACTOR (QFE) + CLASSICAL CLASSIFIER
# ════════════════════════════════════════════════════════════════════
print("\n[SECTION 4]  Quantum Feature Extractor (QFE)")
print("-" * 65)

"""
HOW QFE WORKS:
  Step 1: Take top-4 features, scale to [0, π]
  Step 2: Encode each feature as a rotation angle in a 4-qubit circuit
          Ry(x_i) → CNOT ring → Rx(2*x_i) → reverse CNOT
  Step 3: Measure 7 observables per sample:
          4 single-qubit Z  → individual qubit states
          3 two-qubit ZZ    → entanglement correlations between pairs
  Step 4: These 7 numbers are NEW FEATURES for the sample
  Step 5: Stack 23 classical + 7 quantum = 30 features
  Step 6: Train XGBoost/SVM on 30 features

Why 7 features?
  Each ZZ measurement captures how TWO features interact in
  quantum space — this is NOT the same as x_i * x_j multiplication.
  It captures quantum entanglement correlation.
"""

def build_qfe_circuit(angles):
    """
    4-qubit QFE circuit.
    Input  : 4 angles (one per qubit)
    Layers : Ry encode → CNOT ring → Rx(2x) → reverse CNOT
    """
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(4)
    for i, a in enumerate(angles):  qc.ry(float(a), i)   # encode
    for i in range(4):              qc.cx(i, (i+1) % 4)  # entangle
    for i, a in enumerate(angles):  qc.rx(2.0*float(a), i) # re-encode
    for i in range(3, -1, -1):      qc.cx(i, (i+1) % 4)  # stabilise
    return qc

def extract_qfe_features(X_subset, label=""):
    """
    Extracts 7 quantum features per sample using statevector simulation.
    Observables:
      Z(q0), Z(q1), Z(q2), Z(q3)     — 4 individual qubit states
      ZZ(q0,q1), ZZ(q1,q2), ZZ(q2,q3) — 3 pairwise correlations
    Each returns a value in [-1, +1].
    """
    from qiskit.quantum_info import Statevector, SparsePauliOp

    ops = [
        SparsePauliOp("IIIZ"),  # Z  on qubit 0
        SparsePauliOp("IIZI"),  # Z  on qubit 1
        SparsePauliOp("IZII"),  # Z  on qubit 2
        SparsePauliOp("ZIII"),  # Z  on qubit 3
        SparsePauliOp("IIZZ"),  # ZZ on qubits 0,1
        SparsePauliOp("IZZI"),  # ZZ on qubits 1,2
        SparsePauliOp("ZZII"),  # ZZ on qubits 2,3
    ]

    feats, n = [], len(X_subset)
    for idx, angles in enumerate(X_subset):
        if idx % 200 == 0:
            print(f"    {label}: {idx}/{n} samples processed…")
        sv = Statevector(build_qfe_circuit(angles))
        feats.append([sv.expectation_value(op).real for op in ops])
    print(f"    {label}: {n}/{n} samples — done.")
    return np.array(feats)

# Use top-4 features (scaled to [0, π]) for quantum encoding
X_tr_top4 = X_tr_q[:, top4_idx]
X_te_top4 = X_te_q[:, top4_idx]

Q_tr, Q_te    = None, None
qfe_available = False

# Load saved features if available (avoids ~10-45 min recompute)
if (os.path.exists("results/Q_tr.npy") and
        os.path.exists("results/Q_te.npy")):
    Q_tr = np.load("results/Q_tr.npy")
    Q_te = np.load("results/Q_te.npy")

    if (Q_tr.shape == (X_train_raw.shape[0], 7) and
            Q_te.shape  == (X_test_raw.shape[0],  7)):
        qfe_available = True
        print(f"  Loaded saved QFE features — Q_tr: {Q_tr.shape}, "
              f"Q_te: {Q_te.shape}")
    else:
        print(f"  Saved QFE shapes don't match dataset — recomputing…")
        Q_tr, Q_te = None, None

if not qfe_available:
    print("  Computing QFE features from scratch…")
    try:
        t0   = time.time()
        Q_tr = extract_qfe_features(X_tr_top4, "Train")
        Q_te = extract_qfe_features(X_te_top4, "Test")
        np.save("results/Q_tr.npy", Q_tr)
        np.save("results/Q_te.npy", Q_te)
        qfe_available = True
        print(f"  QFE features computed in {time.time()-t0:.1f}s and saved.")
    except Exception as e:
        print(f"  QFE computation failed: {e}")

if qfe_available:
    # ── Feature sets ─────────────────────────────────────────────
    # A: 23 classical features only          (already done in Section 2)
    # B: 7 quantum features only
    # C: 30 = 23 classical + 7 quantum       (the hybrid model)

    X_tr_B = Q_tr
    X_te_B = Q_te
    X_tr_C = np.hstack([X_tr_c, Q_tr])   # 30 features
    X_te_C = np.hstack([X_te_c, Q_te])

    print(f"\n  Feature sets ready:")
    print(f"    A (classical only)    : {X_tr_c.shape[1]} features")
    print(f"    B (quantum only)      : {X_tr_B.shape[1]} features")
    print(f"    C (classical+quantum) : {X_tr_C.shape[1]} features")

    print(f"\n  {'Setup':<38}  {'Acc':>6}  {'Rec':>6}  {'AUC':>6}  "
          f"{'TP':>4}  {'TN':>4}  {'FP':>4}  {'FN':>4}")
    print("  " + "-" * 61)

    # B — Quantum features only → XGBoost
    m = xgb.XGBClassifier(n_estimators=300, random_state=42,
                           verbosity=0, eval_metric="logloss",
                           scale_pos_weight=pos_weight)
    m.fit(X_tr_B, y_train)
    p = m.predict(X_te_B)
    pr = m.predict_proba(X_te_B)[:, 1]
    qonly_m = compute_metrics(y_test, p, pr)
    qonly_m["model"] = "QFE only — XGBoost (7 feat)"
    print_metrics("QFE only — XGBoost (7 feat)", qonly_m)
    all_results.append(qonly_m)

    # C — Classical + Quantum → SVM
    svm_30 = SVC(kernel="rbf", probability=True,
                 class_weight="balanced", random_state=42)
    svm_30.fit(X_tr_C, y_train)
    p  = svm_30.predict(X_te_C)
    pr = svm_30.predict_proba(X_te_C)[:, 1]
    svm30_m = compute_metrics(y_test, p, pr)
    svm30_m["model"] = "QFE + SVM (30 feat)"
    print_metrics("QFE + SVM (30 feat)", svm30_m)
    all_results.append(svm30_m)
    trained_models["QFE + SVM (30 feat)"] = svm_30

    # C — Classical + Quantum → XGBoost
    xgb_30 = xgb.XGBClassifier(n_estimators=300, random_state=42,
                                verbosity=0, eval_metric="logloss",
                                scale_pos_weight=pos_weight)
    xgb_30.fit(X_tr_C, y_train)
    p  = xgb_30.predict(X_te_C)
    pr = xgb_30.predict_proba(X_te_C)[:, 1]
    xgb30_m = compute_metrics(y_test, p, pr)
    xgb30_m["model"] = "QFE + XGBoost (30 feat)"
    print_metrics("QFE + XGBoost (30 feat)", xgb30_m)
    all_results.append(xgb30_m)
    trained_models["QFE + XGBoost (30 feat)"] = xgb_30

    # C — Classical + Quantum → LightGBM
    lgb_30 = lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=42, verbose=-1)
    lgb_30.fit(X_tr_C, y_train)
    p  = lgb_30.predict(X_te_C)
    pr = lgb_30.predict_proba(X_te_C)[:, 1]
    lgb30_m = compute_metrics(y_test, p, pr)
    lgb30_m["model"] = "QFE + LightGBM (30 feat)"
    print_metrics("QFE + LightGBM (30 feat)", lgb30_m)
    all_results.append(lgb30_m)
    trained_models["QFE + LightGBM (30 feat)"] = lgb_30

else:
    print("  QFE unavailable — skipping hybrid models.")

# ════════════════════════════════════════════════════════════════════
# SECTION 5 — FULL COMPARISON TABLE
# ════════════════════════════════════════════════════════════════════
print("\n[SECTION 5]  Complete Results Comparison")
print("=" * 65)

df_results = pd.DataFrame(all_results)

# Sort by accuracy descending
display_cols = ["model", "accuracy", "recall", "precision", "f1", "auc",
                "TP", "TN", "FP", "FN"]
df_results = df_results[[c for c in display_cols if c in df_results.columns]]
df_results = df_results.sort_values("accuracy", ascending=False).reset_index(drop=True)

print("\n  Rank  Model                                 "
      "Acc     Rec     AUC")
print("  " + "-" * 61)
for i, row in df_results.iterrows():
    marker = "  ← BEST" if i == 0 else ""
    print(f"  #{i+1:<4} {row['model']:<38} "
          f"{row['accuracy']:.4f}  {row['recall']:.4f}  "
          f"{row['auc']:.4f}{marker}")

# Save full table
df_results.to_csv("evaluation/comparison_results.csv", index=False)
print(f"\n  Full table saved: evaluation/comparison_results.csv")

# Identify best model for prediction
best_row        = df_results.iloc[0]
best_model_name = best_row["model"]
print(f"\n  Best model  : {best_model_name}")
print(f"  Accuracy    : {best_row['accuracy']:.4f}")
print(f"  AUC         : {best_row['auc']:.4f}")
print(f"  Recall      : {best_row['recall']:.4f}")

# ════════════════════════════════════════════════════════════════════
# SECTION 6 — PREDICT FOR USER INPUT SEQUENCE
# ════════════════════════════════════════════════════════════════════
print("\n[SECTION 6]  Predict Hemolytic Activity")
print("-" * 45)

sequence = input("\n  Enter peptide sequence: ").strip()

if not sequence:
    print("  No sequence entered — skipping prediction.")
else:
    try:
        raw_features = extract_features_from_sequence(sequence)

        # Scale to [0,1] for classical models
        features_scaled = scaler_c.transform(raw_features.reshape(1, -1))

        if best_model_name in trained_models:
            clf  = trained_models[best_model_name]

            # If best model uses 30 features (QFE hybrid), add quantum features
            if "30 feat" in best_model_name and qfe_available:
                q_angles = scaler_q.transform(raw_features.reshape(1, -1))[:, top4_idx]
                q_feats  = extract_qfe_features(q_angles, "Predict")
                features_scaled = np.hstack([features_scaled, q_feats])

            pred = clf.predict(features_scaled)[0]

            # Probability if available
            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(features_scaled)[0][1]
                prob_str = f" (hemolytic probability: {prob:.2%})"
            else:
                prob_str = ""

            print(f"\n  Sequence : {sequence}")
            print(f"  Model    : {best_model_name}")
            print(f"  Result   : {'HEMOLYTIC' if pred == 1 else 'NON-HEMOLYTIC'}{prob_str}")

            if pred == 1:
                print("\n  WARNING: This peptide is predicted to be HEMOLYTIC.")
                print("  It may damage red blood cells. Not safe for drug use.")
            else:
                print("\n  This peptide is predicted to be NON-HEMOLYTIC.")
                print("  It appears safe with respect to red blood cell damage.")
        else:
            print(f"  Best model '{best_model_name}' not available for prediction.")

    except Exception as e:
        print(f"  Prediction error: {e}")

# ════════════════════════════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"  ALL SECTIONS COMPLETE")
print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65)
print("\n  Output files:")
print("    results/Q_tr.npy                  — quantum features (train)")
print("    results/Q_te.npy                  — quantum features (test)")
print("    results/K_tr_sharp.npy            — QSVM kernel (train)")
print("    results/K_te_sharp.npy            — QSVM kernel (test)")
print("    evaluation/comparison_results.csv — full metrics table")