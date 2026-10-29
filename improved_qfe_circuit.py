"""
improved_qfe_circuit.py
=======================
Compares two QFE circuit designs:

Circuit V1 (current):
  RY → CNOT ring → RX(2x) → Reverse CNOT → Measure
  - Basic encoding + entanglement
  - Limited expressivity

Circuit V2 (improved):
  RY → CNOT ring → RX(2x) → CNOT ring → RY → Measure
  - Extra RY layer = deeper encoding
  - Extra CNOT layer = richer entanglement
  - Higher circuit expressivity

Both circuits are tested with:
  - Classical SVM on 30 features (23 + 7 quantum)
  - XGBoost on 30 features
  - Stacking ensemble

Results saved to:
  results/improved_qfe_comparison.csv
  results/improved_qfe_roc.png
  results/improved_qfe_confusion.png
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
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  IMPROVED QFE CIRCUIT — V1 vs V2 COMPARISON")
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

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print(f"  Train : {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Test  : {X_test.shape[0]}  samples")

# ════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE SELECTION
# ════════════════════════════════════════════════════════════════
print("\n[STEP 2]  Feature selection (LightGBM importance)")
print("-" * 50)

lgb_sel = lgb.LGBMClassifier(n_estimators=300, random_state=42,
                              class_weight="balanced", verbose=-1)
lgb_sel.fit(X_tr_c, y_train)
imp      = lgb_sel.feature_importances_
top4_idx = np.argsort(imp)[::-1][:4]
top4     = [feature_cols[i] for i in top4_idx]
print(f"  Top-4 features for QFE: {top4}")

X_tr_top4 = X_tr_q[:, top4_idx]
X_te_top4 = X_te_q[:, top4_idx]

# ════════════════════════════════════════════════════════════════
# STEP 3 — DEFINE BOTH CIRCUITS
# ════════════════════════════════════════════════════════════════
print("\n[STEP 3]  Quantum circuit definitions")
print("-" * 50)

def build_circuit_v1(angles):
    """
    CIRCUIT V1 — Current design
    ════════════════════════════
    Layer 1: RY(xi)       — encode feature as rotation angle
    Layer 2: CNOT ring    — entangle q0→q1→q2→q3→q0
    Layer 3: RX(2*xi)     — second encoding with doubled angle
    Layer 4: Reverse CNOT — q3→q2→q1→q0 stabilise state
    """
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(4)
    # Layer 1: RY encoding
    for i, a in enumerate(angles):
        qc.ry(float(a), i)
    # Layer 2: CNOT ring
    for i in range(4):
        qc.cx(i, (i + 1) % 4)
    # Layer 3: RX with doubled angle
    for i, a in enumerate(angles):
        qc.rx(2.0 * float(a), i)
    # Layer 4: Reverse CNOT
    for i in range(3, -1, -1):
        qc.cx(i, (i + 1) % 4)
    return qc

def build_circuit_v2(angles):
    """
    CIRCUIT V2 — Improved design
    ════════════════════════════
    Layer 1: RY(xi)        — encode feature as rotation angle
    Layer 2: CNOT ring     — first entanglement q0→q1→q2→q3→q0
    Layer 3: RX(2*xi)      — non-linear transformation
    Layer 4: CNOT ring     — second entanglement (forward again)
    Layer 5: RY(xi)        — third encoding layer (deeper expressivity)

    WHY THIS IS BETTER:
    - Three encoding layers vs two in V1
    - Two forward CNOT layers create richer entanglement patterns
    - Extra RY after entanglement lets circuit refine the encoding
    - Higher circuit depth = more expressive quantum feature space
    """
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(4)
    # Layer 1: RY encoding
    for i, a in enumerate(angles):
        qc.ry(float(a), i)
    # Layer 2: CNOT ring (first entanglement)
    for i in range(4):
        qc.cx(i, (i + 1) % 4)
    # Layer 3: RX non-linearity
    for i, a in enumerate(angles):
        qc.rx(2.0 * float(a), i)
    # Layer 4: CNOT ring again (second entanglement — KEY UPGRADE)
    for i in range(4):
        qc.cx(i, (i + 1) % 4)
    # Layer 5: RY third encoding (KEY UPGRADE — deeper feature encoding)
    for i, a in enumerate(angles):
        qc.ry(float(a), i)
    return qc

# Print circuit info
from qiskit import QuantumCircuit
sample = [0.5, 1.2, 0.8, 1.5]
qc_v1  = build_circuit_v1(sample)
qc_v2  = build_circuit_v2(sample)
print(f"  Circuit V1 — depth: {qc_v1.depth()}, gates: {qc_v1.size()}")
print(f"  Circuit V2 — depth: {qc_v2.depth()}, gates: {qc_v2.size()}")
print(f"  Depth increase: +{qc_v2.depth() - qc_v1.depth()} layers")

# ════════════════════════════════════════════════════════════════
# STEP 4 — EXTRACT FEATURES FROM BOTH CIRCUITS
# ════════════════════════════════════════════════════════════════
print("\n[STEP 4]  Extracting quantum features")
print("-" * 50)

def extract_features(X_subset, circuit_fn, label=""):
    """
    Measures 7 observables per peptide:
      4 single-qubit Z  : Z(q0), Z(q1), Z(q2), Z(q3)
      3 two-qubit ZZ    : ZZ(q0,q1), ZZ(q1,q2), ZZ(q2,q3)
    Values are quantum expectation values in [-1, +1]
    """
    from qiskit.quantum_info import Statevector, SparsePauliOp
    ops = [
        SparsePauliOp("IIIZ"),   # Z  qubit 0
        SparsePauliOp("IIZI"),   # Z  qubit 1
        SparsePauliOp("IZII"),   # Z  qubit 2
        SparsePauliOp("ZIII"),   # Z  qubit 3
        SparsePauliOp("IIZZ"),   # ZZ qubits 0-1
        SparsePauliOp("IZZI"),   # ZZ qubits 1-2
        SparsePauliOp("ZZII"),   # ZZ qubits 2-3
    ]
    feats, n = [], len(X_subset)
    for idx, angles in enumerate(X_subset):
        if idx % 200 == 0:
            print(f"    {label}: {idx}/{n}")
        sv = Statevector(circuit_fn(angles))
        feats.append([sv.expectation_value(op).real for op in ops])
    print(f"    {label}: {n}/{n} done.")
    return np.array(feats)

# ── Extract V1 features ──────────────────────────────────────────
print("\n  Circuit V1 features:")
v1_path_tr = "results/Q_tr.npy"
v1_path_te = "results/Q_te.npy"

if (os.path.exists(v1_path_tr) and os.path.exists(v1_path_te)):
    Q_tr_v1 = np.load(v1_path_tr)
    Q_te_v1 = np.load(v1_path_te)
    if (Q_tr_v1.shape == (X_train.shape[0], 7) and
            Q_te_v1.shape == (X_test.shape[0], 7)):
        print(f"    Loaded saved V1 features — "
              f"Q_tr: {Q_tr_v1.shape}, Q_te: {Q_te_v1.shape}")
    else:
        print("    Shape mismatch, recomputing V1...")
        Q_tr_v1 = extract_features(X_tr_top4, build_circuit_v1, "V1 Train")
        Q_te_v1 = extract_features(X_te_top4, build_circuit_v1, "V1 Test")
        np.save(v1_path_tr, Q_tr_v1)
        np.save(v1_path_te, Q_te_v1)
else:
    print("    Computing V1 features...")
    t0 = time.time()
    Q_tr_v1 = extract_features(X_tr_top4, build_circuit_v1, "V1 Train")
    Q_te_v1 = extract_features(X_te_top4, build_circuit_v1, "V1 Test")
    np.save(v1_path_tr, Q_tr_v1)
    np.save(v1_path_te, Q_te_v1)
    print(f"    Done in {time.time()-t0:.1f}s")

# ── Extract V2 features ──────────────────────────────────────────
print("\n  Circuit V2 features (improved circuit):")
v2_path_tr = "results/Q_tr_v2.npy"
v2_path_te = "results/Q_te_v2.npy"

if (os.path.exists(v2_path_tr) and os.path.exists(v2_path_te)):
    Q_tr_v2 = np.load(v2_path_tr)
    Q_te_v2 = np.load(v2_path_te)
    if (Q_tr_v2.shape == (X_train.shape[0], 7) and
            Q_te_v2.shape == (X_test.shape[0], 7)):
        print(f"    Loaded saved V2 features — "
              f"Q_tr: {Q_tr_v2.shape}, Q_te: {Q_te_v2.shape}")
    else:
        print("    Shape mismatch, recomputing V2...")
        Q_tr_v2 = extract_features(X_tr_top4, build_circuit_v2, "V2 Train")
        Q_te_v2 = extract_features(X_te_top4, build_circuit_v2, "V2 Test")
        np.save(v2_path_tr, Q_tr_v2)
        np.save(v2_path_te, Q_te_v2)
else:
    print("    Computing V2 features (improved circuit)...")
    t0 = time.time()
    Q_tr_v2 = extract_features(X_tr_top4, build_circuit_v2, "V2 Train")
    Q_te_v2 = extract_features(X_te_top4, build_circuit_v2, "V2 Test")
    np.save(v2_path_tr, Q_tr_v2)
    np.save(v2_path_te, Q_te_v2)
    print(f"    Done in {time.time()-t0:.1f}s")

# ── Compare output statistics ────────────────────────────────────
print("\n  Feature distribution comparison:")
print(f"  {'Stat':<10} {'V1 (current)':>16} {'V2 (improved)':>16}")
print("  " + "-" * 44)
for stat, v1_val, v2_val in [
    ("Mean",   Q_tr_v1.mean(),   Q_tr_v2.mean()),
    ("Std",    Q_tr_v1.std(),    Q_tr_v2.std()),
    ("Min",    Q_tr_v1.min(),    Q_tr_v2.min()),
    ("Max",    Q_tr_v1.max(),    Q_tr_v2.max()),
    ("Var",    Q_tr_v1.var(),    Q_tr_v2.var()),
]:
    better = " <- higher variance" if stat == "Var" and v2_val > v1_val else \
             " <- better spread"   if stat == "Std" and v2_val > v1_val else ""
    print(f"  {stat:<10} {v1_val:>16.4f} {v2_val:>16.4f}{better}")

# ════════════════════════════════════════════════════════════════
# STEP 5 — TRAIN AND EVALUATE ALL COMBINATIONS
# ════════════════════════════════════════════════════════════════
print("\n[STEP 5]  Training models on V1 and V2 features")
print("-" * 50)

all_results = []
roc_data    = {}
conf_data   = {}

def evaluate(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    cm  = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  {name:<46} Acc={acc:.4f}  Rec={rec:.4f}  "
          f"AUC={auc:.4f}  TP={tp} TN={tn}")
    all_results.append({
        "Model": name,
        "Accuracy": round(acc, 4), "Recall": round(rec, 4),
        "Precision": round(pre, 4), "F1": round(f1, 4),
        "AUC": round(auc, 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    })
    return cm

# ── Baseline: LightGBM on 23 classical features ──────────────────
print("\n  BASELINE:")
lgb_base = lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                               random_state=42, verbose=-1)
lgb_base.fit(X_tr_c, y_train)
lgb_pred  = lgb_base.predict(X_te_c)
lgb_proba = lgb_base.predict_proba(X_te_c)[:, 1]
cm_base   = evaluate("LightGBM baseline (23 feat)", y_test, lgb_pred, lgb_proba)
roc_data["LightGBM baseline\n(23 feat)"] = (lgb_proba, "#999999", "--", 1.8)

# ── V1 models (current QFE) ──────────────────────────────────────
print("\n  CIRCUIT V1 (current):")
X_tr_30_v1 = np.hstack([X_tr_c, Q_tr_v1])
X_te_30_v1 = np.hstack([X_te_c, Q_te_v1])

svm_v1 = SVC(kernel="rbf", C=1.0, probability=True,
             class_weight="balanced", random_state=42)
svm_v1.fit(X_tr_30_v1, y_train)
svm_v1_pred  = svm_v1.predict(X_te_30_v1)
svm_v1_proba = svm_v1.predict_proba(X_te_30_v1)[:, 1]
cm_v1_svm    = evaluate("V1 QFE + SVM (30 feat)",
                         y_test, svm_v1_pred, svm_v1_proba)
roc_data["V1 QFE+SVM\n(current circuit)"] = (svm_v1_proba, "#FF7F00", "-.", 1.8)

xgb_v1 = xgb.XGBClassifier(n_estimators=300, random_state=42,
                              eval_metric="logloss", verbosity=0,
                              scale_pos_weight=pos_weight)
xgb_v1.fit(X_tr_30_v1, y_train)
xgb_v1_pred  = xgb_v1.predict(X_te_30_v1)
xgb_v1_proba = xgb_v1.predict_proba(X_te_30_v1)[:, 1]
cm_v1_xgb    = evaluate("V1 QFE + XGBoost (30 feat)",
                         y_test, xgb_v1_pred, xgb_v1_proba)
roc_data["V1 QFE+XGBoost\n(current circuit)"] = (xgb_v1_proba, "#FFAA00", ":", 1.8)

# ── V2 models (improved QFE) ─────────────────────────────────────
print("\n  CIRCUIT V2 (improved — RY→CNOT→RX→CNOT→RY):")
X_tr_30_v2 = np.hstack([X_tr_c, Q_tr_v2])
X_te_30_v2 = np.hstack([X_te_c, Q_te_v2])

svm_v2 = SVC(kernel="rbf", C=1.0, probability=True,
             class_weight="balanced", random_state=42)
svm_v2.fit(X_tr_30_v2, y_train)
svm_v2_pred  = svm_v2.predict(X_te_30_v2)
svm_v2_proba = svm_v2.predict_proba(X_te_30_v2)[:, 1]
cm_v2_svm    = evaluate("V2 QFE + SVM (30 feat)  [IMPROVED]",
                         y_test, svm_v2_pred, svm_v2_proba)
roc_data["V2 QFE+SVM\n(improved circuit)"] = (svm_v2_proba, "#1F4E79", "-", 2.5)
conf_data["V2 QFE+SVM\n(Improved Circuit)"] = (cm_v2_svm, plt.cm.Blues)

xgb_v2 = xgb.XGBClassifier(n_estimators=300, random_state=42,
                              eval_metric="logloss", verbosity=0,
                              scale_pos_weight=pos_weight)
xgb_v2.fit(X_tr_30_v2, y_train)
xgb_v2_pred  = xgb_v2.predict(X_te_30_v2)
xgb_v2_proba = xgb_v2.predict_proba(X_te_30_v2)[:, 1]
cm_v2_xgb    = evaluate("V2 QFE + XGBoost (30 feat)  [IMPROVED]",
                         y_test, xgb_v2_pred, xgb_v2_proba)
roc_data["V2 QFE+XGBoost\n(improved circuit)"] = (xgb_v2_proba, "#E41A1C", "-", 2.5)
conf_data["V2 QFE+XGBoost\n(Improved Circuit)"] = (cm_v2_xgb, plt.cm.Reds)

# ── V2 Stacking ensemble ─────────────────────────────────────────
print("\n  CIRCUIT V2 STACKING ENSEMBLE:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(y_train))
oof_v2  = np.zeros(len(y_train))

print("  Building OOF predictions (5-fold)...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_c, y_train)):
    m1 = lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                             random_state=42, verbose=-1)
    m1.fit(X_tr_c[tr_idx], y_train[tr_idx])
    oof_lgb[val_idx] = m1.predict_proba(X_tr_c[val_idx])[:, 1]

    m2 = SVC(kernel="rbf", C=1.0, probability=True,
             class_weight="balanced", random_state=42)
    m2.fit(X_tr_30_v2[tr_idx], y_train[tr_idx])
    oof_v2[val_idx] = m2.predict_proba(X_tr_30_v2[val_idx])[:, 1]
    print(f"    Fold {fold+1}/5 done")

meta_tr = np.column_stack([oof_lgb, oof_v2])
meta_lr = LogisticRegression(random_state=42)
meta_lr.fit(meta_tr, y_train)

lgb_full = lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                               random_state=42, verbose=-1)
lgb_full.fit(X_tr_c, y_train)
svm_full = SVC(kernel="rbf", C=1.0, probability=True,
               class_weight="balanced", random_state=42)
svm_full.fit(X_tr_30_v2, y_train)

meta_te       = np.column_stack([lgb_full.predict_proba(X_te_c)[:, 1],
                                  svm_full.predict_proba(X_te_30_v2)[:, 1]])
stack_pred    = meta_lr.predict(meta_te)
stack_proba   = meta_lr.predict_proba(meta_te)[:, 1]
cm_stack      = evaluate("V2 Stacking: LightGBM + V2 QFE+SVM  [IMPROVED]",
                          y_test, stack_pred, stack_proba)
roc_data["V2 Stacking\n(LightGBM + V2 QFE+SVM)"] = (stack_proba, "#984EA3", "-", 3.0)
conf_data["V2 Stacking\n(LightGBM+V2 QFE+SVM)"]  = (cm_stack, plt.cm.Purples)

# ════════════════════════════════════════════════════════════════
# STEP 6 — RESULTS TABLE
# ════════════════════════════════════════════════════════════════
print("\n[STEP 6]  Results summary")
print("=" * 65)

df = pd.DataFrame(all_results)
df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
df.to_csv("results/improved_qfe_comparison.csv", index=False)

baseline_acc = float(df[df["Model"].str.contains("baseline")]["Accuracy"].values[0])

print(f"\n  {'Rank':<4} {'Model':<48} {'Acc':>6}  {'AUC':>6}  {'vs Baseline':>12}")
print("  " + "-" * 74)
for i, row in df.iterrows():
    diff   = (row["Accuracy"] - baseline_acc) * 100
    beat   = " BEAT BASELINE" if diff > 0 else ""
    marker = f"{diff:+.2f}%"
    print(f"  #{i+1:<3} {row['Model']:<48} "
          f"{row['Accuracy']:>6.4f}  {row['AUC']:>6.4f}  "
          f"{marker:>12}{beat}")

print(f"\n  Saved: results/improved_qfe_comparison.csv")

# ════════════════════════════════════════════════════════════════
# STEP 7 — ROC CURVE
# ════════════════════════════════════════════════════════════════
print("\n[STEP 7]  ROC curve")

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot([0,1],[0,1], "k--", lw=1.2,
        label="Random Baseline (AUC=0.50)", alpha=0.5)

for name, (proba, color, ls, lw) in roc_data.items():
    fpr_p, tpr_p, _ = roc_curve(y_test, proba)
    auc_p           = roc_auc_score(y_test, proba)
    ax.plot(fpr_p, tpr_p, color=color, linestyle=ls, linewidth=lw,
            label=f"{name.replace(chr(10),' ')}  (AUC={auc_p:.4f})")

ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate",  fontsize=13)
ax.set_title(
    "ROC Curves — Circuit V1 vs Circuit V2 (Improved)\n"
    "HemoPI-2 Hemolytic Peptide Classification",
    fontsize=12, fontweight="bold"
)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/improved_qfe_roc.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: results/improved_qfe_roc.png")

# ════════════════════════════════════════════════════════════════
# STEP 8 — CONFUSION MATRICES (V2 models only)
# ════════════════════════════════════════════════════════════════
print("\n[STEP 8]  Confusion matrices")

n   = len(conf_data)
fig2, axes2 = plt.subplots(1, n, figsize=(6 * n, 5))
if n == 1:
    axes2 = [axes2]

fig2.suptitle(
    "Confusion Matrices — Improved Circuit V2 Models\n"
    "HemoPI-2 Hemolytic Peptide Classification",
    fontsize=12, fontweight="bold", y=1.04
)

for ax2, (title, (cm, cmap)) in zip(axes2, conf_data.items()):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Hemolytic (0)", "Hemolytic (1)"]
    )
    disp.plot(ax=ax2, cmap=cmap, colorbar=False, values_format="d")
    tn2, fp2, fn2, tp2 = cm.ravel()
    a2 = (tp2+tn2)/cm.sum()
    r2 = tp2/(tp2+fn2) if (tp2+fn2)>0 else 0
    p2 = tp2/(tp2+fp2) if (tp2+fp2)>0 else 0
    f2 = 2*p2*r2/(p2+r2) if (p2+r2)>0 else 0
    stats = (f"Acc={a2:.3f}  Recall={r2:.3f}\n"
             f"Precision={p2:.3f}  F1={f2:.3f}\n"
             f"TP={tp2}  TN={tn2}  FP={fp2}  FN={fn2}")
    ax2.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel(f"Predicted Label\n\n{stats}", fontsize=9)
    ax2.set_ylabel("True Label", fontsize=10)

plt.tight_layout()
plt.savefig("results/improved_qfe_confusion.png",
            dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: results/improved_qfe_confusion.png")

# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  FINAL SUMMARY — CIRCUIT V1 vs V2")
print("=" * 65)

v1_svm_acc = float(df[df["Model"].str.contains("V1 QFE \\+ SVM")]["Accuracy"].values[0]) \
             if any(df["Model"].str.contains("V1 QFE \\+ SVM")) else 0
v2_svm_acc = float(df[df["Model"].str.contains("V2 QFE \\+ SVM")]["Accuracy"].values[0]) \
             if any(df["Model"].str.contains("V2 QFE \\+ SVM")) else 0
v2_stk_acc = float(df[df["Model"].str.contains("V2 Stacking")]["Accuracy"].values[0]) \
             if any(df["Model"].str.contains("V2 Stacking")) else 0

print(f"""
  LightGBM baseline                : {baseline_acc*100:.2f}%  (target)

  V1 QFE + SVM (current circuit)   : {v1_svm_acc*100:.2f}%
  V2 QFE + SVM (improved circuit)  : {v2_svm_acc*100:.2f}%  ({(v2_svm_acc-v1_svm_acc)*100:+.2f}% vs V1)
  V2 Stacking ensemble             : {v2_stk_acc*100:.2f}%  ({(v2_stk_acc-baseline_acc)*100:+.2f}% vs baseline)

  Circuit V2 improvements:
    Old: RY → CNOT → RX → Reverse CNOT → Measure
    New: RY → CNOT → RX → CNOT → RY → Measure

    Depth     : V1={qc_v1.depth()}  V2={qc_v2.depth()}  (+{qc_v2.depth()-qc_v1.depth()} layers)
    Gate count: V1={qc_v1.size()}   V2={qc_v2.size()}   (+{qc_v2.size()-qc_v1.size()} gates)
    Variance  : V1={Q_tr_v1.var():.4f}  V2={Q_tr_v2.var():.4f}

  Saved files:
    results/Q_tr_v2.npy                — V2 quantum features (train)
    results/Q_te_v2.npy                — V2 quantum features (test)
    results/improved_qfe_comparison.csv
    results/improved_qfe_roc.png
    results/improved_qfe_confusion.png
""")
print("  ALL DONE.")