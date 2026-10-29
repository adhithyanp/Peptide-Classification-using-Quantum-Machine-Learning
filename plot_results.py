"""
plot_results.py
===============
Generates:
  1. Multi-model ROC curve (all 6 models on one plot)
  2. Figure with 3 confusion matrices side-by-side:
       - XGBoost       (best classical model)
       - QSVM          (with kernel sharpening)
       - QFE + SVM     (quantum feature extractor, 30 features)

Saves to results/ folder.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (roc_curve, auc,
                             confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score)
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
train = pd.read_csv('data/processed/train_features.csv')
test  = pd.read_csv('data/processed/test_features.csv')

feature_cols = [c for c in train.columns if c != 'label']
X_train = train[feature_cols].values
y_train = train['label'].values
X_test  = test[feature_cols].values
y_test  = test['label'].values

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# MinMaxScaler for classical models [0,1]
scaler_c = MinMaxScaler()
X_tr_c = scaler_c.fit_transform(X_train)
X_te_c = scaler_c.transform(X_test)

# MinMaxScaler for quantum circuit angles [0, pi]
scaler_q = MinMaxScaler(feature_range=(0, np.pi))
X_tr_q = scaler_q.fit_transform(X_train)
X_te_q = scaler_q.transform(X_test)

# ─────────────────────────────────────────────
# 2. LIGHTGBM — feature importance only
# ─────────────────────────────────────────────
lgb_model = lgb.LGBMClassifier(n_estimators=300, random_state=42,
                                class_weight='balanced', verbose=-1)
lgb_model.fit(X_tr_c, y_train)
importances = lgb_model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]
top8_idx    = sorted_idx[:8]
top4_idx    = sorted_idx[:4]

print(f"Top-4 features (QFE):  {[feature_cols[i] for i in top4_idx]}")
print(f"Top-8 features (QSVM): {[feature_cols[i] for i in top8_idx]}")

# ─────────────────────────────────────────────
# 3. ALL CLASSICAL MODELS  (for ROC curve)
# ─────────────────────────────────────────────
roc_data = {}   # {name: (fpr, tpr, auc)}

# SVM
svm_clf = SVC(kernel='rbf', probability=True,
              class_weight='balanced', random_state=42)
svm_clf.fit(X_tr_c, y_train)
p = svm_clf.predict_proba(X_te_c)[:, 1]
fpr, tpr, _ = roc_curve(y_test, p)
roc_data['SVM (RBF)'] = (fpr, tpr, auc(fpr, tpr))
print(f"SVM           Acc={accuracy_score(y_test, svm_clf.predict(X_te_c)):.4f}  AUC={auc(fpr,tpr):.4f}")

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
rf_clf.fit(X_tr_c, y_train)
p = rf_clf.predict_proba(X_te_c)[:, 1]
fpr, tpr, _ = roc_curve(y_test, p)
roc_data['Random Forest'] = (fpr, tpr, auc(fpr, tpr))
print(f"Random Forest Acc={accuracy_score(y_test, rf_clf.predict(X_te_c)):.4f}  AUC={auc(fpr,tpr):.4f}")

# ── XGBoost  ← BEST CLASSICAL ──
xgb_clf = xgb.XGBClassifier(
    n_estimators=300, random_state=42,
    scale_pos_weight=(y_train==0).sum() / (y_train==1).sum(),
    eval_metric='logloss', verbosity=0
)
xgb_clf.fit(X_tr_c, y_train)
xgb_proba = xgb_clf.predict_proba(X_te_c)[:, 1]
xgb_pred  = xgb_clf.predict(X_te_c)
fpr, tpr, _ = roc_curve(y_test, xgb_proba)
roc_data['XGBoost'] = (fpr, tpr, auc(fpr, tpr))
print(f"XGBoost       Acc={accuracy_score(y_test, xgb_pred):.4f}  AUC={auc(fpr,tpr):.4f}")

# LightGBM
lgb_proba = lgb_model.predict_proba(X_te_c)[:, 1]
fpr, tpr, _ = roc_curve(y_test, lgb_proba)
roc_data['LightGBM'] = (fpr, tpr, auc(fpr, tpr))
print(f"LightGBM      Acc={accuracy_score(y_test, lgb_model.predict(X_te_c)):.4f}  AUC={auc(fpr,tpr):.4f}")

# ── Confusion matrix 1/3 : XGBoost ──
cm_xgb = confusion_matrix(y_test, xgb_pred)
print(f"\nXGBoost Confusion Matrix:\n{cm_xgb}")

# ─────────────────────────────────────────────
# 4. QSVM WITH KERNEL SHARPENING
# ─────────────────────────────────────────────
print("\nBuilding QSVM kernel…")
cm_qsvm    = None
qsvm_proba = None

try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel

    X_tr_q8 = X_tr_q[:, top8_idx]
    X_te_q8 = X_te_q[:, top8_idx]

    feature_map = ZZFeatureMap(feature_dimension=8, reps=2, entanglement='linear')
    kernel_obj  = FidelityStatevectorKernel(feature_map=feature_map)

    K_train_raw = kernel_obj.evaluate(x_vec=X_tr_q8)
    K_test_raw  = kernel_obj.evaluate(x_vec=X_te_q8, y_vec=X_tr_q8)

    gamma = 5.0
    K_tr_sharp = np.exp(gamma * (K_train_raw - 1))
    K_te_sharp = np.exp(gamma * (K_test_raw  - 1))
    np.save('results/K_tr_sharp.npy', K_tr_sharp)
    np.save('results/K_te_sharp.npy', K_te_sharp)

    qsvm = SVC(kernel='precomputed', probability=True,
               class_weight='balanced', random_state=42)
    qsvm.fit(K_tr_sharp, y_train)
    qsvm_pred  = qsvm.predict(K_te_sharp)
    qsvm_proba = qsvm.predict_proba(K_te_sharp)[:, 1]

    cm_qsvm = confusion_matrix(y_test, qsvm_pred)
    fpr, tpr, _ = roc_curve(y_test, qsvm_proba)
    roc_data['QSVM + Sharpening'] = (fpr, tpr, auc(fpr, tpr))
    print(f"QSVM Acc={accuracy_score(y_test, qsvm_pred):.4f}  AUC={auc(fpr,tpr):.4f}")
    print(f"QSVM Confusion Matrix:\n{cm_qsvm}")

except Exception as e:
    print(f"  Qiskit live run failed: {e}")
    if (os.path.exists('results/K_tr_sharp.npy') and
            os.path.exists('results/K_te_sharp.npy')):
        print("  Loading saved kernel matrices from results/")
        K_tr_sharp = np.load('results/K_tr_sharp.npy')
        K_te_sharp = np.load('results/K_te_sharp.npy')
        qsvm = SVC(kernel='precomputed', probability=True,
                   class_weight='balanced', random_state=42)
        qsvm.fit(K_tr_sharp, y_train)
        qsvm_pred  = qsvm.predict(K_te_sharp)
        qsvm_proba = qsvm.predict_proba(K_te_sharp)[:, 1]
        cm_qsvm = confusion_matrix(y_test, qsvm_pred)
        fpr, tpr, _ = roc_curve(y_test, qsvm_proba)
        roc_data['QSVM + Sharpening'] = (fpr, tpr, auc(fpr, tpr))
        print(f"  QSVM Confusion Matrix:\n{cm_qsvm}")
    else:
        print("  !! No saved kernels found — QSVM panel will show placeholder.")

# ─────────────────────────────────────────────
# 5. QUANTUM FEATURE EXTRACTOR (QFE) + SVM
# ─────────────────────────────────────────────
print("\nRunning Quantum Feature Extractor…")
cm_qfe    = None
qfe_proba = None

def build_qfe_circuit(angles):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(4)
    for i, a in enumerate(angles):
        qc.ry(float(a), i)
    for i in range(4):
        qc.cx(i, (i+1) % 4)
    for i, a in enumerate(angles):
        qc.rx(2*float(a), i)
    for i in range(3, -1, -1):
        qc.cx(i, (i+1) % 4)
    return qc

def extract_qfe_features(X_subset):
    from qiskit.quantum_info import Statevector, SparsePauliOp
    ops = [
        SparsePauliOp('IIIZ'), SparsePauliOp('IIZI'),
        SparsePauliOp('IZII'), SparsePauliOp('ZIII'),
        SparsePauliOp('IIZZ'), SparsePauliOp('IZZI'),
        SparsePauliOp('ZZII'),
    ]
    feats = []
    for idx, angles in enumerate(X_subset):
        if idx % 200 == 0:
            print(f"    QFE progress: {idx}/{len(X_subset)}")
        qc = build_qfe_circuit(angles)
        sv = Statevector(qc)
        row = [sv.expectation_value(op).real for op in ops]
        feats.append(row)
    return np.array(feats)

try:
    if os.path.exists('results/Q_tr.npy') and os.path.exists('results/Q_te.npy'):
        print("  Loading saved QFE features from results/")
        Q_tr = np.load('results/Q_tr.npy')
        Q_te = np.load('results/Q_te.npy')
    else:
        print("  Computing QFE features (~45–60 min)…")
        Q_tr = extract_qfe_features(X_tr_q[:, top4_idx])
        Q_te = extract_qfe_features(X_te_q[:, top4_idx])
        np.save('results/Q_tr.npy', Q_tr)
        np.save('results/Q_te.npy', Q_te)
        print("  Saved Q_tr.npy and Q_te.npy")

    X_tr_30 = np.hstack([X_tr_c, Q_tr])
    X_te_30 = np.hstack([X_te_c, Q_te])

    qfe_svm = SVC(kernel='rbf', probability=True,
                  class_weight='balanced', random_state=42)
    qfe_svm.fit(X_tr_30, y_train)
    qfe_pred  = qfe_svm.predict(X_te_30)
    qfe_proba = qfe_svm.predict_proba(X_te_30)[:, 1]

    cm_qfe = confusion_matrix(y_test, qfe_pred)
    fpr, tpr, _ = roc_curve(y_test, qfe_proba)
    roc_data['QFE + SVM (30 features)'] = (fpr, tpr, auc(fpr, tpr))
    print(f"QFE+SVM Acc={accuracy_score(y_test, qfe_pred):.4f}  AUC={auc(fpr,tpr):.4f}")
    print(f"QFE Confusion Matrix:\n{cm_qfe}")

except Exception as e:
    print(f"  !! QFE failed: {e}")
    print("  QFE panel will show placeholder.")

# ─────────────────────────────────────────────
# 6. PLOT 1 — MULTI-MODEL ROC CURVE
# ─────────────────────────────────────────────
COLORS = {
    'SVM (RBF)':               '#E41A1C',
    'Random Forest':           '#377EB8',
    'XGBoost':                 '#4DAF4A',
    'LightGBM':                '#FF7F00',
    'QSVM + Sharpening':       '#984EA3',
    'QFE + SVM (30 features)': '#A65628',
}
STYLES = {
    'SVM (RBF)': '-',  'Random Forest': '--',
    'XGBoost':   '-.',  'LightGBM': ':',
    'QSVM + Sharpening': '-',  'QFE + SVM (30 features)': '-',
}
WIDTHS = {
    'SVM (RBF)': 1.8, 'Random Forest': 1.8,
    'XGBoost': 2.5,   'LightGBM': 1.8,
    'QSVM + Sharpening': 2.0, 'QFE + SVM (30 features)': 2.5,
}

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0,1],[0,1],'k--', lw=1.2, label='Random Baseline (AUC = 0.50)', alpha=0.5)

for name, (fpr, tpr, roc_auc) in roc_data.items():
    ax.plot(fpr, tpr,
            color=COLORS.get(name, 'gray'),
            linestyle=STYLES.get(name, '-'),
            lw=WIDTHS.get(name, 1.8),
            label=f'{name}  (AUC = {roc_auc:.3f})')

ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves — All Models\nHemolytic Peptide Classification (HemoPI-2)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('results/roc_all_models.png', dpi=200, bbox_inches='tight')
plt.close()
print("\nSaved: results/roc_all_models.png")

# ─────────────────────────────────────────────
# 7. PLOT 2 — 3 CONFUSION MATRICES SIDE-BY-SIDE
# ─────────────────────────────────────────────
panel_configs = [
    ("XGBoost\n(Best Classical)",              cm_xgb,  plt.cm.Greens),
    ("QSVM\n(Kernel Sharpening)",              cm_qsvm, plt.cm.Purples),
    ("QFE + SVM\n(Quantum Feature Extractor)", cm_qfe,  plt.cm.Oranges),
]

fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle(
    'Confusion Matrices — XGBoost  |  QSVM  |  Quantum Feature Extractor\n'
    'Hemolytic Peptide Classification (HemoPI-2)',
    fontsize=13, fontweight='bold', y=1.04
)

for ax2, (title, cm, cmap) in zip(axes, panel_configs):
    if cm is None:
        # Placeholder if model was skipped
        ax2.set_facecolor('#f5f5f5')
        ax2.text(0.5, 0.5,
                 f'{title}\n\nNot available\n(model was skipped)',
                 ha='center', va='center', fontsize=11,
                 color='grey', transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title(title, fontsize=12, fontweight='bold')
        continue

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Non-Hemolytic (0)', 'Hemolytic (1)']
    )
    disp.plot(ax=ax2, cmap=cmap, colorbar=False, values_format='d')

    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / cm.sum()
    rec = tp / (tp + fn)    if (tp + fn) > 0 else 0
    pre = tp / (tp + fp)    if (tp + fp) > 0 else 0
    f1  = 2*pre*rec/(pre+rec) if (pre+rec) > 0 else 0

    stats = (f'Acc = {acc:.3f}    Recall = {rec:.3f}\n'
             f'Precision = {pre:.3f}    F1 = {f1:.3f}\n'
             f'TP={tp}   TN={tn}   FP={fp}   FN={fn}')

    ax2.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel(f'Predicted Label\n\n{stats}', fontsize=10)
    ax2.set_ylabel('True Label', fontsize=10)

plt.tight_layout()
plt.savefig('results/confusion_matrices_3models.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: results/confusion_matrices_3models.png")
print("\n✅ All plots complete.")