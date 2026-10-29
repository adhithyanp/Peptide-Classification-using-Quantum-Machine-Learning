"""
combine_kernels.py
==================
Multiple Kernel Learning (MKL) — Teacher's Suggestion Implementation

Combines:
  K_quantum  — precomputed QSVM quantum kernel (already sharpened, saved)
  K_classical — RBF kernel computed on 23 classical features

Formula:
  K_combined = alpha * K_quantum + (1 - alpha) * K_classical

Tests alpha values: 0.1, 0.2, 0.3, 0.5

Outputs saved to results/:
  mkl_alpha_comparison.png     — accuracy/AUC vs alpha bar chart
  mkl_roc_curves.png           — ROC curves for all alpha values
  mkl_best_confusion_matrix.png— confusion matrix of best alpha
  mkl_results_summary.csv      — full metrics table
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score,
                             recall_score, precision_score, f1_score)
from sklearn.metrics.pairwise import rbf_kernel

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

# ════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

train_df = pd.read_csv('data/processed/train_features.csv')
test_df  = pd.read_csv('data/processed/test_features.csv')

feature_cols = [c for c in train_df.columns if c != 'label']
X_train = train_df[feature_cols].values
y_train = train_df['label'].values
X_test  = test_df[feature_cols].values
y_test  = test_df['label'].values

# Scale classical features to [0,1]
scaler = MinMaxScaler()
X_tr_c = scaler.fit_transform(X_train)
X_te_c = scaler.transform(X_test)

print(f"  Train : {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Test  : {X_test.shape[0]} samples,  {X_test.shape[1]} features")

# ════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD SAVED QUANTUM KERNEL (QSVM)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Loading saved QSVM quantum kernel matrices")
print("=" * 60)

qsvm_available = False

if (os.path.exists('results/K_tr_sharp.npy') and
        os.path.exists('results/K_te_sharp.npy')):
    K_q_train = np.load('results/K_tr_sharp.npy')
    K_q_test  = np.load('results/K_te_sharp.npy')

    # Verify shapes match current dataset
    if (K_q_train.shape == (X_train.shape[0], X_train.shape[0]) and
            K_q_test.shape  == (X_test.shape[0], X_train.shape[0])):
        qsvm_available = True
        print(f"  K_quantum train : {K_q_train.shape}  (loaded from K_tr_sharp.npy)")
        print(f"  K_quantum test  : {K_q_test.shape}   (loaded from K_te_sharp.npy)")
        print(f"  K_quantum stats — mean={K_q_train.mean():.4f}, "
              f"std={K_q_train.std():.4f}, "
              f"min={K_q_train.min():.4f}, max={K_q_train.max():.4f}")
    else:
        print(f"  !! Shape mismatch:")
        print(f"     K_tr_sharp: {K_q_train.shape} "
              f"(expected ({X_train.shape[0]}, {X_train.shape[0]}))")
        print(f"     K_te_sharp: {K_q_test.shape}  "
              f"(expected ({X_test.shape[0]}, {X_train.shape[0]}))")
        print("  Recomputing quantum kernel now…")
else:
    print("  Saved kernel files not found — recomputing quantum kernel…")

if not qsvm_available:
    try:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit_machine_learning.kernels import FidelityStatevectorKernel
        import lightgbm as lgb

        # Get top-8 features for QSVM
        scaler_q = MinMaxScaler(feature_range=(0, np.pi))
        X_tr_q   = scaler_q.fit_transform(X_train)
        X_te_q   = scaler_q.transform(X_test)

        lgb_m = lgb.LGBMClassifier(n_estimators=300, random_state=42,
                                    class_weight='balanced', verbose=-1)
        lgb_m.fit(X_tr_c, y_train)
        top8 = np.argsort(lgb_m.feature_importances_)[::-1][:8]

        print("  Building ZZFeatureMap kernel (8 features, reps=2)…")
        fmap = ZZFeatureMap(feature_dimension=8, reps=2, entanglement='linear')
        kobj = FidelityStatevectorKernel(feature_map=fmap)

        K_raw_tr = kobj.evaluate(x_vec=X_tr_q[:, top8])
        K_raw_te = kobj.evaluate(x_vec=X_te_q[:, top8],
                                 y_vec=X_tr_q[:, top8])

        gamma_sharp = 5.0
        K_q_train = np.exp(gamma_sharp * (K_raw_tr - 1))
        K_q_test  = np.exp(gamma_sharp * (K_raw_te - 1))

        np.save('results/K_tr_sharp.npy', K_q_train)
        np.save('results/K_te_sharp.npy', K_q_test)
        qsvm_available = True
        print("  Quantum kernel computed and saved.")

    except Exception as e:
        print(f"  !! Quantum kernel computation failed: {e}")
        print("  Cannot proceed without quantum kernel.")
        print("  Make sure results/K_tr_sharp.npy exists from your QSVM run.")
        exit(1)

# ════════════════════════════════════════════════════════════════
# SECTION 3 — COMPUTE CLASSICAL RBF KERNEL
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Computing classical RBF kernel")
print("=" * 60)

# gamma='scale' = 1 / (n_features * X.var()) — sklearn default
gamma_rbf = 1.0 / (X_tr_c.shape[1] * X_tr_c.var())

K_c_train = rbf_kernel(X_tr_c, X_tr_c, gamma=gamma_rbf)
K_c_test  = rbf_kernel(X_te_c, X_tr_c, gamma=gamma_rbf)

print(f"  K_classical train : {K_c_train.shape}")
print(f"  K_classical test  : {K_c_test.shape}")
print(f"  K_classical stats — mean={K_c_train.mean():.4f}, "
      f"std={K_c_train.std():.4f}, "
      f"min={K_c_train.min():.4f}, max={K_c_train.max():.4f}")

# ════════════════════════════════════════════════════════════════
# SECTION 4 — BASELINE MODELS FOR COMPARISON
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Baseline models (for comparison)")
print("=" * 60)

results = []

def run_svm(name, K_tr, K_te, y_tr, y_te, alpha_val=None):
    svm = SVC(kernel='precomputed', probability=True,
              class_weight='balanced', random_state=42)
    svm.fit(K_tr, y_tr)
    pred  = svm.predict(K_te)
    proba = svm.predict_proba(K_te)[:, 1]

    acc = accuracy_score(y_te, pred)
    rec = recall_score(y_te, pred)
    pre = precision_score(y_te, pred, zero_division=0)
    f1  = f1_score(y_te, pred)
    fpr, tpr, _ = roc_curve(y_te, proba)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_te, pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"  {name:45s}  Acc={acc:.4f}  Rec={rec:.4f}  "
          f"AUC={roc_auc:.4f}  TP={tp} TN={tn} FP={fp} FN={fn}")

    results.append({
        'Model':     name,
        'Alpha':     alpha_val if alpha_val is not None else '—',
        'Accuracy':  round(acc, 4),
        'Recall':    round(rec, 4),
        'Precision': round(pre, 4),
        'F1':        round(f1,  4),
        'AUC':       round(roc_auc, 4),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    })
    return fpr, tpr, roc_auc, pred, proba, cm

# QSVM only (quantum kernel)
print("\n  [1] QSVM only (quantum kernel, sharpened)")
q_fpr, q_tpr, q_auc, q_pred, q_proba, q_cm = run_svm(
    'QSVM only (K_quantum)', K_q_train, K_q_test, y_train, y_test)

# Classical RBF kernel only
print("\n  [2] Classical RBF kernel only")
c_fpr, c_tpr, c_auc, c_pred, c_proba, c_cm = run_svm(
    'Classical RBF only (K_classical)', K_c_train, K_c_test, y_train, y_test)

# ════════════════════════════════════════════════════════════════
# SECTION 5 — MULTIPLE KERNEL LEARNING (MKL)
#   K_combined = alpha * K_quantum + (1 - alpha) * K_classical
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Multiple Kernel Learning — testing alpha values")
print("=" * 60)
print(f"  Formula: K_combined = alpha × K_quantum + (1-alpha) × K_classical\n")

ALPHAS = [0.1, 0.2, 0.3, 0.5]

mkl_roc    = {}   # alpha → (fpr, tpr, auc)
mkl_cms    = {}   # alpha → confusion matrix
mkl_probas = {}   # alpha → probabilities

for alpha in ALPHAS:
    K_comb_tr = alpha * K_q_train + (1 - alpha) * K_c_train
    K_comb_te = alpha * K_q_test  + (1 - alpha) * K_c_test

    name = f'MKL α={alpha} ({int(alpha*100)}% quantum + {int((1-alpha)*100)}% classical)'
    fpr, tpr, roc_auc, pred, proba, cm = run_svm(
        name, K_comb_tr, K_comb_te, y_train, y_test, alpha_val=alpha)

    mkl_roc[alpha]    = (fpr, tpr, roc_auc)
    mkl_cms[alpha]    = cm
    mkl_probas[alpha] = proba

# ════════════════════════════════════════════════════════════════
# SECTION 6 — IDENTIFY BEST ALPHA
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Identifying best alpha")
print("=" * 60)

df = pd.DataFrame(results)
df.to_csv('results/mkl_results_summary.csv', index=False)

print("\n  Full results table:")
print(df.to_string(index=False))

# Best by accuracy among MKL models
mkl_rows   = df[df['Alpha'] != '—']
best_row   = mkl_rows.loc[mkl_rows['Accuracy'].idxmax()]
best_alpha = float(best_row['Alpha'])
best_acc   = float(best_row['Accuracy'])
best_auc   = float(best_row['AUC'])

print(f"\n  Best alpha      : {best_alpha}")
print(f"  Best MKL Acc    : {best_acc:.4f}")
print(f"  Best MKL AUC    : {best_auc:.4f}")

baseline_acc = float(df[df['Model'] == 'Classical RBF only (K_classical)']['Accuracy'])
diff = best_acc - baseline_acc
print(f"\n  Classical RBF baseline Acc : {baseline_acc:.4f}")
print(f"  MKL improvement            : {diff:+.4f} "
      f"({'improved' if diff > 0 else 'no improvement'})")
print("\n  Saved: results/mkl_results_summary.csv")

# ════════════════════════════════════════════════════════════════
# SECTION 7 — PLOT 1: ALPHA COMPARISON BAR CHART
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Plotting alpha comparison bar chart")
print("=" * 60)

mkl_df     = df[df['Alpha'] != '—'].copy()
alpha_vals = mkl_df['Alpha'].astype(float).tolist()
acc_vals   = (mkl_df['Accuracy'] * 100).tolist()
auc_vals   = mkl_df['AUC'].tolist()

x      = np.arange(len(alpha_vals))
width  = 0.35
colors_acc = ['#4C72B0' if a != best_alpha else '#DD4444' for a in alpha_vals]

fig, ax1 = plt.subplots(figsize=(9, 6))

bars = ax1.bar(x - width/2, acc_vals, width, label='Accuracy (%)',
               color=colors_acc, edgecolor='black', linewidth=0.8, alpha=0.85)

for bar, v in zip(bars, acc_vals):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.2, f'{v:.2f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2 = ax1.twinx()
ax2.bar(x + width/2, auc_vals, width, label='AUC',
        color='#55A868', edgecolor='black', linewidth=0.8, alpha=0.85)
for i, (xi, v) in enumerate(zip(x, auc_vals)):
    ax2.text(xi + width/2, v + 0.002, f'{v:.4f}',
             ha='center', va='bottom', fontsize=9, color='darkgreen')

# Reference lines
ax1.axhline(y=baseline_acc * 100, color='blue', ls='--', lw=1.5,
            label=f'Classical RBF baseline ({baseline_acc*100:.2f}%)')
ax1.axhline(y=float(df[df['Model'].str.contains('QSVM only')]['Accuracy']) * 100,
            color='purple', ls=':', lw=1.5,
            label=f'QSVM only baseline')

ax1.set_xlabel('Alpha (quantum kernel weight)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_ylabel('AUC', fontsize=12, color='darkgreen')
ax1.set_title('Multiple Kernel Learning — Alpha Comparison\n'
              'K_combined = α × K_quantum + (1-α) × K_classical',
              fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'α = {a}\n({int(a*100)}% quantum\n{int((1-a)*100)}% classical)'
                     for a in alpha_vals], fontsize=9)
ax1.set_ylim(55, min(100, max(acc_vals) + 10))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='lower right', fontsize=9, framealpha=0.9)
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('results/mkl_alpha_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: results/mkl_alpha_comparison.png")

# ════════════════════════════════════════════════════════════════
# SECTION 8 — PLOT 2: ROC CURVES
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Plotting ROC curves")
print("=" * 60)

COLORS_MKL = {
    0.1: '#1F77B4',
    0.2: '#FF7F0E',
    0.3: '#2CA02C',
    0.5: '#D62728',
}

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0,1],[0,1], 'k--', lw=1.2, label='Random Baseline (AUC=0.50)', alpha=0.5)

# Baselines
ax.plot(q_fpr, q_tpr, color='#9467BD', ls=':', lw=1.8,
        label=f'QSVM only  (AUC={q_auc:.4f})')
ax.plot(c_fpr, c_tpr, color='#8C564B', ls='--', lw=2.0,
        label=f'Classical RBF only  (AUC={c_auc:.4f})')

# MKL curves
for alpha, (fpr, tpr, roc_auc) in mkl_roc.items():
    lw   = 3.0 if alpha == best_alpha else 1.8
    mark = ' ← BEST' if alpha == best_alpha else ''
    ax.plot(fpr, tpr,
            color=COLORS_MKL[alpha], ls='-', lw=lw,
            label=f'MKL α={alpha}  (AUC={roc_auc:.4f}){mark}')

ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate',  fontsize=13)
ax.set_title('ROC Curves — Multiple Kernel Learning\n'
             'QSVM + Classical RBF Kernel Combination (HemoPI-2)',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('results/mkl_roc_curves.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: results/mkl_roc_curves.png")

# ════════════════════════════════════════════════════════════════
# SECTION 9 — PLOT 3: CONFUSION MATRIX OF BEST ALPHA
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9: Confusion matrix for best alpha")
print("=" * 60)

fig3, axes3 = plt.subplots(1, 3, figsize=(17, 5))
fig3.suptitle(
    f'Confusion Matrices — QSVM only  |  Classical RBF only  |  Best MKL (α={best_alpha})\n'
    'Multiple Kernel Learning — HemoPI-2',
    fontsize=12, fontweight='bold', y=1.04
)

panel_data = [
    ('QSVM only\n(K_quantum)',              q_cm,                    plt.cm.Purples),
    ('Classical RBF only\n(K_classical)',   c_cm,                    plt.cm.Blues),
    (f'Best MKL\nα = {best_alpha}\n'
     f'({int(best_alpha*100)}% quantum + '
     f'{int((1-best_alpha)*100)}% classical)',
     mkl_cms[best_alpha],                                             plt.cm.Oranges),
]

for ax2, (title, cm, cmap) in zip(axes3, panel_data):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Non-Hemolytic (0)', 'Hemolytic (1)']
    )
    disp.plot(ax=ax2, cmap=cmap, colorbar=False, values_format='d')

    tn, fp, fn, tp = cm.ravel()
    acc = (tp+tn)/cm.sum()
    rec = tp/(tp+fn) if (tp+fn)>0 else 0
    pre = tp/(tp+fp) if (tp+fp)>0 else 0
    f1  = 2*pre*rec/(pre+rec) if (pre+rec)>0 else 0
    stats = (f'Acc={acc:.3f}    Recall={rec:.3f}\n'
             f'Precision={pre:.3f}    F1={f1:.3f}\n'
             f'TP={tp}  TN={tn}  FP={fp}  FN={fn}')

    ax2.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel(f'Predicted Label\n\n{stats}', fontsize=9)
    ax2.set_ylabel('True Label', fontsize=10)

plt.tight_layout()
plt.savefig('results/mkl_best_confusion_matrix.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: results/mkl_best_confusion_matrix.png")

# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"""
  QSVM only               : Acc={float(df[df['Model'].str.contains('QSVM only')]['Accuracy']):.4f}
  Classical RBF only      : Acc={baseline_acc:.4f}
  Best MKL (α={best_alpha})     : Acc={best_acc:.4f}   AUC={best_auc:.4f}
  Improvement over RBF    : {diff:+.4f}

  Interpretation:
  ─────────────────────────────────────────────────────
  The Multiple Kernel Learning approach combines the
  quantum kernel (capturing entanglement-based peptide
  similarity) with the classical RBF kernel (capturing
  Euclidean feature-space similarity).
  
  Best alpha = {best_alpha} means {int(best_alpha*100)}% weight on quantum kernel
  and {int((1-best_alpha)*100)}% weight on classical kernel gives optimal performance.
  
  This is your 4th novel contribution:
  A systematic MKL study of quantum + classical kernels
  for hemolytic peptide classification.
  ─────────────────────────────────────────────────────
""")

print("Output files in results/:")
print("  mkl_alpha_comparison.png      — bar chart of all alpha values")
print("  mkl_roc_curves.png            — ROC curves all models")
print("  mkl_best_confusion_matrix.png — confusion matrix for best alpha")
print("  mkl_results_summary.csv       — complete metrics table")
print("\nALL DONE.")