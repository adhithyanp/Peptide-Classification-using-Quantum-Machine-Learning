"""
plot_qfe_for_ppt.py
===================
Generates high-quality confusion matrix and ROC curve
specifically for the QFE+SVM model (78.22% accuracy, AUC 0.8593)
and compares it with LightGBM baseline (80.20%).

Saves to results/:
  ppt_qfe_confusion_matrix.png   (single large confusion matrix)
  ppt_comparison_confusion.png   (LightGBM vs QFE+SVM side by side)
  ppt_roc_curve.png              (ROC curve with all key models)

Prerequisites:
  results/Q_tr.npy and results/Q_te.npy must exist.
  Run main_pipeline.py first if they are missing.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score,
    recall_score, precision_score, f1_score,
    roc_auc_score
)
import lightgbm as lgb

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ─── LOAD DATA ───────────────────────────────────────────────────
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

# Load QFE features
if not (os.path.exists("results/Q_tr.npy") and
        os.path.exists("results/Q_te.npy")):
    print("ERROR: results/Q_tr.npy not found.")
    print("Run main_pipeline.py first to generate QFE features.")
    exit(1)

Q_tr    = np.load("results/Q_tr.npy")
Q_te    = np.load("results/Q_te.npy")
X_tr_30 = np.hstack([X_tr_c, Q_tr])
X_te_30 = np.hstack([X_te_c, Q_te])

print("Data loaded successfully.")
print(f"  Classical features : {X_tr_c.shape}")
print(f"  Quantum features   : {Q_tr.shape}")
print(f"  Combined 30-feat   : {X_tr_30.shape}")

# ─── TRAIN MODELS ────────────────────────────────────────────────
print("\nTraining LightGBM baseline (23 features)...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=300, class_weight="balanced",
    random_state=42, verbose=-1
)
lgb_model.fit(X_tr_c, y_train)
lgb_pred  = lgb_model.predict(X_te_c)
lgb_proba = lgb_model.predict_proba(X_te_c)[:, 1]

lgb_acc = accuracy_score(y_test, lgb_pred)
lgb_auc = roc_auc_score(y_test, lgb_proba)
lgb_rec = recall_score(y_test, lgb_pred)
lgb_f1  = f1_score(y_test, lgb_pred)
lgb_cm  = confusion_matrix(y_test, lgb_pred)
print(f"  LightGBM: Acc={lgb_acc:.4f}  AUC={lgb_auc:.4f}")

print("\nTraining QFE + SVM (30 features)...")
svm_model = SVC(
    kernel="rbf", C=1.0, probability=True,
    class_weight="balanced", random_state=42
)
svm_model.fit(X_tr_30, y_train)
svm_pred  = svm_model.predict(X_te_30)
svm_proba = svm_model.predict_proba(X_te_30)[:, 1]

svm_acc = accuracy_score(y_test, svm_pred)
svm_auc = roc_auc_score(y_test, svm_proba)
svm_rec = recall_score(y_test, svm_pred)
svm_f1  = f1_score(y_test, svm_pred)
svm_pre = precision_score(y_test, svm_pred, zero_division=0)
svm_cm  = confusion_matrix(y_test, svm_pred)
tn,fp,fn,tp = svm_cm.ravel()
print(f"  QFE+SVM: Acc={svm_acc:.4f}  AUC={svm_auc:.4f}")
print(f"           TP={tp}  TN={tn}  FP={fp}  FN={fn}")

# Also train XGBoost for ROC comparison
import xgboost as xgb
xgb_model = xgb.XGBClassifier(
    n_estimators=300, random_state=42,
    eval_metric="logloss", verbosity=0,
    scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()
)
xgb_model.fit(X_tr_c, y_train)
xgb_proba = xgb_model.predict_proba(X_te_c)[:, 1]

# ─── PLOT 1: QFE CONFUSION MATRIX ALONE (Large, PPT-ready) ───────
print("\nGenerating QFE confusion matrix (single, large)...")

fig1, ax1 = plt.subplots(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=svm_cm,
    display_labels=["Non-Hemolytic", "Hemolytic"]
)
disp.plot(ax=ax1, cmap=plt.cm.Blues, colorbar=False, values_format="d",
          text_kw={"fontsize": 22, "fontweight": "bold"})

# Style the plot
ax1.set_title(
    "Quantum Feature Extractor + SVM\n"
    "Confusion Matrix (HemoPI-2 Test Set)",
    fontsize=16, fontweight="bold", pad=15, color="#1F4E79"
)
ax1.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
ax1.set_ylabel("True Label", fontsize=14, fontweight="bold")
ax1.tick_params(labelsize=12)

# Add metrics box below the plot
metrics_text = (
    f"Accuracy = {svm_acc:.2%}    Recall = {svm_rec:.2%}    "
    f"Precision = {svm_pre:.2%}    F1 = {svm_f1:.2%}    AUC = {svm_auc:.4f}"
)
fig1.text(0.5, 0.01, metrics_text, ha="center", fontsize=11,
          color="#1F4E79", fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.4", facecolor="#D6E4F0",
                    edgecolor="#2E75B6", linewidth=1.5))

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("results/ppt_qfe_confusion_matrix.png",
            dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: results/ppt_qfe_confusion_matrix.png")

# ─── PLOT 2: SIDE-BY-SIDE COMPARISON ─────────────────────────────
print("\nGenerating side-by-side comparison confusion matrices...")

fig2, (ax_l, ax_q) = plt.subplots(1, 2, figsize=(13, 6))
fig2.suptitle(
    "LightGBM Baseline  vs  Quantum Feature Extractor + SVM\n"
    "HemoPI-2 Hemolytic Peptide Classification",
    fontsize=14, fontweight="bold", color="#1F4E79", y=1.03
)

# LightGBM panel
tn_l,fp_l,fn_l,tp_l = lgb_cm.ravel()
disp_l = ConfusionMatrixDisplay(
    confusion_matrix=lgb_cm,
    display_labels=["Non-Hemolytic", "Hemolytic"]
)
disp_l.plot(ax=ax_l, cmap=plt.cm.Greens, colorbar=False,
            values_format="d",
            text_kw={"fontsize": 18, "fontweight": "bold"})
ax_l.set_title(
    f"LightGBM (23 Classical Features)\n"
    f"Accuracy={lgb_acc:.2%}   AUC={lgb_auc:.4f}",
    fontsize=12, fontweight="bold", color="#1A5E38", pad=10
)
ax_l.set_xlabel("Predicted Label", fontsize=12)
ax_l.set_ylabel("True Label", fontsize=12)
ax_l.tick_params(labelsize=11)

lgb_stats = f"TP={tp_l}  TN={tn_l}  FP={fp_l}  FN={fn_l}   Recall={lgb_rec:.2%}"
ax_l.set_xlabel(f"Predicted Label\n{lgb_stats}", fontsize=11)

# QFE+SVM panel
disp_q = ConfusionMatrixDisplay(
    confusion_matrix=svm_cm,
    display_labels=["Non-Hemolytic", "Hemolytic"]
)
disp_q.plot(ax=ax_q, cmap=plt.cm.Blues, colorbar=False,
            values_format="d",
            text_kw={"fontsize": 18, "fontweight": "bold"})
ax_q.set_title(
    f"QFE + SVM (30 Features: 23 Classical + 7 Quantum)\n"
    f"Accuracy={svm_acc:.2%}   AUC={svm_auc:.4f}  (BEST AUC)",
    fontsize=12, fontweight="bold", color="#1F4E79", pad=10
)
ax_q.set_ylabel("True Label", fontsize=12)
ax_q.tick_params(labelsize=11)

qfe_stats = f"TP={tp}  TN={tn}  FP={fp}  FN={fn}   Recall={svm_rec:.2%}"
ax_q.set_xlabel(f"Predicted Label\n{qfe_stats}", fontsize=11)

plt.tight_layout()
plt.savefig("results/ppt_comparison_confusion.png",
            dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: results/ppt_comparison_confusion.png")

# ─── PLOT 3: ROC CURVE (PPT quality, all key models) ─────────────
print("\nGenerating ROC curve for PPT...")

fig3, ax3 = plt.subplots(figsize=(9, 7))

# Random baseline
ax3.plot([0,1],[0,1], "k--", lw=1.5,
         label="Random Baseline (AUC = 0.500)", alpha=0.5)

# All key models
models_roc = [
    ("LightGBM (23 feat) — Best Accuracy",
     lgb_proba, "#2CA02C", "-", 2.0),
    ("XGBoost (23 feat)",
     xgb_proba, "#FF7F0E", "--", 1.8),
    ("QFE + SVM (30 feat) — Best AUC",
     svm_proba, "#1F4E79", "-", 3.0),
]

for name, proba, color, ls, lw in models_roc:
    fpr_m, tpr_m, _ = roc_curve(y_test, proba)
    auc_m = roc_auc_score(y_test, proba)
    ax3.plot(fpr_m, tpr_m, color=color, linestyle=ls,
             linewidth=lw, label=f"{name}  (AUC = {auc_m:.4f})")

# Mark the operating point of QFE+SVM at threshold 0.5
svm_tn, svm_fp, svm_fn, svm_tp = svm_cm.ravel()
op_tpr = svm_tp / (svm_tp + svm_fn)
op_fpr = svm_fp / (svm_fp + svm_tn)
ax3.plot(op_fpr, op_tpr, "o", color="#1F4E79", markersize=12,
         label=f"QFE+SVM operating point (thresh=0.5)\n"
               f"TPR={op_tpr:.2%}  FPR={op_fpr:.2%}",
         zorder=5)

ax3.set_xlim([-0.01, 1.01])
ax3.set_ylim([-0.01, 1.05])
ax3.set_xlabel("False Positive Rate\n(1 - Specificity)", fontsize=13)
ax3.set_ylabel("True Positive Rate\n(Sensitivity / Recall)", fontsize=13)
ax3.set_title(
    "ROC Curves — Quantum Feature Extractor vs Classical Models\n"
    "HemoPI-2 Hemolytic Peptide Classification (202 Test Peptides)",
    fontsize=13, fontweight="bold", color="#1F4E79"
)
ax3.legend(loc="lower right", fontsize=10, framealpha=0.95,
           edgecolor="#CCCCCC")
ax3.grid(True, alpha=0.3, linestyle="--")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# Annotation box
ax3.annotate(
    "QFE+SVM has the\nHIGHEST AUC (0.8593)\namong all models",
    xy=(0.25, 0.82), xytext=(0.45, 0.60),
    fontsize=10, color="#1F4E79", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#1F4E79", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#D6E4F0",
              edgecolor="#2E75B6", linewidth=1.5)
)

plt.tight_layout()
plt.savefig("results/ppt_roc_curve.png",
            dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: results/ppt_roc_curve.png")

# ─── FINAL PRINT ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  QFE + SVM FINAL NUMBERS (for your PPT)")
print("=" * 60)
print(f"""
  Accuracy   : {svm_acc*100:.2f}%
  Recall     : {svm_rec*100:.2f}%  (catches {tp} out of {tp+fn} hemolytic peptides)
  Precision  : {svm_pre*100:.2f}%
  F1 Score   : {svm_f1*100:.2f}%
  AUC        : {svm_auc:.4f}  <- BEST of all models

  Confusion matrix:
    TP = {tp}  (correctly predicted hemolytic)
    TN = {tn}  (correctly predicted non-hemolytic)
    FP = {fp}  (predicted hemolytic but actually not)
    FN = {fn}  (missed — predicted not hemolytic but actually was)

  3 files saved for your PPT:
    results/ppt_qfe_confusion_matrix.png  (single, large)
    results/ppt_comparison_confusion.png  (LightGBM vs QFE side by side)
    results/ppt_roc_curve.png             (ROC with all models)
""")