"""
reviewer_ablation.py
====================
Experiments requested by the reviewers of ICCSAI paper 2930.

PART A - Official split (812 train / 202 held-out test)
  * every model is evaluated on the SAME 202 test peptides
  * ablation of the quantum features (variants below)
  * 95% bootstrap confidence intervals
  * paired tests: McNemar (accuracy) and paired bootstrap (accuracy, AUC)

PART B - Repeated stratified k-fold CV on all 1014 peptides
  * the WHOLE pipeline (scaling, top-4 selection, QFE, stacking) is re-fitted
    inside every fold -> no label leakage
  * Nadeau-Bengio corrected paired t-test across folds

Ablation variants (what the second base model, an SVM, receives):
  classical only (23)           no extra features
  +7 random features            noise control, same dimension as QFE
  +7 shuffled QFE               real QFE values but rows shuffled
  +7 no-entanglement circuit    same gates WITHOUT CNOTs = classical nonlinear
                                transform of the same 4 features
  +7 QFE (entangled)            proposed method

All labels come from probability >= 0.5 (SVM numbers can differ by a peptide
or two from older runs that used svm.predict()).

Run (PowerShell):   python reviewer_ablation.py
Quick test:         $env:N_REPEATS=1; $env:N_BOOT=300; python reviewer_ablation.py
Needs: data/processed/train_features.csv and test_features.csv
"""

import os, json, time, platform, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             roc_curve)
import lightgbm as lgb
import xgboost as xgb
import sklearn, qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

N_REPEATS = int(os.environ.get("N_REPEATS", 3))
N_FOLDS   = int(os.environ.get("N_FOLDS", 5))
N_BOOT    = int(os.environ.get("N_BOOT", 2000))
INNER     = 5
SEED0     = 42

V_NONE  = "classical only (23)"
V_RAND  = "+7 random features"
V_SHUF  = "+7 shuffled QFE"
V_NOENT = "+7 no-entanglement circuit"
V_QFE   = "+7 QFE (entangled, proposed)"
VARIANTS = [V_NONE, V_RAND, V_SHUF, V_NOENT, V_QFE]

OPS = [SparsePauliOp(s) for s in
       ["IIIZ", "IIZI", "IZII", "ZIII", "IIZZ", "IZZI", "ZZII"]]


# ════════════════════════════════════════════════════════════════
# MODELS AND FEATURES
# ════════════════════════════════════════════════════════════════
def make_lgb():
    return lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                              random_state=42, verbose=-1)


def make_svm():
    return SVC(kernel="rbf", C=1.0, probability=True,
               class_weight="balanced", random_state=42)


def build_circuit(angles, entangle=True):
    """RY -> [CNOT ring] -> RX(2x) -> [CNOT ring] -> RY   (CNOTs optional)"""
    qc = QuantumCircuit(4)
    for i, a in enumerate(angles):
        qc.ry(float(a), i)
    if entangle:
        for i in range(4):
            qc.cx(i, (i + 1) % 4)
    for i, a in enumerate(angles):
        qc.rx(2.0 * float(a), i)
    if entangle:
        for i in range(4):
            qc.cx(i, (i + 1) % 4)
    for i, a in enumerate(angles):
        qc.ry(float(a), i)
    return qc


def qfe(angle_matrix, entangle=True):
    out = []
    for angles in angle_matrix:
        sv = Statevector(build_circuit(angles, entangle))
        out.append([sv.expectation_value(op).real for op in OPS])
    return np.array(out)


def prepare(X_fit, y_fit, X_oth, seed):
    """Everything that learns from data is fitted on X_fit/y_fit ONLY."""
    sc = MinMaxScaler().fit(X_fit)
    Xc_f, Xc_o = sc.transform(X_fit), sc.transform(X_oth)

    selector = make_lgb().fit(Xc_f, y_fit)
    top4 = np.argsort(selector.feature_importances_)[::-1][:4]

    scq = MinMaxScaler(feature_range=(0, np.pi)).fit(X_fit[:, top4])
    A_f, A_o = scq.transform(X_fit[:, top4]), scq.transform(X_oth[:, top4])

    Qf, Qo = qfe(A_f, True), qfe(A_o, True)
    Nf, No = qfe(A_f, False), qfe(A_o, False)

    rng = np.random.default_rng(seed)
    blocks = {
        V_NONE:  (np.empty((len(Xc_f), 0)), np.empty((len(Xc_o), 0))),
        V_RAND:  (rng.uniform(-1, 1, (len(Xc_f), 7)),
                  rng.uniform(-1, 1, (len(Xc_o), 7))),
        V_SHUF:  (Qf[rng.permutation(len(Qf))], Qo[rng.permutation(len(Qo))]),
        V_NOENT: (Nf, No),
        V_QFE:   (Qf, Qo),
    }
    return Xc_f, Xc_o, blocks, top4


def run_models(X_fit, y_fit, X_oth, seed):
    """Fit everything on (X_fit, y_fit); return probabilities for X_oth."""
    Xc_f, Xc_o, blocks, top4 = prepare(X_fit, y_fit, X_oth, seed)
    spw = (y_fit == 0).sum() / (y_fit == 1).sum()
    P = {}

    # classical baselines (23 features)
    P["LightGBM (23)"] = make_lgb().fit(Xc_f, y_fit).predict_proba(Xc_o)[:, 1]
    P["XGBoost (23)"] = xgb.XGBClassifier(
        n_estimators=300, random_state=42, eval_metric="logloss",
        verbosity=0, scale_pos_weight=spw
    ).fit(Xc_f, y_fit).predict_proba(Xc_o)[:, 1]
    P["Random Forest (23)"] = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    ).fit(Xc_f, y_fit).predict_proba(Xc_o)[:, 1]
    P["Logistic Regression (23)"] = LogisticRegression(
        max_iter=3000, class_weight="balanced"
    ).fit(Xc_f, y_fit).predict_proba(Xc_o)[:, 1]

    # single SVMs with each variant of extra features
    svm_p = {}
    for v, (Ef, Eo) in blocks.items():
        m = make_svm().fit(np.hstack([Xc_f, Ef]), y_fit)
        svm_p[v] = m.predict_proba(np.hstack([Xc_o, Eo]))[:, 1]
        P[f"SVM | {v}"] = svm_p[v]

    # leak-free stacking: inner out-of-fold predictions
    skf = StratifiedKFold(INNER, shuffle=True, random_state=seed)
    oof_l = np.zeros(len(y_fit))
    oof_s = {v: np.zeros(len(y_fit)) for v in VARIANTS}
    for tr, va in skf.split(X_fit, y_fit):
        a_f, a_o, bl, _ = prepare(X_fit[tr], y_fit[tr], X_fit[va], seed)
        oof_l[va] = make_lgb().fit(a_f, y_fit[tr]).predict_proba(a_o)[:, 1]
        for v, (Ef, Eo) in bl.items():
            m = make_svm().fit(np.hstack([a_f, Ef]), y_fit[tr])
            oof_s[v][va] = m.predict_proba(np.hstack([a_o, Eo]))[:, 1]

    for v in VARIANTS:
        meta = LogisticRegression(random_state=42).fit(
            np.column_stack([oof_l, oof_s[v]]), y_fit)
        P[f"Stack | {v}"] = meta.predict_proba(
            np.column_stack([P["LightGBM (23)"], svm_p[v]]))[:, 1]
    return P, top4


# ════════════════════════════════════════════════════════════════
# METRICS AND STATISTICS
# ════════════════════════════════════════════════════════════════
def metrics_from_proba(y, p):
    pred = (p >= 0.5).astype(int)
    return {"Accuracy": accuracy_score(y, pred),
            "Recall": recall_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred),
            "MCC": matthews_corrcoef(y, pred),
            "AUC": roc_auc_score(y, p)}


def mcnemar_exact(y, pa, pb):
    a, b = (pa >= 0.5).astype(int), (pb >= 0.5).astype(int)
    only_a = int(((a == y) & (b != y)).sum())
    only_b = int(((a != y) & (b == y)).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else stats.binomtest(min(only_a, only_b), n, 0.5).pvalue
    return only_a, only_b, p


def boot_p(d):
    d = np.asarray(d)
    return min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def bootstrap(y, P, names, pairs, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    pred = {m: (P[m] >= 0.5).astype(int) for m in names}
    acc = {m: [] for m in names}
    auc_ = {m: [] for m in names}
    d_acc = {pr: [] for pr in pairs}
    d_auc = {pr: [] for pr in pairs}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        a_b = {m: float((pred[m][idx] == yb).mean()) for m in names}
        u_b = {m: roc_auc_score(yb, P[m][idx]) for m in names}
        for m in names:
            acc[m].append(a_b[m])
            auc_[m].append(u_b[m])
        for pr in pairs:
            d_acc[pr].append(a_b[pr[0]] - a_b[pr[1]])
            d_auc[pr].append(u_b[pr[0]] - u_b[pr[1]])
    return acc, auc_, d_acc, d_auc


def nadeau_bengio(d, n_train, n_test):
    d = np.asarray(d, dtype=float)
    k = len(d)
    var = d.var(ddof=1) if k > 1 else 0.0
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * var)
    return t, 2 * stats.t.sf(abs(t), k - 1)


STACK = lambda v: f"Stack | {v}"
PAIRS = [
    (STACK(V_QFE), "LightGBM (23)"),
    (STACK(V_QFE), STACK(V_NONE)),
    (STACK(V_QFE), STACK(V_RAND)),
    (STACK(V_QFE), STACK(V_SHUF)),
    (STACK(V_QFE), STACK(V_NOENT)),
    (f"SVM | {V_QFE}", f"SVM | {V_NONE}"),
]
KEY = ["LightGBM (23)", STACK(V_NONE), STACK(V_RAND),
       STACK(V_NOENT), STACK(V_QFE)]

# ════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════
print("=" * 70)
print("  REVIEWER-REQUESTED EXPERIMENTS")
print("=" * 70)
train_df = pd.read_csv("data/processed/train_features.csv")
test_df = pd.read_csv("data/processed/test_features.csv")
feature_cols = [c for c in train_df.columns if c != "label"]
X_tr, y_tr = train_df[feature_cols].values, train_df["label"].values
X_te, y_te = test_df[feature_cols].values, test_df["label"].values
print(f"  Train {X_tr.shape} | Test {X_te.shape} | "
      f"features: {len(feature_cols)}")

# ════════════════════════════════════════════════════════════════
# PART A - OFFICIAL SPLIT
# ════════════════════════════════════════════════════════════════
print("\n[PART A]  Official split: fit on 812, evaluate on the 202 held-out")
t0 = time.time()
P, top4_final = run_models(X_tr, y_tr, X_te, seed=SEED0)
top4_names = [feature_cols[i] for i in top4_final]
print(f"  Models fitted in {time.time() - t0:.0f}s | "
      f"top-4 features (qubit order): {top4_names}")

names = list(P.keys())
acc_b, auc_b, d_acc, d_auc = bootstrap(y_te, P, names, PAIRS, N_BOOT)

rows = []
for m in names:
    r = {"Model": m, **metrics_from_proba(y_te, P[m])}
    r["Acc_CI_low"], r["Acc_CI_high"] = np.percentile(acc_b[m], [2.5, 97.5])
    r["AUC_CI_low"], r["AUC_CI_high"] = np.percentile(auc_b[m], [2.5, 97.5])
    rows.append(r)
dfA = pd.DataFrame(rows)
dfA.to_csv("results/reviewer_A_official_split.csv", index=False)

print(f"\n  All models on the SAME {len(y_te)} test peptides "
      f"(95% bootstrap CI, {N_BOOT} resamples)")
print(f"  {'Model':<42} {'Acc':>6} {'[95% CI]':>14} {'AUC':>6} "
      f"{'[95% CI]':>14} {'Recall':>6}")
print("  " + "-" * 94)
for _, r in dfA.iterrows():
    print(f"  {r['Model']:<42} {r['Accuracy']:>6.3f} "
          f"[{r['Acc_CI_low']:.3f},{r['Acc_CI_high']:.3f}] "
          f"{r['AUC']:>6.3f} [{r['AUC_CI_low']:.3f},{r['AUC_CI_high']:.3f}] "
          f"{r['Recall']:>6.3f}")

prow = []
print("\n  Paired comparisons on the official test set")
print(f"  {'A  vs  B':<70}{'dAcc':>7}{'p(McN)':>8}{'dAUC':>8}"
      f"{'95% CI':>18}{'p(boot)':>9}")
print("  " + "-" * 120)
for pr in PAIRS:
    a, b = pr
    dacc = metrics_from_proba(y_te, P[a])["Accuracy"] - \
        metrics_from_proba(y_te, P[b])["Accuracy"]
    dauc = roc_auc_score(y_te, P[a]) - roc_auc_score(y_te, P[b])
    oa, ob, p_mc = mcnemar_exact(y_te, P[a], P[b])
    lo, hi = np.percentile(d_auc[pr], [2.5, 97.5])
    p_auc = boot_p(d_auc[pr])
    prow.append({"A": a, "B": b, "dAcc": dacc, "McNemar_only_A_correct": oa,
                 "McNemar_only_B_correct": ob, "McNemar_p": p_mc,
                 "dAUC": dauc, "dAUC_CI_low": lo, "dAUC_CI_high": hi,
                 "dAUC_boot_p": p_auc, "dAcc_boot_p": boot_p(d_acc[pr])})
    print(f"  {a + '  vs  ' + b:<70}{dacc:>+7.3f}{p_mc:>8.3f}{dauc:>+8.4f}"
          f"  [{lo:+.4f},{hi:+.4f}]{p_auc:>9.3f}")
pd.DataFrame(prow).to_csv("results/reviewer_A_pairwise_tests.csv", index=False)

# plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
cols = ["#2E75B6", "#7F8C8D", "#E67E22", "#27AE60", "#8E44AD"]
for m, c in zip(KEY, cols):
    fpr, tpr, _ = roc_curve(y_te, P[m])
    ax.plot(fpr, tpr, color=c, lw=2.2,
            label=f"{m} (AUC={roc_auc_score(y_te, P[m]):.3f})")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC - ablation of quantum features (202 test peptides)",
             fontweight="bold")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
sub = dfA.set_index("Model").loc[KEY]
err = np.vstack([sub["AUC"] - sub["AUC_CI_low"], sub["AUC_CI_high"] - sub["AUC"]])
ax.barh(range(len(KEY)), sub["AUC"], xerr=err, color=cols, alpha=0.85,
        capsize=4)
ax.set_yticks(range(len(KEY)))
ax.set_yticklabels(KEY, fontsize=8)
ax.set_xlim(0.7, 0.95)
ax.invert_yaxis()
ax.set_xlabel("AUC with 95% bootstrap CI")
ax.set_title("AUC with confidence intervals", fontweight="bold")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("results/reviewer_A_ablation.png", dpi=220, bbox_inches="tight",
            facecolor="white")
plt.close()
print("\n  Saved: results/reviewer_A_official_split.csv")
print("  Saved: results/reviewer_A_pairwise_tests.csv")
print("  Saved: results/reviewer_A_ablation.png")

# ════════════════════════════════════════════════════════════════
# PART B - REPEATED STRATIFIED K-FOLD ON ALL 1014 PEPTIDES
# ════════════════════════════════════════════════════════════════
print(f"\n[PART B]  Repeated {N_FOLDS}-fold CV on all "
      f"{len(y_tr) + len(y_te)} peptides  ({N_REPEATS} repeats)")
X_all = np.vstack([X_tr, X_te])
y_all = np.concatenate([y_tr, y_te])

fold_rows, nb_sizes = [], []
t0, done, total = time.time(), 0, N_REPEATS * N_FOLDS
for rep in range(N_REPEATS):
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=100 + rep)
    for fold, (tr, te) in enumerate(skf.split(X_all, y_all)):
        Pf, _ = run_models(X_all[tr], y_all[tr], X_all[te],
                           seed=1000 * rep + fold)
        for m, p in Pf.items():
            fold_rows.append({"Repeat": rep, "Fold": fold, "Model": m,
                              **metrics_from_proba(y_all[te], p)})
        nb_sizes.append((len(tr), len(te)))
        done += 1
        el = time.time() - t0
        print(f"  fold {done}/{total} done | elapsed {el / 60:.1f} min | "
              f"ETA {el / done * (total - done) / 60:.1f} min")
        pd.DataFrame(fold_rows).to_csv("results/reviewer_B_cv_folds.csv",
                                       index=False)

dfB = pd.DataFrame(fold_rows)
summ = dfB.groupby("Model")[["Accuracy", "AUC", "Recall", "F1", "MCC"]] \
    .agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ = summ.reset_index()
summ.to_csv("results/reviewer_B_cv_summary.csv", index=False)

print(f"\n  Mean +/- std over {total} outer folds")
print(f"  {'Model':<42} {'Accuracy':>16} {'AUC':>16} {'Recall':>8}")
print("  " + "-" * 86)
for _, r in summ.iterrows():
    print(f"  {r['Model']:<42} {r['Accuracy_mean']:>8.3f}+/-{r['Accuracy_std']:.3f}"
          f" {r['AUC_mean']:>8.3f}+/-{r['AUC_std']:.3f} {r['Recall_mean']:>8.3f}")

n_tr, n_te = np.mean([s[0] for s in nb_sizes]), np.mean([s[1] for s in nb_sizes])
pw = []
print("\n  Paired comparisons across folds (Nadeau-Bengio corrected t-test)")
print(f"  {'A  vs  B':<70}{'dAcc':>8}{'p':>8}{'dAUC':>9}{'p':>8}"
      f"{'AUC wins/ties/losses':>22}")
print("  " + "-" * 125)
for a, b in PAIRS:
    A = dfB[dfB["Model"] == a].sort_values(["Repeat", "Fold"])
    B = dfB[dfB["Model"] == b].sort_values(["Repeat", "Fold"])
    d_a = A["Accuracy"].values - B["Accuracy"].values
    d_u = A["AUC"].values - B["AUC"].values
    _, p_a = nadeau_bengio(d_a, n_tr, n_te)
    _, p_u = nadeau_bengio(d_u, n_tr, n_te)
    w, t_, l = int((d_u > 0).sum()), int((d_u == 0).sum()), int((d_u < 0).sum())
    pw.append({"A": a, "B": b, "mean_dAcc": d_a.mean(), "p_Acc": p_a,
               "mean_dAUC": d_u.mean(), "p_AUC": p_u,
               "AUC_wins": w, "AUC_ties": t_, "AUC_losses": l})
    print(f"  {a + '  vs  ' + b:<70}{d_a.mean():>+8.4f}{p_a:>8.3f}"
          f"{d_u.mean():>+9.4f}{p_u:>8.3f}{f'{w}/{t_}/{l}':>22}")
pd.DataFrame(pw).to_csv("results/reviewer_B_pairwise_tests.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for ax, metric in zip(axes, ["AUC", "Accuracy"]):
    data = [dfB[dfB["Model"] == m][metric].values for m in KEY]
    bp = ax.boxplot(data, vert=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xticklabels([k.replace(" | ", "\n") for k in KEY], fontsize=7)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} over {total} leak-free CV folds", fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("results/reviewer_B_cv_boxplots.png", dpi=220, bbox_inches="tight",
            facecolor="white")
plt.close()
print("\n  Saved: results/reviewer_B_cv_folds.csv / _summary.csv / "
      "_pairwise_tests.csv / _cv_boxplots.png")

# ════════════════════════════════════════════════════════════════
# REPRODUCIBILITY MANIFEST
# ════════════════════════════════════════════════════════════════
manifest = {
    "python": platform.python_version(),
    "scikit-learn": sklearn.__version__, "qiskit": qiskit.__version__,
    "lightgbm": lgb.__version__, "xgboost": xgb.__version__,
    "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
    "features": feature_cols,
    "top4_features_final_model_qubit_order": top4_names,
    "circuit": "RY -> CNOT ring -> RX(2x) -> CNOT ring -> RY, 4 qubits, "
               "statevector simulation (exact, no shot noise)",
    "observables": ["Z0", "Z1", "Z2", "Z3", "Z0Z1", "Z1Z2", "Z2Z3"],
    "angle_scaling": "MinMax to [0, pi], fitted on training part only",
    "classical_scaling": "MinMax to [0, 1], fitted on training part only",
    "lightgbm": {"n_estimators": 300, "class_weight": "balanced",
                 "random_state": 42},
    "svm": {"kernel": "rbf", "C": 1.0, "class_weight": "balanced",
            "probability": True, "random_state": 42},
    "meta_learner": "LogisticRegression (default settings)",
    "stacking": f"{INNER}-fold out-of-fold probabilities; scaling, top-4 "
                "selection and QFE re-fitted inside every fold",
    "decision_threshold": 0.5,
    "official_split_seed": SEED0, "cv_repeats": N_REPEATS,
    "cv_folds": N_FOLDS, "bootstrap_resamples": N_BOOT,
}
with open("results/reviewer_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("  Saved: results/reviewer_manifest.json")

print("\n" + "=" * 70)
print("  HOW TO READ THE P-VALUES")
print("=" * 70)
print("""  p < 0.05  -> difference is statistically supported
  p >= 0.05 -> NOT supported: write 'no statistically significant difference'
  Compare 'Stack | +7 QFE' against 'Stack | classical only (23)',
  '+7 random', and '+7 no-entanglement'. Only if QFE beats those controls can
  you credit the improvement to the quantum features rather than to stacking.
""")
print("  ALL DONE.")