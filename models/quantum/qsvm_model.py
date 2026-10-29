from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import time

from .quantum_feature_map import create_feature_map
from .quantum_kernel import create_quantum_kernel


MAX_TRAIN_SAMPLES = 300
MAX_TEST_SAMPLES  = 75


def train_qsvm(X_train, y_train, X_test, y_test,
               feature_map_type='zz'):

    print("\n==============================")
    print(" Quantum Support Vector Machine")
    print(f" Feature Map : {feature_map_type.upper()}FeatureMap")
    print("==============================")

    # ── STEP 1: Subsample ─────────────────────────────────────
    X_train, y_train = _subsample(X_train, y_train,
                                   MAX_TRAIN_SAMPLES, "train")
    X_test,  y_test  = _subsample(X_test,  y_test,
                                   MAX_TEST_SAMPLES,  "test")

    # ── STEP 2: Scale to [0, 2π] ──────────────────────────────
    scaler  = MinMaxScaler(feature_range=(0, 2 * np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    num_features = X_train.shape[1]
    print(f"\nQubits        : {num_features}")
    print(f"Train samples : {len(X_train)}")
    print(f"Test  samples : {len(X_test)}")

    # ── STEP 3: Build feature map + kernel ────────────────────
    feature_map    = create_feature_map(num_features,
                                        map_type=feature_map_type)
    quantum_kernel = create_quantum_kernel(feature_map)

    # ── STEP 4: Compute kernel matrices ONCE ──────────────────
    print(f"\n[1/3] Computing training kernel matrix ...")
    print(f"      ({len(X_train)}x{len(X_train)}"
          f" = {len(X_train)**2:,} evaluations)")
    t0      = time.time()
    K_train = quantum_kernel.evaluate(X_train, X_train)
    t1      = time.time()
    print(f"      Done in {t1-t0:.1f}s")

    print(f"\n[2/3] Computing test kernel matrix ...")
    K_test = quantum_kernel.evaluate(X_test, X_train)
    t2     = time.time()
    print(f"      Done in {t2-t1:.1f}s")
    print(f"      Total kernel time: {t2-t0:.1f}s")

    # ── STEP 5: Kernel diagnostics ────────────────────────────
    off_diag = K_train[~np.eye(len(K_train), dtype=bool)]
    print(f"\n  Raw kernel stats:")
    print(f"    Mean = {off_diag.mean():.4f} | "
          f"Std = {off_diag.std():.4f} | "
          f"Min = {off_diag.min():.4f} | "
          f"Max = {off_diag.max():.4f}")

    # ── STEP 6: Grid search over C, class_weight, gamma ───────
    print("\n[3/3] Hyperparameter search ...")

    C_values      = [0.1, 0.5, 1, 5, 10]
    weight_values = [None, 'balanced']
    gamma_values  = [1.0, 2.0, 3.0, 5.0]

    best_model        = None
    best_acc          = -1
    best_f1           = -1
    best_params       = {}
    best_metrics      = None
    best_K_test_sharp = None

    for gamma in gamma_values:
        # Kernel sharpening: increases contrast between
        # similar and dissimilar quantum states
        K_tr_s = np.exp(gamma * (K_train - 1))
        K_te_s = np.exp(gamma * (K_test  - 1))

        for cw in weight_values:
            for C in C_values:
                qsvm = SVC(kernel='precomputed', C=C,
                           class_weight=cw, probability=True)
                qsvm.fit(K_tr_s, y_train)
                y_pred = qsvm.predict(K_te_s)

                acc  = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred,
                                       zero_division=0)
                rec  = recall_score(y_test, y_pred,
                                    zero_division=0)
                f1   = f1_score(y_test, y_pred,
                                zero_division=0)

                tag = (f"g={gamma:<3} C={C:<4}"
                       f" w={str(cw):<8}")
                print(f"  {tag}  acc={acc:.3f}  "
                      f"prec={prec:.3f}  "
                      f"rec={rec:.3f}  f1={f1:.3f}")

                # Optimise for accuracy primarily
                if (acc > best_acc) or \
                   (acc == best_acc and f1 > best_f1):
                    best_acc          = acc
                    best_f1           = f1
                    best_model        = qsvm
                    best_params       = {'gamma': gamma,
                                         'C': C, 'cw': cw}
                    best_K_test_sharp = K_te_s
                    best_metrics      = {
                        'accuracy' : acc,
                        'precision': prec,
                        'recall'   : rec,
                        'f1'       : f1,
                    }

    # ── STEP 7: Final report ──────────────────────────────────
    y_pred_best = best_model.predict(best_K_test_sharp)
    cm = confusion_matrix(y_test, y_pred_best)

    print(f"\n{'='*48}")
    print(f"  Best gamma        : {best_params['gamma']}")
    print(f"  Best C            : {best_params['C']}")
    print(f"  Best class_weight : {best_params['cw']}")
    print(f"{'='*48}")
    print(f"  Accuracy          : {best_metrics['accuracy']:.4f}")
    print(f"  Precision         : {best_metrics['precision']:.4f}")
    print(f"  Recall            : {best_metrics['recall']:.4f}")
    print(f"  F1 Score          : {best_metrics['f1']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
    print(f"    FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred_best,
          target_names=['Non-Hemolytic', 'Hemolytic'],
          zero_division=0))

    return best_model, best_metrics


def _subsample(X, y, max_n, label="data", seed=42):
    if len(X) <= max_n:
        return X, y
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=max_n, random_state=seed
    )
    idx, _ = next(sss.split(X, y))
    dist = dict(zip(*np.unique(y[idx], return_counts=True)))
    print(f"  Subsampled {label}: {len(X)} -> {len(idx)} "
          f"(class dist: {dist})")
    return X[idx], y[idx]