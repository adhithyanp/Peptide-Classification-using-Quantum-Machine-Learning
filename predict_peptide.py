"""
=============================================================
  HEMOLYTIC PEPTIDE PREDICTOR
  Enter any peptide sequence → get Hemolytic / Non-Hemolytic
=============================================================

This script:
  1. Trains the best model (Classical + Quantum SVM, 30 features)
  2. Lets you enter any peptide sequence
  3. Predicts whether it is Hemolytic or Non-Hemolytic
  4. Shows confidence score and which features drove the prediction

Usage:
    python predict_peptide.py

Requirements:
    - Run quantum_feature_extractor.py at least once first
      (so that results/Q_tr.npy and results/Q_te.npy exist)
    - Same qsvm_env environment
=============================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# CHECK FILES EXIST
# ─────────────────────────────────────────────────────────
required_files = [
    "data/processed/train_features.csv",
    "data/processed/test_features.csv",
    "results/Q_tr.npy",
    "results/Q_te.npy"
]
for f in required_files:
    if not os.path.exists(f):
        print(f"\n[ERROR] Missing file: {f}")
        print("Please run quantum_feature_extractor.py first.")
        sys.exit(1)

# ─────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import lightgbm as lgb

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

# ─────────────────────────────────────────────────────────
# THE 20 STANDARD AMINO ACIDS
# ─────────────────────────────────────────────────────────
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydrophobicity scale
KD_SCALE = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5,
    'F': 2.8,  'G': -0.4, 'H': -3.2, 'I': 4.5,
    'K': -3.9, 'L': 3.8,  'M': 1.9,  'N': -3.5,
    'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8,
    'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3
}

# Charge values
CHARGE = {
    'K': +1, 'R': +1, 'H': +0.5,
    'D': -1, 'E': -1
}

# ─────────────────────────────────────────────────────────
# FEATURE EXTRACTION FROM SEQUENCE
# ─────────────────────────────────────────────────────────
def extract_features_from_sequence(seq):
    """
    Convert a peptide sequence string into 23 numerical features.
    Same logic as the original feature extraction code.
    Returns: numpy array of shape (23,)
    """
    seq = seq.upper().strip()
    n = len(seq)

    if n == 0:
        return None, "Empty sequence"

    # Validate — only standard amino acids allowed
    invalid = [c for c in seq if c not in AMINO_ACIDS]
    if invalid:
        return None, f"Invalid characters found: {list(set(invalid))}. Use only standard amino acid letters."

    # ── AAC: 20 features ──────────────────────────────
    aac = []
    for aa in AMINO_ACIDS:
        freq = seq.count(aa) / n
        aac.append(freq)

    # ── PCP: 3 features ───────────────────────────────
    # Net charge
    net_charge = sum(CHARGE.get(aa, 0) for aa in seq)

    # Positive/Negative ratio
    pos_count = sum(1 for aa in seq if aa in ['K', 'R', 'H'])
    neg_count = sum(1 for aa in seq if aa in ['D', 'E'])
    if neg_count == 0:
        pos_neg_ratio = float(pos_count) if pos_count > 0 else 0.0
    else:
        pos_neg_ratio = pos_count / neg_count

    # Hydrophobicity (Kyte-Doolittle average)
    hydrophobicity = sum(KD_SCALE.get(aa, 0) for aa in seq) / n

    pcp = [net_charge, pos_neg_ratio, hydrophobicity]

    # Combine all 23 features
    features = np.array(aac + pcp, dtype=float)
    return features, None


# ─────────────────────────────────────────────────────────
# BUILD QUANTUM CIRCUIT (same as quantum_feature_extractor.py)
# ─────────────────────────────────────────────────────────
def build_quantum_circuit():
    n_qubits = 4
    feature_params = ParameterVector("x", n_qubits)

    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(feature_params[i], i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.cx(n_qubits - 1, 0)
    for i in range(n_qubits):
        qc.rx(feature_params[i] * 2, i)
    for i in range(n_qubits - 1, 0, -1):
        qc.cx(i, i - 1)

    # 7 observables: 4 single-qubit Z + 3 nearest-neighbor ZZ
    observables_Z = [
        SparsePauliOp("I" * (n_qubits - 1 - i) + "Z" + "I" * i)
        for i in range(n_qubits)
    ]
    observables_ZZ = [
        SparsePauliOp("I" * (n_qubits - 1 - i) + "Z" + "I" * (i - j - 1) + "Z" + "I" * j)
        for i in range(1, n_qubits) for j in range(i)
        if i - j == 1
    ]
    all_observables = observables_Z + observables_ZZ[:3]

    return qc, list(feature_params), all_observables


def get_quantum_features(x_scaled, qc, params, observables):
    """
    Run one sample through the quantum circuit.
    Returns 7 quantum measurement values.
    """
    estimator = StatevectorEstimator()
    param_dict = dict(zip(params, x_scaled))
    bound_qc = qc.assign_parameters(param_dict)
    pub = [(bound_qc, obs) for obs in observables]
    job = estimator.run(pub)
    results = job.result()
    return np.array([float(results[j].data.evs) for j in range(len(observables))])


# ─────────────────────────────────────────────────────────
# TRAIN THE MODEL
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("  HEMOLYTIC PEPTIDE PREDICTOR")
print("  Loading data and training model...")
print("=" * 60)

# Load data
train_df = pd.read_csv("data/processed/train_features.csv")
test_df  = pd.read_csv("data/processed/test_features.csv")

X_train = train_df.drop(columns=["label"]).values
y_train = train_df["label"].values
X_test  = test_df.drop(columns=["label"]).values
y_test  = test_df["label"].values

feature_names = train_df.drop(columns=["label"]).columns.tolist()

# Load pre-computed quantum features
Q_train = np.load("results/Q_tr.npy")
Q_test  = np.load("results/Q_te.npy")

print(f"  Train samples : {len(X_train)}")
print(f"  Test samples  : {len(X_test)}")
print(f"  Classical features : {X_train.shape[1]}")
print(f"  Quantum features   : {Q_train.shape[1]}")

# Scale classical features
scaler_classical = MinMaxScaler()
X_train_scaled = scaler_classical.fit_transform(X_train)
X_test_scaled  = scaler_classical.transform(X_test)

# Combine classical + quantum = 30 features
X_train_combined = np.hstack([X_train_scaled, Q_train])
X_test_combined  = np.hstack([X_test_scaled,  Q_test])

# Train SVM on combined features
print("\n  Training Classical+Quantum SVM (30 features)...")
model = SVC(C=1, kernel="rbf", probability=True,
            class_weight="balanced", random_state=42)
model.fit(X_train_combined, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train_combined))
test_acc  = accuracy_score(y_test,  model.predict(X_test_combined))
print(f"  Train accuracy : {train_acc:.4f}")
print(f"  Test accuracy  : {test_acc:.4f}  (expected ~0.7822)")

# Get top-4 features used by quantum circuit
lgb_model = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)
importances = lgb_model.feature_importances_
top4_idx    = np.argsort(importances)[::-1][:4]
top4_names  = [feature_names[i] for i in top4_idx]

# Scaler for quantum circuit input (scale to [0, pi])
scaler_quantum = MinMaxScaler(feature_range=(0, np.pi))
scaler_quantum.fit(X_train[:, top4_idx])

# Build quantum circuit
qc, params, observables = build_quantum_circuit()

print(f"\n  Top-4 features for quantum circuit: {top4_names}")
print(f"\n  Model ready.")
print("=" * 60)

# ─────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────
def predict_peptide(sequence):
    """
    Full pipeline: sequence → features → quantum features → prediction
    """
    seq = sequence.upper().strip()

    # Step 1: Extract 23 classical features
    features_23, error = extract_features_from_sequence(seq)
    if error:
        return None, error, None

    # Step 2: Scale classical features
    features_23_scaled = scaler_classical.transform(features_23.reshape(1, -1))[0]

    # Step 3: Extract top-4 features for quantum circuit
    features_4 = features_23[top4_idx]
    features_4_scaled = scaler_quantum.transform(features_4.reshape(1, -1))[0]

    # Step 4: Get 7 quantum features
    print("  Computing quantum features...", end="\r")
    q_features = get_quantum_features(features_4_scaled, qc, params, observables)
    print("  Quantum features computed.   ")

    # Step 5: Combine 23 + 7 = 30 features
    combined = np.hstack([features_23_scaled, q_features]).reshape(1, -1)

    # Step 6: Predict
    prediction   = model.predict(combined)[0]
    probability  = model.predict_proba(combined)[0]
    confidence   = probability[int(prediction)]

    # Feature details for explanation
    details = {
        "hydrophobicity": round(features_23[feature_names.index("hydrophobicity")], 4),
        "net_charge":     round(features_23[feature_names.index("net_charge")], 4),
        "freq_K":         round(features_23[feature_names.index("K")], 4),
        "freq_R":         round(features_23[feature_names.index("R")], 4),
        "freq_W":         round(features_23[feature_names.index("W")], 4),
        "q_features":     [round(float(v), 4) for v in q_features],
        "prob_hemolytic":     round(float(probability[1]), 4),
        "prob_nonhemolytic":  round(float(probability[0]), 4),
    }

    return int(prediction), None, details


# ─────────────────────────────────────────────────────────
# DISPLAY RESULT
# ─────────────────────────────────────────────────────────
def display_result(seq, prediction, details):
    print("\n" + "=" * 60)
    print(f"  PREDICTION RESULT")
    print("=" * 60)
    print(f"  Sequence : {seq.upper()}")
    print(f"  Length   : {len(seq)} amino acids")
    print()

    if prediction == 1:
        print("  VERDICT:  *** HEMOLYTIC ***")
        print("  This peptide is predicted to DESTROY red blood cells.")
        print("  NOT safe for therapeutic use without modification.")
    else:
        print("  VERDICT:  *** NON-HEMOLYTIC ***")
        print("  This peptide is predicted to be SAFE for red blood cells.")

    print()
    print(f"  Probability Hemolytic     : {details['prob_hemolytic']:.1%}")
    print(f"  Probability Non-Hemolytic : {details['prob_nonhemolytic']:.1%}")
    print()
    print("  KEY FEATURES:")
    print(f"    Hydrophobicity : {details['hydrophobicity']:+.4f}  "
          f"({'high - helps membrane insertion' if details['hydrophobicity'] > 0 else 'low - less membrane affinity'})")
    print(f"    Net Charge     : {details['net_charge']:+.4f}  "
          f"({'positive - attracted to membrane' if details['net_charge'] > 0 else 'negative - repelled from membrane'})")
    print(f"    Lysine (K) %   : {details['freq_K']*100:.1f}%")
    print(f"    Arginine (R) % : {details['freq_R']*100:.1f}%")
    print(f"    Tryptophan (W)%: {details['freq_W']*100:.1f}%")
    print()
    print(f"  QUANTUM FEATURES (7 measurements from circuit):")
    qf = details['q_features']
    labels = ["Z-qubit1 (hydro)", "Z-qubit2 (L)", "Z-qubit3 (K)",
              "Z-qubit4 (G)", "ZZ-qubits1+2", "ZZ-qubits2+3", "ZZ-qubits3+4"]
    for label, val in zip(labels, qf):
        bar_len = int((val + 1) / 2 * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {label:<20}: {val:+.4f}  [{bar}]")
    print("=" * 60)


# ─────────────────────────────────────────────────────────
# EXAMPLE PREDICTIONS (run automatically on startup)
# ─────────────────────────────────────────────────────────
print("\n  Running 2 example predictions from the dataset...")

examples = [
    ("KWLRRVWRWWR",   "Known hemolytic peptide from training set"),
    ("RWKIFKKIEKVGQNIRDGIVKAGPAVAV", "Known non-hemolytic peptide from training set"),
]

for seq, description in examples:
    print(f"\n  Example: {description}")
    pred, error, details = predict_peptide(seq)
    if error:
        print(f"  Error: {error}")
    else:
        display_result(seq, pred, details)

# ─────────────────────────────────────────────────────────
# INTERACTIVE PREDICTION LOOP
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  INTERACTIVE PREDICTOR")
print("  Enter any peptide sequence to predict hemolytic activity.")
print("  Type 'quit' or 'exit' to stop.")
print("=" * 60)
print()
print("  Valid amino acid letters:")
print("  A C D E F G H I K L M N P Q R S T V W Y")
print()

while True:
    try:
        user_input = input("  Enter peptide sequence: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n  Exiting predictor. Goodbye!")
        break

    if user_input.lower() in ["quit", "exit", "q", ""]:
        print("\n  Exiting predictor. Goodbye!")
        break

    if len(user_input) < 4:
        print("  Sequence too short. Please enter at least 4 amino acids.")
        continue

    if len(user_input) > 200:
        print("  Sequence too long (max 200). Please enter a shorter sequence.")
        continue

    pred, error, details = predict_peptide(user_input)

    if error:
        print(f"\n  [ERROR] {error}")
        print("  Please try again with valid amino acid letters (A C D E F G H I K L M N P Q R S T V W Y)")
    else:
        display_result(user_input, pred, details)

    print()