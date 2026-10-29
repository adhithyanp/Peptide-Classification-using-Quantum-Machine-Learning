from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap


def create_feature_map(num_features, map_type='zz'):
    """
    Create a quantum feature map for encoding classical data.

    Args:
        num_features : number of qubits (= number of selected features)
        map_type     : 'zz' (default) or 'pauli' (more expressive)

    Returns:
        Qiskit QuantumCircuit feature map

    ── ZZFeatureMap (map_type='zz') ──────────────────────────────
        The standard benchmark in QML literature (Havlíček et al.,
        Nature 2019). Encodes features using Z and ZZ Pauli operators:

          U(x) = exp(i * sum_j x_j Z_j
                   + i * sum_{j<k} (π-x_j)(π-x_k) Z_j Z_k)

        reps=2, entanglement='linear':
          - 2 encoding layers
          - Only adjacent qubits entangled (5 ZZ gates per rep)
          - Fast to simulate, sufficient expressibility for 6 qubits

    ── PauliFeatureMap (map_type='pauli') ────────────────────────
        Generalization of ZZFeatureMap using additional Pauli X gates.
        paulis=['Z', 'X', 'ZZ'] adds X-basis rotations, giving the
        circuit access to a broader region of Hilbert space.

        WHY THIS MAY HELP:
          ZZFeatureMap only rotates around the Z-axis of the Bloch
          sphere. PauliFeatureMap with X terms also rotates around
          the X-axis, creating richer quantum states that may better
          separate hemolytic from non-hemolytic peptides.

        VIVA ANSWER:
          "I compared ZZFeatureMap and PauliFeatureMap as an ablation
           study. PauliFeatureMap with paulis=['Z','X','ZZ'] explores
           a broader Hilbert space by combining Z and X rotations,
           potentially improving kernel separability for peptide data."
    """
    if map_type == 'zz':
        return ZZFeatureMap(
            feature_dimension=num_features,
            reps=2,
            entanglement='linear'
        )

    elif map_type == 'pauli':
        return PauliFeatureMap(
            feature_dimension=num_features,
            reps=2,
            paulis=['Z', 'X', 'ZZ'],   # richer than ZZFeatureMap
            entanglement='linear'
        )

    else:
        raise ValueError(
            f"Unknown map_type '{map_type}'. Choose 'zz' or 'pauli'."
        )