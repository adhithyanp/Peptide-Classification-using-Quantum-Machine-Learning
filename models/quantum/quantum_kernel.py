from qiskit_machine_learning.kernels import FidelityStatevectorKernel

def create_quantum_kernel(feature_map):
    """
    FidelityStatevectorKernel — Exact quantum kernel for simulation.

    CHANGE FROM ORIGINAL:
      FidelityQuantumKernel   →   FidelityStatevectorKernel

    WHAT IS THE DIFFERENCE?

      FidelityQuantumKernel (your original):
        - Uses a Sampler primitive internally
        - Sampler = shot-based measurement (probabilistic)
        - Each kernel entry K(xi, xj) is estimated from N_shots samples
        - Default shots = 1024 per evaluation
        - For 150×150 kernel: 22,500 evaluations × 1024 shots = 23M circuit runs
        - Has statistical noise in every kernel entry
        - Much slower on a classical simulator

      FidelityStatevectorKernel (fixed):
        - Uses exact statevector simulation
        - Computes K(xi, xj) = |<φ(xj)|φ(xi)>|² analytically
        - No shots, no noise, no sampling variance
        - Single exact computation per kernel entry
        - For 150×150 kernel: 22,500 evaluations, each computed once exactly
        - Significantly faster on a classical laptop simulator

    WHEN TO USE WHICH?
      Statevector  → classical simulator, laptop, thesis experiments  ✅
      Sampler      → real quantum hardware (future work)

    VIVA ANSWER:
      "I replaced FidelityQuantumKernel with FidelityStatevectorKernel
       because our experiments run on a classical statevector simulator,
       not real quantum hardware. The statevector kernel computes each
       inner product |<φ(xj)|φ(xi)>|² exactly in a single pass, whereas
       the sampler-based kernel estimates it from 1024 probabilistic
       measurement shots — adding noise and multiplying runtime by ~1000×.
       On a simulator, statevector is always the correct choice."
    """
    quantum_kernel = FidelityStatevectorKernel(
        feature_map=feature_map
    )
    return quantum_kernel