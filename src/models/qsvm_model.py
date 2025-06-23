"""Quantum Support Vector Machine model for credit default prediction.

This module provides a minimal wrapper around Qiskit's `QSVC` implementation.
Classical credit features are encoded into a quantum state using a feature map
(here ``ZZFeatureMap``) and then classified using a support vector machine
running on the ``qasm_simulator`` backend.  The goal is to demonstrate how
quantum kernels can be leveraged for financial risk estimation.
"""
from typing import Optional

try:
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.utils import algorithm_globals
    from qiskit import BasicAer
except ImportError:  # Fallback if qiskit is not available
    QSVC = None
    ZZFeatureMap = None
    algorithm_globals = None
    BasicAer = None

import numpy as np

class QuantumSVM:
    """Simple wrapper around Qiskit's QSVC."""

    def __init__(self, feature_dim: int = 3, random_seed: int = 42) -> None:
        """Initialize the QSVM wrapper.

        Args:
            feature_dim: Dimension of the classical feature vector.
            random_seed: Seed for reproducible circuit initialization.
        """
        self.feature_dim = feature_dim
        self.random_seed = random_seed
        self.model: Optional[QSVC] = None

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the QSVM on provided data."""
        if QSVC is None:
            raise ImportError("Qiskit is required for QuantumSVM")

        # Ensure reproducibility of the quantum circuit
        algorithm_globals.random_seed = self.random_seed

        # ``ZZFeatureMap`` entangles qubits to map classical data to a
        # higher-dimensional quantum Hilbert space. ``reps`` controls the
        # depth of the circuit.
        feature_map = ZZFeatureMap(self.feature_dim, reps=2)

        # Use the basic simulator so no real quantum hardware is required
        backend = BasicAer.get_backend("qasm_simulator")

        # Instantiate and train the quantum SVM classifier
        self.model = QSVC(feature_map=feature_map, quantum_instance=backend)
        self.model.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predictions for a batch of feature vectors."""
        if not self.model:
            raise ValueError("Model not trained")

        # QSVC handles the quantum kernel evaluations internally
        return self.model.predict(features)
