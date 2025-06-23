"""Quantum Support Vector Machine model for credit default prediction."""
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
        self.feature_dim = feature_dim
        self.random_seed = random_seed
        self.model: Optional[QSVC] = None

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        if QSVC is None:
            raise ImportError("Qiskit is required for QuantumSVM")
        algorithm_globals.random_seed = self.random_seed
        feature_map = ZZFeatureMap(self.feature_dim, reps=2)
        backend = BasicAer.get_backend("qasm_simulator")
        self.model = QSVC(feature_map=feature_map, quantum_instance=backend)
        self.model.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not trained")
        return self.model.predict(features)
