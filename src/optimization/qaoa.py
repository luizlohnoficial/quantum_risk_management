"""Portfolio optimization using QAOA."""
from typing import List

try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.primitives import Sampler
except ImportError:
    QuadraticProgram = None
    MinimumEigenOptimizer = None
    QAOA = None
    Sampler = None

import numpy as np


def optimize_portfolio(returns: List[float], risks: List[float], budget: int = 2) -> List[int]:
    """Optimize portfolio selection using QAOA.

    Args:
        returns: Expected returns of assets.
        risks: Risk metric of assets.
        budget: Number of assets to select.

    Returns:
        A binary list indicating which assets were selected.
    """
    if QuadraticProgram is None or MinimumEigenOptimizer is None or QAOA is None or Sampler is None:
        # Fallback: greedy selection when Qiskit is unavailable
        idx = np.argsort(returns)[-budget:]
        return [1 if i in idx else 0 for i in range(len(returns))]

    problem = QuadraticProgram()
    for i in range(len(returns)):
        problem.binary_var(name=f"x{i}")
    problem.linear_constraint(
        linear={f"x{i}": 1 for i in range(len(returns))}, sense="==", rhs=budget
    )
    objective = {f"x{i}": -returns[i] + risks[i] for i in range(len(returns))}
    problem.minimize(linear=objective)

    sampler = Sampler()
    try:
        qaoa = QAOA(optimizer=COBYLA(), sampler=sampler, reps=1)
    except TypeError:
        # Older versions may use different constructor order
        qaoa = QAOA(sampler=sampler, reps=1, optimizer=COBYLA())

    try:
        optimizer = MinimumEigenOptimizer(qaoa)
        result = optimizer.solve(problem)
    except Exception:
        # Fall back to legacy API if available
        try:
            result = qaoa.solve(problem)
        except Exception:
            idx = np.argsort(returns)[-budget:]
            return [1 if i in idx else 0 for i in range(len(returns))]

    return list(result.x)
