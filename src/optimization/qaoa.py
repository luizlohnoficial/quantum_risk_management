"""Portfolio optimization using QAOA.

This function formulates the asset selection problem as a quadratic binary
optimization task and solves it using the Quantum Approximate Optimization
Algorithm (QAOA). When Qiskit is unavailable the function falls back to a
simple greedy heuristic so the rest of the application can continue to work.
"""
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
        # Fallback: select the best expected returns if quantum tools are missing
        idx = np.argsort(returns)[-budget:]
        return [1 if i in idx else 0 for i in range(len(returns))]

    # Build a binary quadratic program representing the portfolio selection
    problem = QuadraticProgram()
    for i in range(len(returns)):
        problem.binary_var(name=f"x{i}")
    problem.linear_constraint(
        linear={f"x{i}": 1 for i in range(len(returns))}, sense="==", rhs=budget
    )
    objective = {f"x{i}": -returns[i] + risks[i] for i in range(len(returns))}
    problem.minimize(linear=objective)

    # QAOA works with a quantum sampler to evaluate the cost function
    sampler = Sampler()
    try:
        qaoa = QAOA(optimizer=COBYLA(), sampler=sampler, reps=1)
    except TypeError:
        # Older versions may use different constructor order
        qaoa = QAOA(sampler=sampler, reps=1, optimizer=COBYLA())

    try:
        # MinimumEigenOptimizer wraps the quantum algorithm so it can be applied
        # to the high-level optimization problem defined above
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
