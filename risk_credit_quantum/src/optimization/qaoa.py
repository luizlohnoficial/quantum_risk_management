"""Portfolio optimization using QAOA."""
from typing import List

try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_algorithms import QAOA
    from qiskit.primitives import Sampler
except ImportError:
    QuadraticProgram = None
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
    if QuadraticProgram is None:
        # Fallback: greedy selection
        idx = np.argsort(returns)[-budget:]
        solution = [1 if i in idx else 0 for i in range(len(returns))]
        return solution

    problem = QuadraticProgram()
    for i in range(len(returns)):
        problem.binary_var(name=f"x{i}")
    problem.linear_constraint(
        linear={f"x{i}": 1 for i in range(len(returns))}, sense="==", rhs=budget
    )
    objective = {f"x{i}": -returns[i] + risks[i] for i in range(len(returns))}
    problem.minimize(linear=objective)

    sampler = Sampler()
    qaoa = QAOA(sampler=sampler, reps=1)
    result = qaoa.solve(problem)
    return list(result.x)
