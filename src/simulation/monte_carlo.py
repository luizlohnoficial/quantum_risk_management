"""Monte Carlo simulation for default scenarios."""
from typing import List
import numpy as np


def simulate_defaults(probabilities: List[float], trials: int = 1000) -> float:
    """Simulate defaults using a binomial model.

    Args:
        probabilities: Default probability for each asset.
        trials: Number of simulation trials.

    Returns:
        Estimated portfolio default rate.
    """
    defaults = np.zeros(trials)
    for t in range(trials):
        events = np.random.binomial(1, probabilities)
        defaults[t] = events.mean()
    return float(defaults.mean())
