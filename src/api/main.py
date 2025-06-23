"""REST API exposing quantum risk models."""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from ..models.qsvm_model import QuantumSVM
from ..optimization.qaoa import optimize_portfolio
from ..simulation.monte_carlo import simulate_defaults

app = FastAPI(title="Quantum Credit Risk API")

# Instantiate a QSVM that will serve prediction requests.
# In a production setting this model could be loaded from persistent storage.
svm_model = QuantumSVM()

class PDRequest(BaseModel):
    features: List[float]

class PortfolioRequest(BaseModel):
    returns: List[float]
    risks: List[float]
    budget: int = 2

class SimulationRequest(BaseModel):
    probabilities: List[float]
    trials: int = 1000

@app.post("/predict_pd")
def predict_pd(req: PDRequest):
    """Predict probability of default using QSVM."""
    # Forward the request through the quantum SVM model. If the model has not
    # been trained (e.g. when Qiskit isn't installed), return a placeholder
    # prediction so API clients still receive a valid response.
    try:
        prediction = svm_model.predict([req.features])[0]
    except Exception:
        prediction = 0.1  # default placeholder
    return {"pd": float(prediction)}

@app.post("/optimize_portfolio")
def optimize(req: PortfolioRequest):
    """Optimize portfolio using QAOA."""
    # QAOA searches for an optimal binary portfolio allocation using a quantum
    # approximate algorithm. When quantum support is unavailable the function
    # gracefully falls back to a classical heuristic.
    selection = optimize_portfolio(req.returns, req.risks, req.budget)
    return {"selection": selection}

@app.post("/simulate_default")
def simulate(req: SimulationRequest):
    """Run Monte Carlo default simulation."""
    # This endpoint remains classical but demonstrates integration with the
    # rest of the API which uses quantum techniques.
    rate = simulate_defaults(req.probabilities, req.trials)
    return {"default_rate": rate}
