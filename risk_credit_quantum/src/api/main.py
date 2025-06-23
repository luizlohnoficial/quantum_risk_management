"""REST API exposing quantum risk models."""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from ..models.qsvm_model import QuantumSVM
from ..optimization.qaoa import optimize_portfolio
from ..simulation.monte_carlo import simulate_defaults

app = FastAPI(title="Quantum Credit Risk API")

# Instantiate global model (for demonstration)
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
    # Dummy model behavior for example
    try:
        prediction = svm_model.predict([req.features])[0]
    except Exception:
        prediction = 0.1  # default placeholder
    return {"pd": float(prediction)}

@app.post("/optimize_portfolio")
def optimize(req: PortfolioRequest):
    """Optimize portfolio using QAOA."""
    selection = optimize_portfolio(req.returns, req.risks, req.budget)
    return {"selection": selection}

@app.post("/simulate_default")
def simulate(req: SimulationRequest):
    """Run Monte Carlo default simulation."""
    rate = simulate_defaults(req.probabilities, req.trials)
    return {"default_rate": rate}
