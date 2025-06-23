"""Streamlit dashboard for quantum credit risk management."""
import streamlit as st
from typing import List

from ..optimization.qaoa import optimize_portfolio
from ..simulation.monte_carlo import simulate_defaults

st.title("Quantum Credit Risk Dashboard")

st.header("PD Distribution")
if st.button("Example PD Chart"):
    st.line_chart([0.1, 0.2, 0.15])

st.header("Portfolio Optimization")
returns = st.text_input("Returns comma separated", "0.05,0.07,0.02")
risks = st.text_input("Risks comma separated", "0.02,0.03,0.01")
if st.button("Optimize"):
    r = [float(x) for x in returns.split(",")]
    risk_vals = [float(x) for x in risks.split(",")]
    selection = optimize_portfolio(r, risk_vals)
    st.write({"selection": selection})

st.header("Monte Carlo Simulation")
probs = st.text_input("Probabilities", "0.1,0.05,0.2")
trials = st.number_input("Trials", 100, 10000, 1000)
if st.button("Simulate"):
    p = [float(x) for x in probs.split(",")]
    rate = simulate_defaults(p, int(trials))
    st.write({"default_rate": rate})
