import numpy as np
import networkx as nx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse

app = FastAPI()

class SimulationParams(BaseModel):
    N: int = 500
    J: float = 0.5
    sigma: float = 0.2
    dt: float = 0.01
    steps: int = 2000
    graph_type: str = "random_regular"  # erdos_renyi, watts_strogatz
    k: int = 6                # Average degree

def generate_graph(params: SimulationParams):
    if params.graph_type == "random_regular":
        return nx.random_regular_graph(d=params.k, n=params.N)
    elif params.graph_type == "erdos_renyi":
        return nx.fast_gnp_random_graph(n=params.N, p=params.k / params.N)
    elif params.graph_type == "watts_strogatz":
        return nx.watts_strogatz_graph(n=params.N, k=params.k, p=params.k / params.N)
    else:
        raise ValueError("Graph type not supported")
    
def run_simulation(params: SimulationParams):
    G = generate_graph(params)
    L = nx.laplacian_matrix(G).toarray()
    
    x = np.ones(params.N)
    sqrt_dt = np.sqrt(params.dt)
    
    for _ in range(params.steps):
        dw = np.random.normal(0, sqrt_dt, params.N)
        interaction = -params.J * (L @ x)
        milstein_correction = 0.5 * (params.sigma**2) * x * (dw**2 - params.dt)
        x += interaction * params.dt + (params.sigma * x * dw) + milstein_correction
        x = np.clip(x, 1e-5, None)

    # Hill estimator for alpha
    x_normalized = x / np.mean(x)
    sorted_x = np.sort(x_normalized)  # Already normalized
    k = max(1, int(0.1 * params.N))
    tail_data = sorted_x[-k:]
    alpha = k / np.sum(np.log(tail_data / tail_data[0]))
    
    return x, alpha

@app.post("/simulate")
def run_simulation_json(params: SimulationParams):
    x, alpha = run_simulation(params)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "tail_exponent": float(alpha),
        "distribution": x.tolist()
    }

@app.post("/simulate/plot")
def run_simulation_plot(params: SimulationParams):
    x, alpha = run_simulation(params)
    
    # Plotting the wealth distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(x / np.mean(x), bins=max(10, int(params.N / 100)), density=True, alpha=0.6, color='teal')
    ax.set_title(f"Normalized Wealth Distribution (alpha={alpha:.2f})")
    ax.set_xlabel("Normalized Wealth")
    ax.set_ylabel("Probability Density")
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close() # Close the plot to free memory
    
    return StreamingResponse(buf, media_type="image/png")