# SPX-VIX Joint Calibration via Optimal Transport

This repository provides a high-performance implementation for the joint calibration of SPX and VIX options, replicating the methodology of **"Joint Modelling and Calibration of SPX and VIX by Optimal Transport"**.

The project leverages a **multi-paradigm design** and **JAX-powered GPU acceleration** to solve complex Hamilton-Jacobi-Bellman (HJB) equations and parabolic PDEs efficiently.

---

## 🚀 Key Design Philosophies

### 1. Multi-Paradigm Architecture

The engine is built on a "Phased Workflow" that separates concerns across different mathematical and computational paradigms:

* **Stochastic Paradigm (Data Generation):** Uses Monte Carlo simulations (Heston model) to generate synthetic market data and ground-truth price surfaces.
* **Variational Paradigm (Optimization):** Formulates the calibration as a dual problem of Optimal Transport. It solves for Lagrange multipliers () using L-BFGS-B to minimize the distance between model and market prices.
* **Differential Paradigm (PDE/HJB):** Solves the HJB system for the value function  and uses the results to derive optimal drift and diffusion coefficients ().

### 2. JAX & GPU Acceleration

To handle the "curse of dimensionality" and the iterative nature of the HJB-PDE system, the core solvers are built using **JAX**:

* **Just-In-Time (JIT) Compilation:** Critical kernels like the `solve_batched_tridiag` (for ADI schemes) and `find_optimal_beta_jax` are JIT-compiled into optimized XLA kernels.
* **Hardware Acceleration:** All tensor operations are offloaded to the **GPU**, enabling the calibration of a full-scale grid (e.g., ) in a fraction of the time required by standard CPU-based NumPy implementations.
* **Automatic Differentiation:** JAX’s `grad` and `jacobian` are utilized to calculate the sensitivities required for the inner optimization loop.
* **Vectorized ADI Solver:** The Alternating Direction Implicit (ADI) method for 2D pricing PDEs is fully vectorized to process the entire spatial grid simultaneously.

---

## 🛠 Project Structure

```bash
├── main.ipynb            # Workflow orchestration (Quick Validation vs. Full Scale)
├── src/
│   ├── hjb_solver.py     # HJB back-propagation and optimal control logic (JAX)
│   ├── pricing_pde.py    # Forward pricing PDE using vectorized ADI (JAX)
│   ├── optimizer.py      # L-BFGS-B loop for dual parameter calibration
│   ├── simulate_heston.py# GPU-accelerated Monte Carlo for market data
│   ├── reference_iter.py # Outer loop for smoothing reference diffusion
│   └── visualize.py      # Implied volatility skew and path plotting

```

---

## 📈 Performance & Results

The system is designed to handle two execution modes:

1. **Quick Validation:** A low-resolution grid (40x40) to verify pipeline integrity in ~60 seconds.
2. **Full Replication:** High-resolution calibration matching the paper’s parameters, utilizing GPU memory to solve the 2D non-linear control problem.

---

## ⚙️ Quick Start

**Prerequisites:**

* Python 3.9+
* CUDA-enabled GPU
* `jax`, `jaxlib`, `scipy`, `matplotlib`, `pandas`

**Execution:**
Open `main.ipynb` and set `RUN_FULL_SCALE = False` for a smoke test, or `True` to begin the intensive calibration process.

