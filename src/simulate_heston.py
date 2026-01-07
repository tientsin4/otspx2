# src/simulate_heston.py
# Generates "market" data using a Heston model via JAX/GPU Monte Carlo.

import jax
import jax.numpy as jnp
import jax.random as random
import pandas as pd
from functools import partial
from utils import CalibrationParams

# --- Monte Carlo Simulation Parameters ---
N_PATHS = 250000
STEPS_PER_DAY = 4

# --- Helper Functions (JAX compatible) ---

def get_nu0_from_X2_0T(X2_0_T, T, kappa, theta):
    """JAX-compatible version to find initial nu_0."""
    A_0_T = jax.lax.cond(
        jnp.abs(kappa) < 1e-8,
        lambda _: T,
        lambda k: (1.0 - jnp.exp(-k * T)) / k,
        kappa
    )
    nu_0 = jax.lax.cond(
        jnp.abs(A_0_T) < 1e-10,
        lambda _: theta,
        lambda A: (2 * X2_0_T - theta * T) / A + theta,
        A_0_T
    )
    return nu_0

def calculate_X2_tT_analytic_jax(nu_t, t, T, kappa, theta):
    """JAX-compatible version of the analytic formula for X^2_{t,T}."""
    time_to_maturity = T - t
    A_t_T = jax.lax.cond(
        jnp.abs(kappa) < 1e-8,
        lambda _: time_to_maturity,
        lambda k: (1.0 - jnp.exp(-k * time_to_maturity)) / k,
        kappa
    )
    return 0.5 * A_t_T * (nu_t - theta) + 0.5 * theta * time_to_maturity

def calculate_vix_from_x2_jax(x2_unscaled, time_to_maturity):
    """JAX-compatible VIX calculation."""
    inside_sqrt = jnp.maximum(0.0, (2.0 / time_to_maturity) * x2_unscaled)
    return 100.0 * jnp.sqrt(inside_sqrt)

# --- JIT-compiled Heston Path Simulation ---

@partial(jax.jit, static_argnames=('p', 'n_steps', 'n_paths'))
def _simulate_heston_paths_jax(key, p: CalibrationParams, n_steps: int, n_paths: int):
    """
    JIT-compiled function to simulate Heston paths on GPU.
    Uses a functional style with jax.lax.scan for performance.
    """
    dt = p.T / n_steps
    sqrt_dt = jnp.sqrt(dt)
    nu0 = get_nu0_from_X2_0T(p.X2_0_T, p.T, p.KAPPA, p.THETA)

    # Generate all random numbers at once
    key, subkey1, subkey2 = random.split(key, 3)
    Z1 = random.normal(subkey1, shape=(n_paths, n_steps))
    Z2_uncorr = random.normal(subkey2, shape=(n_paths, n_steps))
    Z2 = p.ETA * Z1 + jnp.sqrt(1.0 - p.ETA**2) * Z2_uncorr

    # --- Define the scan body ---
    def scan_body(carry, i):
        """Body of the loop for one time step."""
        X1_t, nu_t, results = carry
        
        # Full Truncation scheme for stability
        nu_t_pos = jnp.maximum(nu_t, 0.0)
        sqrt_nu_t = jnp.sqrt(nu_t_pos)
        
        # Evolve paths
        X1_t_next = X1_t - 0.5 * nu_t_pos * dt + sqrt_nu_t * Z1[:, i] * sqrt_dt
        nu_t_next = nu_t + p.KAPPA * (p.THETA - nu_t_pos) * dt + p.OMEGA * sqrt_nu_t * Z2[:, i] * sqrt_dt
        
        # Check if the current step is a maturity date and update results tuple
        t_current = (i + 1) * dt
        
        # Unpack the results tuple
        res_X1_T_SPX1, res_nu_T0, res_X1_T = results

        # Conditionally update the results tuple
        is_T_SPX1_step = jnp.abs(t_current - p.T_SPX1) < dt / 2
        res_X1_T_SPX1 = jax.lax.cond(is_T_SPX1_step, lambda: X1_t_next, lambda: res_X1_T_SPX1)

        is_T0_step = jnp.abs(t_current - p.T0) < dt / 2
        res_nu_T0 = jax.lax.cond(is_T0_step, lambda: nu_t_next, lambda: res_nu_T0)
        
        is_T_step = jnp.abs(t_current - p.T) < dt / 2
        res_X1_T = jax.lax.cond(is_T_step, lambda: X1_t_next, lambda: res_X1_T)

        # Repack the updated results tuple
        updated_results = (res_X1_T_SPX1, res_nu_T0, res_X1_T)
        
        return (X1_t_next, nu_t_next, updated_results), None # Return updated carry

    # --- Initialize and run the scan ---
    # Initial state
    X1_initial = jnp.full(n_paths, p.X1_0)
    nu_initial = jnp.full(n_paths, nu0)
    
    # Initialize results tuple with zero arrays of the correct shape
    results_initial = (
        jnp.zeros(n_paths), # for X1_T_SPX1
        jnp.zeros(n_paths), # for nu_T0
        jnp.zeros(n_paths)  # for X1_T
    )

    initial_carry = (X1_initial, nu_initial, results_initial)
    
    # Run the main loop
    (final_X1, final_nu, final_results_tuple), _ = jax.lax.scan(scan_body, initial_carry, jnp.arange(n_steps))
    
    # Convert final results tuple back to a dictionary
    final_results = {
        'X1_T_SPX1': final_results_tuple[0],
        'nu_T0': final_results_tuple[1],
        'X1_T': final_results_tuple[2]
    }
    
    return final_results

# --- Main Data Generation Function ---

def generate_and_save_market_data(params: CalibrationParams, seed: int = 42, save_path: str = None):
    """
    Main function to generate the "market data" for calibration and save it.
    This simulates paths and then prices all required instruments using JAX/GPU.
    """
    p = params
    n_paths = N_PATHS
    n_steps = p.T_DAYS * STEPS_PER_DAY
    key = random.PRNGKey(seed)
    
    print("--- Generating 'Market' Data using Heston MC (JAX/GPU) ---")
    print(f"Simulating {n_paths} paths on GPU...")
    
    sim_results = _simulate_heston_paths_jax(key, p, n_steps, n_paths)
    
    # --- Price All Instruments from Simulated Paths ---
    prices = {}
    
    # Price SPX 44-day calls
    payoffs_spx1 = jnp.maximum(jnp.exp(sim_results['X1_T_SPX1']) - jnp.array(p.SPX_STRIKES)[:, jnp.newaxis], 0)
    prices_spx1 = jnp.mean(payoffs_spx1, axis=1)
    for i, k in enumerate(p.SPX_STRIKES):
        prices[f'SPX_CALL_{p.T_SPX1_DAYS}D_K{k}'] = prices_spx1[i]

    # Price SPX 79-day calls
    payoffs_spxT = jnp.maximum(jnp.exp(sim_results['X1_T']) - jnp.array(p.SPX_STRIKES)[:, jnp.newaxis], 0)
    prices_spxT = jnp.mean(payoffs_spxT, axis=1)
    for i, k in enumerate(p.SPX_STRIKES):
        prices[f'SPX_CALL_{p.T_DAYS}D_K{k}'] = prices_spxT[i]

    # Price VIX Derivatives at t0=49 days
    nu_t0 = sim_results['nu_T0']
    X2_t0_T_paths = calculate_X2_tT_analytic_jax(nu_t0, p.T0, p.T, p.KAPPA, p.THETA)
    vix_t0_paths = calculate_vix_from_x2_jax(X2_t0_T_paths, p.T - p.T0)
    
    # Price VIX future
    prices[f'VIX_FUTURE_{p.T0_DAYS}D'] = jnp.mean(vix_t0_paths)
    
    # Price VIX calls
    payoffs_vix = jnp.maximum(vix_t0_paths - jnp.array(p.VIX_STRIKES)[:, jnp.newaxis], 0)
    prices_vix = jnp.mean(payoffs_vix, axis=1)
    for i, k in enumerate(p.VIX_STRIKES):
        prices[f'VIX_CALL_{p.T0_DAYS}D_K{k}'] = prices_vix[i]
        
    # Price Singular Contract at T=79 days (theoretically zero)
    prices[f'SINGULAR_CONTRACT_{p.T_DAYS}D'] = 0.0

    # Convert JAX arrays to standard Python floats for saving
    prices_np = {k: float(v) for k, v in prices.items()}
    print("Data generation complete.")

    # --- Save the generated prices to a CSV file ---
    if save_path:
        df = pd.DataFrame(list(prices_np.items()), columns=['Instrument', 'Price'])
        df.to_csv(save_path, index=False)
        print(f"Market data saved to {save_path}")

    return prices_np

if __name__ == '__main__':
    # Standalone execution for generating the data file
    params = CalibrationParams()
    data_path = 'data/simulated_market_data_gpu.csv'
    market_data = generate_and_save_market_data(params, save_path=data_path)
    print("\n--- Generated 'Market' Prices (Sample) ---")
    print(pd.DataFrame(list(market_data.items()), columns=['Instrument', 'Price']).head())

