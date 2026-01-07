# src/visualize.py
# Contains functions for plotting the results of the calibration.

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf  # FIXED: Import erf from scipy instead of using np.math
from utils import Grid, CalibrationParams, black_scholes_vega, calculate_vix_from_x2

# --- Helper function for normal CDF ---
def norm_cdf(x):
    """Cumulative distribution function for the standard normal distribution."""
    # FIXED: Use scipy.special.erf instead of np.math.erf
    return (1.0 + erf(x / np.sqrt(2.0))) / 2.0

# --- Implied Volatility Solver ---
def implied_volatility(price, S0, K, T, r=0.0, option_type='call'):
    """
    Calculates the implied volatility of a European option using a simple Newton-Raphson method.
    """
    MAX_ITERATIONS = 100
    TOLERANCE = 1.0e-5
    
    vol = 0.5  # Initial guess
    for i in range(MAX_ITERATIONS):
        d1 = (np.log(S0 / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        vega = black_scholes_vega(S0, K, T, r, vol)
        
        if vega < 1e-6:
            return np.nan  # Avoid division by zero
            
        # Price difference
        if option_type == 'call':
            d2 = d1 - vol * np.sqrt(T)
            model_price = S0 * np.exp(-r * T) * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
        else:  # put
            d2 = d1 - vol * np.sqrt(T)
            model_price = K * np.exp(-r * T) * norm_cdf(-d2) - S0 * np.exp(-r * T) * norm_cdf(-d1)
            
        diff = price - model_price
        
        if abs(diff) < TOLERANCE:
            return vol
            
        vol = vol + diff / vega
        
    return vol  # Return last guess if not converged

# --- Plotting Functions ---

def plot_volatility_skews(model_prices, market_prices, params: CalibrationParams, save_path=None):
    """
    Plots the SPX and VIX implied volatility skews for model vs. market.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Implied Volatility Skews: Model vs. Market', fontsize=16)

    # --- SPX Skews ---
    spx_maturities = {
        params.T_SPX1_DAYS: params.T_SPX1,
        params.T_DAYS: params.T
    }
    
    for days, T_ann in spx_maturities.items():
        strikes = params.SPX_STRIKES
        market_vols = []
        model_vols = []
        for K in strikes:
            market_p = market_prices[f'SPX_CALL_{days}D_K{K}']
            model_p = model_prices[f'SPX_CALL_{days}D_K{K}']
            market_vols.append(implied_volatility(market_p, params.S0, K, T_ann))
            model_vols.append(implied_volatility(model_p, params.S0, K, T_ann))
        axes[0].plot(strikes, model_vols, 'o-', label=f'Model ({days} days)')
        axes[0].plot(strikes, market_vols, 'x--', label=f'Market ({days} days)')
        
    axes[0].set_title('SPX Volatility Skews')
    axes[0].set_xlabel('Strike')
    axes[0].set_ylabel('Implied Volatility')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- VIX Skew ---
    vix_strikes = params.VIX_STRIKES
    axes[1].plot(vix_strikes, [market_prices[f'VIX_CALL_{params.T0_DAYS}D_K{K}'] for K in vix_strikes], 'x--', label='Market Prices')
    axes[1].plot(vix_strikes, [model_prices[f'VIX_CALL_{params.T0_DAYS}D_K{K}'] for K in vix_strikes], 'o-', label='Model Prices')
    
    # Add VIX futures prices as vertical lines
    market_future = market_prices[f'VIX_FUTURE_{params.T0_DAYS}D']
    model_future = model_prices[f'VIX_FUTURE_{params.T0_DAYS}D']
    axes[1].axvline(market_future, color='gray', linestyle='--', label=f'Market Future: {market_future:.2f}')
    axes[1].axvline(model_future, color='blue', linestyle=':', label=f'Model Future: {model_future:.2f}')
    
    axes[1].set_title(f'VIX Call Prices at {params.T0_DAYS} days')
    axes[1].set_xlabel('Strike')
    axes[1].set_ylabel('VIX Call Price')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
        print(f"Volatility skew plot saved to {save_path}")
    plt.show()

def plot_calibrated_paths(grid: Grid, alpha_t, beta_t, params: CalibrationParams, save_path=None, n_paths=1000):
    """
    Simulates and plots paths for X1 and X2 from the calibrated model.
    """
    X1_paths = np.zeros((n_paths, grid.Np.NT + 1))
    X2_paths = np.zeros((n_paths, grid.Np.NT + 1))
    X1_paths[:, 0] = params.X1_0
    X2_paths[:, 0] = params.X2_0_T

    sqrt_dt = np.sqrt(grid.dt)

    for t_idx in range(grid.Np.NT):
        x1_curr = X1_paths[:, t_idx]
        x2_curr = X2_paths[:, t_idx]
        
        # Find nearest grid indices for current path values
        ix1 = np.clip(np.searchsorted(grid.x1_vec, x1_curr) - 1, 0, grid.Np.NX1 - 1)
        ix2 = np.clip(np.searchsorted(grid.x2_vec, x2_curr) - 1, 0, grid.Np.NX2 - 1)
        
        # Get interpolated drift and diffusion
        alpha_interp = alpha_t[t_idx, ix1, ix2]
        beta_interp = beta_t[t_idx, ix1, ix2]
        
        b11, b12, b22 = beta_interp[:, 0], beta_interp[:, 1], beta_interp[:, 2]
        
        # Cholesky decomposition for correlated random numbers
        # sigma = [[sqrt(b11), 0], [b12/sqrt(b11), sqrt(b22 - b12^2/b11)]]
        L11 = np.sqrt(np.maximum(1e-9, b11))
        L21 = b12 / L11
        L22 = np.sqrt(np.maximum(1e-9, b22 - L21**2))
        
        Z1 = np.random.normal(size=n_paths)
        Z2 = np.random.normal(size=n_paths)

        dW1 = L11 * Z1
        dW2 = L21 * Z1 + L22 * Z2
        
        # Euler-Maruyama step
        X1_paths[:, t_idx+1] = X1_paths[:, t_idx] + alpha_interp[:, 0] * grid.dt + dW1 * sqrt_dt
        X2_paths[:, t_idx+1] = X2_paths[:, t_idx] + alpha_interp[:, 1] * grid.dt + dW2 * sqrt_dt
        X2_paths[:, t_idx+1] = np.maximum(0, X2_paths[:, t_idx+1])  # Ensure non-negativity

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot X1 paths
    percentiles = [5, 25, 50, 75, 95]
    p_values = np.percentile(X1_paths, percentiles, axis=0)
    ax1.plot(grid.t_vec * params.DAYS_PER_YEAR, p_values[2], color='blue', label='Median Path')
    ax1.fill_between(grid.t_vec * params.DAYS_PER_YEAR, p_values[1], p_values[3], color='blue', alpha=0.3, label='25-75 Percentile')
    ax1.fill_between(grid.t_vec * params.DAYS_PER_YEAR, p_values[0], p_values[4], color='blue', alpha=0.1, label='5-95 Percentile')
    ax1.set_title('Simulated Paths for X1 (log-price)')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('X1')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot X2 paths
    p_values = np.percentile(X2_paths, percentiles, axis=0)
    ax2.plot(grid.t_vec * params.DAYS_PER_YEAR, p_values[2], color='green', label='Median Path')
    ax2.fill_between(grid.t_vec * params.DAYS_PER_YEAR, p_values[1], p_values[3], color='green', alpha=0.3, label='25-75 Percentile')
    ax2.fill_between(grid.t_vec * params.DAYS_PER_YEAR, p_values[0], p_values[4], color='green', alpha=0.1, label='5-95 Percentile')
    ax2.set_title('Simulated Paths for X2 (Fwd Var)')
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('X2')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle('Simulations from Calibrated Model')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Path simulation plot saved to {save_path}")
    plt.show()