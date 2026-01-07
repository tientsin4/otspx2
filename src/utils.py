# src/utils.py
# Defines shared, immutable dataclasses and utility functions for the project.
# FIXED: Added calculate_vix_from_x2_jax and corrected numpy/jax usage.

import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from scipy.special import erf # For norm_cdf

# --- Helper Math Functions (centralized here) ---

def norm_cdf(x):
    """Cumulative distribution function for the standard normal distribution."""
    return (1.0 + erf(x / np.sqrt(2.0))) / 2.0

def norm_pdf(x):
    """Probability density function for the standard normal distribution."""
    return np.exp(-x**2 / 2.0) / np.sqrt(2.0 * np.pi)

def black_scholes_vega(S0, K, T, r, vol):
    """Calculates the Black-Scholes vega of a European option."""
    S0, K, T, r, vol = map(float, [S0, K, T, r, vol])
    
    if vol <= 1e-10 or T <= 1e-10 or S0 <= 1e-10 or K <= 1e-10:
        return 0.0
        
    vol_sqrt_T = vol * np.sqrt(T)
    if abs(vol_sqrt_T) < 1e-10:
        return 0.0
        
    d1 = (np.log(S0 / K) + (r + 0.5 * vol**2) * T) / vol_sqrt_T
    # Paper assumes r=0, so vega = S0 * norm_pdf(d1) * np.sqrt(T)
    vega = S0 * norm_pdf(d1) * np.sqrt(T) * np.exp(-r*T) 
    
    if not np.isfinite(vega):
        return 0.0
        
    return vega


# --- Dataclasses for Parameters (Immutable) ---

@dataclass(frozen=True)
class CalibrationParams:
    """Parameters related to the market, instruments, and model dynamics."""
    S0: float = 100.0
    X1_0: float = field(init=False)
    X2_0_T: float = 0.0098
    T_DAYS: int = 79
    T0_DAYS: int = 49
    T_SPX1_DAYS: int = 44
    DAYS_PER_YEAR: float = 365.25 
    T: float = field(init=False)
    T0: float = field(init=False)
    T_SPX1: float = field(init=False)
    KAPPA: float = 0.6
    THETA: float = 0.09
    OMEGA: float = 0.4
    ETA: float = -0.5
    KAPPA_BAR: float = 0.9
    THETA_BAR: float = 0.04
    OMEGA_BAR: float = 0.6
    ETA_BAR: float = -0.3
    BETA11_BAR: float = 0.09
    BETA12_BAR: float = -0.01
    BETA22_BAR: float = 0.04
    SPX_STRIKES: tuple = field(default_factory=lambda: (85, 90, 95, 100, 105, 110, 115))
    VIX_STRIKES: tuple = field(default_factory=lambda: (15, 20, 25, 30, 35))
    R: float = 0.0 
    Q: float = 0.0 

    def __post_init__(self):
        object.__setattr__(self, 'X1_0', np.log(self.S0))
        object.__setattr__(self, 'T', self.T_DAYS / self.DAYS_PER_YEAR)
        object.__setattr__(self, 'T0', self.T0_DAYS / self.DAYS_PER_YEAR)
        object.__setattr__(self, 'T_SPX1', self.T_SPX1_DAYS / self.DAYS_PER_YEAR)

@dataclass(frozen=True)
class NumericalParams:
    """Parameters controlling the numerical PDE solvers and optimization."""
    NX1: int = 100
    NX2: int = 100
    NT: int = 158
    X1_MIN_MULT: float = 0.8
    X1_MAX_MULT: float = 1.2
    X2_MAX_MULT: float = 5.0 
    X2_SCALE_K: int = 40 
    OPT_TOL_INNER: float = 1e-4 
    POLICY_ITER_MAX: int = 40 
    OUTER_ITERATIONS: int = 5 
    SMOOTH_BW_T: int = 3 
    SMOOTH_BW_X1: int = 5
    SMOOTH_BW_X2: int = 5 

# --- Grid Class (Immutable with Aliases) ---

@dataclass(frozen=True)
class Grid:
    """A helper class to manage the discretized grid and its properties."""
    Np: NumericalParams
    Cp: CalibrationParams
    
    t_vec: np.ndarray = field(init=False, repr=False)
    dt: float = field(init=False)
    x1_vec: np.ndarray = field(init=False, repr=False)
    dx1: float = field(init=False)
    x2_vec: np.ndarray = field(init=False, repr=False) # UN SCALED
    dx2: float = field(init=False) # UN SCALED
    x1: np.ndarray = field(init=False, repr=False) # ALIAS
    x2: np.ndarray = field(init=False, repr=False) # ALIAS
    time_idx_map: dict = field(init=False, repr=False)
    
    def __post_init__(self):
        t_vec = np.linspace(0, self.Cp.T, self.Np.NT + 1)
        dt = self.Cp.T / self.Np.NT
        object.__setattr__(self, 't_vec', t_vec)
        object.__setattr__(self, 'dt', dt)

        x1_min_val = self.Cp.X1_0 + np.log(self.Np.X1_MIN_MULT)
        x1_max_val = self.Cp.X1_0 + np.log(self.Np.X1_MAX_MULT)
        min_log_strike = np.log(min(self.Cp.SPX_STRIKES)) if self.Cp.SPX_STRIKES else x1_min_val
        max_log_strike = np.log(max(self.Cp.SPX_STRIKES)) if self.Cp.SPX_STRIKES else x1_max_val
        x1_min_val = min(x1_min_val, min_log_strike - 0.1)
        x1_max_val = max(x1_max_val, max_log_strike + 0.1)

        x1_vec = np.linspace(x1_min_val, x1_max_val, self.Np.NX1)
        dx1 = (x1_max_val - x1_min_val) / (self.Np.NX1 - 1)
        object.__setattr__(self, 'x1_vec', x1_vec)
        object.__setattr__(self, 'dx1', dx1)
        
        x2_max_val = max(1e-6, self.Np.X2_MAX_MULT * self.Cp.X2_0_T)
        x2_vec = np.linspace(0, x2_max_val, self.Np.NX2)
        dx2 = x2_max_val / (self.Np.NX2 - 1) if self.Np.NX2 > 1 else x2_max_val
        object.__setattr__(self, 'x2_vec', x2_vec) # Unscaled vector
        object.__setattr__(self, 'dx2', dx2) # Unscaled dx

        object.__setattr__(self, 'x1', x1_vec)
        object.__setattr__(self, 'x2', x2_vec)

        time_idx_map = {
            'T': self.Np.NT,
            'T0': np.argmin(np.abs(t_vec - self.Cp.T0)),
            'T_SPX1': np.argmin(np.abs(t_vec - self.Cp.T_SPX1))
        }
        object.__setattr__(self, 'time_idx_map', time_idx_map)

    def get_initial_indices(self, x1_val, x2_val):
        ix1_0 = np.argmin(np.abs(self.x1_vec - x1_val))
        ix2_0 = np.argmin(np.abs(self.x2_vec - x2_val))
        return ix1_0, ix2_0
    
    def __hash__(self):
        return hash((
            self.Np.NX1, self.Np.NX2, self.Np.NT,
            self.Cp.T, self.Cp.S0, self.Cp.X2_0_T,
            self.Np.X1_MIN_MULT, self.Np.X1_MAX_MULT, self.Np.X2_MAX_MULT,
            self.Np.X2_SCALE_K
        ))

# --- Payoff Functions ---

# [!! NEW / FIXED !!]
# JAX-compatible version for use in JIT functions (like hjb_solver, simulate_heston)
@jax.jit
def calculate_vix_from_x2_jax(x2_unscaled, time_to_maturity):
    """JAX-compatible VIX calculation from UN SCALED X2."""
    safe_x2 = jnp.maximum(x2_unscaled, 0.0)
    safe_time_to_maturity = jnp.maximum(time_to_maturity, 1e-10)
    inside_sqrt = 2.0 * safe_x2 / safe_time_to_maturity
    return 100.0 * jnp.sqrt(inside_sqrt)

# [!! NEW / FIXED !!]
# NumPy version for setup/payoff generation
def calculate_vix_from_x2(x2_unscaled, time_to_maturity):
    """NumPy VIX calculation from UN SCALED X2 values."""
    if time_to_maturity <= 1e-10:
        return np.zeros_like(x2_unscaled)
    safe_x2 = np.maximum(x2_unscaled, 0.0)
    inside_sqrt = 2.0 * safe_x2 / time_to_maturity
    return 100.0 * np.sqrt(inside_sqrt)


def get_payoff_functions(grid: Grid, params: CalibrationParams):
    """
    Generates a dictionary of all instrument payoffs (ORIGINAL SCALE) on the grid.
    Returns the payoff dictionary and a sorted list of instrument names.
    """
    payoffs = {}
    instrument_names = []
    
    # Use NumPy for grid setup. Grid vectors are unscaled.
    X1, X2_unscaled = np.meshgrid(grid.x1_vec, grid.x2_vec, indexing='ij')
    S_grid = np.exp(X1)
    
    # Singular Contract Payoff
    name_singular = f'SINGULAR_CONTRACT_{params.T_DAYS}D'
    payoffs[name_singular] = 1.0 - np.exp(-(X2_unscaled**2))
    instrument_names.append(name_singular)
    
    # SPX Call Option Payoffs
    for days in [params.T_SPX1_DAYS, params.T_DAYS]:
        for k in params.SPX_STRIKES:
            name_spx = f'SPX_CALL_{days}D_K{k}'
            payoffs[name_spx] = np.maximum(S_grid - k, 0)
            instrument_names.append(name_spx)
            
    # VIX Calculation at T0
    time_to_maturity_vix = params.T - params.T0
    # [!! FIXED !!] Use the NumPy version here
    vix_surface_T0 = calculate_vix_from_x2(X2_unscaled, time_to_maturity_vix)
    
    # VIX Future Payoff (Value of VIX at T0)
    name_vix_fut = f'VIX_FUTURE_{params.T0_DAYS}D'
    payoffs[name_vix_fut] = vix_surface_T0
    instrument_names.append(name_vix_fut)
    
    # VIX Call Option Payoffs
    for k in params.VIX_STRIKES:
        name_vix_call = f'VIX_CALL_{params.T0_DAYS}D_K{k}'
        payoffs[name_vix_call] = np.maximum(vix_surface_T0 - k, 0)
        instrument_names.append(name_vix_call)
        
    instrument_names.sort() # Ensure consistent order
    return payoffs, tuple(instrument_names)