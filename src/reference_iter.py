# src/reference_iter.py
# Implements the outer iteration loop for smoothing the reference beta_bar.

import numpy as np
from scipy.ndimage import uniform_filter
from utils import CalibrationParams, NumericalParams, Grid

def get_initial_beta_bar(grid: Grid, params: CalibrationParams, ref_type: str = 'Heston'):
    """
    Computes the initial reference diffusion matrix beta_bar based on a specified model.
    
    Args:
        grid: The discretized grid object.
        params: Calibration parameters.
        ref_type: 'Heston' or 'Constant'.
    
    Returns:
        A numpy array of shape (NT+1, NX1, NX2, 3) for [beta11, beta12, beta22].
    """
    p, Np = params, grid.Np
    X1, X2_scaled = np.meshgrid(grid.x1_vec, grid.x2_vec, indexing='ij')
    X2 = X2_scaled / Np.X2_SCALE_K
    
    beta_bar_t = np.zeros((Np.NT + 1, Np.NX1, Np.NX2, 3))

    # Loop over each time step to calculate time-dependent values
    # CORRECTED: Looping over grid.t_vec instead of grid.t
    for k, t in enumerate(grid.t_vec):
        if ref_type == 'Heston':
            kappa_ref, theta_ref, omega_ref, eta_ref = p.KAPPA_BAR, p.THETA_BAR, p.OMEGA_BAR, p.ETA_BAR
            
            time_to_mat = p.T - t
            
            # Avoid division by zero if kappa is very small
            if np.abs(kappa_ref) < 1e-8:
                A_t_T = time_to_mat
            else:
                A_t_T = (1 - np.exp(-kappa_ref * time_to_mat)) / kappa_ref
            
            # nu_t is a function of X2_t,T, so we compute it on the grid
            # nu(t, X_t,T^2, k, th) = A(t,k)^-1 * (2*X_t,T^2 - th*(T-t)) + th
            if np.abs(A_t_T) < 1e-10:
                nu_t_grid = theta_ref # Approximation if T is very close to t
            else:
                nu_t_grid = (2 * X2 - theta_ref * time_to_mat) / A_t_T + theta_ref
            
            nu_t_grid = np.maximum(nu_t_grid, 1e-9) # Ensure positivity

            beta_bar_t[k, ..., 0] = nu_t_grid
            beta_bar_t[k, ..., 1] = 0.5 * eta_ref * omega_ref * A_t_T * nu_t_grid
            beta_bar_t[k, ..., 2] = 0.25 * (omega_ref**2) * (A_t_T**2) * nu_t_grid
        
        else: # Constant reference
            beta_bar_t[k, ..., 0] = p.BETA11_BAR
            beta_bar_t[k, ..., 1] = p.BETA12_BAR
            beta_bar_t[k, ..., 2] = p.BETA22_BAR
            
    return beta_bar_t


def smooth_beta_star(beta_star_t: np.ndarray, num_params: NumericalParams):
    """
    Applies a moving average filter to the optimal beta_star to get the next beta_bar.
    """
    print("Smoothing beta_star to create the next beta_bar...")

    size = (num_params.SMOOTH_BW_T, num_params.SMOOTH_BW_X1, num_params.SMOOTH_BW_X2, 1)
    smoothed_beta_bar_t = np.zeros_like(beta_star_t)
    for i in range(3):  # beta11, beta12, beta22
        smoothed_beta_bar_t[..., i] = uniform_filter(
            beta_star_t[..., i], 
            size=size[:-1],  # 对 (t,x1,x2) 三维做均值滤波
            mode='nearest'
        )
    return smoothed_beta_bar_t


