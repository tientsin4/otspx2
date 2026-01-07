# src/optimizer.py
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from functools import partial
from utils import Grid, CalibrationParams, NumericalParams, get_payoff_functions
from hjb_solver import solve_hjb_system_jax
from pricing_pde import solve_pricing_pde_jax

@partial(jax.jit, static_argnames=('params', 'num_params', 'instrument_names', 'nx1', 'nx2'))
def objective_function_jax(lambdas_flat_jax, params, num_params,
                           dx1, dx2, dt, instrument_names, nx1, nx2,
                           idx_T, idx_T0, idx_T_SPX1,
                           scaled_market_prices_arr,
                           scaled_all_payoffs,
                           beta_bar_t_jax, x1_vec, x2_vec):
    
    phi_t, alpha_star_t, beta_star_t = solve_hjb_system_jax(
        params, num_params, dx1, dx2, dt, instrument_names, nx1, nx2,
        idx_T, idx_T0, idx_T_SPX1,
        lambdas_flat_jax, beta_bar_t_jax, x1_vec, x2_vec,
        scaled_all_payoffs # 传递缩放后的 Payoffs
    )

    ix1_0 = jnp.argmin(jnp.abs(x1_vec - params.X1_0))
    ix2_0 = jnp.argmin(jnp.abs(x2_vec - params.X2_0_T))

    phi_0_X0 = phi_t[0, ix1_0, ix2_0]
    objective_val = phi_0_X0 - jnp.dot(lambdas_flat_jax, scaled_market_prices_arr)

    maturity_indices = jnp.array([
        idx_T_SPX1 if f'SPX_CALL_{params.T_SPX1_DAYS}D' in name
        else idx_T if f'SPX_CALL_{params.T_DAYS}D' in name or 'SINGULAR' in name
        else idx_T0
        for name in instrument_names
    ], dtype=jnp.int32)

    def price_single_instrument(scaled_payoff, maturity_idx):
        return solve_pricing_pde_jax(
            scaled_payoff, params, num_params, dx1, dx2, dt, nx1, nx2,
            alpha_star_t, beta_star_t, 
            jnp.array(0, dtype=jnp.int32), maturity_idx
        )

    vmapped_pricer = jax.vmap(price_single_instrument, in_axes=(0, 0))
    all_scaled_price_surfaces = vmapped_pricer(scaled_all_payoffs, maturity_indices)
    scaled_model_prices = all_scaled_price_surfaces[:, ix1_0, ix2_0]

    gradient = scaled_model_prices - scaled_market_prices_arr

    return -objective_val, -gradient


def run_optimization(grid: Grid, params: CalibrationParams, num_params: NumericalParams,
                     instrument_names: tuple,
                     market_prices: dict, # 原始市场价格 (现在未使用，但保留签名以防万一)
                     scaled_market_prices: dict, # <--- 传入缩放后的市场价格
                     vegas_dict: dict, # <--- 传入 Vega 值字典
                     beta_bar_t: np.ndarray,
                     initial_lambdas: np.ndarray):
    
    payoffs_dict, _ = get_payoff_functions(grid, params)

    # 缩放 Payoffs
    scaled_payoffs_list = []
    for name in instrument_names:
        original_payoff = payoffs_dict[name]
        vega = vegas_dict.get(name, 1.0)
        if vega is None or abs(vega) < 1e-10:
            vega = 1.0
        scaled_payoff = jnp.array(original_payoff) / vega
        scaled_payoffs_list.append(scaled_payoff)
    
    scaled_all_payoffs = jnp.stack(scaled_payoffs_list, axis=0)
    
    # 缩放市场价格
    scaled_market_prices_arr = jnp.array([scaled_market_prices[name] for name in instrument_names])
    
    beta_bar_t_jax = jnp.array(beta_bar_t)
    x1_vec_jax = jnp.array(grid.x1_vec)
    x2_vec_jax = jnp.array(grid.x2_vec)

    dx1_jax = jnp.array(grid.dx1, dtype=jnp.float32)
    dx2_jax = jnp.array(grid.dx2, dtype=jnp.float32)
    dt_jax = jnp.array(grid.dt, dtype=jnp.float32)
    nx1 = grid.Np.NX1
    nx2 = grid.Np.NX2
    idx_T_jax = jnp.array(grid.time_idx_map['T'], dtype=jnp.int32)
    idx_T0_jax = jnp.array(grid.time_idx_map['T0'], dtype=jnp.int32)
    idx_T_SPX1_jax = jnp.array(grid.time_idx_map['T_SPX1'], dtype=jnp.int32)
    
    static_instrument_names = tuple(instrument_names)

    def objective_for_scipy(lambdas_np):
        lambdas_jax = jnp.array(lambdas_np)
        val_jax, grad_jax = objective_function_jax(
            lambdas_jax, params, num_params,
            dx1_jax, dx2_jax, dt_jax, static_instrument_names, nx1, nx2,
            idx_T_jax, idx_T0_jax, idx_T_SPX1_jax,
            scaled_market_prices_arr,
            scaled_all_payoffs,
            beta_bar_t_jax, x1_vec_jax, x2_vec_jax
        )
        val_np = np.array(val_jax.block_until_ready())
        grad_np = np.array(grad_jax.block_until_ready())
        # print(f"Objective: {val_np}, Max Grad: {np.max(np.abs(grad_np))}") # 调试
        return val_np, grad_np

    print("Starting L-BFGS-B optimization with scaled objective...")
    opt_result = minimize(
        objective_for_scipy,
        initial_lambdas,
        method='L-BFGS-B',
        jac=True,
        options={'disp': True, 'gtol': num_params.OPT_TOL_INNER, 'maxiter': 50}
    )

    print("Optimization finished. Re-calculating final alpha* and beta*...")
    final_lambdas_jax = jnp.array(opt_result.x)

    @partial(jax.jit, static_argnames=('params', 'num_params', 'instrument_names', 'nx1', 'nx2'))
    def get_final_results(lambdas, params, num_params,
                          dx1, dx2, dt, instrument_names, nx1, nx2,
                          idx_T, idx_T0, idx_T_SPX1,
                          beta_bar, x1_vec, x2_vec,
                          scaled_payoffs):
        _, final_alpha, final_beta = solve_hjb_system_jax(
            params, num_params, dx1, dx2, dt, instrument_names, nx1, nx2,
            idx_T, idx_T0, idx_T_SPX1,
            lambdas, beta_bar, x1_vec, x2_vec,
            scaled_payoffs
        )
        return final_alpha, final_beta

    final_alpha_star, final_beta_star = get_final_results(
        final_lambdas_jax, params, num_params,
        dx1_jax, dx2_jax, dt_jax, static_instrument_names, nx1, nx2,
        idx_T_jax, idx_T0_jax, idx_T_SPX1_jax,
        beta_bar_t_jax, x1_vec_jax, x2_vec_jax,
        scaled_all_payoffs
    )
    print("✓ Final alpha* and beta* calculated.")
    
    return opt_result, np.array(final_alpha_star.block_until_ready()), np.array(final_beta_star.block_until_ready())