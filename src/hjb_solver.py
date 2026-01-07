# src/hjb_solver.py
import jax
import jax.numpy as jnp
from functools import partial
from utils import CalibrationParams, NumericalParams, calculate_vix_from_x2_jax
from pricing_pde import solve_pricing_pde_jax

# --- 修正后的 find_optimal_beta_jax (包含 Lemma A.1 的 Case 3) ---
@partial(jax.jit, static_argnames=('num_params', 'nx1', 'nx2'))
def find_optimal_beta_jax(phi, dx1, dx2, nx1, nx2, num_params, beta_bar_k):
    """
    Calculates optimal beta* and alpha* from the derivatives of phi.
    Vectorized and JIT-compiled for GPU using deconstructed grid info.
    """
    dx2_scaled = dx2 * num_params.X2_SCALE_K 

    phi = phi.reshape(nx1, nx2)
    phi_x1 = jnp.gradient(phi, dx1, axis=0)
    phi_x2 = jnp.gradient(phi, dx2_scaled, axis=1) 
    phi_x1x1 = jnp.gradient(phi_x1, dx1, axis=0)
    phi_x2x2 = jnp.gradient(phi_x2, dx2_scaled, axis=1) 
    phi_x1x2 = jnp.gradient(phi_x1, dx2_scaled, axis=1) 

    a1, a2 = phi_x1, phi_x2
    b11_hjb, b12_hjb, b22_hjb = 0.5 * phi_x1x1, 0.5 * phi_x1x2, 0.5 * phi_x2x2

    beta_bar_k = beta_bar_k.reshape(nx1, nx2, 3) 

    A = beta_bar_k[..., 0] + b11_hjb - 0.5 * a1 - 0.5 * a2 
    B = beta_bar_k[..., 1] + b12_hjb 
    C = beta_bar_k[..., 2] + b22_hjb 

    M_det = A * C - B**2
    M_trace = A + C

    is_M_psd = (M_det >= 0) & (M_trace >= 0) # Case 1
    is_origin_proj = (M_det >= 0) & (M_trace < 0) # Case 2

    beta_star = jnp.zeros_like(beta_bar_k)

    beta_star = beta_star.at[..., 0].set(jnp.where(is_M_psd, A, beta_star[..., 0]))
    beta_star = beta_star.at[..., 1].set(jnp.where(is_M_psd, B, beta_star[..., 1]))
    beta_star = beta_star.at[..., 2].set(jnp.where(is_M_psd, C, beta_star[..., 2]))

    # Case 3: 投影到边界
    is_boundary_proj = ~is_M_psd & ~is_origin_proj

    x_prime_bar = 0.5 * (A - C)
    y_prime_bar = B
    z_prime_bar = 0.5 * (A + C)

    denom_sqrt = jnp.sqrt(x_prime_bar**2 + y_prime_bar**2).clip(1e-9)

    x_plus_prime = 0.5 * x_prime_bar + 0.5 * x_prime_bar * z_prime_bar / denom_sqrt
    y_plus_prime = 0.5 * y_prime_bar + 0.5 * y_prime_bar * z_prime_bar / denom_sqrt
    z_plus_prime = jnp.sqrt(x_plus_prime**2 + y_plus_prime**2)

    x_minus_prime = 0.5 * x_prime_bar - 0.5 * x_prime_bar * z_prime_bar / denom_sqrt
    y_minus_prime = 0.5 * y_prime_bar - 0.5 * y_prime_bar * z_prime_bar / denom_sqrt
    z_minus_prime = jnp.sqrt(x_minus_prime**2 + y_minus_prime**2)

    beta_plus_11 = z_plus_prime + x_plus_prime
    beta_plus_12 = y_plus_prime
    beta_plus_22 = z_plus_prime - x_plus_prime

    beta_minus_11 = z_minus_prime + x_minus_prime
    beta_minus_12 = y_minus_prime
    beta_minus_22 = z_minus_prime - x_minus_prime
    
    # 确保 PSD 条件
    beta_plus_11 = jnp.maximum(0., beta_plus_11)
    beta_plus_22 = jnp.maximum(0., beta_plus_22)
    beta_plus_det = beta_plus_11 * beta_plus_22 - beta_plus_12**2
    beta_plus_11 = jnp.where(beta_plus_det < 0, 0., beta_plus_11) 
    beta_plus_22 = jnp.where(beta_plus_det < 0, 0., beta_plus_22) 

    beta_minus_11 = jnp.maximum(0., beta_minus_11)
    beta_minus_22 = jnp.maximum(0., beta_minus_22)
    beta_minus_det = beta_minus_11 * beta_minus_22 - beta_minus_12**2
    beta_minus_11 = jnp.where(beta_minus_det < 0, 0., beta_minus_11) 
    beta_minus_22 = jnp.where(beta_minus_det < 0, 0., beta_minus_22) 


    dist_sq_plus = (beta_plus_11 - A)**2 + 2 * (beta_plus_12 - B)**2 + (beta_plus_22 - C)**2
    dist_sq_minus = (beta_minus_11 - A)**2 + 2 * (beta_minus_12 - B)**2 + (beta_minus_22 - C)**2

    use_plus = dist_sq_plus <= dist_sq_minus

    beta_star = beta_star.at[..., 0].set(jnp.where(is_boundary_proj, jnp.where(use_plus, beta_plus_11, beta_minus_11), beta_star[..., 0]))
    beta_star = beta_star.at[..., 1].set(jnp.where(is_boundary_proj, jnp.where(use_plus, beta_plus_12, beta_minus_12), beta_star[..., 1]))
    beta_star = beta_star.at[..., 2].set(jnp.where(is_boundary_proj, jnp.where(use_plus, beta_plus_22, beta_minus_22), beta_star[..., 2]))

    alpha_star = jnp.zeros((nx1, nx2, 2))
    alpha_star = alpha_star.at[..., 0].set(-0.5 * beta_star[..., 0])
    alpha_star = alpha_star.at[..., 1].set(-0.5 * beta_star[..., 0])

    return alpha_star, beta_star


@partial(jax.jit, static_argnames=('params', 'num_params', 'nx1', 'nx2'))
def policy_iteration_step_jax(phi, phi_next_t, beta_bar_k, params, num_params,
                              dx1, dx2, dt, nx1, nx2):
    alpha_star, beta_star = find_optimal_beta_jax(
        phi, dx1, dx2, nx1, nx2, num_params, beta_bar_k
    )
    
    source_term = -0.5 * jnp.sum((beta_star - beta_bar_k.reshape(nx1, nx2, 3))**2, axis=-1)
    
    effective_payoff = phi_next_t + dt * source_term

    phi_new = solve_pricing_pde_jax(
        effective_payoff,
        params=params, num_params=num_params,
        dx1=dx1, dx2=dx2, dt=dt, nx1=nx1, nx2=nx2,
        alpha_t=alpha_star[jnp.newaxis,...],
        beta_t=beta_star[jnp.newaxis,...],
        start_time_idx=jnp.array(0, dtype=jnp.int32), 
        maturity_time_idx=jnp.array(1, dtype=jnp.int32)
    )
    return phi_new

@partial(jax.jit, static_argnames=('params', 'num_params', 'nx1', 'nx2'))
def policy_iteration_scan_body(carry, _, phi_next_t_closure, beta_bar_k_closure, 
                               params, num_params, dx1, dx2, dt, nx1, nx2):
    phi_k_iter = carry
    phi_k_next_iter = policy_iteration_step_jax(
        phi_k_iter, phi_next_t_closure, beta_bar_k_closure,
        params, num_params, dx1, dx2, dt, nx1, nx2
    )
    return phi_k_next_iter, phi_k_next_iter


@partial(jax.jit, static_argnames=('params', 'num_params', 'instrument_names', 'nx1', 'nx2'))
def solve_hjb_system_jax(params, num_params,
                         dx1, dx2, dt, instrument_names, nx1, nx2,
                         idx_T, idx_T0, idx_T_SPX1,
                         lambdas_flat_jax, beta_bar_t_jax, x1_vec, x2_vec,
                         scaled_all_payoffs):
    
    X1_grid, X2_grid_unscaled = jnp.meshgrid(x1_vec, x2_vec, indexing='ij')
    
    jump_T = jnp.zeros((nx1, nx2))
    jump_T0 = jnp.zeros((nx1, nx2))
    jump_T_SPX1 = jnp.zeros((nx1, nx2))

    for i, name in enumerate(instrument_names):
        scaled_payoff = scaled_all_payoffs[i]
        lambda_val = lambdas_flat_jax[i]
        
        is_T_mat = (f'SPX_CALL_{params.T_DAYS}D' in name) or ('SINGULAR' in name)
        is_T0_mat = 'VIX' in name
        is_T_SPX1_mat = f'SPX_CALL_{params.T_SPX1_DAYS}D' in name

        jump_T = jnp.where(is_T_mat, jump_T + lambda_val * scaled_payoff, jump_T)
        jump_T0 = jnp.where(is_T0_mat, jump_T0 + lambda_val * scaled_payoff, jump_T0)
        jump_T_SPX1 = jnp.where(is_T_SPX1_mat, jump_T_SPX1 + lambda_val * scaled_payoff, jump_T_SPX1)

    phi_t = jnp.zeros((num_params.NT + 1, nx1, nx2))
    alpha_star_t = jnp.zeros((num_params.NT, nx1, nx2, 2))
    beta_star_t = jnp.zeros((num_params.NT, nx1, nx2, 3))
    
    phi_k_plus_1 = jnp.zeros((nx1, nx2))
    phi_t = phi_t.at[num_params.NT].set(phi_k_plus_1)
    
    # 定义非局部变量
    nonlocal_vars = {
        'phi_next_time_step': None,
        'beta_bar_current_k': None
    }

    def backward_step_body(k_rev, state):
        k = num_params.NT - 1 - k_rev 
        phi_k_plus_1_state, phi_t_storage, alpha_t_storage, beta_t_storage = state

        phi_next_with_jumps = phi_k_plus_1_state
        phi_next_with_jumps = jnp.where(k + 1 == idx_T, phi_next_with_jumps + jump_T, phi_next_with_jumps)
        phi_next_with_jumps = jnp.where(k + 1 == idx_T0, phi_next_with_jumps + jump_T0, phi_next_with_jumps)
        phi_next_with_jumps = jnp.where(k + 1 == idx_T_SPX1, phi_next_with_jumps + jump_T_SPX1, phi_next_with_jumps)

        # 更新闭包变量
        nonlocal_vars['phi_next_time_step'] = phi_next_with_jumps
        nonlocal_vars['beta_bar_current_k'] = beta_bar_t_jax[k]

        policy_iter_partial = partial(
            policy_iteration_scan_body,
            params=params, num_params=num_params,
            dx1=dx1, dx2=dx2, dt=dt, nx1=nx1, nx2=nx2
        )
        
        initial_phi_k_guess = phi_next_with_jumps 
        
        final_phi_k_policy, _ = jax.lax.scan(
            lambda carry, x: policy_iter_partial(carry, x, nonlocal_vars['phi_next_time_step'], nonlocal_vars['beta_bar_current_k']),
            initial_phi_k_guess,
            None,
            length=num_params.POLICY_ITER_MAX
        )
        phi_k = final_phi_k_policy

        phi_t_storage = phi_t_storage.at[k].set(phi_k)
        alpha_k, beta_k = find_optimal_beta_jax(phi_k, dx1, dx2, nx1, nx2, num_params, beta_bar_t_jax[k])
        alpha_t_storage = alpha_t_storage.at[k].set(alpha_k)
        beta_t_storage = beta_t_storage.at[k].set(beta_k)

        return phi_k, phi_t_storage, alpha_t_storage, beta_t_storage

    initial_state = (phi_k_plus_1, phi_t, alpha_star_t, beta_star_t)
    
    _, final_phi_t, final_alpha_t, final_beta_t = jax.lax.fori_loop(
        0, num_params.NT, backward_step_body, initial_state
    )

    return final_phi_t, final_alpha_t, final_beta_t