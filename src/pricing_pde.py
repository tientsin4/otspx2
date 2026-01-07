# src/pricing_pde.py
# Rewritten: fixes ADI indexing direction mismatch and adds beta_12 cross term support

import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def solve_batched_tridiag(a, b, c, d):
    n = b.shape[1]
    c_prime = jnp.zeros_like(c)
    d_prime = jnp.zeros_like(d)
    c_prime = c_prime.at[:, 0].set(c[:, 0] / b[:, 0])
    d_prime = d_prime.at[:, 0].set(d[:, 0] / b[:, 0])

    def forward(i, state):
        c_p, d_p = state
        denom = b[:, i] - a[:, i-1] * c_p[:, i-1]
        c_p = c_p.at[:, i].set(c[:, i] / denom)
        d_p = d_p.at[:, i].set((d[:, i] - a[:, i-1] * d_p[:, i-1]) / denom)
        return c_p, d_p

    c_prime, d_prime = jax.lax.fori_loop(1, n - 1, forward, (c_prime, d_prime))
    d_prime = d_prime.at[:, n-1].set((d[:, n-1] - a[:, n-2] * d_prime[:, n-2]) / (b[:, n-1] - a[:, n-2] * c_prime[:, n-2]))

    x = jnp.zeros_like(d)
    x = x.at[:, n-1].set(d_prime[:, n-1])

    def backward(i, x_):
        j = n - 2 - i
        x_ = x_.at[:, j].set(d_prime[:, j] - c_prime[:, j] * x_[:, j+1])
        return x_

    x = jax.lax.fori_loop(0, n - 1, backward, x)
    return x

@partial(jax.jit, static_argnames=("params", "num_params", "nx1", "nx2"))
def solve_pricing_pde_jax(payoff, params, num_params,
                          dx1, dx2, dt, nx1, nx2,
                          alpha_t, beta_t,
                          start_time_idx, maturity_time_idx):

    dx2_scaled = dx2 * num_params.X2_SCALE_K
    price = payoff

    def time_step_body(k, price_next):
        alpha_k, beta_k = alpha_t[k], beta_t[k]
        a1, a2 = alpha_k[..., 0], alpha_k[..., 1]
        b11, b12, b22 = beta_k[..., 0], beta_k[..., 1], beta_k[..., 2]

        # --- ADI Step 1: Implicit x1, Explicit x2 and x1x2 ---
        v_x2 = (price_next[:, 2:] - price_next[:, :-2]) / (2 * dx2_scaled)
        v_x2x2 = (price_next[:, 2:] - 2 * price_next[:, 1:-1] + price_next[:, :-2]) / dx2_scaled**2

        L_x2 = jnp.zeros_like(price_next)
        L_x2 = L_x2.at[:, 1:-1].set(a2[:, 1:-1] * v_x2 + 0.5 * b22[:, 1:-1] * v_x2x2)

        # beta12 cross term ∂²V/∂x1∂x2
        v_x1x2 = (
            price_next[2:, 2:] - price_next[2:, :-2] - price_next[:-2, 2:] + price_next[:-2, :-2]
        ) / (4 * dx1 * dx2_scaled)
        L_x12 = jnp.zeros_like(price_next)
        L_x12 = L_x12.at[1:-1, 1:-1].set(b12[1:-1, 1:-1] * v_x1x2)

        rhs1 = price_next + 0.5 * dt * (L_x2 + L_x12)

        c1 = 0.5 * dt / dx1**2
        c2 = 0.5 * dt / (2 * dx1)

        lower = -c1 * b11[1:, :] + c2 * a1[1:, :]
        main  = 1 + 2 * c1 * b11
        upper = -c1 * b11[:-1, :] - c2 * a1[:-1, :]

        rhs1_T = rhs1.T
        x_sol = solve_batched_tridiag(lower.T, main.T, upper.T, rhs1_T)
        price_star = x_sol.T

        # --- ADI Step 2: Implicit x2, Explicit x1 ---
        v_x1 = (price_star[2:, :] - price_star[:-2, :]) / (2 * dx1)
        v_x1x1 = (price_star[2:, :] - 2 * price_star[1:-1, :] + price_star[:-2, :]) / dx1**2

        L_x1 = jnp.zeros_like(price_star)
        L_x1 = L_x1.at[1:-1, :].set(a1[1:-1, :] * v_x1 + 0.5 * b11[1:-1, :] * v_x1x1)

        rhs2 = price_star + 0.5 * dt * L_x1

        c3 = 0.5 * dt / dx2_scaled**2
        c4 = 0.5 * dt / (2 * dx2_scaled)
        lower2 = -c3 * b22[:, 1:] + c4 * a2[:, 1:]
        main2  = 1 + 2 * c3 * b22
        upper2 = -c3 * b22[:, :-1] - c4 * a2[:, :-1]

        price_k = solve_batched_tridiag(lower2, main2, upper2, rhs2)

        # Neumann boundary conditions
        price_k = price_k.at[0, :].set(price_k[1, :])
        price_k = price_k.at[-1, :].set(price_k[-2, :])
        price_k = price_k.at[:, 0].set(price_k[:, 1])
        price_k = price_k.at[:, -1].set(price_k[:, -2])

        return price_k

    steps = maturity_time_idx - start_time_idx
    final_price = jax.lax.fori_loop(
        0, steps,
        lambda i, curr: time_step_body(maturity_time_idx - 1 - i, curr),
        price
    )
    return final_price
