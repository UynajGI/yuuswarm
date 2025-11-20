# -*- coding: utf-8 -*-
# simulation.py - 简化版仿真函数
import time
from logging import debug

import numpy as np

from integrator import rk23_adaptive
from model import f_ode, init_states


def run_simulation(
    coeff_mat,
    v0=None,
    omega0=None,
    L=10.0,
    t_span=(0.0, 10.0),
    A=1.0,
    B=1.0,
    J=0.8,
    K=0.3,
    rtol=1e-3,
    atol=1e-6,
    h_min=1e-12,
    h_max=1.0,
    d=None,  # Add d and d_s as explicit parameters
    d_s=None,
):
    """
    Run swarmalator system simulation.
    """
    coeff_mat = np.asarray(coeff_mat, dtype=np.float64)
    N = coeff_mat.shape[0]
    assert coeff_mat.shape == (N, 4), f"coeff_mat should be ({N}, 4) shape"

    # Determine dimensions
    if d is None:
        if v0 is not None:
            d = v0.shape[1]
        else:
            d = 2  # Default
    if d_s is None:
        if omega0 is not None:
            d_s = omega0.shape[1]
        else:
            d_s = 3  # Default

    if v0 is None:
        v0 = np.zeros((N, d), dtype=np.float64)
    if omega0 is None:
        omega0 = np.zeros((N, d_s), dtype=np.float64)

    v0 = np.asarray(v0, dtype=np.float64)
    omega0 = np.asarray(omega0, dtype=np.float64)

    assert v0.shape == (N, d), f"v0 should be ({N}, {d}) shape"
    assert omega0.shape == (N, d_s), f"omega0 should be ({N}, {d_s}) shape"

    coeff_mat, v0, omega0, (n_A, n_B, n_C, n_D) = validate_and_preprocess_params(
        coeff_mat, v0, omega0, d, d_s
    )

    debug("Validated parameters.")
    debug(f"Initial positions will be in a box of size L={L}.")
    debug(f"Initial velocities: {v0}")
    debug(f"Initial orientations: {omega0}")
    debug(f"Particle types: A={n_A}, B={n_B}, C={n_C}, D={n_D}")

    states = init_states(n_A, n_B, n_C, n_D, d, d_s, L)
    x_A, v_A, s_A, ω_A, x_B, v_B, s_B, x_C, s_C, ω_C, x_D, s_D = states

    y0 = pack_state(x_A, v_A, s_A, ω_A, x_B, v_B, s_B, x_C, s_C, ω_C, x_D, s_D)

    def ode_rhs(t, y):
        return f_ode(
            t, y, n_A, n_B, n_C, n_D, d, d_s, v0, omega0, coeff_mat, A, B, J, K
        )

    t_start, t_end = t_span
    print(f"Starting simulation: N={N}, t_span=[{t_start}, {t_end}]")
    print(f"Particle type distribution: A={n_A}, B={n_B}, C={n_C}, D={n_D}")
    start_time = time.time()  # 记录开始时间

    # Pass dimensions to the integrator
    T, Y = rk23_adaptive(
        ode_rhs,
        t_span,
        y0,
        rtol=rtol,
        atol=atol,
        h_min=h_min,
        h_max=h_max,
        n_A=n_A,
        n_B=n_B,
        n_C=n_C,
        n_D=n_D,
        d=d,
        d_s=d_s,
    )

    elapsed_time = time.time() - start_time  # 计算耗时
    print(
        f"Simulation finished: {len(T)} time steps, elapsed time: {elapsed_time:.2f} seconds"
    )
    print(f"Final time: T[-1] = {T[-1]:.6f}")  # 显示最终时间点

    # --- Optimized unpacking: Use list comprehension once ---
    # Unpack all time steps at once into a list of tuples
    print("Unpacking simulation results...", end="")  # 提示开始解包
    Y_unpacked_list_of_tuples = [
        unpack_state(y_i, n_A, n_B, n_C, n_D, d, d_s) for y_i in Y
    ]

    # Convert the list of tuples into a tuple of lists (each list corresponds to a state component)
    unpacked_components = list(zip(*Y_unpacked_list_of_tuples))

    # Now convert each list of arrays into a single 3D array (n_steps, n_particles, n_dims)
    unpacked_sequences = {
        "x_A": np.array(unpacked_components[0]),  # [state[0] for state in Y_unpacked]
        "v_A": np.array(unpacked_components[1]),  # [state[1] for state in Y_unpacked]
        "s_A": np.array(unpacked_components[2]),  # [state[2] for state in Y_unpacked]
        "ω_A": np.array(unpacked_components[3]),  # [state[3] for state in Y_unpacked]
        "x_B": np.array(unpacked_components[4]),  # [state[4] for state in Y_unpacked]
        "v_B": np.array(unpacked_components[5]),  # [state[5] for state in Y_unpacked]
        "s_B": np.array(unpacked_components[6]),  # [state[6] for state in Y_unpacked]
        "x_C": np.array(unpacked_components[7]),  # [state[7] for state in Y_unpacked]
        "s_C": np.array(unpacked_components[8]),  # [state[8] for state in Y_unpacked]
        "ω_C": np.array(unpacked_components[9]),  # [state[9] for state in Y_unpacked]
        "x_D": np.array(unpacked_components[10]),  # [state[10] for state in Y_unpacked]
        "s_D": np.array(unpacked_components[11]),  # [state[11] for state in Y_unpacked]
    }
    print(" Done.")  # 解包完成提示

    metadata = {
        "n_types": (n_A, n_B, n_C, n_D),
        "dims": (d, d_s),
        "coeff_mat": coeff_mat,
        "v0": v0,
        "omega0": omega0,
        "params": {"A": A, "B": B, "J": J, "K": K},
    }

    return T, unpacked_sequences, metadata
