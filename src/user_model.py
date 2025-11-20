# -*- coding: utf-8 -*-
# user_model.py
import numpy as np
from numba import njit

EPS = 1e-12


# === 模型 A: 你原始的 Attraction-Repulsion 模型 ===
@njit(inline="always")
def original_interaction(Xi, Xj, Si, Sj, params):
    """
    Returns:
        force_pos (d,): 作用在粒子 i 上的位置力
        force_spin (d_s,): 作用在粒子 i 上的自旋力矩/力
    """
    A, B, J, K = params[0], params[1], params[2], params[3]

    rij = Xj - Xi
    dist_sq = np.sum(rij**2)
    dist = np.sqrt(dist_sq)

    if dist < EPS:
        return np.zeros_like(Xi), np.zeros_like(Si)

    inv_dist = 1.0 / dist
    n_ij = rij * inv_dist  # 单位向量

    # 位置力: (A + J cosθ) - B / r
    dot_s = np.vdot(Si, Sj)
    mag_att = A + J * dot_s
    mag_rep = B * inv_dist  # B / r

    # F_pos = (Att - Rep) * n_ij
    # 注意：你原代码里 Repulsion 是 B * (Xj-Xi)/dist^2 = B/r * n_ij，这里保持一致
    f_pos = (mag_att - mag_rep) * n_ij

    # 自旋力
    # Torque ~ K * (Sj_proj) / r
    proj_sj = Sj - dot_s * Si
    f_spin = (K * inv_dist) * proj_sj

    return f_pos, f_spin


# === 模型 B: 比如你想换成 Lennard-Jones + Vicsek ===
@njit(inline="always")
def lennard_jones_vicsek(Xi, Xj, Si, Sj, params):
    epsilon, sigma, eta = params[0], params[1], params[2]

    rij = Xj - Xi
    r2 = np.sum(rij**2)

    if r2 > 9.0 * sigma**2:  # 截断半径
        return np.zeros_like(Xi), np.zeros_like(Si)

    r6 = (sigma**2 / r2) ** 3
    r12 = r6 * r6

    # LJ Force: F = 24*eps/r * (2*r12 - r6) * n_ij
    # Simplified logic for demo
    f_mag = 24 * epsilon * (2 * r12 - r6) / r2
    f_pos = f_mag * rij

    # Vicsek Alignment: J * Sj
    f_spin = eta * Sj

    return f_pos, f_spin
