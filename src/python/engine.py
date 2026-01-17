# -*- coding: utf-8 -*-
# engine.py
import numpy as np
from numba import njit, prange

EPS = 1e-12


def create_solver(interaction_kernel):
    """
    工厂函数：接收一个相互作用核函数，返回编译好的 ODE 函数和积分器。
    """

    # --- 1. 定义通用的力计算循环 (闭包) ---
    @njit(fastmath=True, parallel=True)
    def compute_all_forces(X, S, v0, omega0, params):
        n = X.shape[0]
        d = X.shape[1]
        d_s = S.shape[1]

        total_fx = np.zeros((n, d))
        total_fs = np.zeros((n, d_s))

        # 并行计算力
        for i in prange(n):
            fx_i = np.zeros(d)
            fs_i = np.zeros(d_s)

            for j in range(n):
                if i == j:
                    continue

                # === 这里调用用户传入的核函数 ===
                # Numba 会自动内联这段代码，不会有函数调用开销
                dv, ds = interaction_kernel(X[i], X[j], S[i], S[j], params)

                fx_i += dv
                fs_i += ds

            total_fx[i] = fx_i / n + v0[i]
            total_fs[i] = fs_i / n + omega0[i]

        return total_fx, total_fs

    # --- 2. 定义 ODE 函数 ---
    @njit(fastmath=True, parallel=True)
    def f_ode_dynamic(t, y, d, d_s, v0, omega0, coeff_mat, params):
        n = y.shape[0]
        # 零拷贝切片
        X = np.ascontiguousarray(y[:, :d])
        V = np.ascontiguousarray(y[:, d : 2 * d])
        S = np.ascontiguousarray(y[:, 2 * d : 2 * d + d_s])
        W = np.ascontiguousarray(y[:, 2 * d + d_s :])

        m = coeff_mat[:, 0]
        beta_m = coeff_mat[:, 1]
        ms = coeff_mat[:, 2]
        beta_s = coeff_mat[:, 3]

        # 计算力
        fx, fs = compute_all_forces(X, S, v0, omega0, params)

        dydt = np.empty_like(y)

        # 填充导数 (支持有质量/无质量混合)
        for i in prange(n):
            # 位置部分
            if m[i] == 0:
                dydt[i, :d] = fx[i]
                dydt[i, d : 2 * d] = 0.0
            else:
                dydt[i, :d] = V[i]
                dydt[i, d : 2 * d] = (fx[i] - beta_m[i] * V[i]) / m[i]

            # 自旋部分
            if ms[i] == 0:
                # 过阻尼自旋：投影切向
                s_curr = S[i]
                ds_raw = fs[i]
                # 投影: dS/dt = F - (F·S)S
                proj = ds_raw - np.vdot(ds_raw, s_curr) * s_curr
                dydt[i, 2 * d : 2 * d + d_s] = proj
                dydt[i, 2 * d + d_s :] = 0.0
            else:
                dydt[i, 2 * d : 2 * d + d_s] = W[i]
                dydt[i, 2 * d + d_s :] = (fs[i] - beta_s[i] * W[i]) / ms[i]

        return dydt

    return f_ode_dynamic


# --- 辅助工具：自旋归一化 ---
@njit(parallel=True)
def normalize_inplace(y, d, d_s):
    n = y.shape[0]
    start = 2 * d
    end = start + d_s
    for i in prange(n):
        s = y[i, start:end]
        norm = np.sqrt(np.sum(s**2))
        if norm > 1e-12:
            y[i, start:end] = s / norm


def init_states(n, d, d_s, L):
    """
    初始化位置和自旋状态

    Parameters
    ----------
    n : int
        粒子数
    d : int
        位置维度
    d_s : int
        自旋维度
    L : float
        位置空间边长

    Returns
    -------
    y : np.ndarray, shape (n , d + d + d_s + d_s,)
        初始状态向量
    """
    X = np.random.uniform(0, L, size=(n, d))
    S = np.random.normal(0, 1, size=(n, d_s))
    V = np.zeros((n, d))
    W = np.zeros((n, d_s))
    y = np.concatenate([X, V, S, W], axis=1)
    normalize_inplace(y, d, d_s)
    return y
