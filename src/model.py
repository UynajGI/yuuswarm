# -*- coding: utf-8 -*-
# model.py
from math import isinf

import numpy as np
from numba import njit, prange

EPS = 1e-12


@njit(fastmath=True, parallel=True)
def norm_vec(x, d_s):
    """
    归一化向量
    Parameters
    ----------
    x : ndarray, shape (n, d_s)
        输入向量
    d_s : int
        向量维度

    Returns
    -------
    x_normalized : ndarray, shape (n, d_s)
        归一化后的向量
    """
    n = x.shape[0]
    x_normalized = np.empty_like(x)
    for i in prange(n):
        norm = np.linalg.norm(x[i])
        if norm > EPS:
            x_normalized[i] = x[i] / norm
        else:
            x_normalized[i] = np.zeros(d_s)
    return x_normalized


@njit(fastmath=True, parallel=True)
def Lpdist(x, y, p=2.0):
    """
    Compute Lp distance matrix with Numba JIT.

    Supports:
      - p = 1.0     → Manhattan (L1)
      - p = 2.0     → Euclidean (L2)
      - p = np.inf  → Chebyshev (L∞)
      - p > 0       → Minkowski (Lp)

    Parameters
    ----------
    x : ndarray, shape (n, d)
    y : ndarray, shape (m, d)
    p : float, default=2.0

    Returns
    -------
    dist : ndarray, shape (n, m)
    """
    n, d = x.shape
    m = y.shape[0]
    dist = np.empty((n, m), dtype=np.float64)

    if isinf(p):  # p == np.inf → L∞
        for i in prange(n):
            for j in prange(m):
                max_val = 0.0
                for k in range(d):
                    val = abs(x[i, k] - y[j, k])
                    if val > max_val:
                        max_val = val
                dist[i, j] = max_val

    else:  # General Lp (p > 0)
        inv_p = 1.0 / p
        for i in prange(n):
            for j in prange(m):
                s = 0.0
                for k in range(d):
                    s += abs(x[i, k] - y[j, k]) ** p
                dist[i, j] = s**inv_p

    return dist


@njit(fastmath=True)
def I_att(X, i, j, *args):
    """
    计算两个粒子的位置差矢量相互作用
    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        位置变量
    i : int
        粒子i的索引
    j : int
        粒子j的索引
    *args : tuple
        额外参数，包含距离矩阵

    Returns
    -------
    I_att : np.ndarray, shape (d,)
        两个粒子位置差矢量的相互作用
    """
    dist = args[0]

    return (X[j] - X[i]) / (dist[i, j] + EPS)


@njit(fastmath=True)
def F(X, i, j, *args):
    """
    计算两个粒子的自旋向量相互作用
    Parameters
    ----------
    X : np.ndarray, shape (n, d_s)
        自旋变量
    i : int
        粒子i的索引
    j : int
        粒子j的索引
    *args : tuple
        额外参数，包含系数A和J
    Returns
    -------
    F : np.ndarray, shape (d_s,)
        两个粒子自旋向量的相互作用
    """
    A = args[0]
    J = args[1]
    # 自旋向量的余弦值
    return A + J * np.vdot(X[i], X[j])  # cos(theta) = s_i · s_j


@njit(fastmath=True)
def I_rep(X, i, j, *args):
    """
    计算两个粒子的位置差矢量斥力
    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        位置变量
    i : int
        粒子i的索引
    j : int
        粒子j的索引
    *args : tuple
        额外参数，包含距离矩阵

    Returns
    -------
    I_rep : np.ndarray, shape (d,)
    """
    B = args[0]
    dist = args[1]
    return B * (X[j] - X[i]) / (dist[i, j] + EPS) ** 2


@njit(fastmath=True)
def H(X, i, j, *args):
    """
    计算高维自旋相互作用的 torque（切空间向量）

    Parameters
    ----------
    X : np.ndarray, shape (n, d_s) — 每行是单位自旋向量
    i, j : int
    args : (K,)

    Returns
    -------
    torque : np.ndarray, shape (d_s,) — 作用在 X[i] 上的切空间向量
    """
    si = X[i]
    sj = X[j]
    dot = np.vdot(si, sj)
    # 投影 sj 到 si 的切空间
    proj = sj - dot * si
    return proj


@njit(fastmath=True)
def G(X, i, j, *args):
    """
    计算位置差的

    Parameters
    ----------
    X : np.ndarray, shape (n, d) — 位置变量
    i, j : int

    Returns
    -------
    energy : float
    """
    dist = args[0]
    return 1 / (dist[i, j] + EPS)


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
    S = norm_vec(S, d_s)
    V = np.zeros((n, d))
    W = np.zeros((n, d_s))
    y = np.concatenate([X, V, S, W], axis=1)
    return y


@njit(fastmath=True, parallel=True)
def _compute_forces(X, S, dist, v0, omega0, A, B, J, K):
    """
    计算所有粒子的总力和总自旋力

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        位置变量
    S : np.ndarray, shape (n, d_s)
        自旋变量
    dist : np.ndarray, shape (n, n)
        距离矩阵
    v0 : np.ndarray, shape (n, d)
        外加速度
    omega0 : np.ndarray, shape (n, d_s)
        外加自旋角速度
    A : float
        吸引力系数
    B : float
        排斥力系数
    J : float
        自旋相互作用系数
    K : float
        自旋相互作用系数
    n : int
        粒子数
    d : int
        位置维度

    Returns
    -------
    total_fx : np.ndarray, shape (n, d)
        每个粒子的总力
    total_fs : np.ndarray, shape (n, d_s)
        每个粒子的总自旋力
    """
    n = X.shape[0]
    d = X.shape[1]  # 位置维度
    d_s = S.shape[1]  # 自旋维度

    total_fx = np.zeros((n, d))
    total_fs = np.zeros((n, d_s))

    for i in prange(n):
        # 初始化与输出维度相同的零数组
        fx = np.zeros(d)  # 形状: (d,)
        fs = np.zeros(d_s)  # 形状: (d_s,)

        for j in range(n):
            if i != j:
                fx += (
                    I_att(X, i, j, dist) * F(S, i, j, A, J)
                    - I_rep(X, i, j, B, dist)
                    + v0[i]
                )
                fs += K * H(S, i, j, K) * G(X, i, j, dist)
        fx += v0[i]
        fs /= n
        fs += omega0[i]
        total_fx[i] = fx
        total_fs[i] = fs

    return total_fx, total_fs


@njit(fastmath=True, parallel=True)
def f_ode(
    t,
    y,
    d,
    d_s,
    v0,
    omega0,
    coeff_mat,
    A,
    B,
    J,
    K,
):
    """
    ODE 函数

    Parameters
    ----------
    t : float
        时间
    y : np.ndarray, shape (n , d + d + d_s + d_s,)
        状态向量
    d : int
        位置维度
    d_s : int
        自旋维度
    v0 : np.ndarray, shape (n, d)
        速度向量
    omega0 : np.ndarray, shape (n, d_s)
        自旋角速度向量
    coeff_mat : np.ndarray, shape (n, 4)
        系数矩阵
    A : float
        吸引力系数
    B : float
        排斥力系数
    J : float
        自旋相互作用系数
    K : float
        自旋相互作用系数
    """
    n = y.shape[0]
    X = np.ascontiguousarray(y[:, :d])
    V = np.ascontiguousarray(y[:, d : d + d])
    S = np.ascontiguousarray(norm_vec(y[:, d + d : d + d + d_s], d_s))
    W = np.ascontiguousarray(y[:, d + d + d_s :])

    m = coeff_mat[:, 0]
    beta_m = coeff_mat[:, 1]
    ms = coeff_mat[:, 2]
    beta_s = coeff_mat[:, 3]

    dist = Lpdist(X, X, p=2.0)

    dXdt = np.zeros_like(X)
    dVdt = np.zeros_like(V)
    dSdt = np.zeros_like(S)
    dWdt = np.zeros_like(W)

    total_fx, total_fs = _compute_forces(X, S, dist, v0, omega0, A, B, J, K)

    for i in prange(n):
        if m[i] == 0:
            dXdt[i] = total_fx[i]
        else:
            dXdt[i] = V[i]
            dVdt[i] = (total_fx[i] - beta_m[i] * V[i]) / m[i]
        if ms[i] == 0:
            dSdt[i] = total_fs[i]
        else:
            dSdt[i] = W[i]
            dWdt[i] = (total_fs[i] - beta_s[i] * W[i]) / ms[i]
    # 关键修复：直接传入数组，不先组成列表

    for i in prange(n):
        s_i = S[i]
        dsdt_i = dSdt[i]
        # 投影到切空间：ds/dt = ds/dt - (ds/dt · s) s
        dot_product = np.vdot(dsdt_i, s_i)
        dSdt[i] = dsdt_i - dot_product * s_i
    dydt = np.concatenate((dXdt, dVdt, dSdt, dWdt), axis=1)
    return dydt
