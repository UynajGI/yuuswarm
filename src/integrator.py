# -*- coding: utf-8 -*-
# integrator.py
import numpy as np
from numba import njit, prange


@njit(fastmath=True, parallel=True)
def rk23_step(fun, t, y, h):
    """
    Perform a single RK23 step with embedded error estimate.

    Parameters
    ----------
    fun : function
        RHS of ODE: dy/dt = fun(t, y)
    t : float
        Current time
    y : ndarray, shape (n, dim)
        Current state
    h : float
        Step size

    Returns
    -------
    y3 : ndarray, shape (n, dim)
        3rd-order solution
    error : float
        Estimated error norm
    """
    # Butcher tableau for RK23 (Heun-Euler)
    k1 = fun(t, y)
    k2 = fun(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = fun(t + h, y + h * k2)

    # 2nd-order (Heun) + 3rd-order (RK3)
    y2 = y + h * (0.5 * k1 + 0.5 * k2)
    y3 = y + h * (k1 / 6.0 + 2.0 * k2 / 3.0 + k3 / 6.0)

    # RMS error (并行计算提升效率)
    error = 0.0
    n, dim = y.shape
    for i in prange(n):
        for j in range(dim):
            error += (y3[i, j] - y2[i, j]) ** 2
    error = np.sqrt(error / (n * dim))

    return y3, error


@njit(fastmath=True)
def rk23(fun, t_span, y0):
    """
    RK23 自适应步长积分器（Numba 兼容版）

    Parameters
    ----------
    fun : callable
        ODE 闭包函数（已包含所有参数）
    t_span : tuple (t0, tf)
        积分时间区间
    y0 : ndarray, shape (n, dim)
        初始状态

    Returns
    -------
    t : ndarray, shape (n_steps,)
        积分时间点
    y : ndarray, shape (n_steps, n, dim)
        状态演化结果
    """
    t0, tf = t_span
    n, dim = y0.shape  # n: 粒子数, dim: 每个粒子的状态维度

    # 初始化参数
    t = t0
    y = y0.copy()
    h = 1e-3  # 初始步长
    rtol = 1e-3  # 相对误差容限
    atol = 1e-6  # 绝对误差容限
    h_min = 1e-12  # 最小步长
    h_max = 1.0  # 最大步长

    # 预分配结果数组（预估最大步数，避免动态扩容）
    max_steps = int(np.ceil((tf - t0) / h_min)) + 2
    t_array = np.empty(max_steps, dtype=np.float64)
    y_array = np.empty((max_steps, n, dim), dtype=np.float64)

    # 存储初始状态
    step_idx = 0
    t_array[step_idx] = t
    y_array[step_idx] = y

    while t < tf - 1e-10:  # 避免浮点误差导致多算一步
        # 调整步长以不超过终点
        if t + h > tf:
            h = tf - t

        # 执行 RK23 步
        y_new, error = rk23_step(fun, t, y, h)

        # 误差控制与步长调整
        tol = atol + rtol * np.maximum(np.abs(y).mean(), np.abs(y_new).mean())
        if error <= tol:
            # 误差满足要求，更新状态并存储
            t += h
            y = y_new
            step_idx += 1
            t_array[step_idx] = t
            y_array[step_idx] = y

        # 计算下一步长（三阶方法，步长缩放因子为 (tol/error)^(1/3)）
        if error == 0.0:
            s = 2.0
        else:
            s = 0.9 * (tol / error) ** (1 / 3)
        h = np.clip(h * s, h_min, h_max)

        # 防止步数超出预分配数组（极端情况扩容）
        if step_idx >= max_steps - 1:
            max_steps *= 2
            t_array = np.resize(t_array, max_steps)
            y_array = np.resize(y_array, (max_steps, n, dim))

    # 截取有效数据（去除未使用的预分配空间）
    t_array = t_array[: step_idx + 1]
    y_array = y_array[: step_idx + 1]

    return t_array, y_array
