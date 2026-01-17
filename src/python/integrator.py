# -*- coding: utf-8 -*-
# integrator.py
import numpy as np
from numba import njit

# 从 engine 导入原地归一化函数，避免重复造轮子和内存分配
from engine import normalize_inplace


@njit(fastmath=True)
def rk23_step(fun, t, y, h, args):
    """
    执行单步 RK23，并将 args 透传给 ODE 函数 fun
    """
    # 解包参数以传递给 fun
    d, d_s, v0, omega0, coeff_mat, params = args

    # --- K1 ---
    k1 = fun(t, y, d, d_s, v0, omega0, coeff_mat, params)

    # --- K2 ---
    # y + 0.5 * h * k1
    y_k2 = y + 0.5 * h * k1
    k2 = fun(t + 0.5 * h, y_k2, d, d_s, v0, omega0, coeff_mat, params)

    # --- K3 ---
    # Heun-Euler / Bogacki-Shampine 标准形式通常用 y - k1 + 2k2，
    # 但这里为了兼容你之前的数学逻辑 (y + h*k2)
    y_k3 = y + h * k2
    k3 = fun(t + h, y_k3, d, d_s, v0, omega0, coeff_mat, params)

    # --- 组合结果 ---
    # 二阶解 (用于误差估计)
    y2 = y + h * (0.5 * k1 + 0.5 * k2)
    # 三阶解 (作为下一步状态)
    y3 = y + h * (k1 / 6.0 + 2.0 * k2 / 3.0 + k3 / 6.0)

    # --- 计算误差 (RMS) ---
    # 直接在寄存器中累加，无需申请 error 数组
    error_sq_sum = 0.0
    n, dim = y.shape

    # Flatten loop for simpler reduction in Numba (optional, but simple loops are fast)
    # 使用 ravel() 视图或者直接双重循环
    for i in range(n):
        for j in range(dim):
            diff = y3[i, j] - y2[i, j]
            error_sq_sum += diff * diff

    error = np.sqrt(error_sq_sum / (n * dim))

    return y3, error


@njit(fastmath=True)
def rk23_adaptive(ode_func, t_span, y0, dt, args):
    """
    RK23 自适应积分器 - 专为 Engine 工厂模式优化

    Parameters
    ----------
    ode_func : function
        由 engine.create_solver 生成的编译后 ODE 函数
    t_span : tuple
        (start_time, end_time)
    y0 : ndarray
        初始状态
    args : tuple
        (d, d_s, v0, omega0, coeff_mat, params)

    Returns
    -------
    t_array, y_array
    """
    d, d_s = args[0], args[1]  # 提取维度用于归一化
    t0, tf = t_span
    n, dim = y0.shape

    # --- 初始化 ---
    t = t0
    y = y0.copy()

    # 初始归一化，防止非法输入
    normalize_inplace(y, d, d_s)

    h = 1e-3  # 初始步长
    rtol = 1e-3  # 相对误差
    atol = 1e-6  # 绝对误差
    h_min = 1e-6  # 最小步长
    h_max = 0.5  # 最大步长

    # --- 预分配内存 ---
    # 估算步数，避免频繁 resize
    est_steps = int((tf - t0) / 0.01) + 100
    max_steps = max(1000, est_steps)

    t_array = np.empty(max_steps, dtype=np.float64)
    y_array = np.empty((max_steps, n, dim), dtype=np.float64)

    # 存储第0步
    step_idx = 0
    t_array[step_idx] = t
    y_array[step_idx] = y

    # --- 积分主循环 ---
    while t < tf - 1e-12:
        # 步长对齐终点
        if t + h > tf:
            h = tf - t

        # 尝试一步
        y_new, error = rk23_step(ode_func, t, y, h, args)

        # 误差限
        tol = atol + rtol * np.maximum(np.abs(y).mean(), np.abs(y_new).mean())

        # 接受/拒绝
        if error <= tol:
            # === 步骤接受 ===
            t += h
            y = y_new

            # 关键：原地归一化，不产生新数组
            normalize_inplace(y, d, d_s)

            step_idx += 1

            # 数组扩容 (Dynamic Resizing)
            if step_idx >= max_steps:
                new_size = max_steps * 2
                # Numba 中 resize 不如重新分配+拷贝安全，但在此处做手动扩容
                new_t = np.empty(new_size, dtype=np.float64)
                new_y = np.empty((new_size, n, dim), dtype=np.float64)

                new_t[:max_steps] = t_array
                new_y[:max_steps] = y_array

                t_array = new_t
                y_array = new_y
                max_steps = new_size

            t_array[step_idx] = t
            y_array[step_idx] = y

            # 计算下一步长 (稍微激进一点的增长)
            if error < 1e-15:
                s = 2.0
            else:
                s = 0.9 * (tol / error) ** (1.0 / 3.0)

            # 限制步长增长过快
            h = min(h_max, h * s)

        else:
            # === 步骤拒绝 ===
            # 缩小步长
            if error < 1e-15:
                error = 1e-15  # 避免除0
            s = 0.9 * (tol / error) ** (1.0 / 3.0)
            h = max(h_min, h * s)

            # 如果步长已经压到最小还是不满足，强制向前一步（防止死循环）
            if h <= h_min:
                # 警告：精度丢失，但在物理模拟中通常优于卡死
                t += h_min
                # 重新计算以更新状态
                y_new, _ = rk23_step(ode_func, t, y, h_min, args)
                y = y_new
                normalize_inplace(y, d, d_s)

                step_idx += 1
                if step_idx < max_steps:  # 边界检查
                    t_array[step_idx] = t
                    y_array[step_idx] = y
                h = h_min * 2  # 尝试恢复

    return t_array[: step_idx + 1], y_array[: step_idx + 1]


@njit(fastmath=True)
def rk2_fixed(ode_func, t_span, y0, dt, args):
    """
    固定步长 RK2 (Heun's Method) 积分器

    Parameters
    ----------
    ode_func : function
        由 engine.create_solver 生成的编译后 ODE 函数
    t_span : tuple
        (start_time, end_time)
    y0 : ndarray
        初始状态
    dt : float
        固定时间步长
    args : tuple
        (d, d_s, v0, omega0, coeff_mat, params)

    Returns
    -------
    t_array, y_array
    """
    # 解包参数
    d, d_s, v0, omega0, coeff_mat, params = args
    t0, tf = t_span
    n, dim = y0.shape

    # 计算总步数
    num_steps = int(np.ceil((tf - t0) / dt)) + 1

    # 预分配结果数组 (一次性分配，无需 resize，速度最快)
    t_array = np.empty(num_steps, dtype=np.float64)
    y_array = np.empty((num_steps, n, dim), dtype=np.float64)

    # 初始化
    t = t0
    y = y0.copy()

    # 初始归一化
    normalize_inplace(y, d, d_s)

    # 存储第0步
    t_array[0] = t
    y_array[0] = y

    # 循环积分
    for i in range(1, num_steps):
        # --- RK2 (Heun's Method) ---
        # k1 = f(t, y)
        k1 = ode_func(t, y, d, d_s, v0, omega0, coeff_mat, params)

        # k2 = f(t + h, y + h * k1)
        y_pred = y + dt * k1
        k2 = ode_func(t + dt, y_pred, d, d_s, v0, omega0, coeff_mat, params)

        # y_new = y + (h/2) * (k1 + k2)
        y_next = y + 0.5 * dt * (k1 + k2)

        # 更新时间 (使用乘法避免浮点累加误差)
        t = t0 + i * dt

        # 原地归一化
        normalize_inplace(y_next, d, d_s)

        # 更新状态
        y = y_next

        # 存储
        t_array[i] = t
        y_array[i] = y

    return t_array, y_array
