import json  # 用于保存元数据
import os
import sys
import time
from datetime import datetime

import h5py
import numpy as np

# 确保能导入你的模块
sys.path.append("/home/jiangyuan/swarmlators/src")
# 导入你定义好的模块
from engine import create_solver, init_states
from integrator import rk2_fixed, rk23_adaptive
from user_model import original_interaction


def save_results(t_array, y_array, config_data, output_dir="sim_results"):
    """
    保存模拟结果到指定的输出目录，使用 HDF5 (.h5) 格式存储数据和元数据。
    同时生成JSON格式的元数据文件。
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义 HDF5 文件路径
    h5_path = os.path.join(output_dir, "simulation_data.h5")

    # 定义 JSON 元数据文件路径
    json_path = os.path.join(output_dir, "metadata.json")

    # --- 1. 保存数据和元数据 (HDF5) ---
    print(f"Saving data and metadata to HDF5 file: {h5_path}")

    # 使用 'w' 模式创建或覆盖文件
    with h5py.File(h5_path, "w") as hf:
        # 1.1 保存数据集 (t_history 和 y_history)
        # 启用 GZIP 压缩，压缩等级为 4 (0-9，数字越大压缩率越高，但写入越慢)
        hf.create_dataset("t", data=t_array, compression="gzip", compression_opts=4)

        # 状态空间路径通常最大，必须压缩
        hf.create_dataset("y", data=y_array, compression="gzip", compression_opts=4)

        # 1.2 保存配置为根属性 (Attributes)
        metadata = config_data.copy()

        # 确保 NumPy 数组可以被存储为 HDF5 属性 (需转换为 list/Python native 或 NumPy 数组)
        metadata["params"] = metadata["params"].tolist()

        root_attrs = hf.attrs
        print("\nSaving Metadata Attributes:")
        for key, value in metadata.items():
            # HDF5 属性只能是标量、字符串或 NumPy 数组。
            if isinstance(value, (list, tuple)):
                root_attrs[key] = np.array(value)
            elif isinstance(value, str):
                # HDF5 有时对 Unicode 有限制，但通常可以直接存
                root_attrs[key] = value
            else:
                root_attrs[key] = value

            print(f"  - {key}: {value}")  # 打印确认保存的属性

    # --- 2. 保存 JSON 元数据 ---
    print(f"Saving metadata to JSON file: {json_path}")
    json_metadata = config_data.copy()
    json_metadata["params"] = json_metadata["params"].tolist()  # 转换为list以兼容JSON
    json_metadata["timestamp"] = datetime.now().isoformat()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_metadata, f, indent=2, ensure_ascii=False)

    print(f"JSON metadata saved to: {json_path}")

    # 打印文件大小，确认压缩效果
    size_mb = os.path.getsize(h5_path) / (1024 * 1024)
    print(f"\nHDF5 file saved successfully.")
    print(f"Final file size: {size_mb:.2f} MB")

    # 提示如何读取文件
    print("To read the data later, use Python/h5py:")
    print("  >>> import h5py")
    print("  >>> with h5py.File(h5_path, 'r') as hf:")
    print("  >>>     y = hf['y'][:]")
    print("  >>>     N = hf.attrs['N']")
    print("  >>>     # or read JSON metadata:")
    print("  >>>     with open('metadata.json', 'r') as f:")
    print("  >>>         metadata = json.load(f)")


def run_simulation(
    N, d, d_s, L, T_end, dt_fixed, params, integrator_name, output_dir, seed=42
):
    """
    执行模拟并保存结果。

    Parameters
    ----------
    N : int
        粒子数
    d : int
        空间维度 (新增参数)
    d_s : int
        自旋维度 (新增参数)
    L : float
        空间大小
    T_end : float
        结束时间
    dt_fixed : float
        RK2的固定步长。若使用RK23，此值被忽略。
    params : np.ndarray
        物理模型参数 [A, B, J, K]
    integrator_name : str
        积分器名称 ('rk2_fixed' 或 'rk23_adaptive')
    output_dir : str
        结果保存目录

    Returns
    -------
    None
    """
    # 记录开始时间
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Starting simulation with parameters:")
    print(f"  - N: {N}, d: {d}, d_s: {d_s}")
    print(f"  - L: {L}, T_end: {T_end}")
    print(f"  - params: {params}")
    print(f"  - integrator: {integrator_name}")
    print(f"  - output_dir: {output_dir}")

    np.random.seed(seed)
    t_span = (0.0, T_end)

    # ==========================================
    # 1. 初始化状态 & 物理参数
    # ==========================================
    print("Step 1: Initializing states...")
    init_start = time.time()
    # 使用传入的 d 和 d_s 进行初始化
    y0 = init_states(N, d, d_s, L)
    v0 = np.zeros((N, d))
    omega0 = np.zeros((N, d_s))
    coeff_mat = np.zeros((N, 4))
    init_time = time.time() - init_start
    print(f"  - States initialized in {init_time:.4f}s")
    print(
        f"  - y0 shape: {y0.shape}, v0 shape: {v0.shape}, omega0 shape: {omega0.shape}"
    )

    # 打包 ODE 函数所需的参数
    args = (d, d_s, v0, omega0, coeff_mat, params)

    # 确定积分器函数
    if integrator_name == "rk2_fixed":
        ode_integrator = rk2_fixed
        print(f"Using integrator: RK2 Fixed (d={d}, d_s={d_s}, dt={dt_fixed})")
    elif integrator_name == "rk23_adaptive":
        ode_integrator = rk23_adaptive
        print(f"Using integrator: RK23 Adaptive (d={d}, d_s={d_s})")
    else:
        raise ValueError(f"Unknown integrator: {integrator_name}")

    # ==========================================
    # 2. 构建求解器 (编译)
    # ==========================================
    print("Step 2: Compiling Solver...")
    t_compile_start = time.time()
    ode_solver = create_solver(original_interaction)
    print("Warming up JIT compiler...")
    _ = ode_integrator(ode_solver, (0.0, 0.2), y0.copy(), dt_fixed, args)
    compile_time = time.time() - t_compile_start
    print(f"Warm-up completed in {compile_time:.4f}s")

    # ==========================================
    # 3. 运行模拟
    # ==========================================
    print("Step 3: Running simulation...")
    sim_start = time.time()
    ts, ys = ode_integrator(ode_solver, t_span, y0, dt_fixed, args)
    sim_time = time.time() - sim_start

    t_end = time.time()
    total_time = t_end - start_time

    print(f"Simulation completed in {sim_time:.4f}s")
    print(f"Final time array shape: {ts.shape}")
    print(f"Final state array shape: {ys.shape}")

    # ==========================================
    # 4. 结果记录与保存
    # ==========================================

    config = {
        "N": N,
        "d": d,  # 记录维度
        "d_s": d_s,  # 记录维度
        "L": L,
        "T_end": T_end,
        "dt_fixed": dt_fixed,
        "params": params,
        "integrator": integrator_name,
        "total_time_s": total_time,
        "compile_time_s": compile_time,
        "simulation_time_s": sim_time,
        "init_time_s": init_time,
        "simulation_steps": len(ts),
        "seed": seed,
        "timestamp": timestamp,
        "status": "completed",
    }

    print("-" * 60)
    print("SIMULATION SUMMARY:")
    print(f"  Total run time: {config['total_time_s']:.4f}s")
    print(f"  Compilation time: {config['compile_time_s']:.4f}s")
    print(f"  Simulation time: {config['simulation_time_s']:.4f}s")
    print(f"  Initialization time: {config['init_time_s']:.4f}s")
    print(f"  Simulation steps: {config['simulation_steps']}")
    print(f"  Final time: {ts[-1]:.4f}")
    print("-" * 60)

    save_results(ts, ys, config, output_dir)
    print(f"Results saved successfully to directory: {output_dir}")

    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{end_timestamp}] Simulation completed successfully")


if __name__ == "__main__":
    # --- 示例调用 ---

    # 定义核心参数
    N_sim = 1000
    D_sim = 2  # 使用 2D 空间
    DS_sim = 2  # 使用 2D 自旋
    T_sim = 100.0
    DT_sim = 0.1
    L_sim = 5.0
    PARAMS_sim = np.array([1.0, 1.0, 1.0, -0.1])  # A, B, J, K

    # 配置 1: RK2 Fixed 积分
    run_simulation(
        N=N_sim,
        d=D_sim,
        d_s=DS_sim,
        L=L_sim,
        T_end=T_sim,
        dt_fixed=DT_sim,
        params=PARAMS_sim,
        integrator_name="rk2_fixed",
        output_dir=f"results_N{N_sim}_D{D_sim}DS{DS_sim}_RK2",
    )
