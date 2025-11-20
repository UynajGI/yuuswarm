# scripts/main.py
import hashlib
import json  # 用于保存元数据
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from dotenv import load_dotenv

# 获取当前脚本所在目录，然后向上一级得到项目根目录（proj/）
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent  # 因为 script/ 在 proj/ 下

# 加载 proj/.env
dotenv_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path)

# 确保能导入你的模块
sys.path.append(str(PROJECT_ROOT / "src"))
# 导入你定义好的模块
from engine import create_solver, init_states  # noqa: E402
from integrator import rk2_fixed, rk23_adaptive  # noqa: E402
from user_model import original_interaction  # noqa: E402

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "output")).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_config_by_id(exp_id: int):
    """从 experiments.json 文件加载指定 ID 的配置。"""
    # 假设 experiments.json 位于脚本运行目录或项目根目录
    CONFIG_PATH = SCRIPT_DIR / "experiments.json"

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        all_configs = json.load(f)

    # 数组索引刚好对应 ID
    if exp_id >= len(all_configs):
        raise IndexError(f"Experiment ID {exp_id} out of bounds.")

    config = all_configs[exp_id]

    # 将 params 列表转换回 NumPy 数组
    config["params"] = np.array(config["params"])

    return config


def get_run_id(config):
    """根据核心参数生成一个简短的、唯一的运行ID。"""
    # 提取所有影响模拟结果的关键参数
    # 注意：必须将 NumPy 数组转换为列表，以保证 JSON 可序列化和哈希一致性

    # 移除非核心/变动的元数据，如时间、运行时长等
    core_config = {
        "N": config["N"],
        "d": config["d"],
        "d_s": config["d_s"],
        "L": config["L"],
        "T_end": config["T_end"],
        "dt_fixed": config["dt_fixed"],
        "params": config["params"].tolist(),  # 转换为列表
        "integrator": config["integrator"],
        "seed": config["seed"],
    }

    # 将字典规范化为字符串，进行哈希计算
    config_str = json.dumps(core_config, sort_keys=True)

    # 使用 SHA-1 或 SHA-256 生成哈希值，取前几位作为短ID
    run_hash = hashlib.sha1(config_str.encode("utf-8")).hexdigest()[:8]

    # 构造可读性强的目录名
    dirname = (
        f"sim_N{config['N']}_T{config['T_end']:.0f}_{config['integrator']}_{run_hash}"
    )
    return dirname


def save_results(t_array, y_array, config_data, output_dir: Path):
    """
    保存模拟结果到指定的输出目录，使用 HDF5 (.h5) 格式存储数据和元数据。
    同时生成JSON格式的元数据文件。
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 定义 HDF5 文件路径
    h5_path = output_dir / "results.h5"

    # 定义 JSON 元数据文件路径
    json_path = output_dir / "metadata.json"

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
    print("\nHDF5 file saved successfully.")
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
        "d": d,
        "d_s": d_s,
        "L": L,
        "T_end": T_end,
        "dt_fixed": dt_fixed,
        "params": params,  # <-- 原始 NumPy 数组
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

    run_id = get_run_id(config)
    output_dir_new = Path(output_dir) / str(run_id)
    print(f"Generated Run ID: {run_id}")
    print(f"Saving results to: {output_dir_new}")

    save_results(ts, ys, config, output_dir_new)
    print(f"Results saved successfully to directory: {output_dir_new}")

    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{end_timestamp}] Simulation completed successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a Swarmlators N-body simulation based on an Experiment ID."
    )
    # 只需要一个参数：实验 ID
    parser.add_argument(
        "experiment_id",
        type=int,
        help="The ID of the experiment to run from experiments.json.",
    )

    args = parser.parse_args()
    exp_id = args.experiment_id

    # 1. 加载配置
    try:
        config = load_config_by_id(exp_id)
    except Exception as e:
        print(f"FATAL: Failed to load config for ID {exp_id}: {e}")
        sys.exit(1)

    print(
        f"Loaded Configuration for ID {exp_id}: K={config['params'][3]}, Seed={config['seed']}"
    )

    # 2. 调用主函数
    # 注意：这里需要确保 run_simulation 函数的参数顺序和名称一致
    run_simulation(
        N=config["N"],
        d=config["d"],
        d_s=config["d_s"],
        L=config["L"],
        T_end=config["T_end"],
        dt_fixed=config["dt_fixed"],
        params=config["params"],
        integrator_name=config["integrator"],
        output_dir=OUTPUT_DIR,
        seed=config["seed"],
    )
