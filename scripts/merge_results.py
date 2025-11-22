#!/usr/bin/env python3
# scripts/merge_results.py
"""
增量式结构化合并 Swarmlator 模拟结果，特别适用于 J-K 参数扫描。
核心特性：绝不将完整的轨迹 y 加载到内存中，仅加载最后一帧用于计算序参量。
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np


def collect_runs(job_dir: Path) -> List[Path]:
    """收集所有包含 results.h5 的 run 目录"""
    if not job_dir.is_dir():
        raise ValueError(f"Job directory not found: {job_dir}")
    runs = [
        item
        for item in job_dir.iterdir()
        if item.is_dir() and (item / "results.h5").exists()
    ]
    return sorted(runs, key=lambda x: x.name)


def load_run_metadata(run_dir: Path) -> Dict[str, Any]:
    """加载单个 run 的元数据"""
    meta_path = run_dir / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_scan_grid(
    runs: List[Path], tolerance: float = 1e-6
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    尝试从 runs 中推断 J-K 扫描网格。
    返回: (J_unique, K_unique) 或 None（非网格扫描）
    """
    J_vals, K_vals = [], []
    for run in runs:
        meta = load_run_metadata(run)
        params = meta["params"]
        if len(params) < 4:
            return None
        J, K = float(params[2]), float(params[3])
        J_vals.append(J)
        K_vals.append(K)

    J_arr = np.array(J_vals)
    K_arr = np.array(K_vals)

    J_unique = np.unique(np.round(J_arr / tolerance).astype(int)) * tolerance
    K_unique = np.unique(np.round(K_arr / tolerance).astype(int)) * tolerance

    if len(J_unique) * len(K_unique) == len(runs):
        expected = {(round(j, 6), round(k, 6)) for j in J_unique for k in K_unique}
        actual = {(round(j, 6), round(k, 6)) for j, k in zip(J_arr, K_arr)}
        if expected == actual:
            return J_unique, K_unique

    return None


def compute_synchrony(y_snapshot: np.ndarray, d: int, d_s: int) -> float:
    """
    y_snapshot: (N, 2*d + 2*d_s) — 单一时间步的状态
    同步性: r = |(1/N) Σ exp(i θ_j)|
    """
    spin = y_snapshot[:, 2 * d : 2 * d + d_s]
    r = np.linalg.norm(np.mean(spin, axis=0))
    return float(r)


def compute_alignment(y_snapshot: np.ndarray, d: int, d_s: int) -> float:
    """
    对齐性: v_align = |(1/N) Σ (v_j / ||v_j||)|
    """
    v = y_snapshot[:, d : 2 * d]
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    v_hat = v / norms
    return float(np.linalg.norm(np.mean(v_hat, axis=0)))


def compute_clustering(y_snapshot: np.ndarray, d: int, d_s: int) -> float:
    """
    可选：若在周期性边界 [0, L)^d 中，需做最小镜像处理。
    这里先给无边界/大系统近似版本。
    """
    pos = y_snapshot[:, :d]  # (N, d)
    centroid = np.mean(pos, axis=0)  # 系统质心
    mean_sq_dist = np.mean(np.sum((pos - centroid) ** 2, axis=1))
    # 可返回 1 / (1 + mean_sq_dist) 作为聚集度（值越大越聚集）
    # 或直接返回 rms 半径（越小越聚集）
    return float(np.sqrt(mean_sq_dist))  # 返回空间弥散尺度，非 [0,1]，但有意义


def compute_speed_cv(y_snapshot: np.ndarray, d: int, d_s: int) -> float:
    v = y_snapshot[:, d : 2 * d]
    speeds = np.linalg.norm(v, axis=1)  # (N,)
    mean_speed = np.mean(speeds)
    if mean_speed == 0:
        return 0.0
    cv = np.std(speeds) / mean_speed  # 变异系数（无量纲涨落）
    return float(cv)


def compute_spin_omega_cv(y_snapshot: np.ndarray, d: int, d_s: int) -> float:
    omega = y_snapshot[:, 2 * d + d_s : 2 * d + 2 * d_s]  # (N, d_s)
    mag = np.linalg.norm(omega, axis=1)
    if np.mean(mag) == 0:
        return 0.0
    return float(np.std(mag) / np.mean(mag))


def compute_spin_anisotropy(y_snapshot: np.ndarray, d: int, d_s: int) -> float:
    s = y_snapshot[:, 2 * d : 2 * d + d_s]  # (N, d_s), assumed normalized
    cov = np.cov(s, rowvar=False)  # (d_s, d_s) 协方差矩阵
    eigs = np.linalg.eigvalsh(cov)
    eigs = np.sort(eigs)[::-1]  # 降序
    # 各向异性：最大特征值占总和的比例
    return float(eigs[0] / (np.sum(eigs) + 1e-12))  # ∈ [1/d_s, 1]


def compute_spin_angular_dispersion(
    y_snapshot: np.ndarray, d: int, d_s: int, max_pairs=5000
) -> float:
    s = y_snapshot[:, 2 * d : 2 * d + d_s]  # assumed normalized
    N = s.shape[0]
    if N < 2:
        return 0.0

    if N * (N - 1) // 2 > max_pairs:
        # 随机采样
        n_sample = int(np.ceil(np.sqrt(2 * max_pairs)))
        n_sample = min(n_sample, N)
        idx = np.random.choice(N, size=n_sample, replace=False)
        s_sub = s[idx]
    else:
        s_sub = s

    dot_matrix = np.dot(s_sub, s_sub.T)
    dot_matrix = np.clip(dot_matrix, -1.0, 1.0)
    triu_mask = np.triu(np.ones_like(dot_matrix, dtype=bool), k=1)
    if not np.any(triu_mask):
        return 0.0
    mean_cos = np.mean(dot_matrix[triu_mask])
    return float(1.0 - mean_cos)


def compute_winding_number_1d(y_snapshot: np.ndarray, d: int, d_s: int) -> float:
    if d != 1 or d_s != 2:
        return np.nan
    s = y_snapshot[:, 2 * d : 2 * d + d_s]  # (N, 2)
    theta = np.arctan2(s[:, 1], s[:, 0])
    dtheta = np.diff(np.unwrap(theta))
    winding = np.sum(dtheta) / (2 * np.pi)
    return float(winding)


ORDER_PARAMETER_FUNCTIONS = {
    "synchrony": compute_synchrony,
    "alignment": compute_alignment,
    "clustering": compute_clustering,
    "speed_cv": compute_speed_cv,
    "spin_omega_cv": compute_spin_omega_cv,
    "spin_anisotropy": compute_spin_anisotropy,
    "spin_angular_dispersion": compute_spin_angular_dispersion,
    "winding_number_1d": compute_winding_number_1d,
}


def merge_as_grid(
    job_dir: Path,
    output_file: Path,
    runs: List[Path],
    J_unique: np.ndarray,
    K_unique: np.ndarray,
    cleanup: bool = False,
) -> None:
    """按 J-K 网格结构化合并，增量式处理"""
    nJ, nK = len(J_unique), len(K_unique)
    print(f"Detected J-K grid: {nJ} J values × {nK} K values")

    meta0 = load_run_metadata(runs[0])
    N = int(meta0["N"])
    d = int(meta0["d"])
    d_s = int(meta0["d_s"])
    total_state_dim = 2 * d + 2 * d_s

    J_to_idx = {round(val, 6): i for i, val in enumerate(J_unique)}
    K_to_idx = {round(val, 6): k for k, val in enumerate(K_unique)}

    # **关键增量步骤 1: 确定时间轴长度 T**
    # 仅读取 shape，不加载数据
    T = 0
    t_ref = None
    for run in runs:
        with h5py.File(run / "results.h5", "r") as src:
            T = max(T, src["t"].shape[0])
            if t_ref is None:
                t_ref = src["t"][:]

    y_final_shape = (nJ, nK, T, N, total_state_dim)

    with h5py.File(output_file, "w") as h5f:
        h5f.create_dataset("J_values", data=J_unique, dtype="f8")
        h5f.create_dataset("K_values", data=K_unique, dtype="f8")
        h5f.attrs.update(
            {
                "N": N,
                "d": d,
                "d_s": d_s,
                "state_dim": total_state_dim,
                "scan_type": "J_K_grid",
                "total_runs": nJ * nK,
            }
        )

        # 创建主数据集
        y_ds = h5f.create_dataset(
            "y",
            y_final_shape,
            dtype="f8",
            compression="gzip",
            compression_opts=4,
            fillvalue=np.nan,
        )
        h5f.create_dataset(
            "t", data=t_ref if t_ref is not None else np.arange(T), dtype="f8"
        )

        order_params = {}
        for name in ORDER_PARAMETER_FUNCTIONS:
            order_params[name] = np.full((nJ, nK), np.nan, dtype="f8")

        # 逐个处理 run
        for run in runs:
            meta = load_run_metadata(run)
            J, K = float(meta["params"][2]), float(meta["params"][3])
            j_idx = J_to_idx[round(J, 6)]
            k_idx = K_to_idx[round(K, 6)]

            with h5py.File(run / "results.h5", "r") as src:
                T_run = src["y"].shape[0]
                write_len = min(T, T_run)

                if T_run > 0:
                    y_last = src["y"][-1]  # (N, total_state_dim)
                    for name, func in ORDER_PARAMETER_FUNCTIONS.items():
                        try:
                            order_params[name][j_idx, k_idx] = func(y_last, d, d_s)
                        except Exception as e:
                            print(
                                f"⚠️  Failed to compute {name} for run {run.name}: {e}"
                            )
                            order_params[name][j_idx, k_idx] = np.nan

                # 复制轨迹（不变）
                temp_group = h5f.create_group(f"_temp_{j_idx}_{k_idx}")
                src.copy("y", temp_group, name="y_temp")
                y_ds[j_idx, k_idx, :write_len] = temp_group["y_temp"][:write_len]
                del h5f[f"_temp_{j_idx}_{k_idx}"]

        # 存储所有序参量
        for name, data in order_params.items():
            h5f.create_dataset(name, data=data, dtype="f8", compression="gzip")

    if cleanup:
        _cleanup_runs(runs)


def merge_as_jk_scatter(
    job_dir: Path,
    output_file: Path,
    runs: List[Path],
    cleanup: bool = False,
) -> None:
    """合并非网格 J-K 扫描：始终按 (J, K) 构建稀疏结构，不退化为无序列表"""
    print("Merging as sparse J-K scan (non-grid supported)...")

    # Step 1: 提取所有 (J, K)
    J_vals, K_vals = [], []
    run_jk_list = []
    for run in runs:
        meta = load_run_metadata(run)
        params = meta["params"]
        if len(params) < 4:
            raise ValueError(f"Run {run} missing J/K in params")
        J, K = float(params[2]), float(params[3])
        J_vals.append(J)
        K_vals.append(K)
        run_jk_list.append((run, J, K))

    # Step 2: 构建唯一 J, K（排序以保持可读性）
    J_unique = np.array(sorted(set(np.round(J_vals, 6))))
    K_unique = np.array(sorted(set(np.round(K_vals, 6))))
    nJ, nK = len(J_unique), len(K_unique)

    J_to_idx = {round(val, 6): i for i, val in enumerate(J_unique)}
    K_to_idx = {round(val, 6): k for k, val in enumerate(K_unique)}

    # Step 3: 获取系统维度
    sample_meta = load_run_metadata(runs[0])
    N = int(sample_meta["N"])
    d = int(sample_meta["d"])
    d_s = int(sample_meta["d_s"])
    total_state_dim = 2 * d + 2 * d_s

    # Step 4: 确定最大时间长度 T
    T = 0
    t_ref = None
    for run, _, _ in run_jk_list:
        with h5py.File(run / "results.h5", "r") as src:
            T = max(T, src["t"].shape[0])
            if t_ref is None:
                t_ref = src["t"][:]

    y_final_shape = (nJ, nK, T, N, total_state_dim)

    # Step 5: 创建 HDF5 文件
    with h5py.File(output_file, "w") as h5f:
        h5f.create_dataset("J_values", data=J_unique, dtype="f8")
        h5f.create_dataset("K_values", data=K_unique, dtype="f8")
        h5f.attrs.update(
            {
                "N": N,
                "d": d,
                "d_s": d_s,
                "state_dim": total_state_dim,
                "scan_type": "J_K_sparse",  # 注意：不再是 flat_list
                "total_runs": len(runs),
            }
        )

        # 主轨迹数据集（缺失点保持 fillvalue=np.nan）
        y_ds = h5f.create_dataset(
            "y",
            y_final_shape,
            dtype="f8",
            compression="gzip",
            compression_opts=4,
            fillvalue=np.nan,
        )
        h5f.create_dataset(
            "t", data=t_ref if t_ref is not None else np.arange(T), dtype="f8"
        )

        # 初始化所有序参量
        order_params = {
            name: np.full((nJ, nK), np.nan, dtype="f8")
            for name in ORDER_PARAMETER_FUNCTIONS
        }

        # Step 6: 处理每个 run
        for run, J, K in run_jk_list:
            j_idx = J_to_idx[round(J, 6)]
            k_idx = K_to_idx[round(K, 6)]

            with h5py.File(run / "results.h5", "r") as src:
                T_run = src["y"].shape[0]
                if T_run > 0:
                    y_last = src["y"][-1]  # (N, total_state_dim)
                    for name, func in ORDER_PARAMETER_FUNCTIONS.items():
                        try:
                            order_params[name][j_idx, k_idx] = func(y_last, d, d_s)
                        except Exception as e:
                            print(
                                f"⚠️  Failed to compute {name} for (J={J}, K={K}): {e}"
                            )
                            order_params[name][j_idx, k_idx] = np.nan

                    # 增量复制轨迹
                    temp_group = h5f.create_group(f"_temp_{j_idx}_{k_idx}")
                    src.copy("y", temp_group, name="y_temp")
                    write_len = min(T, T_run)
                    y_ds[j_idx, k_idx, :write_len] = temp_group["y_temp"][:write_len]
                    del h5f[f"_temp_{j_idx}_{k_idx}"]

        # 存储所有序参量
        for name, data in order_params.items():
            h5f.create_dataset(name, data=data, dtype="f8", compression="gzip")

    if cleanup:
        _cleanup_runs(runs)


def _cleanup_runs(runs: List[Path]) -> None:
    if runs:
        for run in runs:
            shutil.rmtree(run)
            print(f"🧹 Removed {run}")
        print(f"🧹 Removed {len(runs)} runs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Swarmlator results (incremental & grid-aware)."
    )
    parser.add_argument("job_dir", type=str, help="Directory containing run folders")
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output HDF5 path"
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Delete original runs after merge"
    )
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    output = Path(args.output) if args.output else job_dir / "merged_results.h5"
    runs = collect_runs(job_dir)

    if not runs:
        raise ValueError(f"No runs found in {job_dir}")

    print(f"Found {len(runs)} runs. Attempting to detect parameter grid...")

    grid_info = infer_scan_grid(runs)
    if grid_info is not None:
        J_unique, K_unique = grid_info
        merge_as_grid(job_dir, output, runs, J_unique, K_unique, cleanup=args.cleanup)
    else:
        merge_as_jk_scatter(job_dir, output, runs, cleanup=args.cleanup)

    print(f"✅ Merge completed: {output}")


if __name__ == "__main__":
    main()
