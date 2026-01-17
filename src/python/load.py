# scripts/read_results_lazy.py
import json
from pathlib import Path
from typing import Dict, Any, Optional

import h5py
import numpy as np


class MergedResults:
    """
    一个用于懒加载（非内存读取）Swarmlator 合并结果的容器类。
    它保持 HDF5 文件句柄打开，并在需要时通过切片访问数据。
    """
    def __init__(self, h5_file_path: Path, global_meta: Dict[str, Any]):
        self._h5_file_path = h5_file_path
        self._h5f: h5py.File = None
        self.global_meta = global_meta
        self.h5_attrs: Dict[str, Any] = {}

        # 懒加载的数据集占位符
        self.J_values: h5py.Dataset = None
        self.K_values: h5py.Dataset = None
        self.t: h5py.Dataset = None
        self.trajectory_y: h5py.Dataset = None
        self.order_parameters: Dict[str, h5py.Dataset] = {}

    def __enter__(self):
        """进入上下文管理器时打开 HDF5 文件"""
        self._h5f = h5py.File(self._h5_file_path, "r")

        # 立即加载不可变属性
        self.h5_attrs = dict(self._h5f.attrs)

        # 存储数据集引用 (不触发读取)
        self.J_values = self._h5f["J_values"]
        self.K_values = self._h5f["K_values"]
        self.t = self._h5f["t"]
        self.trajectory_y = self._h5f["y"]

        # 存储序参量引用 (不触发读取)
        for name in self._h5f:
            if name not in ["y", "t", "J_values", "K_values"]:
                self.order_parameters[name] = self._h5f[name]

        print(f"Loaded HDF5 file references from {self._h5_file_path.name}. Data remains on disk.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器时关闭 HDF5 文件"""
        if self._h5f:
            self._h5f.close()
            print(f"Closed HDF5 file {self._h5_file_path.name}.")

    # --- 辅助方法：按需切片读取（示例）---

    def get_op_data(self, op_name: str) -> np.ndarray:
        """获取并加载指定的序参量数组（通常较小，可加载）"""
        if op_name in self.order_parameters:
            return self.order_parameters[op_name][:]  # 触发内存读取
        raise KeyError(f"Order parameter '{op_name}' not found.")

    def get_run_trajectory(self, j_idx: int, k_idx: int) -> np.ndarray:
        """
        根据 J-K 索引获取并加载单个 run 的完整轨迹。
        注意：这会加载一个 (T, N, state_dim) 的大数组到内存。
        """
        print(f"Reading full trajectory for J_idx={j_idx}, K_idx={k_idx}...")
        # 触发内存读取：只读取指定的 (j, k) 索引切片
        return self.trajectory_y[j_idx, k_idx, :, :, :]


def load_lazy_results(job_dir_or_file: Path) -> MergedResults:
    """
    懒加载 Swarmlator 模拟合并结果，返回一个 MergedResults 对象。
    使用 'with' 语句确保 HDF5 文件正确关闭。
    """
    # 确定 HDF5 文件路径和元数据文件路径
    if job_dir_or_file.is_dir():
        h5_path = job_dir_or_file / "merged_results.h5"
        meta_path = job_dir_or_file / "global_meta.json"
    else:
        h5_path = job_dir_or_file
        meta_path = h5_path.parent / "global_meta.json"

    if not h5_path.exists():
        raise FileNotFoundError(f"Merged HDF5 file not found: {h5_path}")

    # 读取全局元数据 (通常较小，可以直接加载)
    global_meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            global_meta = json.load(f)

    # 返回 MergedResults 类的实例，该类将管理 HDF5 文件 I/O
    return MergedResults(h5_path, global_meta)


# --- 示例使用 ---

if __name__ == "__main__":
    # --- 模拟创建 HDF5 文件（与之前示例相同）---
    temp_dir = Path("./temp_test_job_lazy")
    temp_dir.mkdir(exist_ok=True)

    # 模拟创建 global_meta.json
    with open(temp_dir / "global_meta.json", "w") as f:
        json.dump({"scan_type": "J_K_grid", "fixed_N": 1000}, f)

    # 模拟创建 merged_results.h5
    try:
        h5_path = temp_dir / "merged_results.h5"
        with h5py.File(h5_path, "w") as hf:
            hf.attrs["scan_type"] = "J_K_grid"
            hf.attrs["N"] = 1000

            J_vals = np.array([0.1, 0.2])
            K_vals = np.array([-0.5, 0.0, 0.5])
            nJ, nK = len(J_vals), len(K_vals)
            T, N, D_state = 100, 1000, 8

            hf.create_dataset("J_values", data=J_vals)
            hf.create_dataset("K_values", data=K_vals)
            hf.create_dataset("t", data=np.linspace(0, 100, T))

            # 轨迹数据很大 (nJ, nK, T, N, D_state)
            y_data = np.random.rand(nJ, nK, T, N, D_state)
            hf.create_dataset("y", data=y_data, compression="gzip")

            # 序参量数据 (nJ, nK)
            sync_data = np.array([[0.9, 0.9, 0.2], [0.8, 0.85, 0.3]])
            hf.create_dataset("synchrony", data=sync_data)

        print("\n--- Starting Lazy Loading Example ---")

        # 使用 'with' 语句和新的懒加载函数
        with load_lazy_results(temp_dir) as data:

            # 1. 访问属性和元数据（已在 __enter__ 中加载）
            print(f"HDF5 N: {data.h5_attrs['N']}")
            print(f"Global Meta Scan Type: {data.global_meta['scan_type']}")

            # 2. 访问 J_values, K_values, t - 它们是 h5py.Dataset 对象，数据仍未加载
            print(f"\nJ_values is a {type(data.J_values)}")
            print(f"K-axis length (from disk metadata): {data.K_values.shape[0]}")

            # 3. 访问序参量 - 懒访问，直到显式调用 [:]
            # 序参量数组相对较小，我们通常会加载它进行绘图
            sync_map = data.get_op_data("synchrony")
            print(f"\nLoaded Synchrony Map (Shape: {sync_map.shape})")
            print(f"Synchronization at (0, 0): {sync_map[0, 0]:.4f}")

            # 4. 访问大轨迹 y - 仅切片读取，避免整体加载
            print(f"\nTrajectory 'y' Shape (from disk metadata): {data.trajectory_y.shape}")

            # 只加载 J=0.1, K=0.0 (索引 0, 1) 的完整轨迹
            # 整个 (100, 1000, 8) 的数组现在才被加载到内存
            run_y = data.get_run_trajectory(0, 1)
            print(f"Specific run trajectory loaded (Shape: {run_y.shape})")

        print("\nSuccessfully used context manager. File is now closed.")

    finally:
        # 清理临时文件
        Path(temp_dir / "global_meta.json").unlink()
        Path(temp_dir / "merged_results.h5").unlink()
        temp_dir.rmdir()
        print(f"\nCleanup: Removed temporary directory {temp_dir}")