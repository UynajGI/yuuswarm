import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import numpy as np
import pandas as pd

# 尝试导入进度条
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterator, **kwargs):
        return iterator


def get_slurm_cores():
    """尝试获取 Slurm 分配的核数，如果没在 Slurm 里则返回物理核数"""
    # 1. 优先尝试读取 Linux 进程亲和性 (最准确，能看到实际能用的核)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        pass

    # 2. 尝试读取 Slurm 环境变量
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)

    # 3.不仅如此，防止 OOM，还是保守一点
    return max(1, os.cpu_count() - 1)


# 设置 Worker 数量
NUM_WORKERS = get_slurm_cores()


# ================= 配置 =================
# [修正路径] 假设脚本在 scripts/analysis/ 下，根目录是向上 3 层
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "processed_data_lake"
OUTPUT_DIR = PROJECT_ROOT / "output" / "analysis" / "animations"

# 目标参数列表
TARGETS = [
    {"n": 2000, "J": 0.6, "K": -0.9},
    {"n": 2000, "J": 0.6, "K": -0.4},
    {"n": 2000, "J": 0.6, "K": -0.1},
    {"n": 2000, "J": 0.6, "K": 0},
    {"n": 2000, "J": 0.6, "K": 0.1},
]

# 动画设置
FPS = 10
DOWNSAMPLE = 1
DOT_SIZE = 5


def get_closest_experiment(index_df, target):
    """在索引中查找最接近 target 参数的实验"""
    mask = (
        (index_df["n"] == target["n"])
        & (np.isclose(index_df["J"], target["J"], atol=0.01))
        & (np.isclose(index_df["K"], target["K"], atol=0.01))
    )
    matches = index_df[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def render_worker(exp_meta):
    """
    单个 Worker 进程：负责加载数据、画图、保存视频
    """
    # [关键] 设置非交互式后端，防止多进程绘图冲突
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exp_id = exp_meta["exp_id"]
    file_path = DATA_DIR / exp_meta["file_path"]
    L = exp_meta.get("L", 3.0)

    try:
        # 1. 读取 Parquet
        df = pd.read_parquet(file_path)
        df = df.sort_values(["t", "p_id"])

        n_particles = int(exp_meta["n"])
        n_steps = len(df) // n_particles

        # 2. 数据处理
        pos = df[["x_0", "x_1"]].values.reshape(n_steps, n_particles, 2)
        spin = df[["s_0", "s_1"]].values.reshape(n_steps, n_particles, 2)
        angles = np.arctan2(spin[:, :, 1], spin[:, :, 0])
        times = df["t"].unique()

        # 3. 降采样
        pos = pos[::DOWNSAMPLE]
        angles = angles[::DOWNSAMPLE]
        times = times[::DOWNSAMPLE]
        n_frames = len(times)

        # 4. 绘图设置
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")

        title_template = f"N={n_particles}, J={exp_meta['J']:.2f}, K={exp_meta['K']:.2f}\nTime: {{:.2f}}"
        title = ax.set_title(title_template.format(times[0]))

        scat = ax.scatter(
            pos[0, :, 0],
            pos[0, :, 1],
            s=DOT_SIZE,
            c=angles[0],
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
            alpha=0.8,
            edgecolors="none",
        )

        def update(frame_idx):
            scat.set_offsets(pos[frame_idx])
            scat.set_array(angles[frame_idx])
            title.set_text(title_template.format(times[frame_idx]))
            return scat, title

        # 5. 生成并保存
        output_filename = (
            OUTPUT_DIR
            / f"video_n{n_particles}_J{exp_meta['J']:.4f}_K{exp_meta['K']:.4f}_{exp_id[:6]}.mp4"
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
        writer = animation.FFMpegWriter(
            fps=FPS, metadata=dict(artist="Yuuswarm"), bitrate=1800
        )
        ani.save(output_filename, writer=writer)

        # [关键] 必须关闭图像释放内存
        plt.close(fig)
        return f"✅ Done: {output_filename.name}"

    except Exception as e:
        plt.close("all")  # 出错也要清理
        return f"❌ Error {exp_id}: {e}"


def main():
    # 1. 加载索引
    index_path = DATA_DIR / "_metadata_index.parquet"
    if not index_path.exists():
        print(f"❌ Index not found at: {index_path}")
        print("   Please run 'srun ... python scripts/npz_to_parquet.py' first.")
        return

    index_df = pd.read_parquet(index_path)
    print(f"📂 Loaded index: {len(index_df)} records.")

    # 2. 准备任务列表
    tasks = []
    print("🔍 Matching experiments...")
    for target in TARGETS:
        meta = get_closest_experiment(index_df, target)
        if meta is not None:
            tasks.append(meta)
        else:
            print(f"   ⚠️ No match for {target}")

    if not tasks:
        print("No tasks to run.")
        return

    print(f"🚀 Starting {len(tasks)} animation tasks with {NUM_WORKERS} workers...")

    # 3. 并行执行
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(render_worker, meta) for meta in tasks]

        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=len(tasks), unit="vid"):
            print(future.result())


if __name__ == "__main__":
    # 必须加这行，防止 Linux/Windows 进程生成方式导致的错误
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
