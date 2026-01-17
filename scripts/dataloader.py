import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# 尝试导入进度条，没有就用简易版
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterator, **kwargs):
        return iterator


# ================= 配置 =================
INPUT_ROOT = Path(__file__).resolve().parent.parent / "output" / "scan_JK"
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "processed_data_lake"


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


def flatten_json(y):
    """通用递归函数：把任意嵌套的 JSON 展平。"""
    out = {}

    def flatten(x, name=""):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + ".")
        else:
            out[name[:-1]] = x

    flatten(y)
    return out


def generate_dynamic_cols(d, d_s):
    """生成 x_0, x_1... 这种列名"""
    cols = []
    for p in ["x", "v"]:
        for k in range(d):
            cols.append(f"{p}_{k}")
    for p in ["s", "w"]:
        for k in range(d_s):
            cols.append(f"{p}_{k}")
    return cols


def process_experiment_data(exp_dir):
    """
    核心处理逻辑：读取 -> 合并 -> 生成 DataFrame
    """
    exp_id = exp_dir.name

    # 1. 读取 Config
    cfg_path = exp_dir / "config_snapshot.json"
    if not cfg_path.exists():
        return None

    with open(cfg_path, "r") as f:
        config = json.load(f)

    flat_config = flatten_json(config)
    d = config.get("d", 2)
    d_s = config.get("d_s", 2)
    expected_dims = 2 * d + 2 * d_s

    # 2. 读取 NPZ
    npz_files = sorted(
        list(exp_dir.glob("simulation_part_*.npz")),
        key=lambda f: int(re.search(r"part_(\d+)", f.name).group(1)),
    )
    if not npz_files:
        return None

    # 合并数组
    all_t, all_traj = [], []
    for f in npz_files:
        try:
            dat = np.load(f)
            all_t.append(dat["time"])
            all_traj.append(dat["trajectory"])
        except Exception:
            continue

    if not all_traj:
        return None

    t_arr = np.concatenate(all_t)
    traj_arr = np.concatenate(all_traj)  # (Steps, N, Dims)

    steps, n, total_dims = traj_arr.shape
    if total_dims != expected_dims:
        return None  # 维度不匹配直接放弃

    # 3. 构建 DataFrame
    flat_traj = traj_arr.reshape(-1, total_dims)

    data_dict = {
        "t": np.repeat(t_arr, n).astype(np.float32),
        "p_id": np.tile(np.arange(n), steps).astype(np.int32),
        "exp_id": exp_id,
    }

    col_names = generate_dynamic_cols(d, d_s)
    for idx, name in enumerate(col_names):
        data_dict[name] = flat_traj[:, idx].astype(np.float32)

    df = pd.DataFrame(data_dict)

    # 4. 广播参数
    for key, val in flat_config.items():
        if isinstance(val, (int, float, bool, str)) and len(str(val)) < 100:
            clean_key = key.replace("params.", "")
            if clean_key == "n":
                continue  # 避免覆盖实际的 n

            df[clean_key] = val
            if isinstance(val, float):
                df[clean_key] = df[clean_key].astype(np.float32)
            elif isinstance(val, int):
                df[clean_key] = df[clean_key].astype(np.int32)

    df["n"] = n
    return df, flat_config


def worker_task(exp_dir):
    """
    单个 Worker 进程执行的任务：处理 -> 保存 -> 返回元数据
    注意：不在进程间传递 huge DataFrame，直接在子进程存盘！
    """
    try:
        res = process_experiment_data(exp_dir)
        if res is None:
            return None

        df, config_dict = res
        exp_id = exp_dir.name

        # 保存 Parquet (这是耗时操作，要在 Worker 里做)
        save_path = OUTPUT_ROOT / f"{exp_id}.parquet"
        df.to_parquet(save_path, index=False, compression="zstd")

        # 准备返回给主进程的轻量级元数据
        config_dict["exp_id"] = exp_id
        config_dict["file_path"] = str(save_path.name)

        return config_dict

    except Exception as e:
        # 捕获所有异常，防止炸坏 Pool
        print(f"\n❌ Error in {exp_dir.name}: {e}")
        return None


# ... (前面的 import 和函数定义保持不变) ...

# 全局默认配置
FORCE_UPDATE = False


def main():
    # --- 1. 命令行参数控制 ---
    global FORCE_UPDATE
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        FORCE_UPDATE = True
        print("⚠️  FORCE MODE: All existing files will be overwritten!")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not INPUT_ROOT.exists():
        print("Input dir not found")
        return

    # 获取所有实验文件夹
    exp_dirs = [d for d in INPUT_ROOT.iterdir() if d.is_dir()]

    # --- 2. 筛选需要处理的任务 (增量逻辑) ---
    tasks_to_run = []
    skipped_count = 0

    print(f"🔍 Scanning {len(exp_dirs)} directories...")

    for d in exp_dirs:
        exp_id = d.name
        target_parquet = OUTPUT_ROOT / f"{exp_id}.parquet"

        # 核心判断逻辑
        if target_parquet.exists() and not FORCE_UPDATE:
            # 如果文件存在，且没有开启强制更新，就跳过
            skipped_count += 1
            continue
        else:
            # 否则加入任务列表
            tasks_to_run.append(d)

    print(
        f"📋 Plan: Process {len(tasks_to_run)} new/changed, Skip {skipped_count} existing."
    )

    if not tasks_to_run:
        print("✅ Nothing to do.")
        return

    # --- 3. 并行执行 (只跑筛选出来的任务) ---
    metadata_list = []

    # 注意：这里我们只提交 tasks_to_run
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(worker_task, d) for d in tasks_to_run]

        for future in tqdm(as_completed(futures), total=len(tasks_to_run), unit="exp"):
            result = future.result()
            if result:
                metadata_list.append(result)

    print(f"\n💾 Saved {len(metadata_list)} new parquet files.")

    # --- 4. 索引文件的处理 (注意！) ---
    # 如果是增量更新，metadata_list 只包含“本次新跑”的实验。
    # 为了保证 _metadata_index.parquet 包含所有（旧+新）数据，我们需要把旧的索引读进来合并

    index_path = OUTPUT_ROOT / "_metadata_index.parquet"
    final_meta_df = pd.DataFrame(metadata_list)

    if index_path.exists() and not FORCE_UPDATE:
        try:
            print("🔗 Merging with existing index...")
            old_index = pd.read_parquet(index_path)
            # 合并新旧索引，并根据 exp_id 去重 (保留新的)
            if not final_meta_df.empty:
                # 清理列名以匹配
                final_meta_df.columns = [
                    c.replace("params.", "") for c in final_meta_df.columns
                ]

                combined = pd.concat([old_index, final_meta_df])
                # drop_duplicates: 比如你强制更新了某个文件，这里要把旧索引里的它删掉
                final_meta_df = combined.drop_duplicates(subset=["exp_id"], keep="last")
            else:
                final_meta_df = old_index
        except Exception as e:
            print(f"⚠️ Failed to read old index, creating new one: {e}")

    # 清理列名 (如果是全新生成的)
    if not final_meta_df.empty:
        # 确保列名没有 'params.' 前缀 (如果是从 metadata_list 新建的)
        final_meta_df.columns = [
            c.replace("params.", "") for c in final_meta_df.columns
        ]

        final_meta_df.to_parquet(index_path, index=False)
        print(f"✅ Index updated: {index_path} ({len(final_meta_df)} experiments)")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
