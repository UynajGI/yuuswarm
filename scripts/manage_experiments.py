import copy
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np

# ================= 配置区域 =================

# 1. 基础物理配置模板
BASE_CONFIG = {
    "job_name": "scan_JK",  # 作业名称 (决定了 output 下的一级子目录)
    "output_base_dir": "",  # [程序将自动填入你的 shared output 路径]
    "n": 1000,
    "d": 2,
    "d_s": 2,
    "L": 3.0,
    "t_end": 5000.0,
    "dt": 0.01,
    "seed": 42,
    "params": {
        "A": 1.0,
        "B": 1.0,  # 固定参数
        "J": 0.0,
        "K": 0.0,  # 扫描参数占位符
        "epsilon": 1.0,
        "sigma": 1.0,
        "eta": 1.0,
    },
    "save_interval": 25,
    "chunk_size": 2000,
}

# 2. 扫描参数空间
SCAN_PARAMS = {
    "n": [256, 512, 1024, 2048],
    "J": [0.5],
    "K": np.arange(-0.7, 0.1 + 1e-6, 0.01).round(4).tolist(),
}

# 3. Slurm 资源配置
SLURM_CONFIG = {
    "cpus_per_task": 6,
    "time": "04:00:00",
    "partition": "cpu_amd",
    "exec_path": "./build/simulation",
    "num_workers": 12,
}

# ================= 自动化逻辑 =================


def get_project_root():
    """根据脚本位置自动寻找项目根目录 (假设脚本在 scripts/ 下)"""
    script_path = Path(__file__).resolve()
    return script_path.parent.parent


def update_recursive(config, key, value):
    """
    递归查找 config 中的 key 并更新为 value。
    如果找到了并更新成功，返回 True；否则返回 False。
    """
    if key in config:
        config[key] = value
        return True

    for v in config.values():
        if isinstance(v, dict):
            if update_recursive(v, key, value):
                return True
    return False


def main():
    project_root = get_project_root()
    print(f"🏠 Project Root: {project_root}")

    shared_output_dir = project_root / "output"
    shared_output_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    batch_name = f"batch_{timestamp}"
    batch_dir = project_root / "experiments" / batch_name

    dirs = {
        "root": batch_dir,
        "configs": batch_dir / "configs",
        "logs": batch_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"📂 Batch Directory: {batch_dir}")

    # 生成配置
    task_list = []
    keys = list(SCAN_PARAMS.keys())
    values = list(SCAN_PARAMS.values())
    combinations = list(itertools.product(*values))

    print(f"⚙️ Generating {len(combinations)} configurations...")

    for i, combo in enumerate(combinations):
        config = copy.deepcopy(BASE_CONFIG)
        config["output_base_dir"] = str(shared_output_dir.resolve())
        config["seed"] = 42 + i

        # --- [核心修改] 自动递归注入 ---
        for k, v in zip(keys, combo):
            found = update_recursive(config, k, v)
            if not found:
                # 如果模板里根本没这个参数，说明拼写错了，必须报错！
                print(f"❌ Error: Parameter '{k}' not found in BASE_CONFIG_TEMPLATE!")
                return

        # 文件名格式化
        def fmt(val):
            return f"{val:.4g}" if isinstance(val, float) else str(val)

        param_str = "_".join([f"{k}{fmt(v)}" for k, v in zip(keys, combo)])
        filename = f"cfg_{i:04d}_{param_str}.json"
        file_path = dirs["configs"] / filename

        with open(file_path, "w") as f:
            json.dump(config, f, indent=2)

        task_list.append(str(file_path.resolve()))

    # 生成任务列表
    tasks_file_path = dirs["root"] / "tasks.txt"
    with open(tasks_file_path, "w") as f:
        f.write("\n".join(task_list))

    # 生成 Slurm 脚本
    exec_abs_path = (project_root / SLURM_CONFIG["exec_path"]).resolve()
    if not exec_abs_path.exists():
        print(f"❌ Error: Executable not found at {exec_abs_path}")
        return

    slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name={config["job_name"]}
#SBATCH --output={dirs["logs"]}/%A_%a.out
#SBATCH --error={dirs["logs"]}/%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={SLURM_CONFIG["cpus_per_task"]}
#SBATCH --time={SLURM_CONFIG["time"]}
#SBATCH --array=1-{len(task_list)}%{SLURM_CONFIG["num_workers"]}
#SBATCH --partition={SLURM_CONFIG["partition"]}

module load aocl4.2

CONFIG_FILE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "{tasks_file_path}")

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: Config file not found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running Task: $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG_FILE"

export OMP_NUM_THREADS={SLURM_CONFIG["cpus_per_task"]}

"{exec_abs_path}" "$CONFIG_FILE"
"""

    slurm_script_path = dirs["root"] / "run.sh"
    with open(slurm_script_path, "w") as f:
        f.write(slurm_script_content)

    print("-" * 40)
    print(f"✅ Ready! {len(task_list)} jobs prepared.")

    user_input = input("🚀 Submit to Slurm now? (y/n): ").strip().lower()
    if user_input == "y":
        try:
            result = subprocess.run(
                ["sbatch", str(slurm_script_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"\n🎉 {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Submission failed: {e.stderr}")


if __name__ == "__main__":
    main()
