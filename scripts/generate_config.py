# scripts/generate_config.py
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# --- 1. 定义参数空间 ---

# 待扫描的参数
K_values = np.linspace(-0.2, 1.0, 11)  # 11个K值
J_values = np.linspace(-0.2, 1.0, 11)

# 固定参数
N_fixed = 1000
D_fixed = 2
DS_fixed = 2
L_fixed = 2.0
T_end_fixed = 100.0
DT_fixed_fixed = 0.1
A_fixed, B_fixed = 1.0, 1.0
SEED_fixed = 42
Integrator_fixed = "rk2_fixed"

# --- 2. 生成实验配置列表 ---
experiments = []
exp_id_counter = 0

for K_val in K_values:
    for J_val in J_values:
        config = {
            "id": exp_id_counter,
            "N": N_fixed,
            "d": D_fixed,
            "d_s": DS_fixed,
            "L": L_fixed,
            "T_end": T_end_fixed,
            "dt_fixed": DT_fixed_fixed,
            # 将 params 作为列表保存
            "params": [
                A_fixed,
                B_fixed,
                round(J_val, 4),
                round(K_val, 4),
            ],  # round确保JSON可读性
            "integrator": Integrator_fixed,
            "seed": SEED_fixed,
        }
        experiments.append(config)
        exp_id_counter += 1

# --- 3. 保存到 JSON 文件 ---
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = SCRIPT_DIR / "experiments.json"
with open(CONFIG_FILE, "w") as f:
    json.dump(experiments, f, indent=4)

print(f"Generated {len(experiments)} experiments.")
print(f"Saved configuration to {CONFIG_FILE}")

# 获取最大ID (Slurm Array Job 范围用)
max_id = len(experiments) - 1
print(f"\nMax Array Job Index (SLURM_ARRAY_TASK_MAX): {max_id}")
print(f"\nSBATCH array range: --array=0-{max_id}")
print(f"Total experiments: {len(experiments)} (J: {len(J_values)}, K: {len(K_values)})")
