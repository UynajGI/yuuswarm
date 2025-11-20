#!/bin/bash
#SBATCH --job-name=Swarmlator_Scan
#SBATCH --partition=standard
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-32
#SBATCH --output=logs/array_%A_%a.out

# ==============================
# 自动定位项目根目录（关键！）
# ==============================
# 获取当前 Slurm 脚本的绝对路径（即使从其他目录提交也能正确解析）
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
# scripts/ 的父目录就是项目根
PROJECT_ROOT="$(dirname "$SCRIPT_PATH")/.."
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"  # 标准化为绝对路径

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 确保 logs 目录存在
mkdir -p logs

# 激活虚拟环境（在项目根下）
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment '.venv' not found in $PROJECT_ROOT"
    exit 1
fi

# 设置线程数
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 实验 ID
EXP_ID=$SLURM_ARRAY_TASK_ID

echo "--- Starting Swarmlators Simulation ---"
echo "Detected project root: $PROJECT_ROOT"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $EXP_ID"
echo "Using $(nproc) CPUs, NUMBA threads: $NUMBA_NUM_THREADS"

# 运行主程序（相对路径，从项目根调用）
python scripts/main.py $EXP_ID

echo "--- Simulation Finished ---"