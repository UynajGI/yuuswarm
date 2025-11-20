#!/bin/bash
#SBATCH --job-name=Swarmlator_Scan
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-120
#SBATCH --output=/share/home/jiangyuan/yuuswarm/logs/array_%A_%a.out  # 使用绝对路径

# ==============================
# 固定项目路径（使用绝对路径）
# ==============================
PROJECT_ROOT="/share/home/jiangyuan/yuuswarm"

# 直接切换到项目目录
cd "$PROJECT_ROOT"

# 确保logs目录存在
mkdir -p logs

# 激活虚拟环境
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment '.venv' not found in $PROJECT_ROOT"
    echo "Available files in project root:"
    ls -la
    exit 1
fi

# 设置线程数
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

EXP_ID=$SLURM_ARRAY_TASK_ID

echo "=== 环境信息 ==="
echo "用户: $USER"
echo "项目目录: $PROJECT_ROOT"
echo "当前目录: $(pwd)"
echo "Python: $(which python)"
echo "虚拟环境: $VIRTUAL_ENV"
echo "任务ID: $EXP_ID"

# 运行主程序
python scripts/main.py $EXP_ID --job-id $SLURM_ARRAY_JOB_ID --array-id $SLURM_ARRAY_TASK_ID

echo "--- Simulation Finished ---"