#!/bin/bash
# 任务名称和分区
#SBATCH --job-name=Swarmlator_Scan
#SBATCH --partition=standard

# --- 资源配置优化 ---
# 节点数：对于单任务（Array Task）可以省略或指定为 1
# #SBATCH --nodes=1
# 每个任务的进程数 (保持为 1，因为您的 Python 脚本是一个进程)
#SBATCH --ntasks-per-node=1

# 每个任务请求的核心数（用于Numba多线程）
# 建议值：根据您的集群和Numba的效率，选择 4, 8, 16 等
#SBATCH --cpus-per-task=8

# 内存请求：核心数增加，内存也应增加以安全运行
#SBATCH --mem=8G

# 运行时间限制
#SBATCH --time=01:00:00

# --- 核心配置：使用 Array Job ---
# 请根据 generate_config.py 的输出修改这个范围!
#SBATCH --array=0-32
# 日志文件命名：使用主 Job ID (%A) 和 Array Task ID (%a)
#SBATCH --output=logs/array_%A_%a.out

# --- 环境变量设置 ---
# 将 Numba/OpenMP 的线程数设置为 Slurm 分配给当前任务的核心数
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 定义配置文件的路径 (确保路径正确)
CONFIG_FILE=experiments.json

# 加载环境和模块
source /path/to/your/conda/miniconda3/bin/activate your_env_name

# 核心逻辑
# $SLURM_ARRAY_TASK_ID 自动匹配 --array 中的索引，即我们的实验 ID
EXP_ID=$SLURM_ARRAY_TASK_ID

echo "--- Starting Simulation ---"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Array Task ID (Experiment ID): $EXP_ID"
echo "Requested Cores (NUMBA_NUM_THREADS): $NUMBA_NUM_THREADS"

# 运行 Python 脚本，将 $EXP_ID 作为命令行参数传入
cd /home/jiangyuan/swarmlators
python scripts/main.py $EXP_ID

echo "--- Simulation Finished ---"