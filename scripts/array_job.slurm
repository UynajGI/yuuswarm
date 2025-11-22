#!/bin/bash
#SBATCH --job-name=Swarmlator_Scan
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-199
#SBATCH --output=/share/home/jiangyuan/yuuswarm/logs/array_%A_%a.out

PROJECT_ROOT="/share/home/jiangyuan/yuuswarm"
cd "$PROJECT_ROOT"

mkdir -p logs
source .venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

EXP_ID=$SLURM_ARRAY_TASK_ID
echo "Running experiment $EXP_ID"

python scripts/main.py $EXP_ID --job-id $SLURM_ARRAY_JOB_ID --array-id $SLURM_ARRAY_TASK_ID

echo "--- Simulation Finished ---"