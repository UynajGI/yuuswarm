#!/bin/bash
#SBATCH --job-name=Merge_Swarmlator
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=72:30:00
#SBATCH --output=logs/merge_%j.out

PROJECT_ROOT="/share/home/jiangyuan/yuuswarm"
cd "$PROJECT_ROOT"

source .venv/bin/activate

# 👇 只用 $1，不要尝试解析 SLURM_JOB_DEPENDENCY
JOB_ID=$1

if [ -z "$JOB_ID" ]; then
    echo "Error: No job ID provided. Usage: $0 <ARRAY_JOB_ID>"
    exit 1
fi

echo "Merging results from job $JOB_ID..."
python scripts/merge_results.py "output/$JOB_ID" -o "output/${JOB_ID}/merged_job_${JOB_ID}.h5" --cleanup

echo "✅ Merge completed."