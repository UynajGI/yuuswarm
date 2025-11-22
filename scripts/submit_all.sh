#!/bin/bash
# scripts/submit_all.sh

# 提交数组任务，获取纯数字 ID
ARRAY_JOB_ID=$(sbatch --parsable scripts/array_job.slurm)

echo "Submitted array job: $ARRAY_JOB_ID"

# 👇 关键：把纯数字 ARRAY_JOB_ID 传给 merge 脚本
sbatch --dependency=afterok:$ARRAY_JOB_ID scripts/merge_job.slurm $ARRAY_JOB_ID

echo "Submitted merge job dependent on $ARRAY_JOB_ID"