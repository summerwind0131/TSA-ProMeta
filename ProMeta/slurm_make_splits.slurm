#!/bin/bash
#SBATCH -J make_prometa_splits
#SBATCH -p cpuPartition
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o make_prometa_splits-%j.out
#SBATCH -e make_prometa_splits-%j.err

set -eo pipefail

source /mnt/hpc/home/lihan/miniconda3/etc/profile.d/conda.sh
conda activate fengyuan

set -u

cd /mnt/hpc/home/lihan/fengyuan/task_prometa_code/proMeta/official_ProMeta/ProMeta

python make_prevalent_splits.py \
  --data_dir /mnt/hpc/home/lihan/fengyuan/ProMeta/data/out \
  --proteomics_date_csv /mnt/hpc/home/lihan/fengyuan/ProMeta/data/proteomics_test_date.csv \
  --output_dir /mnt/hpc/home/lihan/fengyuan/ProMeta/data/out \
  --seed 42 \
  --overwrite
