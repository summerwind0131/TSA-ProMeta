#!/bin/bash
#SBATCH -J prometa_smoke
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o /mnt/hpc/home/lihan/fengyuan/task_prometa_output/logs/%x_%j.out
#SBATCH -e /mnt/hpc/home/lihan/fengyuan/task_prometa_output/logs/%x_%j.err

set -euo pipefail

source /mnt/hpc/home/lihan/miniconda3/etc/profile.d/conda.sh
conda activate ProMeta

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/require_rtx3090.py"

cd /mnt/hpc/home/lihan/fengyuan/task_prometa_code/proMeta/official_ProMeta/ProMeta

export DATA_DIR=/mnt/hpc/home/lihan/fengyuan/ProMeta/data/out
export PROTEOMICS_CSV=/mnt/hpc/home/lihan/fengyuan/ProMeta/data/preprocessed_proteomics_data.csv
export CPDB_FILE=/mnt/hpc/home/lihan/fengyuan/task_prometa_code/proMeta/official_ProMeta/resource/CPDB_pathways_genes.tab
export OUTPUT_DIR=/mnt/hpc/home/lihan/fengyuan/task_prometa_output

export GPU_ID=0
export BATCH_SIZE=8
export MAX_SUPPORT_SIZE=32
export EPOCHS=1
export SHOT_LIST="4"
export SEED_LIST="42"

mkdir -p "$OUTPUT_DIR/logs"

bash run_benchmark.sh
python summarize_benchmark.py --output_dir "$OUTPUT_DIR"
