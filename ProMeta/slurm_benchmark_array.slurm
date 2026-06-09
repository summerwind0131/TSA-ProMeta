#!/bin/bash
#SBATCH -J prometa_array
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --array=0-19%4
#SBATCH -o /mnt/hpc/home/lihan/fengyuan/task_prometa_output/logs/%x_%A_%a.out
#SBATCH -e /mnt/hpc/home/lihan/fengyuan/task_prometa_output/logs/%x_%A_%a.err

set -euo pipefail

source /mnt/hpc/home/lihan/miniconda3/etc/profile.d/conda.sh
conda activate ProMeta

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/require_rtx3090.py"

cd /mnt/hpc/home/lihan/fengyuan/task_prometa_code/proMeta/official_ProMeta/ProMeta

PAIRS=(
"4 42"
"4 43"
"4 44"
"4 45"
"4 46"
"8 42"
"8 43"
"8 44"
"8 45"
"8 46"
"16 42"
"16 43"
"16 44"
"16 45"
"16 46"
"32 42"
"32 43"
"32 44"
"32 45"
"32 46"
)

PAIR="${PAIRS[$SLURM_ARRAY_TASK_ID]}"
SHOT="${PAIR%% *}"
SEED="${PAIR##* }"

export DATA_DIR=/mnt/hpc/home/lihan/fengyuan/ProMeta/data/out
export PROTEOMICS_CSV=/mnt/hpc/home/lihan/fengyuan/ProMeta/data/preprocessed_proteomics_data.csv
export CPDB_FILE=/mnt/hpc/home/lihan/fengyuan/task_prometa_code/proMeta/official_ProMeta/resource/CPDB_pathways_genes.tab
export OUTPUT_DIR=/mnt/hpc/home/lihan/fengyuan/task_prometa_output

export GPU_ID=0
export BATCH_SIZE=8
export MAX_SUPPORT_SIZE=32
export EPOCHS=100

export SHOT_LIST="$SHOT"
export SEED_LIST="$SEED"

mkdir -p "$OUTPUT_DIR/logs"

echo "Running shot=${SHOT}, seed=${SEED}"
bash run_benchmark.sh
