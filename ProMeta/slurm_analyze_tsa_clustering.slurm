#!/bin/bash
#SBATCH -J tsa_cluster
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -o /mnt/hpc/home/lihan/fengyuan/%x_%j.out
#SBATCH -e /mnt/hpc/home/lihan/fengyuan/%x_%j.err

set -eo pipefail

source /mnt/hpc/home/lihan/miniconda3/etc/profile.d/conda.sh
conda activate fengyuan
set -u

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_GLOB="${CHECKPOINT_GLOB:?Set CHECKPOINT_GLOB to TSA checkpoint files for one variant.}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/hpc/home/lihan/fengyuan/task_prometa_cluster_analysis}"

cd "$CODE"
python "$CODE/require_rtx3090.py"

python analyze_tsa_clustering.py \
  --checkpoint_glob "$CHECKPOINT_GLOB" \
  --data_dir /mnt/hpc/home/lihan/fengyuan/ProMeta/data/out \
  --proteomics_csv /mnt/hpc/home/lihan/fengyuan/ProMeta/data/preprocessed_proteomics_data.csv \
  --cpdb_path /mnt/hpc/home/lihan/fengyuan/TSA-ProMeta/resource/CPDB_pathways_genes.tab \
  --output_dir "$OUTPUT_DIR" \
  --gpu_id 0 \
  --null_permutations 100 \
  --silhouette_sample_size 300
