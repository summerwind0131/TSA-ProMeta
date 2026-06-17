#!/bin/bash
#SBATCH -J tsa_mlp4
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --array=0-2%3
#SBATCH -o /mnt/hpc/home/lihan/fengyuan/%x_%A_%a.out
#SBATCH -e /mnt/hpc/home/lihan/fengyuan/%x_%A_%a.err

set -eo pipefail

source /mnt/hpc/home/lihan/miniconda3/etc/profile.d/conda.sh
conda activate fengyuan
set -u

CODE="${CODE:-/mnt/hpc/home/lihan/fengyuan/TSA-ProMeta/ProMeta}"
MLP_WARMUP_ROOT="${MLP_WARMUP_ROOT:-/mnt/hpc/home/lihan/fengyuan/task_prometa_mlp_baseline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/hpc/home/lihan/fengyuan/task_prometa_mlp_tsa}"
EPOCHS="${EPOCHS:-100}"

test -f "$CODE/require_rtx3090.py" || {
  echo "Missing GPU guard: $CODE/require_rtx3090.py" >&2
  exit 1
}

python "$CODE/require_rtx3090.py"

SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
WARMUP="$MLP_WARMUP_ROOT/checkpoints/support_4/ProMeta_best_seed${SEED}.pth"

test -f "$WARMUP" || {
  echo "Missing MLP warmup checkpoint: $WARMUP" >&2
  exit 1
}

cd "$CODE"
mkdir -p "$OUTPUT_ROOT"

echo "Running TSA-ProMeta-MLP seed=$SEED"
echo "Warmup=$WARMUP"

python main.py \
  --data_dir /mnt/hpc/home/lihan/fengyuan/ProMeta/data/out \
  --proteomics_csv /mnt/hpc/home/lihan/fengyuan/ProMeta/data/preprocessed_proteomics_data.csv \
  --cpdb_path /mnt/hpc/home/lihan/fengyuan/TSA-ProMeta/resource/CPDB_pathways_genes.tab \
  --output_dir "$OUTPUT_ROOT" \
  --gpu_id 0 \
  --random_seed "$SEED" \
  --support_size 4 \
  --max_support_size 32 \
  --batch_size 8 \
  --outer_lr 1e-4 \
  --inner_lr 0.005 \
  --epochs "$EPOCHS" \
  --patience 10 \
  --encoder_type mlp \
  --experiment_name TSA_MLP \
  --tsa_enable \
  --num_task_groups 5 \
  --tsa_selector_steps 10 \
  --tsa_param_keys "classifier,tokenizer.gate_logits" \
  --tsa_selector_source frozen_warmup \
  --tsa_assignment_source current_group \
  --tsa_distance_mode block_mean_l2 \
  --tsa_gate_distance_weight 1.0 \
  --tsa_selector_l1_lambda 0.001 \
  --tsa_routing_schedule epoch_snapshot \
  --tsa_switch_threshold 0.05 \
  --tsa_min_group_fraction 0.05 \
  --tsa_max_group_fraction 0.50 \
  --tsa_warmup_checkpoint "$WARMUP"
