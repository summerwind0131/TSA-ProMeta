#!/bin/bash
#SBATCH -J tsa_stable_4
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --array=0-8%6
#SBATCH -o /mnt/hpc/home/lihan/fengyuan/task_prometa_v3_logs/%x_%A_%a.out
#SBATCH -e /mnt/hpc/home/lihan/fengyuan/task_prometa_v3_logs/%x_%A_%a.err

set -eo pipefail

source /mnt/hpc/home/lihan/miniconda3/etc/profile.d/conda.sh
conda activate fengyuan
set -u

<<<<<<< Updated upstream
CODE=/mnt/hpc/home/lihan/fengyuan/TSA-ProMeta/ProMeta
=======
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
>>>>>>> Stashed changes
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/hpc/home/lihan/fengyuan/task_prometa_v3_screen}"
EPOCHS="${EPOCHS:-100}"

python "$CODE/require_rtx3090.py"

SEEDS=(42 43 44)
VARIANTS=(
  F_epoch_only
  G_epoch_hysteresis
  H_epoch_bounded
)
SWITCH_THRESHOLDS=(0.0 0.05 0.05)
MIN_GROUP_FRACTIONS=(0.0 0.0 0.05)
MAX_GROUP_FRACTIONS=(1.0 1.0 0.50)

VARIANT_ID=$((SLURM_ARRAY_TASK_ID / 3))
SEED_ID=$((SLURM_ARRAY_TASK_ID % 3))

SEED="${SEEDS[$SEED_ID]}"
VARIANT="${VARIANTS[$VARIANT_ID]}"
OUT="$OUTPUT_ROOT/$VARIANT"
WARMUP="/mnt/hpc/home/lihan/fengyuan/task_prometa_benchmark_output/checkpoints/support_4/ProMeta_best_seed${SEED}.pth"

test -f "$WARMUP" || {
  echo "Missing warmup checkpoint: $WARMUP"
  exit 1
}

cd "$CODE"
mkdir -p "$OUT"

echo "Variant=$VARIANT seed=$SEED"
echo "Switch threshold=${SWITCH_THRESHOLDS[$VARIANT_ID]}"
echo "Group fraction bounds=${MIN_GROUP_FRACTIONS[$VARIANT_ID]}..${MAX_GROUP_FRACTIONS[$VARIANT_ID]}"

python main.py \
  --data_dir /mnt/hpc/home/lihan/fengyuan/ProMeta/data/out \
  --proteomics_csv /mnt/hpc/home/lihan/fengyuan/ProMeta/data/preprocessed_proteomics_data.csv \
  --cpdb_path /mnt/hpc/home/lihan/fengyuan/TSA-ProMeta/resource/CPDB_pathways_genes.tab \
  --output_dir "$OUT" \
  --gpu_id 0 \
  --random_seed "$SEED" \
  --support_size 4 \
  --max_support_size 32 \
  --batch_size 8 \
  --outer_lr 1e-4 \
  --inner_lr 0.005 \
  --epochs "$EPOCHS" \
  --patience 10 \
  --experiment_name "$VARIANT" \
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
  --tsa_switch_threshold "${SWITCH_THRESHOLDS[$VARIANT_ID]}" \
  --tsa_min_group_fraction "${MIN_GROUP_FRACTIONS[$VARIANT_ID]}" \
  --tsa_max_group_fraction "${MAX_GROUP_FRACTIONS[$VARIANT_ID]}" \
  --tsa_warmup_checkpoint "$WARMUP"
