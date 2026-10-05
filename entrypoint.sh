#!/bin/bash
# =============================================================================
# Skyfall-GS  —  Cloud Run Job entrypoint
#
# Env vars (set via --set-env-vars or Secret Manager):
#   SCENE          Scene name, e.g. JAX_068 or NYC_004   (default: JAX_068)
#   STAGE          1 = Reconstruction | 2 = IDU Synthesis (default: 1)
#   DATASET        datasets_JAX | datasets_NYC            (default: datasets_JAX)
#   GCS_BUCKET     GCS bucket URI, e.g. gs://my-bucket   (required)
#   CLOUD_RUN_TASK_INDEX   auto-set by Cloud Run Jobs parallelism
# =============================================================================
set -euo pipefail

SCENE="${SCENE:-JAX_068}"
STAGE="${STAGE:-1}"
DATASET="${DATASET:-datasets_JAX}"
GCS_BUCKET="${GCS_BUCKET:?GCS_BUCKET env var is required}"
DATA_DIR="/mnt/gcs"
OUTPUT_DIR="/mnt/outputs"
CKPT_DIR="/tmp/ckpts"

echo "========================================"
echo "  Skyfall-GS  |  Cloud Run Job"
echo "  SCENE   = $SCENE"
echo "  STAGE   = $STAGE"
echo "  DATASET = $DATASET"
echo "  BUCKET  = $GCS_BUCKET"
echo "  TASK    = ${CLOUD_RUN_TASK_INDEX:-0}"
echo "========================================"

# ── Mount GCS bucket via gcsfuse ──────────────────────────────────────────────
BUCKET_NAME="${GCS_BUCKET#gs://}"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$CKPT_DIR"

echo "[INFO] Mounting GCS bucket: $BUCKET_NAME → $DATA_DIR"
gcsfuse \
    --implicit-dirs \
    --file-mode=0777 \
    --dir-mode=0777 \
    --stat-cache-ttl=60s \
    --type-cache-ttl=60s \
    "$BUCKET_NAME" "$DATA_DIR"

echo "[INFO] GCS mount successful"

# ── Stage 1: Reconstruction ───────────────────────────────────────────────────
if [ "$STAGE" = "1" ]; then
  echo "[INFO] Starting Stage 1 — Reconstruction"

  python train.py \
    -s "$DATA_DIR/datasets/$DATASET/$SCENE" \
    -m "$OUTPUT_DIR/$SCENE" \
    --eval \
    --kernel_size 0.1 \
    --resolution 1 \
    --sh_degree 1 \
    --appearance_enabled \
    --lambda_depth 0 \
    --lambda_opacity 10 \
    --densify_until_iter 21000 \
    --densify_grad_threshold 0.0001 \
    --lambda_pseudo_depth 0.5 \
    --start_sample_pseudo 1000 \
    --end_sample_pseudo 21000 \
    --size_threshold 20 \
    --scaling_lr 0.001 \
    --rotation_lr 0.001 \
    --opacity_reset_interval 3000 \
    --sample_pseudo_interval 10

  echo "[INFO] Stage 1 complete — uploading checkpoint to GCS"
  gsutil -m cp -r "$OUTPUT_DIR/$SCENE" "${GCS_BUCKET}/checkpoints/"

# ── Stage 2: IDU Synthesis ────────────────────────────────────────────────────
elif [ "$STAGE" = "2" ]; then
  echo "[INFO] Starting Stage 2 — IDU Synthesis"

  # Download Stage 1 checkpoint from GCS
  echo "[INFO] Fetching Stage 1 checkpoint from GCS"
  gsutil -m cp -r "${GCS_BUCKET}/checkpoints/$SCENE" "$CKPT_DIR/"

  python train.py \
    -s "$DATA_DIR/datasets/$DATASET/$SCENE" \
    -m "$OUTPUT_DIR/${SCENE}_idu" \
    --start_checkpoint "$CKPT_DIR/$SCENE/chkpnt30000.pth" \
    --iterative_datasets_update \
    --eval \
    --kernel_size 0.1 \
    --resolution 1 \
    --sh_degree 1 \
    --appearance_enabled \
    --lambda_depth 0 \
    --lambda_opacity 0 \
    --idu_opacity_reset_interval 5000 \
    --idu_refine \
    --idu_num_samples_per_view 2 \
    --densify_grad_threshold 0.0002 \
    --idu_num_cams 6 \
    --idu_use_flow_edit \
    --idu_render_size 1024 \
    --idu_flow_edit_n_min 4 \
    --idu_flow_edit_n_max 10 \
    --idu_grid_size 3 \
    --idu_grid_width 512 \
    --idu_grid_height 512 \
    --idu_episode_iterations 10000 \
    --idu_iter_full_train 0 \
    --idu_opacity_cooling_iterations 500 \
    --lambda_pseudo_depth 0.5 \
    --idu_densify_until_iter 9000 \
    --idu_train_ratio 0.75

  echo "[INFO] Stage 2 complete — uploading IDU outputs to GCS"
  gsutil -m cp -r "$OUTPUT_DIR/${SCENE}_idu" "${GCS_BUCKET}/outputs/"

# ── Eval mode ─────────────────────────────────────────────────────────────────
elif [ "$STAGE" = "eval" ]; then
  echo "[INFO] Running evaluation"
  python eval.py \
    --data_dir "$DATA_DIR/results_eval/data_eval_JAX" \
    --temp_dir /tmp/temp_frames \
    --methods ours_stage1 ours_stage2 \
    --output_file "/tmp/metrics_${SCENE}.csv" \
    --frame_rate 30 \
    --resolution 1024 \
    --batch_size 32

  gsutil cp "/tmp/metrics_${SCENE}.csv" "${GCS_BUCKET}/eval/"

else
  echo "[ERROR] Unknown STAGE=$STAGE — must be 1, 2, or eval"
  exit 1
fi

echo "========================================"
echo "  Job complete!"
echo "========================================"
