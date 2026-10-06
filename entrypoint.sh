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
GCS_BUCKET="${GCS_BUCKET:-}"
DATA_DIR="/mnt/gcs"
OUTPUT_DIR="/mnt/outputs"
CKPT_DIR="/tmp/ckpts"

echo "========================================"
echo "  Skyfall-GS  |  Cloud Run Job"
echo "  SCENE   = $SCENE"
echo "  STAGE   = $STAGE"
echo "  DATASET = $DATASET"
echo "  BUCKET  = ${GCS_BUCKET:-[native volume mount]}"
echo "  TASK    = ${CLOUD_RUN_TASK_INDEX:-0}"
if [ -n "${HF_TOKEN:-}" ]; then
  echo "  HF_TOKEN= [configured]"
else
  echo "  HF_TOKEN= [not set]"
fi
echo "========================================"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$CKPT_DIR"

# ── Mount GCS bucket ──────────────────────────────────────────────────────────
# Case A: Already mounted by Cloud Run native volume
if mountpoint -q "$DATA_DIR" 2>/dev/null; then
  echo "[INFO] GCS bucket is already mounted via Cloud Run Volume at $DATA_DIR"
# Case B: Mount inside container using gcsfuse from GCS_BUCKET env var
elif [ -n "$GCS_BUCKET" ]; then
  BUCKET_NAME="${GCS_BUCKET#gs://}"
  echo "[INFO] Mounting GCS bucket via gcsfuse: $BUCKET_NAME → $DATA_DIR"
  gcsfuse \
      --implicit-dirs \
      --file-mode=0777 \
      --dir-mode=0777 \
      --stat-cache-ttl=60s \
      --type-cache-ttl=60s \
      "$BUCKET_NAME" "$DATA_DIR"
  echo "[INFO] GCS mount successful"
else
  echo "[ERROR] Neither /mnt/gcs is mounted nor GCS_BUCKET env var is provided."
  exit 1
fi

# Check if pre-cached model weights exist in GCS to avoid downloading from Hugging Face
if [ -z "${MOGE_MODEL_PATH:-}" ]; then
  for cand in "$DATA_DIR/weights/moge-vitl/model.pt" "$DATA_DIR/models/moge-vitl/model.pt" "$DATA_DIR/weights/model.pt" "$DATA_DIR/moge-vitl/model.pt"; do
    if [ -f "$cand" ]; then
      export MOGE_MODEL_PATH="$cand"
      echo "[INFO] Found MoGe weights in GCS: $MOGE_MODEL_PATH"
      break
    fi
  done
fi

# ── Resolve Scene Directory ───────────────────────────────────────────────────
SCENE_DIR=""
for cand in \
  "$DATA_DIR/datasets/$DATASET/$SCENE" \
  "$DATA_DIR/$DATASET/$SCENE" \
  "$DATA_DIR/datasets/$DATASET/$SCENE/outputs_skew" \
  "$DATA_DIR/$DATASET/$SCENE/outputs_skew" \
  "$DATA_DIR/$SCENE" \
  "$DATA_DIR/datasets/$SCENE"; do
  if [ -d "$cand" ] && { [ -f "$cand/transforms_train.json" ] || [ -d "$cand/sparse" ]; }; then
    SCENE_DIR="$cand"
    echo "[INFO] Found valid scene at: $SCENE_DIR"
    break
  fi
done

if [ -z "$SCENE_DIR" ]; then
  SCENE_DIR="$DATA_DIR/datasets/$DATASET/$SCENE"
  echo "[WARN] Could not auto-detect transforms_train.json. Using fallback: $SCENE_DIR"
fi

# ── Stage 1: Reconstruction ───────────────────────────────────────────────────
if [ "$STAGE" = "1" ]; then
  echo "[INFO] Starting Stage 1 — Reconstruction"

  python train.py \
    -s "$SCENE_DIR" \
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
  if command -v gsutil &> /dev/null && [ -n "$GCS_BUCKET" ]; then
    gsutil -m cp -r "$OUTPUT_DIR/$SCENE" "${GCS_BUCKET}/checkpoints/"
  else
    echo "[INFO] Saving checkpoint to mounted GCS: $DATA_DIR/checkpoints/"
    mkdir -p "$DATA_DIR/checkpoints"
    cp -r "$OUTPUT_DIR/$SCENE" "$DATA_DIR/checkpoints/"
  fi

# ── Stage 2: IDU Synthesis ────────────────────────────────────────────────────
elif [ "$STAGE" = "2" ]; then
  echo "[INFO] Starting Stage 2 — IDU Synthesis"

  # Download Stage 1 checkpoint from GCS
  echo "[INFO] Fetching Stage 1 checkpoint from GCS"
  mkdir -p "$CKPT_DIR"
  if [ -d "$DATA_DIR/checkpoints/$SCENE" ]; then
    echo "[INFO] Restoring checkpoint from mounted GCS: $DATA_DIR/checkpoints/$SCENE"
    cp -r "$DATA_DIR/checkpoints/$SCENE" "$CKPT_DIR/"
  elif command -v gsutil &> /dev/null && [ -n "$GCS_BUCKET" ]; then
    gsutil -m cp -r "${GCS_BUCKET}/checkpoints/$SCENE" "$CKPT_DIR/"
  fi

  python train.py \
    -s "$SCENE_DIR" \
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
  if command -v gsutil &> /dev/null && [ -n "$GCS_BUCKET" ]; then
    gsutil -m cp -r "$OUTPUT_DIR/${SCENE}_idu" "${GCS_BUCKET}/outputs/"
  else
    echo "[INFO] Saving outputs to mounted GCS: $DATA_DIR/outputs/"
    mkdir -p "$DATA_DIR/outputs"
    cp -r "$OUTPUT_DIR/${SCENE}_idu" "$DATA_DIR/outputs/"
  fi

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
