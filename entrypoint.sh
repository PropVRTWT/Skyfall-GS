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

# Fast multi-threaded weights loading
export HF_ENABLE_PARALLEL_LOADING=yes
export HF_HUB_ENABLE_HF_TRANSFER=0

# Point Hugging Face cache to mounted GCS volume (avoids consuming container RAM)
export HF_HOME="$DATA_DIR/hf_cache"
# Direct Hub cache to hf_cache root where blobs and models--* are uploaded
if [ -d "$DATA_DIR/hf_cache/models--black-forest-labs--FLUX.1-dev" ]; then
  export HF_HUB_CACHE="$DATA_DIR/hf_cache"
  export HUGGINGFACE_HUB_CACHE="$DATA_DIR/hf_cache"
elif [ -d "$DATA_DIR/hf_cache/hub/models--black-forest-labs--FLUX.1-dev" ]; then
  export HF_HUB_CACHE="$DATA_DIR/hf_cache/hub"
  export HUGGINGFACE_HUB_CACHE="$DATA_DIR/hf_cache/hub"
else
  export HF_HUB_CACHE="$DATA_DIR/hf_cache"
  export HUGGINGFACE_HUB_CACHE="$DATA_DIR/hf_cache"
fi
mkdir -p "$HF_HOME"
echo "[INFO] Using Hugging Face cache from GCS: $HF_HOME (Hub Cache: $HF_HUB_CACHE)"

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

  # Step 2 strictly starts from iteration 30,000 checkpoint only (never auto-resume from 36000)
  START_CKPT="$CKPT_DIR/$SCENE/chkpnt30000.pth"
  if [ ! -f "$START_CKPT" ] && [ -f "$DATA_DIR/checkpoints/$SCENE/chkpnt30000.pth" ]; then
    echo "[INFO] Copying chkpnt30000.pth directly from $DATA_DIR/checkpoints/$SCENE/"
    cp "$DATA_DIR/checkpoints/$SCENE/chkpnt30000.pth" "$START_CKPT"
  fi

  if [ ! -f "$START_CKPT" ]; then
    echo "[ERROR] Required checkpoint chkpnt30000.pth not found! Stage 2 must start from chkpnt30000.pth."
    exit 1
  fi
  echo "[INFO] Stage 2 starting strictly from checkpoint: $START_CKPT (ignoring any existing higher checkpoints)"

  # Assemble FLUX pipeline in fast container RAM disk (/tmp/flux_pipeline)
  if [ ! -f "/tmp/flux_pipeline/ae.safetensors" ]; then
    echo "[INFO] Setting up FLUX pipeline in RAM disk (/tmp/flux_pipeline)..."
    GCS_SRC="${GCS_BUCKET:-gs://stereo-images}"
    if [[ "$GCS_SRC" != gs://* ]]; then
      GCS_SRC="gs://${GCS_SRC}"
    fi
    GCS_SRC="${GCS_SRC%/}"

    mkdir -p /tmp/flux_blobs
    if command -v gcloud &> /dev/null; then
      echo "[INFO] Fast-streaming FLUX weights directly from $GCS_SRC into RAM disk at >200 MB/s..."
      gcloud storage cp -r "${GCS_SRC}/hf_cache/blobs/*" /tmp/flux_blobs/ || true
    fi

    # Fallback to mounted GCS if needed
    if [ ! -f "/tmp/flux_blobs/f73eecf7c469ff442523dc712cc161d631df071bf4d9d793494fbf00cdd80a82" ] && [ ! -d "/tmp/flux_blobs/f7" ]; then
      if [ -d "$DATA_DIR/hf_cache/blobs" ]; then
        echo "[INFO] Linking FLUX blobs from mounted GCS: $DATA_DIR/hf_cache/blobs"
        ln -s "$DATA_DIR/hf_cache/blobs"/* /tmp/flux_blobs/ 2>/dev/null || true
      fi
    fi

    echo "[INFO] Assembling FLUX pipeline at /tmp/flux_pipeline..."
    python3 /app/setup_flux.py --target /tmp/flux_pipeline --blobs /tmp/flux_blobs
  fi

  python train.py \
    -s "$SCENE_DIR" \
    -m "$OUTPUT_DIR/${SCENE}_idu" \
    --start_checkpoint "$START_CKPT" \
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
    --idu_num_samples_per_view "${IDU_NUM_SAMPLES_PER_VIEW:-1}" \
    --densify_grad_threshold 0.0002 \
    --idu_num_cams "${IDU_NUM_CAMS:-6}" \
    --idu_use_flow_edit \
    --idu_render_size 1024 \
    --idu_flow_edit_n_min 4 \
    --idu_flow_edit_n_max 10 \
    --idu_grid_size 3 \
    --idu_grid_width 512 \
    --idu_grid_height 512 \
    --idu_episode_iterations "${IDU_EPISODE_ITERATIONS:-3000}" \
    --idu_iter_full_train 0 \
    --idu_opacity_cooling_iterations 500 \
    --lambda_pseudo_depth 0.5 \
    --idu_densify_until_iter "${IDU_DENSIFY_UNTIL_ITER:-2250}" \
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
elif [ "$STAGE" = "eval" ] || [ "$STAGE" = "3" ]; then
  echo "[INFO] Running Stage 3: Model Export & Rendering for $SCENE"
  
  MODEL_DIR="$DATA_DIR/outputs/${SCENE}_idu"
  if [ ! -d "$MODEL_DIR" ]; then
    MODEL_DIR="$DATA_DIR/checkpoints/${SCENE}"
  fi
  
  LATEST_CKPT=$(ls -v "$MODEL_DIR"/chkpnt*.pth 2>/dev/null | tail -n 1 || true)
  EVAL_ITER=30000
  if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT" ]; then
    EVAL_ITER=$(basename "$LATEST_CKPT" | grep -o '[0-9]\+' || echo "30000")
  fi
  echo "[INFO] Using model at $MODEL_DIR (iteration $EVAL_ITER)"

  # 1. Export colored 3D Gaussian Splat PLY
  echo "[INFO] Exporting fused 3D PLY model..."
  python create_fused_ply.py \
    -m "$MODEL_DIR" \
    --iteration "$EVAL_ITER" \
    --load_from_checkpoints \
    --output_ply "$MODEL_DIR/fused_${SCENE}_iter${EVAL_ITER}.ply" || echo "[WARN] create_fused_ply non-zero exit"

  # 2. Render orbit fly-through video if trajectory exists
  SCENE_NUM="${SCENE#JAX_}"
  SCENE_NUM="${SCENE_NUM#NYC_}"
  CAM_PATH="camera_paths/JAX/${SCENE_NUM}/r488_e50_fov20.json"
  if [ ! -f "$CAM_PATH" ]; then
    CAM_PATH=$(find camera_paths -name "*.json" 2>/dev/null | grep -i "${SCENE_NUM}" | head -n 1 || true)
  fi

  if [ -n "$CAM_PATH" ] && [ -f "$CAM_PATH" ]; then
    echo "[INFO] Rendering orbit flight video using $CAM_PATH..."
    python render_video.py \
      -m "$MODEL_DIR" \
      --iteration "$EVAL_ITER" \
      --load_from_checkpoints \
      --camera_path "$CAM_PATH" || echo "[WARN] render_video non-zero exit"
  fi

  # 3. If benchmark comparison dataset exists in GCS, run eval.py
  if [ -d "$DATA_DIR/results_eval/data_eval_JAX" ]; then
    echo "[INFO] Found benchmark dataset at $DATA_DIR/results_eval/data_eval_JAX - running eval.py"
    python eval.py \
      --data_dir "$DATA_DIR/results_eval/data_eval_JAX" \
      --temp_dir /tmp/temp_frames \
      --methods ours_stage1 ours_stage2 \
      --output_file "/tmp/metrics_${SCENE}.csv" \
      --frame_rate 30 \
      --resolution 1024 \
      --batch_size 32 || true
    if [ -f "/tmp/metrics_${SCENE}.csv" ] && [ -n "$GCS_BUCKET" ]; then
      gsutil cp "/tmp/metrics_${SCENE}.csv" "${GCS_BUCKET}/eval/" || true
    fi
  else
    echo "[INFO] Benchmark dataset not found at $DATA_DIR/results_eval/data_eval_JAX - skipping academic eval.py"
  fi

  # 4. Sync outputs back to GCS if needed
  if [ -d "$OUTPUT_DIR/${SCENE}_idu" ] && [ -d "$DATA_DIR/outputs" ]; then
    cp -r "$OUTPUT_DIR/${SCENE}_idu" "$DATA_DIR/outputs/" 2>/dev/null || true
  fi

else
  echo "[ERROR] Unknown STAGE=$STAGE — must be 1, 2, or eval"
  exit 1
fi

echo "========================================"
echo "  Job complete!"
echo "========================================"
