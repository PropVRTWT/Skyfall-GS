"""
Skyfall-GS Dashboard  —  FastAPI backend
Serves the single-page dashboard, handles multi-image uploads to GCS,
and orchestrates the 4-step pipeline on Cloud Run Jobs with auto-resume.
"""

import os
import time
import json
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.cloud import storage
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dashboard")

# ── Config (injected by Cloud Run service env vars) ──────────────────────────
PROJECT_ID  = os.environ.get("PROJECT_ID", "")
REGION      = os.environ.get("REGION", "us-central1")
JOB_NAME    = os.environ.get("JOB_NAME", "skyfall-gs-job")
GCS_BUCKET  = os.environ.get("GCS_BUCKET", "stereo-images").replace("gs://", "").strip("/")

app = FastAPI(title="Skyfall-GS Pipeline Dashboard", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Google auth helper ────────────────────────────────────────────────────────
def get_access_token() -> str:
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(GoogleAuthRequest())
    return creds.token

def cr_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "Content-Type": "application/json"
    }

CR_BASE = "https://run.googleapis.com/v2"

# ── In-Memory Pipeline State ──────────────────────────────────────────────────
PIPELINE_STATES: Dict[str, Dict[str, Any]] = {}
CANCEL_FLAGS: Dict[str, bool] = {}

# ── Request models ────────────────────────────────────────────────────────────
class TriggerRequest(BaseModel):
    scene: str = "JAX_068"
    stage: str = "1"
    dataset: str = "datasets_JAX"
    extra_env: Optional[Dict[str, str]] = None

class PipelineStartRequest(BaseModel):
    scene: str
    dataset: str = "datasets_custom"
    steps: List[int] = [0, 1, 2, 3]
    episodes_per_run: int = 2
    total_episodes: int = 6


# ── Storage Helper ────────────────────────────────────────────────────────────
def get_storage_client() -> Optional[storage.Client]:
    try:
        return storage.Client()
    except Exception as e:
        logger.warning(f"Could not initialize storage client: {e}")
        return None


# ── Cloud Run Jobs Trigger & Polling Helpers ──────────────────────────────────
async def run_cloud_job_async(scene: str, stage: str, dataset: str, extra_env: Optional[Dict[str, str]] = None) -> str:
    """Trigger a Cloud Run Job execution and return execution resource name."""
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID not configured")

    url = f"{CR_BASE}/projects/{PROJECT_ID}/locations/{REGION}/jobs/{JOB_NAME}:run"
    env_vars = [
        {"name": "SCENE",      "value": scene},
        {"name": "STAGE",      "value": str(stage)},
        {"name": "DATASET",    "value": dataset},
        {"name": "GCS_BUCKET", "value": f"gs://{GCS_BUCKET}"},
    ]
    if extra_env:
        for k, v in extra_env.items():
            env_vars.append({"name": k, "value": str(v)})

    payload = {
        "overrides": {
            "containerOverrides": [{
                "env": env_vars
            }]
        }
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=cr_headers())

    if resp.status_code not in (200, 202):
        raise RuntimeError(f"Cloud Run Jobs run failed ({resp.status_code}): {resp.text}")

    data = resp.json()
    exec_name = data.get("name", "")
    return exec_name


async def poll_execution_async(exec_name: str, scene: str, poll_interval: int = 25) -> bool:
    """Poll execution until finished. Check cancel flag periodically."""
    short_name = exec_name.split("/")[-1]
    url = f"https://run.googleapis.com/v2/{exec_name}"
    start_time = time.time()

    while True:
        if CANCEL_FLAGS.get(scene, False):
            logger.info(f"Pipeline cancelled for {scene}. Halting wait for {short_name}.")
            return False

        try:
            async with httpx.AsyncClient(timeout=25) as client:
                resp = await client.get(url, headers=cr_headers())
            if resp.status_code == 200:
                data = resp.json()
                conditions = data.get("conditions", [])
                ready_cond = next((c for c in conditions if c.get("type") == "Ready"), None)
                state = ready_cond.get("state") if ready_cond else "UNKNOWN"
                succeeded = data.get("succeededCount", 0)
                failed = data.get("failedCount", 0)
                elapsed = int(time.time() - start_time)

                if state == "True":
                    logger.info(f"Execution {short_name} succeeded in {elapsed}s.")
                    return True
                elif state == "False":
                    msg = ready_cond.get("message", "Execution failed") if ready_cond else "Failed"
                    logger.error(f"Execution {short_name} failed: {msg}")
                    return False
                else:
                    # Still running
                    if scene in PIPELINE_STATES:
                        PIPELINE_STATES[scene]["elapsed_seconds"] = elapsed
                        PIPELINE_STATES[scene]["active_execution"] = short_name
            else:
                logger.warning(f"Poll check {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"Error polling execution: {e}")

        await asyncio.sleep(poll_interval)


# ── Background Pipeline Orchestrator Worker ──────────────────────────────────
async def run_pipeline_orchestrator(scene: str, dataset: str, steps: List[int], episodes_per_run: int = 2, total_episodes: int = 6):
    """Executes the full 4-step pipeline asynchronously with Option A auto-resume for Step 2."""
    state = PIPELINE_STATES[scene]
    state["status"] = "running"
    state["start_time"] = datetime.utcnow().isoformat()
    state["error"] = None
    CANCEL_FLAGS[scene] = False

    try:
        for step in steps:
            if CANCEL_FLAGS.get(scene, False):
                state["status"] = "cancelled"
                state["detail"] = "Pipeline cancelled by user."
                return

            if step == 0:
                state["current_step"] = 0
                state["step_name"] = "Step 0: SfM (COLMAP Pose Estimation & Triangulation)"
                state["detail"] = "Extracting camera poses and initial 3D tie points from images..."
                exec_name = await run_cloud_job_async(scene, stage="0", dataset=dataset)
                state["active_execution"] = exec_name.split("/")[-1]
                ok = await poll_execution_async(exec_name, scene)
                if not ok:
                    raise RuntimeError(f"Step 0 (SfM) execution {exec_name.split('/')[-1]} failed.")

            elif step == 1:
                state["current_step"] = 1
                state["step_name"] = "Step 1: Base Gaussian Splatting (30,000 iters)"
                state["detail"] = "Optimizing initial 3D Gaussians from sparse COLMAP tie points..."
                exec_name = await run_cloud_job_async(scene, stage="1", dataset=dataset)
                state["active_execution"] = exec_name.split("/")[-1]
                ok = await poll_execution_async(exec_name, scene)
                if not ok:
                    raise RuntimeError(f"Step 1 (Base GS) execution {exec_name.split('/')[-1]} failed.")

            elif step == 2:
                # Option A: Sequential chunks of Cloud Run executions with auto-resume
                total_chunks = (total_episodes + episodes_per_run - 1) // episodes_per_run
                state["current_step"] = 2

                for chunk_idx in range(1, total_chunks + 1):
                    if CANCEL_FLAGS.get(scene, False):
                        state["status"] = "cancelled"
                        return

                    start_ep = (chunk_idx - 1) * episodes_per_run + 1
                    end_ep = min(chunk_idx * episodes_per_run, total_episodes)
                    start_iter = 30000 + (start_ep - 1) * 10000
                    target_iter = 30000 + end_ep * 10000

                    state["step_name"] = f"Step 2: IDU Curriculum Refinement [Chunk {chunk_idx}/{total_chunks}]"
                    state["detail"] = f"Running Episodes {start_ep}-{end_ep} (Iters {start_iter} -> {target_iter}) with auto-resume..."
                    state["chunk_info"] = f"Chunk {chunk_idx}/{total_chunks} ({start_ep}-{end_ep})"

                    exec_name = await run_cloud_job_async(
                        scene,
                        stage="2",
                        dataset=dataset,
                        extra_env={
                            "MAX_EPISODES_PER_RUN": str(episodes_per_run),
                            "IDU_EPISODE_ITERATIONS": "10000",
                            "IDU_DENSIFY_UNTIL_ITER": "9000",
                        }
                    )
                    state["active_execution"] = exec_name.split("/")[-1]
                    ok = await poll_execution_async(exec_name, scene)
                    if not ok:
                        raise RuntimeError(f"Step 2 Chunk {chunk_idx} execution {exec_name.split('/')[-1]} failed.")

            elif step == 3:
                state["current_step"] = 3
                state["step_name"] = "Step 3: Fused PLY Model Export & 360° Orbit Video"
                state["detail"] = "Generating colored 3D Gaussian Splat PLY and rendering orbit video..."
                exec_name = await run_cloud_job_async(scene, stage="3", dataset=dataset)
                state["active_execution"] = exec_name.split("/")[-1]
                ok = await poll_execution_async(exec_name, scene)
                if not ok:
                    raise RuntimeError(f"Step 3 (Export & Render) execution {exec_name.split('/')[-1]} failed.")

        state["status"] = "completed"
        state["detail"] = "🎉 All pipeline stages successfully completed!"
        state["completion_time"] = datetime.utcnow().isoformat()

    except Exception as e:
        logger.exception(f"Pipeline error for {scene}: {e}")
        state["status"] = "failed"
        state["error"] = str(e)
        state["detail"] = f"Failed: {e}"


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path) as f:
        return f.read()


@app.get("/api/config")
async def get_config():
    return {
        "project_id": PROJECT_ID,
        "region":     REGION,
        "job_name":   JOB_NAME,
        "gcs_bucket": GCS_BUCKET,
    }


@app.get("/api/status")
async def job_status():
    """List recent executions for the Cloud Run Job."""
    if not PROJECT_ID:
        return JSONResponse({"executions": [], "warning": "PROJECT_ID not set"}, status_code=200)
    url = f"{CR_BASE}/projects/{PROJECT_ID}/locations/{REGION}/jobs/{JOB_NAME}/executions"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers=cr_headers(), params={"pageSize": 15})
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.post("/api/trigger")
async def trigger_job(req: TriggerRequest):
    """Trigger a single Cloud Run Job execution."""
    try:
        exec_name = await run_cloud_job_async(req.scene, req.stage, req.dataset, req.extra_env)
        return {"name": exec_name, "status": "submitted", "scene": req.scene, "stage": req.stage}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_images(
    background_tasks: BackgroundTasks,
    scene: str = Form(...),
    dataset: str = Form("datasets_custom"),
    auto_start: bool = Form(False),
    files: List[UploadFile] = File(...)
):
    """
    Direct multi-image upload endpoint.
    Uploads 12-15 (or more) scene images directly to Google Cloud Storage
    at gs://<GCS_BUCKET>/datasets/<dataset>/<scene>/images/<filename>
    Optionally auto-starts the full 4-step pipeline immediately upon completion.
    """
    clean_scene = scene.strip().replace(" ", "_")
    if not clean_scene:
        raise HTTPException(status_code=400, detail="Invalid scene name")

    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")

    allowed_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    uploaded_files = []
    skipped_files = []

    storage_client = get_storage_client()
    if not storage_client or not GCS_BUCKET:
        raise HTTPException(status_code=500, detail="Google Cloud Storage is not configured or credentials missing.")

    bucket = storage_client.bucket(GCS_BUCKET)
    prefix = f"datasets/{dataset}/{clean_scene}/images"

    for file in files:
        fname = os.path.basename(file.filename or "")
        ext = os.path.splitext(fname)[1]
        if ext not in allowed_exts:
            skipped_files.append(fname)
            continue

        blob_path = f"{prefix}/{fname}"
        blob = bucket.blob(blob_path)
        content = await file.read()
        blob.upload_from_string(content, content_type=file.content_type or "image/jpeg")
        uploaded_files.append(fname)

    logger.info(f"Uploaded {len(uploaded_files)} images for scene {clean_scene} to gs://{GCS_BUCKET}/{prefix}")

    pipeline_started = False
    if auto_start and len(uploaded_files) > 0:
        PIPELINE_STATES[clean_scene] = {
            "scene": clean_scene,
            "dataset": dataset,
            "status": "queued",
            "steps": [0, 1, 2, 3],
            "current_step": 0,
            "step_name": "Step 0: SfM (COLMAP Initializing)",
            "detail": f"{len(uploaded_files)} images uploaded. Launching Step 0 automatically...",
            "active_execution": None,
            "start_time": datetime.utcnow().isoformat(),
            "elapsed_seconds": 0,
            "error": None
        }
        background_tasks.add_task(
            run_pipeline_orchestrator,
            scene=clean_scene,
            dataset=dataset,
            steps=[0, 1, 2, 3],
            episodes_per_run=2,
            total_episodes=6
        )
        pipeline_started = True

    return {
        "status": "success",
        "scene": clean_scene,
        "dataset": dataset,
        "uploaded_count": len(uploaded_files),
        "uploaded_files": uploaded_files,
        "skipped_files": skipped_files,
        "gcs_uri": f"gs://{GCS_BUCKET}/{prefix}/",
        "pipeline_started": pipeline_started
    }


@app.post("/api/pipeline/start")
async def start_pipeline(req: PipelineStartRequest, background_tasks: BackgroundTasks):
    """Initiates the 4-step pipeline orchestration with Option A chunking."""
    clean_scene = req.scene.strip().replace(" ", "_")
    if not clean_scene:
        raise HTTPException(status_code=400, detail="Scene name cannot be empty")

    if clean_scene in PIPELINE_STATES and PIPELINE_STATES[clean_scene].get("status") == "running":
        return JSONResponse({
            "status": "already_running",
            "scene": clean_scene,
            "message": "Pipeline is already running for this scene."
        })

    PIPELINE_STATES[clean_scene] = {
        "scene": clean_scene,
        "dataset": req.dataset,
        "status": "queued",
        "steps": req.steps,
        "current_step": req.steps[0] if req.steps else 0,
        "step_name": "Initializing...",
        "detail": "Queueing pipeline execution...",
        "active_execution": None,
        "start_time": datetime.utcnow().isoformat(),
        "elapsed_seconds": 0,
        "error": None
    }

    background_tasks.add_task(
        run_pipeline_orchestrator,
        scene=clean_scene,
        dataset=req.dataset,
        steps=req.steps,
        episodes_per_run=req.episodes_per_run,
        total_episodes=req.total_episodes
    )

    return {
        "status": "started",
        "scene": clean_scene,
        "dataset": req.dataset,
        "steps": req.steps
    }


@app.get("/api/pipeline/status/{scene}")
async def get_pipeline_status(scene: str):
    """Get the current progress of the 4-step pipeline for a specific scene."""
    if scene not in PIPELINE_STATES:
        # Check if outputs already exist in GCS for this scene
        return {
            "status": "idle",
            "scene": scene,
            "current_step": None,
            "step_name": "No active pipeline running",
            "detail": "Ready to launch 4-step pipeline."
        }
    return PIPELINE_STATES[scene]


@app.post("/api/pipeline/cancel/{scene}")
async def cancel_pipeline(scene: str):
    """Signal cancellation for the pipeline execution."""
    if scene in PIPELINE_STATES:
        CANCEL_FLAGS[scene] = True
        PIPELINE_STATES[scene]["status"] = "cancelling"
        PIPELINE_STATES[scene]["detail"] = "Cancellation requested..."
        return {"status": "cancelled", "scene": scene}
    return {"status": "not_found", "scene": scene}


@app.get("/api/scenes")
async def list_scenes():
    """List available scenes in GCS and standard presets."""
    catalog: Dict[str, List[str]] = {
        "datasets_JAX": ["JAX_068", "JAX_004", "JAX_164", "JAX_168", "JAX_175", "JAX_214", "JAX_260", "JAX_264"],
        "datasets_NYC": ["NYC_004", "NYC_010", "NYC_219", "NYC_336"],
        "datasets_custom": []
    }

    storage_client = get_storage_client()
    if storage_client and GCS_BUCKET:
        try:
            bucket = storage_client.bucket(GCS_BUCKET)
            blobs = bucket.list_blobs(prefix="datasets/", delimiter="/")
            # Iterate prefixes
            for page in blobs.pages:
                for prefix in page.prefixes:
                    # e.g. datasets/datasets_custom/ or datasets/CUSTOM_001/
                    parts = prefix.strip("/").split("/")
                    if len(parts) >= 2:
                        dataset_name = parts[1]
                        if dataset_name not in catalog:
                            catalog[dataset_name] = []
                        # list subdirectories under dataset
                        sub_blobs = bucket.list_blobs(prefix=f"{prefix}", delimiter="/")
                        for sub_page in sub_blobs.pages:
                            for sub_prefix in sub_page.prefixes:
                                sc_name = sub_prefix.strip("/").split("/")[-1]
                                if sc_name and sc_name not in catalog[dataset_name]:
                                    catalog[dataset_name].append(sc_name)
        except Exception as e:
            logger.warning(f"Error querying GCS for scenes: {e}")

    return catalog


@app.get("/api/outputs/{scene}/ply")
async def get_ply_model(scene: str):
    """Download the fused 3D Gaussian Splat PLY model from GCS."""
    storage_client = get_storage_client()
    if not storage_client or not GCS_BUCKET:
        raise HTTPException(status_code=500, detail="GCS storage not configured")

    bucket = storage_client.bucket(GCS_BUCKET)
    # Search candidates
    candidates = [
        f"outputs/{scene}_idu/fused_model.ply",
        f"outputs/{scene}_idu/fused_{scene}_iter90000.ply",
        f"outputs/{scene}_idu/fused_{scene}_iter30000.ply",
    ]
    # Also search by glob/prefix
    for cand in candidates:
        blob = bucket.blob(cand)
        if blob.exists():
            return StreamingResponse(
                blob.open("rb"),
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename=fused_{scene}.ply"}
            )

    # Search any .ply file in outputs/{scene}_idu
    blobs = list(bucket.list_blobs(prefix=f"outputs/{scene}_idu/"))
    for b in blobs:
        if b.name.endswith(".ply"):
            return StreamingResponse(
                b.open("rb"),
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={os.path.basename(b.name)}"}
            )

    raise HTTPException(status_code=404, detail=f"No PLY model found for scene {scene}")


@app.get("/api/outputs/{scene}/video")
async def get_orbit_video(scene: str):
    """Stream or download the rendered circular orbit MP4 video from GCS."""
    storage_client = get_storage_client()
    if not storage_client or not GCS_BUCKET:
        raise HTTPException(status_code=500, detail="GCS storage not configured")

    bucket = storage_client.bucket(GCS_BUCKET)
    candidates = [
        f"outputs/{scene}_idu/orbit_flight.mp4",
        f"outputs/{scene}_idu/video/ours_90000/r488_e50_fov20.mp4",
        f"outputs/{scene}_idu/video/ours_30000/r488_e50_fov20.mp4",
    ]
    for cand in candidates:
        blob = bucket.blob(cand)
        if blob.exists():
            return StreamingResponse(
                blob.open("rb"),
                media_type="video/mp4",
                headers={"Content-Disposition": f"inline; filename=orbit_{scene}.mp4"}
            )

    # Search any .mp4 file in outputs/{scene}_idu
    blobs = list(bucket.list_blobs(prefix=f"outputs/{scene}_idu/"))
    for b in blobs:
        if b.name.endswith(".mp4"):
            return StreamingResponse(
                b.open("rb"),
                media_type="video/mp4",
                headers={"Content-Disposition": f"inline; filename={os.path.basename(b.name)}"}
            )

    raise HTTPException(status_code=404, detail=f"No orbit video found for scene {scene}")


@app.get("/healthz")
async def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}
