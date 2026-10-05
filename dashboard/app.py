"""
Skyfall-GS Dashboard  —  FastAPI backend
Serves the single-page dashboard and proxies Cloud Run Jobs API calls.
"""
import os, datetime, json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
import httpx

# ── Config (injected by Cloud Run service env vars) ──────────────────────────
PROJECT_ID  = os.environ.get("PROJECT_ID", "")
REGION      = os.environ.get("REGION", "us-central1")
JOB_NAME    = os.environ.get("JOB_NAME", "skyfall-gs-job")
GCS_BUCKET  = os.environ.get("GCS_BUCKET", "")

app = FastAPI(title="Skyfall-GS Dashboard", version="1.0.0")

# ── Google auth helper ────────────────────────────────────────────────────────
def get_access_token() -> str:
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(GoogleAuthRequest())
    return creds.token

def cr_headers():
    return {"Authorization": f"Bearer {get_access_token()}"}

CR_BASE = "https://run.googleapis.com/v2"

# ── Request models ────────────────────────────────────────────────────────────
class TriggerRequest(BaseModel):
    scene: str = "JAX_068"
    stage: str = "1"
    dataset: str = "datasets_JAX"


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path) as f:
        return f.read()


@app.get("/api/status")
async def job_status():
    """List recent executions for the Cloud Run Job."""
    if not PROJECT_ID:
        return JSONResponse({"error": "PROJECT_ID not configured"}, status_code=500)
    url = (
        f"{CR_BASE}/projects/{PROJECT_ID}/locations/{REGION}"
        f"/jobs/{JOB_NAME}/executions"
    )
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers=cr_headers(), params={"pageSize": 10})
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.post("/api/trigger")
async def trigger_job(req: TriggerRequest):
    """Trigger a new Cloud Run Job execution."""
    if not PROJECT_ID:
        return JSONResponse({"error": "PROJECT_ID not configured"}, status_code=500)
    url = (
        f"{CR_BASE}/projects/{PROJECT_ID}/locations/{REGION}"
        f"/jobs/{JOB_NAME}:run"
    )
    payload = {
        "overrides": {
            "containerOverrides": [{
                "env": [
                    {"name": "SCENE",      "value": req.scene},
                    {"name": "STAGE",      "value": req.stage},
                    {"name": "DATASET",    "value": req.dataset},
                    {"name": "GCS_BUCKET", "value": GCS_BUCKET},
                ]
            }]
        }
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, json=payload, headers=cr_headers())
    if resp.status_code not in (200, 202):
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.get("/api/config")
async def get_config():
    return {
        "project_id": PROJECT_ID,
        "region":     REGION,
        "job_name":   JOB_NAME,
        "gcs_bucket": GCS_BUCKET,
    }


@app.get("/healthz")
async def health():
    return {"status": "ok", "ts": datetime.datetime.utcnow().isoformat()}
