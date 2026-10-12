#!/usr/bin/env python3
"""
scripts/orchestrate_pipeline.py
================================
End-to-end pipeline orchestrator for Skyfall-GS on Google Cloud Run Jobs.

Implements Option A (Orchestrated Cloud Run Jobs with Auto-Resume):
  - Step 0: Camera Pose Estimation (COLMAP SfM) from 12-15 images
  - Step 1: Base Gaussian Splatting (30,000 iterations)
  - Step 2: IDU Curriculum Refinement (6 episodes x 10,000 iterations = 60,000 iters -> iter 90,000)
            Auto-chunked into 3 sequential Cloud Run executions (2 episodes / ~52 min each)
            with auto-resume from checkpoints (chkpnt50000.pth, chkpnt70000.pth, chkpnt90000.pth)
            to guarantee zero timeout crashes against Cloud Run's 1-hour GPU limit.
  - Step 3: Fused 3D PLY Model Export & 360° Circular Orbit Video Rendering

Usage:
  python scripts/orchestrate_pipeline.py \
    --scene CUSTOM_001 \
    --dataset datasets_custom \
    --project my-gcp-project \
    --region us-central1 \
    --job skyfall-gs-job \
    --bucket gs://my-bucket \
    --steps 0,1,2,3
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("orchestrator")

try:
    import google.auth
    from google.auth.transport.requests import Request as GoogleAuthRequest
    import httpx
except ImportError:
    logger.error("Missing dependencies. Please install: pip install google-auth httpx")
    sys.exit(1)


def get_access_token() -> str:
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(GoogleAuthRequest())
    return creds.token


def get_cr_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "Content-Type": "application/json"
    }


class PipelineOrchestrator:
    def __init__(
        self,
        project_id: str,
        region: str,
        job_name: str,
        gcs_bucket: str,
        scene: str,
        dataset: str,
        poll_interval: int = 30
    ):
        self.project_id = project_id
        self.region = region
        self.job_name = job_name
        self.gcs_bucket = gcs_bucket if gcs_bucket.startswith("gs://") else f"gs://{gcs_bucket}"
        self.scene = scene
        self.dataset = dataset
        self.poll_interval = poll_interval
        self.cr_base = f"https://run.googleapis.com/v2/projects/{project_id}/locations/{region}/jobs/{job_name}"

    def trigger_execution(self, stage: str, extra_env: Optional[Dict[str, str]] = None) -> str:
        """Trigger a Cloud Run Job execution for the given stage and return the execution name."""
        env_vars = [
            {"name": "SCENE", "value": self.scene},
            {"name": "STAGE", "value": str(stage)},
            {"name": "DATASET", "value": self.dataset},
            {"name": "GCS_BUCKET", "value": self.gcs_bucket},
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

        url = f"{self.cr_base}:run"
        with httpx.Client(timeout=30) as client:
            resp = client.post(url, headers=get_cr_headers(), json=payload)

        if resp.status_code not in (200, 202):
            raise RuntimeError(f"Failed to trigger Cloud Run Job (status {resp.status_code}): {resp.text}")

        data = resp.json()
        exec_name = data.get("name", "")
        short_name = exec_name.split("/")[-1] if exec_name else "unknown"
        logger.info(f"Triggered execution: {short_name} for Stage {stage}")
        return exec_name

    def wait_for_execution(self, exec_name: str) -> bool:
        """Poll Cloud Run Job execution until completion. Return True if succeeded, False otherwise."""
        short_name = exec_name.split("/")[-1]
        url = f"https://run.googleapis.com/v2/{exec_name}"
        start_time = time.time()

        logger.info(f"Monitoring execution {short_name} (polling every {self.poll_interval}s)...")
        while True:
            try:
                with httpx.Client(timeout=30) as client:
                    resp = client.get(url, headers=get_cr_headers())
                if resp.status_code == 200:
                    data = resp.json()
                    conditions = data.get("conditions", [])
                    ready_cond = next((c for c in conditions if c.get("type") == "Ready"), None)
                    state = ready_cond.get("state") if ready_cond else "UNKNOWN"
                    succeeded = data.get("succeededCount", 0)
                    failed = data.get("failedCount", 0)
                    elapsed = int(time.time() - start_time)
                    elapsed_str = f"{elapsed // 60}m {elapsed % 60}s"

                    if state == "True":
                        logger.info(f"Execution {short_name} SUCCEEDED in {elapsed_str}! (succeededCount={succeeded})")
                        return True
                    elif state == "False":
                        msg = ready_cond.get("message", "No error message provided") if ready_cond else "Failed"
                        logger.error(f"Execution {short_name} FAILED after {elapsed_str}: {msg}")
                        return False
                    else:
                        logger.info(f"Execution {short_name} running... (elapsed: {elapsed_str}, runningCount={data.get('runningCount', 1)})")
                else:
                    logger.warning(f"Poll check returned status {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.warning(f"Error checking execution status: {e}")

            time.sleep(self.poll_interval)

    def run_step_0(self) -> bool:
        """Stage 0: COLMAP Camera Pose Estimation & Tie Points."""
        logger.info(f"=== Starting Step 0: Camera Pose Estimation (COLMAP SfM) for {self.scene} ===")
        exec_name = self.trigger_execution(stage="0")
        success = self.wait_for_execution(exec_name)
        if not success:
            logger.error("Step 0 failed. Pipeline aborted.")
            return False
        logger.info("=== Step 0 Completed Successfully! ===")
        return True

    def run_step_1(self) -> bool:
        """Stage 1: Base Gaussian Splatting (30k iterations)."""
        logger.info(f"=== Starting Step 1: Base Gaussian Splatting (30k iters) for {self.scene} ===")
        exec_name = self.trigger_execution(stage="1")
        success = self.wait_for_execution(exec_name)
        if not success:
            logger.error("Step 1 failed. Pipeline aborted.")
            return False
        logger.info("=== Step 1 Completed Successfully! ===")
        return True

    def run_step_2(self, total_episodes: int = 6, episodes_per_run: int = 2) -> bool:
        """Stage 2: IDU Curriculum Refinement (6 episodes x 10k iters = 60k iters)."""
        total_runs = (total_episodes + episodes_per_run - 1) // episodes_per_run
        logger.info(f"=== Starting Step 2: IDU Synthesis ({total_episodes} Episodes, {total_runs} Chunks) for {self.scene} ===")

        for chunk_idx in range(1, total_runs + 1):
            start_ep = (chunk_idx - 1) * episodes_per_run + 1
            end_ep = min(chunk_idx * episodes_per_run, total_episodes)
            logger.info(f"--- Launching Step 2 [Chunk {chunk_idx}/{total_runs}] (Episodes {start_ep}-{end_ep}) ---")

            exec_name = self.trigger_execution(
                stage="2",
                extra_env={
                    "MAX_EPISODES_PER_RUN": str(episodes_per_run),
                    "IDU_EPISODE_ITERATIONS": "10000",
                    "IDU_DENSIFY_UNTIL_ITER": "9000",
                }
            )
            success = self.wait_for_execution(exec_name)
            if not success:
                logger.error(f"Step 2 Chunk {chunk_idx} failed. Pipeline aborted.")
                return False

            logger.info(f"Chunk {chunk_idx}/{total_runs} completed successfully. Checkpoint synced to GCS.")
            if chunk_idx < total_runs:
                logger.info("⏳ Pausing 60 seconds cooldown before launching next Step 2 chunk in fresh container...")
                time.sleep(60)

        logger.info("=== Step 2 Completed Successfully! All 6 episodes reached iter 90,000 ===")
        return True

    def run_step_3(self) -> bool:
        """Stage 3: Fused PLY Model Export & Circular Orbit Video Rendering."""
        logger.info(f"=== Starting Step 3: Fused PLY Export & Orbit Video for {self.scene} ===")
        exec_name = self.trigger_execution(stage="3")
        success = self.wait_for_execution(exec_name)
        if not success:
            logger.error("Step 3 failed. Pipeline aborted.")
            return False
        logger.info("=== Step 3 Completed Successfully! Model PLY and orbit_flight.mp4 ready ===")
        return True

    def run_pipeline(self, steps: List[int]) -> bool:
        """Execute selected steps in order with 60s cooldown between steps."""
        logger.info(f"Starting Skyfall-GS Pipeline with Steps: {steps}")
        for i, step in enumerate(steps):
            if step == 0:
                if not self.run_step_0(): return False
            elif step == 1:
                if not self.run_step_1(): return False
            elif step == 2:
                if not self.run_step_2(): return False
            elif step == 3:
                if not self.run_step_3(): return False
            else:
                logger.error(f"Unknown step: {step}")
                return False

            if i < len(steps) - 1:
                logger.info("⏳ Cooldown break: Pausing 60 seconds before next step (allows GCS sync & fresh GPU allocation)...")
                time.sleep(60)

        logger.info("🎉 All requested pipeline stages finished successfully!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Orchestrate Skyfall-GS 4-Step Pipeline on Cloud Run")
    parser.add_argument("--scene", default=os.environ.get("SCENE", "JAX_068"), help="Scene name")
    parser.add_argument("--dataset", default=os.environ.get("DATASET", "datasets_JAX"), help="Dataset name")
    parser.add_argument("--project", default=os.environ.get("PROJECT_ID", ""), required=not os.environ.get("PROJECT_ID"), help="GCP Project ID")
    parser.add_argument("--region", default=os.environ.get("REGION", "us-central1"), help="GCP Region")
    parser.add_argument("--job", default=os.environ.get("JOB_NAME", "skyfall-gs-job"), help="Cloud Run Job Name")
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET", "stereo-images"), help="GCS Bucket")
    parser.add_argument("--steps", default="0,1,2,3", help="Comma-separated steps to run (0, 1, 2, 3)")
    parser.add_argument("--episodes-per-run", type=int, default=2, help="Episodes per Cloud Run execution")
    parser.add_argument("--total-episodes", type=int, default=6, help="Total curriculum episodes")
    parser.add_argument("--poll-interval", type=int, default=30, help="Poll interval in seconds")

    args = parser.parse_args()

    steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]

    orchestrator = PipelineOrchestrator(
        project_id=args.project,
        region=args.region,
        job_name=args.job,
        gcs_bucket=args.bucket,
        scene=args.scene,
        dataset=args.dataset,
        poll_interval=args.poll_interval
    )

    success = orchestrator.run_pipeline(steps)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
