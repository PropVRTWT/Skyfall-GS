# =============================================================================
# Skyfall-GS  —  Dashboard (Cloud Run Service)
# Lightweight FastAPI app — shows job status, triggers executions, tails logs
# Image: python:3.11-slim   (no GPU needed)
# =============================================================================
FROM python:3.11-slim

LABEL maintainer="PropVRTWT"
LABEL description="Skyfall-GS minimal dashboard — Cloud Run Service"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /dashboard

# Install only what the dashboard needs
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    google-cloud-run==0.10.13 \
    google-cloud-logging==3.11.3 \
    google-auth==2.35.0 \
    httpx==0.27.2 \
    jinja2==3.1.4

COPY dashboard/ .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
