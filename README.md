# Hybrid Kubernetes ML Orchestrator

Queue GPU training runs on Kubernetes using a small job API + worker pattern backed by MongoDB.

## What’s inside
- **jobs-api (FastAPI)**: `POST /jobs` creates a job document with hyperparameters and `Status: Pending`.
- **jobs-worker**: polls MongoDB, claims jobs, launches training as a subprocess, streams logs, and writes per-epoch metrics to MongoDB.
- **train.py (PyTorch)**: CUDA-first training script with CLI hyperparameters and per-epoch metric logging (`EPOCH ... trainloss=...`).
- **MongoDB**: in-cluster deployment + PVC + ClusterIP service.
- **Kubernetes manifests**: API NodePort service + GPU worker deployment (`runtimeClassName: nvidia`, `nvidia.com/gpu` requests/limits).

## Quickstart (Kubernetes)
```
kubectl apply -f k8s/mongo.yaml
kubectl apply -f k8s/job-api-deployment.yaml
kubectl apply -f k8s/jobs-worker-deployment.yaml

```

## Repo layout
- `api/app.py` — FastAPI app + job creation endpoint.
- `worker/src/main.py` — worker loop (claim/run/stream logs + metrics).
- `worker/src/train.py` — training entrypoint + CLI hyperparameters.
- `k8s/mongo.yaml` — MongoDB (PVC, Deployment, Service).
- `k8s/job-api-deployment.yaml` — API Deployment + NodePort Service.
- `k8s/jobs-worker-deployment.yaml` — GPU worker Deployment (CUDA/GPU scheduling).

