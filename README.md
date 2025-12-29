# Hybrid Kubernetes ML Orchestrator

Queue GPU training runs on Kubernetes using a small job API + worker pattern backed by MongoDB.

## What’s inside
- **jobs-api (Go + Gin)**: `POST /jobs` inserts a job document in Mongo with `status: "Pending"` and a `hyperparameters` object.
- **jobs-worker (Python + PyTorch)**: polls MongoDB, claims `Pending` jobs, launches `train.py` as a subprocess, streams logs, and writes per-epoch metrics to MongoDB.
- **train.py (PyTorch)**: CUDA-first training script with CLI hyperparameters and per-epoch metric logging (`EPOCH ... train_loss=...`).
- **MongoDB**: in-cluster Deployment + PVC + ClusterIP service.
- **Kubernetes manifests**: API NodePort service + GPU worker deployment (`nvidia.com/gpu` requests/limits; enable `runtimeClassName: nvidia` if your cluster requires it).

## Quickstart (minikube + Kubernetes)
The manifests use `imagePullPolicy: Never`, so build the images into minikube’s Docker daemon:

```bash
eval "$(minikube docker-env)"
docker build -t jobs-api:0.2 -f api/Dockerfile api
docker build -t jobs-worker:0.2 -f worker/Dockerfile worker

kubectl apply -f k8s/mongo.yaml
kubectl apply -f k8s/job-api-deployment.yaml
kubectl apply -f k8s/jobs-worker-deployment.yaml
```

If you change tags, update `k8s/job-api-deployment.yaml` and `k8s/jobs-worker-deployment.yaml` accordingly.

## Submitting a job
The API is exposed as a NodePort on `30080`:

```bash
curl -sS -X POST "http://$(minikube ip):30080/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "gateOptimizer": "sparsemax_noise",
    "gates": 3332,
    "networkLayers": 4,
    "grouping": 10,
    "groupSumTau": 100,
    "residualLayers": 4,
    "noiseTemp": 0.2,
    "epochs": 5
  }'
```

### `gateOptimizer` values
- `softmax`
- `gumbel_softmax`
- `sparsemax`
- `gumbel_sparsemax`
- `sparsemax_noise`
- `log_softmax`
- `entmax`
- `relu_normalized`
- `balenas`

## Repo layout
- `api/main.go`, `api/models.go` — Go API (Gin + Mongo).
- `worker/src/main.py` — worker loop (claim/run/stream logs + metrics).
- `worker/src/train.py` — training entrypoint + CLI hyperparameters.
- `worker/Dockerfile` — GPU worker image (CUDA + PyTorch + difflogic extension).
- `k8s/mongo.yaml` — MongoDB (PVC, Deployment, Service).
- `k8s/job-api-deployment.yaml` — API Deployment + NodePort Service.
- `k8s/jobs-worker-deployment.yaml` — GPU worker Deployment.

