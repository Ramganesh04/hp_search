# HP Search - GPU Training Queue

Distributed hyperparameter search for logic networks on CIFAR-10. FastAPI API submits jobs to MongoDB queue. Concurrent GPU workers claim + execute training runs.

## Architecture
```
POST /jobs → FastAPI API → MongoDB (jobs)
                    ↓
Workers ← MongoDB (Pending) → train.py (CIFAR-10)
                    ↓
MongoDB (job_epochs) ← epoch metrics + logs
```

## Quick Start (Local)
```
# Mongo
docker run -d --name mongo -p 27017:27017 mongo:7

# API
cd api && docker build -t jobs-api . && cd ..
docker run -d -p 8000:8000 --network host jobs-api

# Workers (--gpus all)
cd worker && docker build -t jobs-worker . && cd ..
docker run -d --gpus all --network host jobs-worker
```

**Submit job:**
```
curl -X POST http://localhost:8000/jobs -d '{"GATES":12000,"NETWORK_LAYERS":8,"EPOCHS":200,...}'
```

## Kubernetes Deployment
```
Namespace: jobs
Services: mongo (ClusterIP), api (NodePort/Ingress)
Deployments: api (CPU), worker (GPU nodes, replicas=N)
Collections: jobs (status/logs), job_epochs (metrics)
```

## Key Features
- **Concurrent workers** (`MAX_CONCURRENT=2`) stream epoch metrics to Mongo
- **VRAM checks** before launching training subprocesses
- **Log tail** (200 lines) + structured `job_epochs` collection
- **Timeout/retry** logic with atomic Mongo job claims

## Files
```
├── api/           # FastAPI /jobs → Mongo
│   ├── app.py
│   └── Dockerfile
└── worker/src        # PyTorch executor + train.py
    ├── main.py
    ├── train.py
    └── Dockerfile (nvidia/cuda:12.1.1 + torch cu121)
```

## Config (Env Vars)
```
MONGO_URI=mongodb://mongo:27017
MAX_CONCURRENT=2
POLL_SECONDS=1
MAX_LOG_LINES=200
```

**Monitor:** `docker logs`, `db.jobs.find()`, `db.job_epochs.aggregate()`
```
