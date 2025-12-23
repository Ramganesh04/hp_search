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

## Key Features
- **Concurrent workers** (`MAX_CONCURRENT=2`) stream epoch metrics to Mongo
- **VRAM checks** before launching training subprocesses
- **Log tail** (200 lines) + structured `job_epochs` collection
- **Timeout/retry** logic with atomic Mongo job claims

