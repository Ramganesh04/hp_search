import os
import sys
import time
import asyncio
from collections import deque
from typing import Optional

import torch
from pymongo import AsyncMongoClient, ReturnDocument


# ------------ Config ------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "jobsdb")

JOBS_COLLECTION = os.getenv("JOBS_COLLECTION", "jobs")
EPOCHS_COLLECTION = os.getenv("EPOCHS_COLLECTION", "job_epochs")

WORKER_ID = os.getenv("WORKER_ID", os.getenv("HOSTNAME", "worker-1"))

POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.0"))
IDLE_EXIT_SECONDS = float(os.getenv("IDLE_EXIT_SECONDS", "300"))

MAX_RUNTIME_SECONDS = int(os.getenv("MAX_RUNTIME_SECONDS", str(6 * 60 * 60)))
VRAM_BUFFER_BYTES = int(os.getenv("VRAM_BUFFER_BYTES", str(1 * 1024**3)))

MAX_LOG_LINES = int(os.getenv("MAX_LOG_LINES", "200"))
FLUSH_EVERY = int(os.getenv("FLUSH_EVERY", "20"))

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "2"))  # <-- concurrency knob

TRAIN_SCRIPT = os.getenv("TRAIN_SCRIPT", "train.py")


# ------------ Small logging helper ------------
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] [{WORKER_ID}] {msg}", flush=True)


# ------------ GPU helpers ------------
def free_vram_bytes(device: int = 0) -> int:
    with torch.cuda.device(device):
        free_b, _total_b = torch.cuda.memory.mem_get_info()
    return int(free_b)


def usable_vram_bytes(device: int = 0) -> int:
    return max(0, free_vram_bytes(device) - VRAM_BUFFER_BYTES)


def estimate_required_vram_bytes(job: dict) -> int:
    hp = job.get("Hyperparameters", {})
    gates = int(hp.get("GATES", 0))
    layers = int(hp.get("NETWORK_LAYERS", 0))
    return 2 * 1024**3 + (2 * 1024) * gates * layers


# ------------ Mongo helpers ------------
async def claim_one_job(jobs_coll) -> Optional[dict]:
    job = await jobs_coll.find_one_and_update(
        {"Status": "Pending"},
        {"$set": {"Status": "Running", "WorkerId": WORKER_ID, "ClaimedAt": time.time()}},
        sort=[("TimeCreated", 1)],
        return_document=ReturnDocument.AFTER,
    )
    if job:
        log(f"CLAIMED job={job['_id']}")
    return job


async def release_job(jobs_coll, job: dict, reason: str):
    log(f"RELEASING job={job['_id']} reason={reason}")
    await jobs_coll.update_one(
        {"_id": job["_id"], "WorkerId": WORKER_ID, "Status": "Running"},
        {
            "$set": {"Status": "Pending", "ReleaseReason": reason, "ReleasedAt": time.time()},
            "$unset": {"WorkerId": "", "ClaimedAt": ""},
        },
    )


def build_cmd(job: dict) -> list[str]:
    hp = job["Hyperparameters"]
    return [
        sys.executable, TRAIN_SCRIPT,
        "--job-id", str(job["_id"]),
        "--gate-optimizer", str(hp["GATE_OPTIMIZER"]),
        "--gates", str(hp["GATES"]),
        "--network-layers", str(hp["NETWORK_LAYERS"]),
        "--grouping", str(hp["GROUPING"]),
        "--group-sum-tau", str(hp["GROUP_SUM_TAU"]),
        "--residual-layers", str(hp["RESIDUAL_LAYERS"]),
        "--noise-temp", str(hp["NOISE_TEMP"]),
        "--epochs", str(hp["EPOCHS"]),
    ]


def parse_epoch_line(line: str) -> Optional[dict]:
    if not line.startswith("EPOCH "):
        return None
    try:
        parts = line.split()
        epoch_cur, epoch_total = parts[1].split("/")
        metrics = {}
        for token in parts[2:]:
            if "=" in token:
                k, v = token.split("=", 1)
                metrics[k] = float(v)
        return {
            "epoch": int(epoch_cur),
            "epochs": int(epoch_total),
            "train_loss": metrics.get("train_loss"),
            "train_acc": metrics.get("train_acc"),
            "float_eval_acc": metrics.get("float_eval_acc"),
            "discrete_eval_acc": metrics.get("discrete_eval_acc"),
        }
    except Exception:
        return None


# ------------ Async subprocess runner (per job) ------------
async def run_train_and_stream(jobs_coll, epochs_coll, job: dict, cmd: list[str]) -> int:
    await jobs_coll.update_one(
        {"_id": job["_id"], "WorkerId": WORKER_ID},
        {"$set": {"Cmd": cmd, "StartedAt": time.time(), "LogTail": []}},
    )
    log(f"START subprocess job={job['_id']} cmd={' '.join(cmd[:3])} ...")

    log_tail = deque(maxlen=MAX_LOG_LINES)
    batch: list[str] = []

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )  # asyncio subprocess API [web:350]

    async def reader():
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode(errors="ignore").rstrip("\n")
            log_tail.append(text)
            batch.append(text)

            parsed = parse_epoch_line(text)
            if parsed is not None:
                await epochs_coll.insert_one({"jobId": job["_id"], "ts": time.time(), **parsed})
                log(f"WROTE epoch job={job['_id']} epoch={parsed['epoch']}")

            if len(batch) >= FLUSH_EVERY:
                await jobs_coll.update_one(
                    {"_id": job["_id"], "WorkerId": WORKER_ID},
                    {
                        "$set": {"LastLogAt": time.time()},
                        "$push": {"LogTail": {"$each": batch, "$slice": -MAX_LOG_LINES}},
                    },
                )
                log(f"FLUSHED logs job={job['_id']} lines={len(batch)}")
                batch.clear()

    reader_task = asyncio.create_task(reader())

    try:
        await asyncio.wait_for(proc.wait(), timeout=MAX_RUNTIME_SECONDS)
    except asyncio.TimeoutError:
        log(f"TIMEOUT job={job['_id']} killing subprocess")
        proc.kill()
        await proc.wait()
        await jobs_coll.update_one(
            {"_id": job["_id"], "WorkerId": WORKER_ID},
            {"$set": {"Status": "TimedOut", "FailedAt": time.time()}},
        )
        return 124
    finally:
        await reader_task
        if batch:
            await jobs_coll.update_one(
                {"_id": job["_id"], "WorkerId": WORKER_ID},
                {"$push": {"LogTail": {"$each": batch, "$slice": -MAX_LOG_LINES}}},
            )
            log(f"FINAL-FLUSH logs job={job['_id']} lines={len(batch)}")

    return int(proc.returncode or 0)


async def handle_job(sem: asyncio.Semaphore, jobs_coll, epochs_coll, job: dict):
    async with sem:  # limits concurrent running jobs [web:526]
        job_id = job["_id"]

        # Re-check VRAM right before launch
        available = usable_vram_bytes(0)
        required = estimate_required_vram_bytes(job)
        if required > available:
            await release_job(jobs_coll, job, f"Not enough VRAM at start: required={required} available={available}")
            return

        cmd = build_cmd(job)
        rc = await run_train_and_stream(jobs_coll, epochs_coll, job, cmd)

        if rc == 0:
            log(f"FINISHED job={job_id}")
            await jobs_coll.update_one(
                {"_id": job_id, "WorkerId": WORKER_ID},
                {"$set": {"Status": "Finished", "FinishedAt": time.time(), "ReturnCode": rc}},
            )
        else:
            log(f"FAILED job={job_id} rc={rc}")
            await jobs_coll.update_one(
                {"_id": job_id, "WorkerId": WORKER_ID},
                {"$set": {"Status": "Failed", "FailedAt": time.time(), "ReturnCode": rc}},
            )


async def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    client = AsyncMongoClient(MONGO_URI)
    await client.aconnect()
    db = client[DB_NAME]
    jobs_coll = db[JOBS_COLLECTION]
    epochs_coll = db[EPOCHS_COLLECTION]

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    running: set[asyncio.Task] = set()

    idle_since = time.time()
    log(f"BOOT MAX_CONCURRENT={MAX_CONCURRENT}")

    while True:
        # Clean up finished tasks
        done = {t for t in running if t.done()}
        for t in done:
            try:
                t.result()
            except Exception as e:
                log(f"TASK ERROR: {e}")
            running.remove(t)

        # If we have capacity, try to claim more jobs
        if sem.locked() and sem._value == 0:
            log("AT CAPACITY (no slots), not claiming new jobs")

        if sem._value > 0:
            job = await claim_one_job(jobs_coll)
            if job is None:
                log("NO JOB (pending queue empty)")
                if time.time() - idle_since > IDLE_EXIT_SECONDS and not running:
                    log("IDLE EXIT")
                    break
                await asyncio.sleep(POLL_SECONDS)
            else:
                idle_since = time.time()
                log(f"ACCEPT job={job['_id']} launching task (running={len(running)+1})")
                task = asyncio.create_task(handle_job(sem, jobs_coll, epochs_coll, job))
                running.add(task)
        else:
            await asyncio.sleep(POLL_SECONDS)

    # wait for remaining tasks to finish
    if running:
        log(f"WAITING for {len(running)} running tasks to finish")
        await asyncio.gather(*running, return_exceptions=True)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
