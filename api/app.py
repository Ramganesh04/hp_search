
import os
import time
from typing import Any, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pymongo import AsyncMongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "jobsdb")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "jobs")

client: AsyncMongoClient | None = None


class Hyperparameters(BaseModel):
    GATE_OPTIMIZER: str = Field(...)

    GATES: int = Field(..., ge=1)
    NETWORK_LAYERS: int = Field(..., ge=1)

    GROUPING: int = Field(..., ge=1)
    GROUP_SUM_TAU: int = Field(..., ge=1)

    RESIDUAL_LAYERS: int = Field(..., ge=0)
    NOISE_TEMP: float = Field(..., gt=0)

    EPOCHS: int = Field(..., ge=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = AsyncMongoClient(MONGO_URI)
    await client.aconnect()
    yield
    if client is not None:
        await client.close()


app = FastAPI(lifespan=lifespan)


@app.post("/jobs")
async def create_job(hyperparams: Hyperparameters) -> Dict[str, Any]:
    if client is None:
        raise HTTPException(status_code=500, detail="Mongo client not initialized")

    doc = {
        "TimeCreated": time.time(),
        "TotalParameters": hyperparams.GATES * hyperparams.NETWORK_LAYERS,
        "Hyperparameters": hyperparams.model_dump(),
        "Status": "Pending",
    }

    coll = client[MONGO_DB][MONGO_COLLECTION]
    result = await coll.insert_one(doc)

    return {"jobId": str(result.inserted_id), "status": "Pending"}
