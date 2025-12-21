FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121 



RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/


CMD ["python3", "/app/executor_concurrent_async.py"]

