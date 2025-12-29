FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
# apt-get stuff first (stable)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app
COPY src/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY . /app
ENV TORCH_CUDA_ARCH_LIST="8.9+PTX"
RUN pip3 install /app/lib/difflogic

CMD ["python3", "/app/src/main.py"]
