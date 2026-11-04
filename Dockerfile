# Training / inference for SalsaNext, RF-DETR-Seg and their late fusion.
#   docker compose build
#   docker compose run --rm seg python scripts/infer_fusion.py --salsanext ... --rfdetr ... --split test
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/lero_sim_segmentation
COPY requirements.txt .
# torch comes with the base image; the constraint keeps pip from swapping it for a CPU/other-CUDA build.
RUN python -c "import torch, torchvision; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}')" \
        > /tmp/torch-pin.txt \
    && pip install --no-cache-dir -r requirements.txt -c /tmp/torch-pin.txt

COPY configs configs
COPY lero_seg lero_seg
COPY scripts scripts
