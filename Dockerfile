# Dockerfile
# --------------------------------------------------------------
# 1️⃣  Base image (CUDA 12.1)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

# --------------------------------------------------------------
# 2️⃣  Install ROCm  (optional)
# Uncomment the block below if you need ROCm support.
# --------------------------------------------------------------
# RUN wget https://repo.radeon.com/rocm/rocm-nightly/rocm-repo-22.04-7.7.0-1.2.1_amd64.deb && \
#     dpkg -i rocm-repo-22.04-7.7.0-1.2.1_amd64.deb && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends \
#         rocm-dkms rocm-dev \
#         && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------
# 3️⃣  Generic dependencies
# --------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --------------------------------------------------------------
# 4️⃣  Copy sources
# --------------------------------------------------------------
COPY . .

# --------------------------------------------------------------
# 5️⃣  Build the Python env
# --------------------------------------------------------------
RUN python3 -m venv .venv && \
    .venv/bin/pip install --upgrade pip setuptools wheel && \
    .venv/bin/pip install -r requirements.txt

# --------------------------------------------------------------
# 6️⃣  Entrypoint
# --------------------------------------------------------------
CMD ["python3", "-u", "run_demo.py"]
