# -------------------------------------------------------------
# 1. Build stage: CUDA 12.1 (or ROCm if needed)
# -------------------------------------------------------------
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS build

# -------------------------------------------------------------
# 2. System dependencies
# -------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# -------------------------------------------------------------
# 3. Python virtual environment + dependencies
# -------------------------------------------------------------
RUN python3 -m venv .venv && \
    .venv/bin/pip install --upgrade pip setuptools wheel && \
    .venv/bin/pip install -r requirements.txt && \
    .venv/bin/pip install gunicorn && \
    .venv/bin/pip install . --no-deps

# Cleanup
RUN apt-get purge -y --auto-remove git && \
    rm -rf ~/.cache/pip

# -------------------------------------------------------------
# 4. Runtime stage (slim)
# -------------------------------------------------------------
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS runtime

# Install Python runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 curl && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV VRM_LOG_JSON=1
ENV VRM_API_PORT=5030

WORKDIR /app

# Copy venv and application code
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/core /app/core
COPY --from=build /app/vramancer /app/vramancer
COPY --from=build /app/dashboard /app/dashboard
COPY --from=build /app/scripts /app/scripts
COPY --from=build /app/requirements.txt /app/requirements.txt

ENV PATH="/app/.venv/bin:$PATH"

# -------------------------------------------------------------
# 5. Non-root user for security
# -------------------------------------------------------------
RUN groupadd -r vramancer && \
    useradd -r -g vramancer -d /app -s /sbin/nologin vramancer && \
    chown -R vramancer:vramancer /app
USER vramancer

# Expose the correct port
EXPOSE 5030

# -------------------------------------------------------------
# 6. Health check (correct endpoint and port)
# -------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:5030/health || exit 1

# -------------------------------------------------------------
# 7. Start with production server (gunicorn)
# -------------------------------------------------------------
CMD ["gunicorn", "--bind", "0.0.0.0:5030", "--workers", "1", "--threads", "4", \
     "--timeout", "120", "--graceful-timeout", "30", "--keep-alive", "5", \
     "--access-logfile", "-", "--error-logfile", "-", \
     "core.production_api:app"]
