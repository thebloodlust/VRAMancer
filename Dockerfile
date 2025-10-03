# -------------------------------------------------------------
# 1️⃣  Image de base : CUDA 12.1 (ou ROCm 7.7 si besoin)
# -------------------------------------------------------------
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS build

# -------------------------------------------------------------
# 2️⃣  Installation de ROCm (optionnel) – à décommenter si nécessaire
# -------------------------------------------------------------
# RUN wget https://repo.radeon.com/rocm/rocm-nightly/rocm-repo-22.04-7.7.0-1.2.1_amd64.deb && \
#     dpkg -i rocm-repo-22.04-7.7.0-1.2.1_amd64.deb && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends \
#         rocm-dkms rocm-dev \
#         && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------
# 3️⃣  Installation des dépendances systèmes
# -------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git \
        && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------
# 4️⃣  Point d’entrée du conteneur
# -------------------------------------------------------------
WORKDIR /app

# -------------------------------------------------------------
# 5️⃣  Copie du code source
# -------------------------------------------------------------
COPY . .

# -------------------------------------------------------------
# 6️⃣  Création d’un environnement virtuel Python
# -------------------------------------------------------------
RUN python3 -m venv .venv && \
    .venv/bin/pip install --upgrade pip setuptools wheel

# -------------------------------------------------------------
# 7️⃣  Installation des dépendances Python (torch≥2.1+CUDA)
# -------------------------------------------------------------
# La ligne ci‑dessous installe la dernière version compatible
# avec CUDA 12.1 (ou ROCm si vous avez activé la section ROCm)
# Vous pouvez commenter le suffixe +cu121 si vous utilisez ROCm
RUN .venv/bin/pip install -r requirements.txt && \
    pip install . --no-deps

# -------------------------------------------------------------
# 8️⃣  Nettoyage (facultatif mais recommandé)
# -------------------------------------------------------------
RUN apt-get purge -y --auto-remove git && \
    rm -rf ~/.cache/pip

# -------- Image runtime mince --------
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS runtime
ENV PYTHONUNBUFFERED=1
ENV VRM_DISABLE_SOCKETIO=0 \
    VRM_LOG_JSON=1
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/core /app/core
COPY --from=build /app/run_demo.py /app/run_demo.py
COPY --from=build /app/vramancer /app/vramancer
COPY --from=build /app/requirements.txt /app/requirements.txt
ENV PATH="/app/.venv/bin:$PATH"

# -------------------------------------------------------------
# 9️⃣  Commande par défaut
# -------------------------------------------------------------
# Vous pouvez l’échanger selon le script que vous voulez exécuter
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python - <<'PY' || exit 1
import os,urllib.request,json
url=os.environ.get('VRM_HEALTH_URL','http://localhost:5000/api/version')
try:
    with urllib.request.urlopen(url,timeout=2) as r:
        data=json.loads(r.read().decode())
        assert 'version' in data
except Exception as e:
    print('healthcheck failed',e)
    raise
PY
CMD ["python", "-u", "run_demo.py"]
