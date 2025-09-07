# -------------------------------------------------------------
# 1️⃣  Image de base : CUDA 12.1 (ou ROCm 7.7 si besoin)
# -------------------------------------------------------------
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

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
    # Si vous voulez forcer la version GPU exacte :
    # .venv/bin/pip install "torch>=2.1+cu121" transformers>=4.34 accelerate>=0.27 flask>=2.3

# -------------------------------------------------------------
# 8️⃣  Nettoyage (facultatif mais recommandé)
# -------------------------------------------------------------
RUN apt-get purge -y --auto-remove git && \
    rm -rf ~/.cache/pip

# -------------------------------------------------------------
# 9️⃣  Commande par défaut
# -------------------------------------------------------------
# Vous pouvez l’échanger selon le script que vous voulez exécuter
CMD ["python3", "-u", "run_demo.py"]
