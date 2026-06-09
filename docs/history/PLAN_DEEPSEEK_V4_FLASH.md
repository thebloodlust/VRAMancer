# Plan d'installation et test : DeepSeek-V4-Flash sur VRAMancer

**Cible** : Agent moins capable. Suivre **chaque étape dans l'ordre**. Ne pas sauter d'étape. Ne pas improviser.

**Hardware confirmé** :
- RTX 3090 (CUDA 1, SM86, 24 GB VRAM)
- RTX 5070 Ti (CUDA 0, SM120, 16 GB VRAM)
- **P2P actif au niveau driver** (`nvidia-smi topo -p2p` = OK bidirectionnel, topologie PIX). Proxmox configuré pour ReBAR + IOMMU bypass.
- **PyTorch P2P natif refusé** (`can_device_access_peer = False` à cause SM86/SM120 mismatch ou KVM), MAIS **Rust `vramancer_rust.GpuPipeline.transfer()` fait 23.9 GB/s** réels (PCIe gen4 x16 saturé) via `cuMemcpyPeerAsync`. **C'est précisément le bypass VRAMancer.**
- 191 GB RAM, 142 GB disque libre (INSUFFISANT, voir Phase 0)
- PyTorch 2.11.0+cu130, Python 3.12, venv : `/home/jeremie/VRAMancer/VRAMancer/.venv`

**Modèle cible** : `deepseek-ai/DeepSeek-V4-Flash` (architecture custom `DeepseekV4ForCausalLM`, 159.6 GB FP8, MoE 256 experts / 6 actifs).

**Repo VRAMancer** : `/home/jeremie/VRAMancer/VRAMancer`. Branche : `chore/sonnet-plan-v5`. **Ne JAMAIS pusher sans confirmation utilisateur.**

---

## Règle de gating (CRITIQUE)

Chaque phase produit un **artefact log** (`/tmp/dsv4_phase_NN.log`) ou un fichier de statut.

**Si une phase échoue (exit code != 0 ou critère "STOP" déclenché), NE PAS continuer. Demander à l'utilisateur.**

Toutes les commandes doivent être lancées depuis `/home/jeremie/VRAMancer/VRAMancer` avec le venv activé :

```bash
cd /home/jeremie/VRAMancer/VRAMancer && source .venv/bin/activate
```

---

## Phase 0 — Pré-requis disque (5 min)

**But** : Libérer assez de place pour 159.6 GB de poids + 20 GB de marge convert/safety.

### 0.1 Vérifier disque libre

```bash
df -h ~ | tail -1 | tee /tmp/dsv4_phase_00.log
```

**Critère** : colonne "Avail" doit être >= **180 GB**.

**Si déjà >= 180 GB** : sauter directement à Phase 1.

**Si < 180 GB** : passer à Phase 0.5 (nettoyage automatique).

### 0.2 Vérifier RAM libre

```bash
free -g | head -3 | tee -a /tmp/dsv4_phase_00.log
```

**Critère** : "disponible" >= 150 GB. Sinon STOP (la conversion Phase 4 chargera tout en RAM).

---

## Phase 0.5 — Nettoyage cache HuggingFace (10 min)

**But** : Libérer ~40 GB en supprimant des modèles obsolètes/redondants. **Liste explicite, ne supprimer QUE ces dossiers.**

### 0.5.1 Inventaire avant nettoyage

```bash
du -sh ~/.cache/huggingface/hub/* 2>/dev/null | sort -hr > /tmp/dsv4_phase_005_before.log
df -h ~ | tail -1 >> /tmp/dsv4_phase_005_before.log
cat /tmp/dsv4_phase_005_before.log
```

### 0.5.2 Modèles SAFE à supprimer (par ordre de priorité)

Ces modèles sont **redondants** ou **obsolètes** par rapport aux benchmarks V5 actuels :

| Modèle | Taille | Raison de suppression |
|--------|--------|----------------------|
| `models--Qwen--Qwen2.5-14B` | 28 GB | Base model, on garde uniquement Instruct |
| `models--Qwen--Qwen2.5-14B-Instruct` | 28 GB | Déjà testé V4, plus utilisé |
| `models--Qwen--Qwen2.5-7B-Instruct` | 15 GB | On garde la version GGUF Q4_K_M (4.4 GB) |
| `models--mistralai--Mistral-7B-v0.1` | 14 GB | Bench archivé V4 |
| `models--gpt2-medium` | 1.5 GB | Bench V3 archivé |
| `models--gpt2` | 0.5 GB | Bench V3 archivé |

**Total récupérable : ~87 GB** (largement suffisant pour atteindre 180 GB libres).

### 0.5.3 Suppression itérative jusqu'à 180 GB libres

**À FAIRE PAS À PAS, en vérifiant après chaque suppression** :

```bash
# Étape A : modèles 14B redondants (récupère ~56 GB)
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct
df -h ~ | tail -1 | tee -a /tmp/dsv4_phase_005.log
echo "After step A: $(df -BG --output=avail ~ | tail -1)" | tee -a /tmp/dsv4_phase_005.log
```

**Si Avail >= 180 GB après étape A → STOP, passer à Phase 1.**

```bash
# Étape B (uniquement si A insuffisant) : doublons 7B (récupère ~15 GB)
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
df -h ~ | tail -1 | tee -a /tmp/dsv4_phase_005.log
```

**Si Avail >= 180 GB après étape B → STOP, passer à Phase 1.**

```bash
# Étape C (uniquement si B insuffisant) : Mistral-7B base (récupère ~14 GB)
rm -rf ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1
df -h ~ | tail -1 | tee -a /tmp/dsv4_phase_005.log
```

```bash
# Étape D (dernier recours) : gpt2 (récupère ~2 GB)
rm -rf ~/.cache/huggingface/hub/models--gpt2-medium
rm -rf ~/.cache/huggingface/hub/models--gpt2
df -h ~ | tail -1 | tee -a /tmp/dsv4_phase_005.log
```

### 0.5.4 NE JAMAIS SUPPRIMER

Ces modèles sont **utilisés par les benchmarks V5 en cours** ou irremplaçables :

- ❌ `models--unsloth--Mistral-Medium-3.5-128B-GGUF` (33 GB) — bench V5 P15 actif
- ❌ `models--unsloth--Devstral-Small-2505-bnb-4bit` (27 GB) — bench V5 P16 actif
- ❌ `models--unsloth--Qwen3-30B-A3B-GGUF` (18 GB) — référence MoE
- ❌ `models--bartowski--Qwen2.5-7B-Instruct-GGUF` (4.4 GB) — bench rapide
- ❌ `models--bartowski--Qwen2.5-14B-Instruct-GGUF` (1.7 GB) — bench rapide
- ❌ `models--TinyLlama--TinyLlama-1.1B-Chat-v1.0` (2.1 GB) — tests CI/smoke

### 0.5.5 Vérification finale

```bash
df -h ~ | tail -1 | tee -a /tmp/dsv4_phase_005.log
free -g | head -3 | tee -a /tmp/dsv4_phase_005.log
```

**Critère** : Avail >= 180 GB. Si non atteint après étape D, **STOP et demander à l'utilisateur** quels autres dossiers (hors liste "ne jamais supprimer") il accepte de sacrifier.

---

## Phase 1 — Tilelang (le go/no-go, ~10 min)

**But** : Vérifier que le DSL `tilelang` (utilisé par `kernel.py` du modèle) compile un kernel FP8.

**Stratégie SM86/SM120** :
- On teste **d'abord SM120 (RTX 5070 Ti)** car c'est le compute GPU dans la stratégie split (voir Phase 6a).
- Si SM120 OK mais SM86 KO → on bascule en mode **"3090 = storage VRAM via VRAMancer ReBAR + lending, 5070 Ti = compute"**. Ce n'est PAS un échec, c'est le scénario emblématique de VRAMancer.
- Si SM120 KO → STOP, plan B (DeepSeek-V2-Lite Q4).

### 1.1 Installer tilelang

```bash
cd /home/jeremie/VRAMancer/VRAMancer && source .venv/bin/activate
pip install tilelang 2>&1 | tee /tmp/dsv4_phase_01_install.log
```

**Critère** : exit code 0 ET ligne `Successfully installed tilelang-*`.
**Si échec** : STOP. Coller le log à l'utilisateur.

### 1.2 Test minimal de compilation FP8 sur RTX 3090

Créer le fichier `/tmp/test_tilelang_fp8.py` avec :

```python
import os
# Force usage du RTX 3090 (SM86) - le 5070 Ti est CUDA 0, le 3090 CUDA 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import tilelang
import tilelang.language as T

print(f"torch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
cap = torch.cuda.get_device_capability(0)
print(f"compute_cap: {cap}")
assert cap[0] >= 8, f"Need SM80+, got {cap}"

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

@tilelang.jit(pass_configs=pass_configs)
def simple_gemm(M, N, K, dtype="float8_e4m3", accum_dtype="float32", out_dtype="bfloat16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, 64), T.ceildiv(M, 64), threads=128) as (bx, by):
            A_s = T.alloc_shared((64, 32), dtype)
            B_s = T.alloc_shared((64, 32), dtype)
            C_l = T.alloc_fragment((64, 64), accum_dtype)
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, 32), num_stages=2):
                T.copy(A[by*64:(by+1)*64, k*32:(k+1)*32], A_s)
                T.copy(B[bx*64:(bx+1)*64, k*32:(k+1)*32], B_s)
                T.gemm(A_s, B_s, C_l, transpose_B=True)
            T.copy(C_l, C[by*64:(by+1)*64, bx*64:(bx+1)*64])
    return main

print("Compiling FP8 GEMM kernel for SM86...")
kernel = simple_gemm(128, 128, 128)
print("Kernel compiled OK")

a = torch.randn(128, 128, device="cuda").to(torch.float8_e4m3fn)
b = torch.randn(128, 128, device="cuda").to(torch.float8_e4m3fn)
c = torch.empty(128, 128, device="cuda", dtype=torch.bfloat16)
kernel(a, b, c)
torch.cuda.synchronize()
print(f"Output mean: {c.float().mean().item():.4f}")
print("PHASE_01_OK")
```

Lancer :

```bash
python /tmp/test_tilelang_fp8.py 2>&1 | tee /tmp/dsv4_phase_01_test.log
```

**Critère** : la dernière ligne du log doit être `PHASE_01_OK`.
**Si erreur "no kernel image is available" / "unsupported architecture" / autre** : STOP. C'est le hard blocker. Coller le log à l'utilisateur et passer au plan B (DeepSeek-V2-Lite Q4_K_M, voir Annexe).

---

## Phase 2 — Cloner le code d'inférence officiel (2 min)

**But** : Récupérer `model.py`, `kernel.py`, `generate.py`, `convert.py` du repo HF (déjà fait dans `/tmp/dsv4/` lors de la session précédente, à recréer).

### 2.1 Cloner dans le repo

```bash
cd /home/jeremie/VRAMancer/VRAMancer
mkdir -p third_party/deepseek_v4
cd third_party/deepseek_v4
for f in inference/kernel.py inference/model.py inference/generate.py inference/convert.py encoding/encoding_dsv4.py; do
  mkdir -p "$(dirname "$f")"
  curl -sLf "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/$f" -o "$f" || { echo "FAIL: $f"; exit 1; }
done
ls -la inference/ encoding/ | tee /tmp/dsv4_phase_02.log
```

**Critère** : 4 fichiers dans `inference/` + 1 dans `encoding/`, tous > 1 KB.

### 2.2 Vérifier qu'aucune dépendance bloquante n'est ajoutée

```bash
cd /home/jeremie/VRAMancer/VRAMancer/third_party/deepseek_v4
grep -rn "deep_gemm\|^import deep_gemm\|from deep_gemm" inference/ encoding/ | tee -a /tmp/dsv4_phase_02.log
```

**Critère** : aucune ligne (sortie vide).
**Si lignes trouvées** : STOP. Le modèle dépend de `deep_gemm` (H100/B200 only), plan annulé.

---

## Phase 3 — Téléchargement des poids (LONG, 2-6h selon réseau)

**But** : Récupérer les 46 shards safetensors (~159.6 GB) depuis HuggingFace.

### 3.1 Vérifier que `huggingface-cli` est dispo

```bash
cd /home/jeremie/VRAMancer/VRAMancer && source .venv/bin/activate
which huggingface-cli && huggingface-cli --version
```

**Si absent** : `pip install -U "huggingface_hub[cli]"`

### 3.2 Lancer le download en arrière-plan

```bash
cd /home/jeremie/VRAMancer/VRAMancer
mkdir -p ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash
nohup huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash \
  --max-workers 4 \
  > /tmp/dsv4_phase_03_download.log 2>&1 &
echo "DL_PID=$!" | tee /tmp/dsv4_phase_03_pid.log
```

### 3.3 Surveiller la progression (NE PAS BLOQUER)

Vérifier toutes les 15-30 min :

```bash
ls -la ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/*.safetensors 2>/dev/null | wc -l
du -sh ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/
tail -3 /tmp/dsv4_phase_03_download.log
```

**Critère final** :
- 46 fichiers `model-XXXXX-of-00046.safetensors`
- Taille totale entre 155 et 165 GB
- `tail` doit montrer `Download complete` ou pas d'erreur dans les dernières lignes

**Si erreur réseau / partiel** : relancer la même commande (huggingface-cli reprend automatiquement).

### 3.4 Vérification d'intégrité

```bash
cd ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash
ls *.safetensors | wc -l | tee -a /tmp/dsv4_phase_03.log   # doit afficher 46
test -f config.json && test -f model.safetensors.index.json && echo "INDEX_OK" | tee -a /tmp/dsv4_phase_03.log
```

**Critère** : `46` ET `INDEX_OK`. Sinon STOP.

---

## Phase 4 — Conversion checkpoint HF → format custom (30 min)

**But** : Le `model.py` officiel attend un format custom (clés renommées, sharding par MP). Utiliser `convert.py` du repo.

### 4.1 Lancer la conversion avec MP=2 (2 GPUs)

```bash
cd /home/jeremie/VRAMancer/VRAMancer/third_party/deepseek_v4/inference
python convert.py \
  --hf-ckpt-path ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash \
  --save-path ~/.cache/huggingface/hub/dsv4_converted_mp2 \
  --n-experts 256 \
  --model-parallel 2 \
  2>&1 | tee /tmp/dsv4_phase_04.log
```

**Note** : si `convert.py` n'a pas exactement ces flags, regarder son `ArgumentParser` :
```bash
python convert.py --help 2>&1 | head -30
```
Et adapter. Les valeurs (`n_experts=256`, `mp=2`) sont fixes.

**Critère** : exit code 0, dossier `~/.cache/huggingface/hub/dsv4_converted_mp2/` contient 2 fichiers `.safetensors` shardés (`model0-mp2.safetensors`, `model1-mp2.safetensors` ou similaire).

**Si erreur "out of memory" sur CPU** : STOP. La conversion charge tout en RAM. Avec 191 GB RAM cela devrait passer mais marge serrée.

---

## Phase 5 — Test minimal d'inférence single-GPU (30 min)

**But** : Lancer `generate.py` officiel SANS optimisation VRAMancer pour valider que le code marche. Sur 1 seul GPU avec `mp=1` (mais alors le modèle ne tient pas — ce test SERA en OOM, c'est attendu).

**Cette phase confirme juste que le code charge le tokenizer et tente de loader, pas qu'il tient en VRAM.** L'OOM ici est NORMAL.

### 5.1 Test "load only" (sans génération)

Créer `/tmp/test_dsv4_load.py` :

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"  # 3090 puis 5070 Ti
import sys
sys.path.insert(0, "/home/jeremie/VRAMancer/VRAMancer/third_party/deepseek_v4/inference")
sys.path.insert(0, "/home/jeremie/VRAMancer/VRAMancer/third_party/deepseek_v4/encoding")

import json
import torch
from model import Transformer, ModelArgs

with open(os.path.expanduser("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/config.json")) as f:
    cfg = json.load(f)

print(f"Config keys: {list(cfg.keys())[:10]}...")
print(f"hidden_size={cfg.get('hidden_size')} n_layers={cfg.get('num_hidden_layers')}")
print(f"n_experts={cfg.get('n_routed_experts')}")
print("PHASE_05_CONFIG_OK")
```

```bash
cd /home/jeremie/VRAMancer/VRAMancer && source .venv/bin/activate
python /tmp/test_dsv4_load.py 2>&1 | tee /tmp/dsv4_phase_05.log
```

**Critère** : `PHASE_05_CONFIG_OK` dans le log.

### 5.2 Lancer `generate.py` officiel avec `torchrun` (TENTATIVE)

```bash
cd /home/jeremie/VRAMancer/VRAMancer/third_party/deepseek_v4/inference
CUDA_VISIBLE_DEVICES=1,0 torchrun --nproc-per-node 2 generate.py \
  --ckpt-path ~/.cache/huggingface/hub/dsv4_converted_mp2 \
  --config <CONFIG_TROUVÉ_DANS_LE_REPO> \
  --interactive \
  --max-new-tokens 32 \
  --temperature 0.7 \
  2>&1 | tee /tmp/dsv4_phase_05_gen.log
```

**Note importante** : il faut récupérer le fichier de config attendu par `generate.py`. Regarder l'aide :

```bash
python generate.py --help 2>&1 | head -30
```

Le repo HF a généralement un dossier `configs/` à inspecter via :

```bash
curl -sL "https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash" | python -c "import json, sys; d=json.load(sys.stdin); print('\n'.join(s['rfilename'] for s in d['siblings'] if 'config' in s['rfilename'].lower() or s['rfilename'].endswith('.json')))"
```

**Résultat attendu** : OOM (out of memory) sur 1 ou les 2 GPUs. **C'est normal**. Le modèle 80 GB par rank ne tient pas dans 24+16 GB.

**Critère "succès" de la phase** : le code arrive au moins à appeler `load_model()` AVANT l'OOM. Si erreur dès l'import (manque tilelang, attribut manquant, etc.), STOP.

---

## Phase 6 — STOP. Récapitulation pour décision humaine.

**À ce stade, tu as confirmé :**
- ✅ tilelang compile sur SM86
- ✅ Code d'inférence officiel téléchargé et valide
- ✅ 159 GB de poids présents
- ✅ Conversion MP=2 fonctionne
- ❌ Inférence native impossible (OOM, 80 GB/GPU requis vs 24+16 dispo)

**LA SUITE EST DU DÉVELOPPEMENT, PAS DE L'INSTALLATION.** Il faut patcher `model.py` pour ajouter le **CPU expert offloading** (charger les experts à la demande depuis les 191 GB de RAM, en utilisant les modules existants de VRAMancer : `core/stream_manager.py`, `core/vram_lending.py`, `core/hierarchical_memory.py`).

Cela représente **plusieurs jours de dev**. Ne pas tenter sans plan détaillé séparé.

**Action de l'agent à ce stade** :
1. Écrire un récap dans `/tmp/dsv4_status.md` listant phases réussies/échouées.
2. Coller à l'utilisateur les logs `/tmp/dsv4_phase_*.log`.
3. Demander : *"Phases 1-5 terminées. Veux-tu que je propose le plan de patch d'expert offloading (Phase 6+) ou qu'on bascule sur le plan B (DeepSeek-V2-Lite Q4) ?"*

---

## Plan B — Repli si Phase 1 échoue (DeepSeek-V2-Lite Q4)

**Quand l'utiliser** : tilelang ne compile pas, OU disque insuffisant, OU phases 3/4 échouent et il est impossible de débloquer.

**Modèle** : `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF` (16B MoE, 2.4B actifs, ~10 GB en Q4_K_M, tient sur 1 GPU).

```bash
cd /home/jeremie/VRAMancer/VRAMancer && source .venv/bin/activate
huggingface-cli download bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF \
  --include "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf" \
  --local-dir ~/.cache/huggingface/hub/models--bartowski--DeepSeek-Coder-V2-Lite-Instruct-GGUF \
  2>&1 | tee /tmp/dsv4_planB_dl.log
```

Puis tester via `llama-cpp-python` (déjà installé dans le venv) :

```python
from llama_cpp import Llama
llm = Llama(
    model_path="/home/jeremie/.cache/huggingface/hub/models--bartowski--DeepSeek-Coder-V2-Lite-Instruct-GGUF/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=8192,
    tensor_split=[0.6, 0.4],  # 3090 / 5070 Ti
    flash_attn=True,
)
print(llm("Write a Python quicksort:", max_tokens=128, echo=False)["choices"][0]["text"])
```

---

## Tableau récapitulatif

| Phase | Durée | Bloquant ? | Critère succès |
|-------|-------|------------|----------------|
| 0. Disque/RAM | 5 min | Oui | 180 GB libre, 150 GB RAM dispo |
| 0.5. Cleanup HF cache | 10 min | Si <180 GB | Avail >= 180 GB après suppressions ciblées |
| 1. Tilelang | 10 min | **CRITIQUE** | `PHASE_01_OK` (au moins SM120) |
| 2. Clone code | 2 min | Oui | 5 fichiers .py présents |
| 3. Download | 2-6h | Oui | 46 shards, 155-165 GB |
| 4. Convert | 30 min | Oui | dossier `dsv4_converted_mp2/` valide |
| 5. Load test | 30 min | Non (OOM attendu) | code arrive à `load_model()` |
| 6. Décision | — | — | Plan dev offloading OU plan B |

---

## Annexe : variables d'environnement utilisées

| Variable | Valeur recommandée | Rôle |
|----------|---------------------|------|
| `CUDA_VISIBLE_DEVICES` | `"1,0"` | 3090 d'abord (plus de VRAM = rank 0) |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` | Ordre stable |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` (optionnel) | Download plus rapide si `pip install hf_transfer` |
| `TOKENIZERS_PARALLELISM` | `false` | Évite warnings au load |

---

## Ne PAS faire

- ❌ Ne pas pusher sur `chore/sonnet-plan-v5` ou autre branche distante sans confirmation utilisateur.
- ❌ Ne pas supprimer de fichiers dans `~/.cache/huggingface/` **hors de la liste explicite Phase 0.5.2**. La liste "Ne jamais supprimer" (Phase 0.5.4) est intouchable.
- ❌ Ne pas lancer `pip install --upgrade torch` ou similaire (casse la stack CUDA actuelle).
- ❌ Ne pas modifier `core/`, `dashboard/`, etc. tant que la Phase 5 n'est pas validée.
- ❌ Ne pas lancer phases 3-4 en parallèle (impact disque/IO).

---

**Fin du plan. Suivre dans l'ordre. En cas de doute → STOP et demander.**
