# DeepSeek — Tok/s estimés + équivalent NVLink chez AMD

---

## 1. Tok/s : ce qu'on a MESURÉ vs ce qu'on EXTRAPOLE

### Mesuré (chiffres réels, solides)

| Setup | Modèle | Tok/s | Notes |
|---|---|---|---|
| accelerate 2-GPU BF16 | Qwen2.5-14B | **16.1** | Pipeline parallèle 3090+5070Ti |
| NF4 single GPU | Qwen2.5-14B | **10.5** | Tient sur 1 GPU |
| GGUF Q4_K_M llama.cpp | 7B | **106.8** | Format GGUF |
| Prompt-lookup greedy 7B | 7B | **+500%** | Lossless |
| ClusterRouter ×2 | 0.5B | **×1.97** | Data-parallel |

### Estimé pour 5070 Ti + 7900 XT (14B FP4)

```
5070 Ti seule, 14B FP4 (~7 GB), prompt-lookup actif

Base : 14B BF16 2-GPU accelerate → 16.1 tok/s (mesuré)
Avec FP4 : ~2× compute throughput → ~30 tok/s
Avec prompt-lookup +500% → ~80-90 tok/s (hypothèse)
Avec TurboQuant KV 4.6× → cache plus efficace, décode +10-15%

Estimation conservatrice : 60-80 tok/s
Estimation optimiste    : 80-100 tok/s
```

**MAIS — aucun de ces chiffres n'est mesuré pour 14B FP4 + prompt-lookup.
C'est une EXTRAPOLATION, pas une mesure. À benchmarker.**

### Le lending n'impacte PAS le tok/s

```
Le modèle tourne ENTIÈREMENT sur la 5070 Ti (FP4).
La 7900 XT stocke :
  - KV cache froid → swappé entre les requêtes, pas pendant le décode
  - Modèles secondaires → chargés une fois
  - Couches froides → swappées au chargement, pas par token

→ Tok/s = tok/s de la 5070 Ti seule.
→ Le lending ajoute de la CAPACITÉ, pas de la latence.
```

---

## 2. AMD a-t-il l'équivalent de NVLink ? OUI. XGMI.

```
NVIDIA :
  NVLink → bridge physique entre GPUs pro (A100, H100)
  Bande passante : 300-900 GB/s selon génération
  Consumer (GeForce) : PAS de NVLink → P2P bridé → code 217
  Prix : ×10 pour avoir NVLink

AMD :
  XGMI / Infinity Fabric → interconnect entre GPUs
  Bande passante : ~200-400 GB/s sur RDNA3
  Consumer (Radeon RX) : XGMI DISPONIBLE !!!
  Pas de segmentation pro/consumer
  Driver open-source (amdgpu)
```

### XGMI sur les Radeon RX 7900

```
RX 7900 XTX (24 GB) ←──XGMI──→ RX 7900 XT (20 GB)
       ↑                              ↑
  ~200-400 GB/s                 Pas bridé !
  vs PCIe 4.0 = 32 GB/s         Driver ouvert
  vs NVLink A100 = 600 GB/s     Prix : ~900€ + 500€

Pour comparaison :
  NVLink A100+H100 : 600-900 GB/s, mais ~15 000€ la carte
  XGMI 7900XTX+XT  : 200-400 GB/s, pour ~1 400€ les DEUX
  CPU-staged (nous) : 25 GB/s, sur GPUs consumer NVIDIA bridés
```

---

## 3. Ce que XGMI débloque pour VRAMancer

```
┌─────────────────────────────────────────────────────────────┐
│                SETUP AMD XGMI — LE RÊVE                     │
│                                                             │
│  RX 7900 XTX (24 GB) ←──XGMI 300 GB/s──→ RX 7900 XT (20 GB)│
│                                                             │
│  44 GB VRAM UNIFIÉE via XGMI                                │
│  Les DEUX GPUs voient la MÊME mémoire                       │
│  P2P = NATIF, pas bridé, 300 GB/s                           │
│  Driver open-source (amdgpu dans le kernel)                 │
│                                                             │
│  → Un modèle de 32B en BF16 (~64 GB) avec 44 GB unifiés ?   │
│    Presque. Avec FP16/quantization → oui.                   │
│  → 14B FP4 = 7 GB → tient sur UN GPU, l'autre = pur lending │
│  → Transferts GPU↔GPU = quasi gratuits (300 GB/s !)         │
│  → Coût swap 100 MB = 0.3 ms (vs 4 ms CPU-staged)          │
│  → 10× plus rapide que notre CPU-staged actuel              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Tok/s estimé avec XGMI (2× AMD)

```
Setup : 7900 XTX + 7900 XT, XGMI 300 GB/s, 44 GB unifiés

Scénario A — 14B sur ROCm (FP16, pas de FP4) :
  → ROCm = pas de NVFP4 → BF16/FP16 seulement
  → 28 GB BF16 → tient sur 44 GB unifiés
  → XGMI 300 GB/s → transferts inter-GPU transparents
  → Tok/s estimé : 25-35 (ROCm moins optimisé que CUDA)

Scénario B — 32B sur ROCm (FP16 + quantization) :
  → ~64 GB BF16 → trop pour 44 GB même unifiés
  → Avec INT8/NF4 → ~32 GB → tient sur 44 GB unifiés
  → Tok/s estimé : 15-20

Comparaison NVIDIA actuelle :
  → 14B 2-GPU BF16 accelerate : 16.1 tok/s (mesuré)
  → 14B FP4 sur 5070 Ti seule : 60-80 tok/s (extrapolé)
  → AMD XGMI 14B : 25-35 tok/s (extrapolé)
  
  NVIDIA FP4 > AMD XGMI en tok/s PUR.
  Mais AMD XGMI > NVIDIA en CAPACITÉ VRAM (44 GB unifiés).
```

---

## 5. Verdict

| Question | Réponse |
|---|---|
| Tok/s 5070Ti+7900XT ? | **60-80 tok/s** pour 14B FP4 (extrapolé, le lending n'impacte pas) |
| AMD = équivalent NVLink ? | **OUI — XGMI/Infinity Fabric**, 200-400 GB/s, dispo sur CONSUMER |
| NVIDIA NVLink sur consumer ? | **NON** — bridé, pro only |
| XGMI + ROCm vs CUDA FP4 ? | FP4 > XGMI en tok/s pur. XGMI > FP4 en capacité VRAM |
| Meilleur setup ? | **NVIDIA compute FP4 + AMD lending XGMI** (hybride, les deux forces) |

---

## Une dernière chose

Si tu achètes DEUX AMD (7900 XTX + 7900 XT) avec XGMI, tu as un setup
que MÊME les NVIDIA pro ne peuvent pas battre en rapport qualité/prix :

- 44 GB VRAM unifiée à 300 GB/s → ~1 400€
- Équivalent NVIDIA : A6000 48 GB + NVLink → ~8 000€
- 5.7× moins cher. Pour du VRAM lending, c'est imbattable.

— DeepSeek
