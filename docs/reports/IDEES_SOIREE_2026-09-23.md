# Soirée du 23/09 : les « autres idées », mesurées

Machine : VM Proxmox, EPYC 7402 (16 vCPU, AVX2), 172 Go de RAM, RTX 3090 + RX 7900 XT,
llama.cpp b11112 (Vulkan). Tout est mesuré, sauf mention contraire.

## 1. Modèles ternaires PrismML (Bonsai 2 27B) — intégré

Poids ternaires / 1 bit, fork llama.cpp obligatoire (le llama.cpp officiel refuse le fichier).

| Bonsai 2 27B | Taille | 3090 CUDA | 3090 Vulkan | 7900 XT Vulkan | CPU seul (16 vCPU) |
|---|---:|---:|---:|---:|---:|
| PQ2_0 (2 bits) | 6.7 Gio | **74.0 tok/s** (71.2 à 8K) | 0.9 (*) | 0.9 (*) | 5.3 |
| PTQ1_0 (1 bit) | 5.5 Gio | 61.5 | 6.6 | 7.6 | 0.8 |

(*) noyaux Vulkan du fork pas au point pour PQ2_0 (test court, `-p 16 -n 8`).

**Sur AMD, c'est le pilote, pas la carte** (mesuré le 24/09 après l'ajout de l'accès
`/dev/kfd`) : build **ROCm** du fork sur la RX 7900 XT, PQ2_0 **53.5 tok/s** (51.8 à 8K,
prefill 450), PTQ1_0 33.9 — contre 0.9 en Vulkan. Via `vramancer serve` : 52 tok/s, réponse
juste, calcul sur la 7900 XT seule (8.6 Go). Intégré : sur une machine AMD sans NVIDIA,
le fork est pris en build ROCm et le runtime ROCm 7 est installé sans sudo depuis les
paquets pip d'AMD (TheRock, ~4 Go, `~/.cache/vramancer/rocm`). Seul prérequis système :
`sudo usermod -aG render $USER` (sinon repli Vulkan, avec ce message).

Pour comparer : un 32B classique en Q4_K_M (18.5 Gio) fait 38.3 tok/s sur la même 3090 et
**4.7 tok/s en CPU seul** — le ternaire n'y gagne presque rien (5.3) : sur AVX2 il est borné
par le calcul, pas par la mémoire. Qualité à l'œil, sans mode réflexion : fluide, petites
erreurs (heure d'arrivée d'un train 16h03 au lieu de 16h05 ; un exemple de palindrome faux).

Intégré : `serve` détecte les tenseurs PrismML dans l'en-tête, télécharge la release
précompilée du fork (build CUDA dès qu'une NVIDIA est là), et sert le modèle sans réglage
(8.4 Go de VRAM, réponse correcte). Verdict : **utile sur GPU NVIDIA, AMD (ROCm) et Mac**, pas
en CPU pour l'instant.

## 2. Cache KV quantifié — intégré (`VRM_KV_TYPE=auto`)

Qwen2.5-Coder-32B Q4_K_M, 3090 seule :

| Cache KV | Perplexité wikitext-2 | tg @ 8K | Contexte max tenu |
|---|---:|---:|---|
| f16 | 6.2103 | 34.2 | 16K (31.2 tok/s) ; 24K : mémoire insuffisante |
| q8_0 | 6.2120 (+0.03 %) | 32.9 | 24K (26.9 tok/s) |
| q4_0 | 6.2246 (+0.2 %) | 32.3 | 48K (20.2 tok/s) |

`serve` estime le cache KV depuis l'en-tête (couches d'attention pleine seulement sur les
hybrides Qwen3.5/3.6) et passe en q8_0 puis q4_0 quand le contexte ne tient pas, au lieu de
planter. Vérifié de bout en bout : 24K de contexte sur une 3090 → q8_0 choisi, document de
10 500 tokens, bonne réponse.

TurboQuant (KV 3 bits, Google) : refusé dans llama.cpp officiel, forks seulement ; le gain
sur q4_0 serait ~1/3 de contexte en plus. Non intégré.

## 3. Étage disque : plus gros que la RAM, lent mais sans planter

DeepSeek-V4-Flash (81 Gio), placement du planificateur (3090 + 7900 XT + RAM), chargement en
mmap, RAM plafonnée par cgroup, cache disque vidé avant chaque essai :

| RAM autorisée | tg | prefill | Lu sur disque |
|---|---:|---:|---:|
| illimitée (référence) | 11.8 | 38.9 | — |
| 48 Go | 3.40 | 3.0 | 106 Gio |
| 32 Go | 3.37 | 2.9 | 125 Gio |
| **16 Go** | **2.12** | 2.4 | 144 Gio |

Un modèle de 284 milliards de paramètres répond à 2 tok/s avec 16 Go de RAM. Corrigé
dans la foulée : `serve` chargeait sans mmap, donc un modèle plus gros que la RAM aurait été
tué par l'OOM killer ; il passe maintenant en mmap dans ce cas.

## 4. `vramancer invite` : ajouter une machine en une commande, quel que soit l'OS — intégré

```
Linux / macOS : curl -fsSL 'http://<nœud>:5055/join.sh?t=<jeton>' | sh
Windows       : irm 'http://<nœud>:5055/join.ps1?t=<jeton>' | iex
```

Le script télécharge le `rpc-server` llama.cpp de la même version que le nœud (Vulkan :
NVIDIA / AMD / Intel ; Metal sur Mac ; CPU sinon), le lie à l'adresse vue par le nœud
(jamais 0.0.0.0 : rpc-server n'a pas d'authentification), et s'enregistre ; `serve` utilise
ensuite les nœuds joignables. Testé de bout en bout sur la machine : nœud « 7900 XT »
ajouté par `curl | sh`, Qwen3.6 Q6_K (29 Go, trop gros pour la 3090) servi sur 3090 + nœud
RPC à **78 tok/s** ; mauvais jeton → 403. Script PowerShell non testé (pas de Windows ici).

## 5. `vramancer predict` : prédire avant de télécharger — intégré

Lit l'en-tête d'un GGUF distant par requêtes HTTP partielles (**11 Mo lus sur 34 Go**, 4-6 s,
tailles identiques au fichier local), puis applique le modèle de coût (§2 et §5 ter de
PLANIFICATEUR_COUT).

| Cas | Prédit | Mesuré | Écart |
|---|---:|---:|---:|
| Qwen2.5-32B Q4_K_M, 3090 | 36.0 | 38.3 | −6 % |
| Qwen2.5-32B Q6_K, paire | 27.9 | 27.9 | −0.1 % |
| Qwen3.6 Q4_K_M, 3090 | 123.9 | 136.5 | −9 % |
| Qwen3.6 Q6_K, paire | 115.5 | 107.3 | +8 % |
| Qwen3.6 Q8_0, paire | 112.6 | 96.8 | +16 % |
| Bonsai 27B PQ2_0, 3090 (CUDA) | 66.4 | 74.0 | −10 % |
| DeepSeek-V4-Flash, 3090 + RAM | 27.6 | 11.7 | **+136 %** |
| DeepSeek-V4-Flash, 3090 + 7900 XT + RAM | 33.4 | 11.8 | **+183 %** |

Bon sur les architectures calibrées ; DeepSeek-V4 (attention compressée + indexeur) est
surestimé ×2.4-2.8 → la commande l'affiche comme borne haute, « architecture non calibrée ».

## 6. Non fait / pas pour nous maintenant

- **Engram (DeepSeek)** : tables de ~200 Go consultées par n-grammes, déterministes donc
  préchargeables depuis RAM/NVMe — exactement l'idée des étages. Mais seulement dans vLLM
  et pour DeepSeek-V4.1 ; rien à tester ici (et 33 Go de disque libre).
- **WebNPU / WebNN** : non (Chrome seul, expérimental, NPU faits pour petits modèles).
