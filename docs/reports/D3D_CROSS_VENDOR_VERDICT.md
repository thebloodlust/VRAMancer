# D3.d — Verdict cross-vendor : llama.cpp Vulkan sur la paire RTX 3090 + RX 7900 XT

> Mesuré le 2026-09-22, VM QEMU, les deux cartes en passthrough VFIO, build llama.cpp
> **b11112** Vulkan. Chiffres bruts : `benchmarks/results/vulkan_pair_20260922_2313.md`.
> Le garde-fou du plan D disait : *« si llama.cpp Vulkan fait déjà bien le travail sur la
> paire mixte, le cross-vendor maison est un non-sujet »*. Voici la réponse.

## Ce qui a été mesuré

Les deux cartes sont vues ensemble sans configuration particulière :

```
Vulkan0: NVIDIA GeForce RTX 3090 (24576 MiB)   — matrix cores NV_coopmat2, bf16
Vulkan1: AMD Radeon RX 7900 XT (RADV NAVI31)   — matrix cores KHR_coopmat, pas de bf16
```

### Modèle qui TIENT sur une seule carte (Qwen3.6-35B-A3B Q4_K_M, 20.60 GiB)

| Configuration | Prefill pp512 | Génération tg128 |
|---|---:|---:|
| 7900 XT seule (`-ngl 38`, le max qui tient) | 805.49 ± 6.33 | 29.51 ± 4.52 |
| **3090 seule** (`-ngl 99`) | **3679.66 ± 21.82** | **135.90 ± 0.39** |
| Paire mixte, split par défaut | 2796.08 ± 57.03 | 92.02 ± 3.53 |
| Paire mixte, split 24:20 (proportionnel VRAM) | 2784.42 ± 50.22 | 91.53 ± 4.14 |
| Paire mixte, split par lignes (`-sm row`) | ❌ `failed to load model` | ❌ |

**La paire fonctionne, et elle est plus lente que la meilleure carte seule : −32 % en
génération (92.0 vs 135.9 tok/s), −24 % en prefill.** C'est attendu et sain : le modèle
tient entièrement dans les 24 GB de la 3090, donc ajouter la carte AMD n'apporte aucune
VRAM utile et ne fait qu'insérer des transferts PCIe entre deux couches.

À noter aussi : le **split par lignes échoue** sur cette paire mixte (le modèle ne charge
même pas), alors qu'il fonctionne sur des cartes homogènes. À ne pas proposer en
cross-vendor.

Remarque de mesure : la 7900 XT donne 29.51 tok/s ici contre 37.78 mesurés plus tôt dans
la journée, même commande. L'écart vient du fait que le pilote NVIDIA est maintenant
chargé et que les deux backends Vulkan coexistent dans le même processus. C'est un coût
réel du mode mixte, pas une erreur de mesure.

## Le seul cas où la paire peut payer : un modèle trop gros pour une carte

DeepSeek-V4-Flash IQ2_XS — **81.03 GiB, 284 B paramètres, 43 couches** (~1.88 GiB/couche).
Ni la 3090 (24 GB) ni la 7900 XT (20 GB) ne peuvent en prendre plus d'une poignée de
couches ; ensemble elles offrent 44 GB.

| Configuration | Couches sur GPU | Prefill pp128 | Génération tg32 |
|---|---:|---:|---:|
| CPU seul (`-ngl 0`) — référence | 0 / 43 | 24.26 | 0.97 |
| 3090 seule (`-ngl 11`, ~24 GB) | 11 / 43 | 27.37 | 1.08 |
| **Paire 3090+7900XT** (`-ngl 16`, ~44 GB) | **16 / 43** | **32.88** | **1.38** |
| Paire, `-ngl 21` | — | ❌ `Device memory allocation of size 33554432 failed` | ❌ |
| Paire, `-ngl 21 -ts 24/20` | — | ❌ même échec | ❌ |

Balayage du maximum que la paire encaisse (`-n 8`) : ngl 14 → 1.39, **ngl 16 → 1.45**,
ngl 18 → 1.40, ngl 21 → échec d'allocation. La paire plafonne donc vers **16-18 couches
sur 43**, pas les 21 que laisserait espérer le simple cumul de VRAM : les tampons de
calcul et le KV cache prennent leur part sur chaque carte.

**Ici la paire gagne vraiment** : +28 % en génération contre la 3090 seule (1.38 vs 1.08)
et +42 % contre le CPU seul ; +20 % en prefill contre la 3090 seule. Le gain est réel et
reproductible.

**Et pourtant ça ne sert à rien** : 1.38 tok/s reste inutilisable. Passer de 1.08 à
1.38 tok/s ne transforme aucun usage — ni chat, ni agent de code, ni batch. Avec 16
couches sur 43 déportées, le CPU continue de dominer le temps de calcul, et la VRAM
supplémentaire ne change pas ce rapport de force.

## Verdict

**Le cross-vendor maison est un non-sujet.** Trois raisons, toutes mesurées :

1. **llama.cpp Vulkan le fait déjà**, sans une ligne de code de notre part : les deux
   cartes sont détectées et utilisées ensemble d'office.
2. **Quand le modèle tient sur une carte, mélanger coûte 32 %** de débit. Le bon réflexe
   est d'utiliser la carte la plus rapide seule, pas d'additionner du matériel.
3. **Quand le modèle ne tient pas, le gain est réel (+28 %) mais sans conséquence** :
   on passe de 1.08 à 1.38 tok/s, c'est-à-dire d'inutilisable à inutilisable.

Autrement dit, il n'existe pas, sur cette paire, de régime où un pont cross-vendor maison
apporterait quoi que ce soit que llama.cpp ne fasse pas déjà — et le seul régime où la
VRAM cumulée compte donne des débits qui ne servent à personne. C'est exactement le
scénario A1 : **ne pas re-payer cette leçon.**

### Conséquences concrètes

- `experimental/cross_vendor_bridge.py` : ne pas investir. Ses débits annoncés
  (25-50 GB/s, ~20 GB/s) restent **non mesurés** et le resteront ; ils sont marqués comme
  tels dans le fichier.
- Conseil matériel honnête pour le README : pour faire tourner un modèle plus gros, une
  **deuxième carte du même vendeur** (ou une carte unique avec plus de VRAM) bat une
  paire dépareillée. Le cross-vendor ne devient intéressant que si un jour un modèle
  tient *entièrement* dans la VRAM cumulée de deux cartes de marques différentes — cas
  que cette session n'a pas pu produire.
- Le split par lignes (`-sm row`) est inutilisable en cross-vendor : le modèle ne charge
  même pas.

### Ce que ce verdict ne couvre pas

- Une paire cross-vendor où le modèle tient **entièrement** dans la VRAM cumulée
  (ex. un 30-35 GB sur 24+20 GB). Non testé : le seul modèle de cette taille disponible
  ici tient déjà sur la 3090 seule.
- ROCm / torch-hip : jamais installé. Tout ce rapport passe par Vulkan.
- Une machine réelle (hors VM) : le passthrough VFIO ajoute un surcoût PCIe connu.
