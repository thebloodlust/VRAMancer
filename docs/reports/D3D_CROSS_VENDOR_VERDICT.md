# D3.d — Verdict cross-vendor : llama.cpp Vulkan sur la paire RTX 3090 + RX 7900 XT

> Mesuré le 2026-09-22, VM QEMU, les deux cartes en passthrough VFIO, build llama.cpp
> **b11112** Vulkan. Chiffres bruts : `benchmarks/results/vulkan_pair_20260922_2313.md`.
> Le garde-fou du plan D disait : *« si llama.cpp Vulkan fait déjà bien le travail sur la
> paire mixte, le cross-vendor maison est un non-sujet »*. Voici la réponse.
>
> **⚠️ Ce rapport a été RÉVISÉ le même soir.** Une première version concluait que la
> paire mixte n'apportait rien. Cette conclusion était **fausse par omission** : les deux
> premiers modèles testés tombaient de part et d'autre du seul régime qui compte. Un
> troisième test (§ « Le régime qui change tout ») montre un facteur **4.4×** en faveur
> de la paire. La conclusion finale est en bas de page et remplace la précédente.

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

## Le régime qui change tout : le modèle ne tient QUE dans la VRAM cumulée

Les deux modèles précédents encadraient la vraie question sans la poser : l'un tenait
déjà sur une carte (la paire ne pouvait qu'ajouter du transfert), l'autre ne tenait
nulle part (la paire ne pouvait pas sauver un calcul dominé par le CPU). Entre les deux
il y a la fenêtre qui compte : **27.29 GiB sur des cartes de 24 et 20 GB**. Trop gros
pour chacune, confortable pour les deux ensemble.

Qwen3.6-35B-A3B **Q6_K — 27.29 GiB, 40 couches** (même modèle que plus haut, quantifié
moins agressivement) :

| Configuration | Couches sur GPU | Prefill pp512 | Génération tg128 |
|---|---:|---:|---:|
| CPU seul | 0 / 40 | 195.37 ± 1.29 | 7.46 ± 0.04 |
| 3090 seule (`-ngl 30`, le reste déborde) | 30 / 40 | 506.97 ± 0.93 | 20.74 ± 2.51 |
| 7900 XT seule (`-ngl 26`, le reste déborde) | 26 / 40 | 211.56 ± 3.89 | 18.03 ± 0.10 |
| **PAIRE, modèle ENTIER en VRAM** | **40 / 40** | **2141.22 ± 7.28** | **91.68 ± 3.64** |
| PAIRE, split 24:20 | 40 / 40 | 2138.57 ± 4.41 | **92.74 ± 2.51** |

**La paire va 4.4× plus vite que la meilleure carte seule** (91.7 contre 20.7 tok/s) et
12× plus vite que le CPU. En prefill, 4.2× la 3090 seule.

Et le chiffre brut sous-estime l'effet réel : 20.7 tok/s, c'est trop lent pour un agent
de code ; 92 tok/s, c'est confortable. La paire ne rend pas le modèle « plus rapide »,
elle le rend **utilisable**. C'est précisément la promesse de VRAMancer — faire tourner
sur des GPU dépareillés un modèle qu'aucun d'eux ne peut héberger seul — et ça marche
aussi quand les vendeurs diffèrent.

## Verdict (révisé — remplace la conclusion initiale)

**Mélanger NVIDIA et AMD vaut le coup, mais seulement dans une fenêtre précise**, et
cette fenêtre est celle de tout le projet :

| Le modèle… | Verdict | Mesure |
|---|---|---|
| tient sur la carte la plus rapide | ❌ **ne pas mélanger** — la paire coûte 32 % | 136 → 92 tok/s |
| **ne tient QUE dans la VRAM cumulée** | ✅ **mélanger, largement** — 4.4× | 20.7 → 91.7 tok/s |
| dépasse même la VRAM cumulée | ⚠️ gain réel (+28 %) mais sans usage | 1.08 → 1.38 tok/s |

**Le pont cross-vendor MAISON reste un non-sujet** — mais pour une raison différente de
celle annoncée d'abord :

**llama.cpp Vulkan livre déjà le 4.4×**, sans une ligne de code de notre part, sans
ROCm, sans configuration : les deux cartes sont détectées et remplies d'office. Écrire
notre propre transport cross-vendor ne rattraperait rien — il faudrait battre ce que
llama.cpp fait déjà très bien. C'est le scénario A1 : **ne pas re-payer cette leçon.**

En revanche, **l'orchestration, elle, a un vrai travail à faire** : encore faut-il
*savoir* qu'il existe une deuxième carte, et l'utiliser. VRAMancer ne le savait pas —
son calcul de répartition passait par `torch.cuda`, aveugle aux cartes AMD, donc il
aurait servi ce modèle sur la 3090 seule à 20.7 tok/s au lieu de 91.7. Corrigé dans la
foulée (`backend_devices()` interroge le binaire, qui voit tout le monde). **C'est ça,
la valeur du projet : pas réécrire le transport, mais ne pas laisser 4.4× sur la table.**

### Vérifié aussi à travers VRAMancer lui-même

Pas seulement `llama-bench` : `vramancer serve --model <Q6_K> --gpus 1` (on passe
volontairement 1, puisque torch ne voit qu'une carte) charge désormais le modèle sur les
deux GPU — **NVIDIA 15 990 MiB + AMD 12 360 MiB** — et répond entre **68.6 et 75.8 tok/s**
en régime chaud (5 requêtes de 128 tokens ; la première, à froid, tombe à 21.9 le temps
du préchauffage). L'écart avec les 91.7 de `llama-bench` est le coût HTTP + prompt sur
des requêtes courtes, le même ratio que celui déjà observé en mono-carte.

Sans le correctif, ce même serveur aurait tourné sur la 3090 seule — et aurait même
échoué à charger, puisque 27 GB ne tiennent pas dans 24.

### Conséquences concrètes

- `experimental/cross_vendor_bridge.py` : ne pas investir. Ses débits annoncés
  (25-50 GB/s, ~20 GB/s) restent **non mesurés** et le resteront ; ils sont marqués comme
  tels dans le fichier.
- Conseil matériel honnête pour le README : **une carte d'un autre vendeur est un ajout
  parfaitement valable** si elle fait passer ton modèle sous la barre de la VRAM cumulée.
  Choisis le quant en conséquence : ici, passer de Q4_K_M (20.6 GB, tient sur la 3090) à
  Q6_K (27.3 GB, ne tient que sur la paire) échange un peu de vitesse brute contre
  beaucoup de qualité, à 92 tok/s — un arbitrage qui n'existe pas sans la 2e carte.
- Le split par lignes (`-sm row`) est inutilisable en cross-vendor : le modèle ne charge
  même pas.

### Ce que ce verdict ne couvre pas

- ROCm / torch-hip : jamais installé. Tout ce rapport passe par Vulkan.
- Le chemin HuggingFace/accelerate en cross-vendor : non testé (il demanderait torch-hip).
- La qualité des sorties Q4_K_M vs Q6_K : non évaluée ici, seulement les débits.
- Une machine réelle (hors VM) : le passthrough VFIO ajoute un surcoût PCIe connu.
