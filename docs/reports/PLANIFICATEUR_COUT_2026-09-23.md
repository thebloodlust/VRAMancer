# Planificateur par modèle de coût — étude de faisabilité (2026-09-23)

> Question posée : au lieu de tâtonner modèle par modèle, peut-on **calibrer la machine une
> fois** puis placer les poids de n'importe quel modèle sur les étages mémoire (GPU rapide,
> GPU lent, GPU distant, DRAM, NVMe…) sans essai ? Réponse : oui pour les GPU (erreur
> médiane 2 %) ; pour le CPU, le bon ordre de grandeur mais pas plus (voir §5 bis) —
> d'où une conception hybride : prédire pour classer, mesurer les 2-3 meilleures options.

## 1. Ce qui compte : les octets lus par token, pas la taille du fichier

| Modèle | Fichier | Lu par token |
|---|---:|---:|
| Qwen3.6-35B-A3B Q6_K (MoE 8/256) | 27.3 GiB | **3.27 GiB** |
| Qwen3.6-35B-A3B Q8_0 | 34.4 GiB | 3.48 GiB |
| Qwen2.5-Coder-32B Q6_K (dense) | 25.0 GiB | 25.03 GiB |

Calculé depuis l'en-tête GGUF : tenseurs non-experts + (experts utilisés / experts) × experts.

## 2. Étages GPU : 5 paramètres, 2 % d'erreur médiane hors calibration

Modèle : temps/token = Σ par GPU [octets lus sur ce GPU / bande passante effective + couches
sur ce GPU × coût fixe par couche] + coût de frontière. Calibré sur 9 mesures (balayages de
partage d'un MoE et d'un dense sur la paire), testé sur 7 mesures jamais vues :

| Mesure jamais vue | Mesuré | Prédit | Écart |
|---|---:|---:|---:|
| MoE Q8_0, paire prorata | 89.2 | 89.1 | −0.1 % |
| MoE Q8_0, paire 64/36 | 94.9 | 93.0 | −2.0 % |
| MoE Q8_0, paire 68/32 | 96.8 | 94.8 | −2.1 % |
| MoE Q4_K_M, 3090 seule | 136.5 | 124.0 | −9.2 % |
| dense Q4_K_M, 3090 seule | 36.1 | 36.0 | −0.3 % |
| dense Q4_K_M, 7900 XT seule | 32.4 | 28.0 | −13.7 % |
| MoE Q6_K, paire (répétition) | 91.6 | 90.9 | −0.7 % |

Et **sans aucun essai**, il retrouve les partages que `tune-split` avait trouvés en mesurant
pendant 10 minutes chacun : 83/17 (MoE Q6_K), 92/8 (dense Q6_K), 66/34 (MoE Q8_0).

Lecture physique des paramètres : les deux cartes ont une bande passante effective proche ;
c'est le **coût fixe par couche** qui les sépare (3090 122 µs, 7900 XT 248 µs — surcoût de
lancement des noyaux Vulkan/RADV). D'où le fait que la 7900 XT pénalise surtout les MoE,
qui enchaînent beaucoup de petits noyaux par couche.

## 3. Étage CPU/DRAM : borné par la mémoire OU par le calcul

Un premier modèle « bande passante seule » ratait de 56 à 66 %. Un CPU est souvent borné par
le calcul (le dense y est proportionnellement bien plus lent que le MoE). Avec
temps = max(octets / bande passante, opérations / débit de calcul), calibré sur 2 mesures :

| Mesure jamais vue | Mesuré | Prédit |
|---|---:|---:|
| MoE Q6_K, 3090 seule, 10 couches en CPU | 20.74 | 22.15 (+6.8 %) |
| MoE Q8_0, 3090 seule, 14 couches en CPU | 14.91 | 15.75 (+5.6 %) |
| dense Q6_K, 7900 XT seule, 18 couches en CPU | 2.77 | 2.74 (−1.2 %) |

(Une répétition de la 1re mesure avait donné 16.11 : le régime de débordement est bruité.)

Étage réseau (RPC llama.cpp, mesures netem) : ~1.1 à 2.3 aller-retour par token et par
frontière — un terme simple k × RTT suffit.

## 4. La règle de placement qui en découle : l'intensité de lecture

Ce qui doit aller sur l'étage rapide, ce n'est pas « la couche 0 avant la couche 40 », ce sont
les tenseurs les plus **lus par octet stocké** : attention, normes, expert partagé, sortie
(100 % lus à chaque token) d'abord ; experts routés d'un MoE (~3 % lus) ensuite ; tables
peu consultées (type Engram) en dernier.

Vérifié sur la 3090 seule, Qwen3.6-35B-A3B Q6_K (27 GB, ne tient pas en 24) :

| Placement | Prefill | Génération |
|---|---:|---:|
| 10 couches entières en RAM (ce que tout le monde fait) | 490 | 17.1 ± 1.8 |
| **tout sur GPU, seuls les experts de 10 couches en RAM** | **621** | **39.8 ± 0.5 (×2.3)** |
| … experts de 14 couches en RAM | 494 | 32.6 |
| … experts de 20 couches en RAM | 394 | 25.5 |

## 5. Là où la prédiction a échoué, et pourquoi

DeepSeek-V4-Flash (81 GiB, **92 % d'experts routés**, 6 actifs sur 256, 6.2 GiB hors
experts) : experts en RAM → 1.04 tok/s (3090) / 1.33 (paire), **pas mieux** que la
répartition par couches (1.08 / 1.38). Le modèle annonçait ~10.

Cause : le CPU de la VM est un **« QEMU Virtual CPU version 2.5+ »** — SSE4.2 uniquement,
**ni AVX, ni AVX2, ni FMA, ni AVX-512**, 8 cœurs. llama.cpp charge sa variante CPU la plus
lente (`libggml-cpu-sse42.so`) ; les quants i-quant (IQ2_XS) y sont particulièrement coûteux.
Le build contient une variante Zen 4 (AVX-512) inutilisée. **Tous les chiffres CPU de ce
dépôt mesurés dans cette VM sont bridés** et devront être refaits avec `cpu: host`.

C'est aussi la meilleure démonstration de l'intérêt d'un planificateur *calibré* plutôt que
*présupposé* : les coûts de l'étage CPU dépendent de la machine réelle, VM comprise.

## 5 bis. Après correction de la VM (`cpu: host`, 16 vCPU, EPYC 7402 confirmé)

`/proc/cpuinfo` affiche désormais « AMD EPYC 7402 24-Core Processor », AVX2 + FMA + F16C
(pas d'AVX-512 : Zen 2), et llama.cpp charge `libggml-cpu-haswell` au lieu de `sse42`.
Mêmes commandes qu'avant (`benchmarks/bench_cpu_tier_host.sh`) :

| Mesure | Avant | Après | Gain |
|---|---:|---:|---:|
| MoE Q6_K, CPU seul | 7.46 | 14.13 | ×1.9 |
| MoE Q6_K, 10 couches entières en CPU (3090) | 16.1–20.7 | 27.11 | ×1.3–1.7 |
| dense Q6_K, 8 couches en CPU (3090) | 3.72 | 14.15 | **×3.8** |
| MoE Q6_K, experts de 10 couches en RAM (3090) | 39.8 | 53.38 | ×1.3 |
| MoE Q6_K, experts de 20 couches en RAM (3090) | 25.5 | 40.64 | ×1.6 |
| **DeepSeek-V4-Flash 81 GiB, 3090 seule, tous experts en RAM** | 1.04 | **10.67** | **×10.3** |
| DeepSeek-V4-Flash, paire, experts de 36 couches en RAM | 1.33 | 8.35 | ×6.3 |

- **La prédiction du §5 (« ~10 tok/s ») était juste** : c'est l'environnement qui la faussait.
- Un modèle de **284 milliards de paramètres à 10.7 tok/s sur une seule RTX 3090**. Réserves :
  quant IQ2_XS (2.45 bits par poids, perte de qualité sensible) et **prefill à 24 tok/s** —
  un prompt de 4 000 tokens prend ~3 minutes. Utilisable en conversation courte, pas pour un
  agent de code qui renvoie tout un dépôt à chaque tour.
- Avec les experts en RAM, **la paire fait moins bien que la 3090 seule** (8.35 contre
  10.67) : quand l'essentiel du travail est côté CPU, ajouter la carte lente n'ajoute que son
  coût fixe par couche. Règle pour le planificateur : experts en RAM ⇒ tout le reste sur le
  GPU le plus rapide, seul.

**Précision du modèle de coût pour l'étage CPU réel** : recalibré sur 3 mesures, il prédit
les 3 autres à +80 %, −15 % et +28 %. Le bon ordre de grandeur, pas assez précis pour
décider seul. (Le point « CPU seul » est probablement biaisé : sur un build Vulkan,
llama.cpp délègue une partie des opérations au GPU même avec `-ngl 0`.)

→ **Conception retenue : hybride.** Le modèle classe toutes les options en millisecondes
(il est fiable à 2 % pour les étages GPU), puis on ne mesure que les 2 ou 3 meilleures,
en quelques passes courtes. On garde la rapidité de la prédiction et la sûreté de la mesure.

## 5 ter. La hiérarchie complète, mesurée (VM en `cpu: host`)

Principe validé : **amener le calcul aux données, jamais l'inverse.** Utiliser la VRAM d'un
2e GPU comme simple stockage pour le 1er (poids recopiés à chaque token par PCIe) avait été
réfuté (61-73 % du débit de référence) ; faire calculer chaque étage sur ce qu'il détient
fonctionne.

DeepSeek-V4-Flash 81 GiB (284B, 92 % d'experts routés, 1.74 GiB d'experts par couche) :

| Placement des experts | Prefill | Génération |
|---|---:|---:|
| tous en RAM, 3090 seule | 23.8 | 10.67 |
| 3090 remplie (9 couches), 34 en RAM | 28.8 | 11.66 |
| 3090 : 9 · 7900 XT : 5 · RAM : 29 | 33.5 | 11.42 |
| **3090 : 9 · 7900 XT : 10 · RAM : 24** | **38.9** | **12.07** |

La hiérarchie à trois étages est la meilleure, mais la 7900 XT n'y apporte que +3.5 % en
génération (+35 % en prefill) : chaque couche qui lui est confiée impose un aller-retour
3090 → hôte → 7900 XT → 3090 (pas de lien direct entre cartes), qui coûte presque autant que
le calcul CPU qu'il évite. Piège rencontré : `-sm none` retire le 2e GPU du planificateur de
llama.cpp (« buffer that cannot run the operation ») ; il faut le garder déclaré
(`-ts 99/1`) et y envoyer les experts par `-ot`.

| Étage | Débit effectif mesuré |
|---|---|
| L0 VRAM RTX 3090 | ~925 GiB/s + 122 µs/couche |
| L1 VRAM RX 7900 XT | ~928 GiB/s + 248 µs/couche + aller-retour entre cartes |
| L2 RAM + CPU EPYC 7402 (AVX2, 16 vCPU) | ~58 GiB/s effectifs |
| L3 GPU distant (RPC llama.cpp) | −4 % en 1 GbE, −25 % en Wi-Fi, −72 % derrière un VPN à 20 ms |
| L4 disque de la VM | 2.6 Go/s séquentiel, 1.8 Go/s en lectures de 4 Mo aléatoires |

Au-delà de la RAM (modèle > 172 Go), des experts lus depuis le disque donneraient un ordre
de ~1 tok/s pour DeepSeek (1.75 GiB d'experts actifs par token ; l'usage des experts étant
quasi uniforme, le cache n'aide pas). Lent mais fonctionnel.

## 6. Proposition

1. `vramancer calibrate` (une fois par machine, quelques minutes) : par GPU, bande passante
   effective + coût par couche ; CPU, bande passante + débit de calcul ; liens RPC, RTT.
2. `vramancer plan <modèle.gguf | dépôt HF>` : lit l'en-tête GGUF (octets lus par tenseur),
   classe les tenseurs par intensité de lecture, remplit les étages du plus rapide au plus
   lent, prédit tok/s pour chaque option (et chaque quant) et produit les arguments
   llama-server (`--tensor-split`, `-ot` / `--n-cpu-moe`, `-ngl`). L'en-tête d'un GGUF
   distant se lit par une requête HTTP partielle : **on peut prédire avant de télécharger.**
3. Les 2-3 meilleures options prédites sont vérifiées par des passes courtes (le
   `tune-split` actuel devient cette étape de vérification, ciblée au lieu d'exhaustive).

## 7. Implémenté : `vramancer plan` (core/planner.py)

Choix fait après les mesures du §5 : **pas de calibration machine générique** (46 % d'erreur
avec de petits modèles), mais une calibration *par modèle* en 2–3 passes courtes, qui
prédit les autres placements à 1–3 %.

- **Modèle qui tient en VRAM cumulée** : une passe par GPU, ajustement linéaire du coût par
  couche, remplissage du GPU le moins cher d'abord. Qwen3.6-35B Q6_K : prédit 106.8, vérifié
  **107.3 tok/s** (+0.4 %), `-ngl 99 -ts 0.83/0.17`.
- **MoE plus gros que la VRAM** : chaud sur le GPU principal, experts par étages (principal,
  puis 2e GPU, puis RAM) via `-ot …=Vulkan1,…=CPU`. Le planificateur mesure si le 2e GPU
  bat la RAM *pour ce modèle* et recule d'un cran sur OOM. DeepSeek-V4-Flash IQ2_XS (81 GiB) :
  prédit 12.22, vérifié **11.82 tok/s** (−3.3 %) ; 7900 XT jugée utile (0.44 ms gagnés par
  couche d'experts).
- **Réutilisé par `serve`** (cache `~/.cache/vramancer/plans.json`, clé nom + taille) avec
  repli en cascade : plan → répartition par défaut → `--cpu-moe` pour un MoE. Délai de
  démarrage proportionnel à la taille (8 s/GiB, min. 120 s).
- **De bout en bout** : DeepSeek via `vramancer serve` → `/v1/chat/completions`, 120 tokens
  en 12.4 s prompt compris (≈ 9.7 tok/s côté client), réponse correcte en français.

Pièges corrigés en route : llama-bench sépare les règles `-ot` par `;`, llama-server par `,` ;
l'ancien repli `-ngl -1` sur un modèle de 81 GiB finissait en `ErrorOutOfDeviceMemory` ;
le stderr de llama-server était un PIPE jamais lu (blocage possible après 64 Ko de logs).
