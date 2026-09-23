# Planificateur par modèle de coût — étude de faisabilité (2026-09-23)

> Question posée : au lieu de tâtonner modèle par modèle, peut-on **calibrer la machine une
> fois** puis placer les poids de n'importe quel modèle sur les étages mémoire (GPU rapide,
> GPU lent, GPU distant, DRAM, NVMe…) sans essai ? Réponse : oui pour les GPU (erreur
> médiane 2 %), oui pour le CPU une fois modélisé correctement (6 %), avec une limite
> découverte en route — le CPU de la VM est bridé.

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

## 6. Proposition

1. `vramancer calibrate` (une fois par machine, quelques minutes) : par GPU, bande passante
   effective + coût par couche ; CPU, bande passante + débit de calcul ; liens RPC, RTT.
2. `vramancer plan <modèle.gguf | dépôt HF>` : lit l'en-tête GGUF (octets lus par tenseur),
   classe les tenseurs par intensité de lecture, remplit les étages du plus rapide au plus
   lent, prédit tok/s pour chaque option (et chaque quant) et produit les arguments
   llama-server (`--tensor-split`, `-ot` / `--n-cpu-moe`, `-ngl`). L'en-tête d'un GGUF
   distant se lit par une requête HTTP partielle : **on peut prédire avant de télécharger.**
3. `tune-split` devient la vérification optionnelle du plan, pas le point de départ.
