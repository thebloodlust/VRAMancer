# Mesure cross-nœud — le transfert n'est PAS le tueur (Thunderbolt vindiqué)

> `benchmarks/bench_cross_node_transfer.py` — 1 machine, plancher loopback + extrapolation
> par débit. Profil Qwen2.5-14B (hidden 5120, 48 couches). Date : 2026-06-15.

## Mesuré
| | activation | loopback 1-way | 1 GbE | 10 GbE | USB4/Thunderbolt |
|---|---|---|---|---|---|
| **décode** (1 token) | 10 KB | 0.008 ms | 0.28 ms (**1%** du calcul) | 0.06 ms | 0.05 ms |
| **prefill** (216 tok) | 2.2 MB | 1.2 ms | **17.9 ms** | 1.8 ms | 0.9 ms |

(calcul/token ≈ 27 ms mesuré ; décode = 1 crossing/token en pipeline contigu)

## Ce que ça corrige (7e fois)
Mon a priori « cross-nœud = mort » était **trop large**. La mesure affine :
- **Par-couche interleavé** (design naïf de DeepSeek) : ~N crossings/token → mort. ✅ confirmé.
- **Pipeline contigu** (1 crossing/token) : le **transfert est minuscule** vs le calcul
  (décode 10 KB, latency-bound, <1% même en 1 GbE). Le transfert n'est PAS le goulot.
  Le vrai frein = **décode autorégressif SÉRIEL** : nœud B attend A → **0 speedup
  single-requête**, juste split mémoire + latence en plus. Gain seulement en
  **multi-requêtes** (micro-batches pipeline) ou **data-parallel** (requête/nœud).
- **Prefill** : 2.2 MB → lourd en **1 GbE (18 ms)**, négligeable en 10 GbE/Thunderbolt.

## Thunderbolt/USB4 : vindiqué par la mesure
~20 Gbps (2.5 GB/s effectif) rapproche le transfert de la **vitesse locale**
(CPU-staged 11–25 GB/s). C'est **le seul transport** qui rend le cross-nœud non-stupide :
prefill 2 MB en ~1 ms, décode négligeable. 1 GbE est **100× plus lent** → prefill lourd.
→ Le tier réseau L4 gelé < 8 GB/s reste juste ; **Thunderbolt (16–20 Gbps) passe la barre.**

## Verdict honnête
Cross-nœud **viable pour le DÉBIT** (multi-requêtes : pipeline ou data-parallel) sur un
**lien rapide (Thunderbolt/10 GbE)** — **PAS pour la latence single-requête** (sérialisation).
Mode recommandé : **data-parallel** (requête entière/nœud, 0 crossing) quand le modèle
tient sur 1 nœud ; **pipeline** seulement si le modèle dépasse 1 nœud ET lien rapide ET
charge multi-requêtes. À mesurer pour de vrai quand la 2e machine (Windows) + le lien
Thunderbolt seront prêts.
