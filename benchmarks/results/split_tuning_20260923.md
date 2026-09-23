# Split mesuré vs prorata — RTX 3090 + RX 7900 XT (2026-09-23)

## Balayage llama-bench, Qwen3.6-35B-A3B Q6_K (27.3 GB), pp512/tg128
```
défaut (llama.cpp)                           pp512 =       2155.16 ± 2.55             tg128 =         91.40 ± 3.12 
24:20  (∝ VRAM, AMD 45%)                    pp512 =       2155.38 ± 9.26             tg128 =         91.40 ± 2.82 
70:30  (AMD 30%)                              pp512 =      2141.85 ± 11.30             tg128 =        100.01 ± 0.45 
60:40  (AMD 40%)                              pp512 =       2134.10 ± 3.67             tg128 =         92.30 ± 1.80 
50:50                                         pp512 =       2151.45 ± 0.93             tg128 =         85.12 ± 8.59 
40:60  (AMD 60%)                              pp512 =      2174.95 ± 13.30             tg128 =         78.45 ± 8.56 
défaut, GPU principal = AMD                  pp512 =       2150.31 ± 5.40             tg128 =         88.33 ± 1.69 
75:25 (AMD 25%)                        pp512 =      2145.06 ± 10.21              tg128 =         98.28 ± 3.64 
80:20 (AMD 20%)                        pp512 =      2148.53 ± 18.72              tg128 =        102.75 ± 2.03 
84:16 (AMD 16%)                        pp512 =      2156.32 ± 16.37              tg128 =        106.35 ± 4.99 
88:12 (AMD 12%)             ÉCHEC  ÉCHEC  ÉCHEC
70:30 (contrôle)                      pp512 =      2152.22 ± 16.11              tg128 =         98.12 ± 1.21 
```

## vramancer tune-split (profondeur 16K)
```
                             /home/jeremie/.cache/vramancer/bin/b11112-linux-vul
2 GPU : NVIDIA GeForce RTX 3090, AMD Radeon RX 7900 XT (RADV NAVI31)
Modèle 27.3 GiB, contexte 16384
Étape 1 — quelle carte est la plus rapide ?
  55% / 45%                81.8 tok/s
  83% / 17%                99.4 tok/s
  31% / 69%                73.7 tok/s
Étape 2 — affinage autour de NVIDIA GeForce RTX 3090
  79% / 21%                97.2 tok/s
  75% / 25%                95.3 tok/s
  71% / 29%                91.9 tok/s
Retenu : 83% / 17% → 99.4 tok/s (+21 % vs prorata VRAM)

[exited with code 0]
```

## A/B à travers vramancer serve (8 requêtes ~400 tokens, médiane)
- prorata VRAM (55/45) : 74.5 tok/s (72.2–76.9)
- split mesuré (83/17) : **95.7 tok/s** (88.4–96.5) → **+28 %**

## Modèle dense Qwen2.5-Coder-32B Q6_K (25 GB)
```
print_info: file size   = 25.03 GiB (6.56 BPW) 
print_info: n_layer               = 64
3090 seule (ngl 56, débordement)                    pp512 =        570.33 ± 0.64              tg128 =          3.72 ± 0.34 
7900 XT seule (ngl 46, débordement)                 pp512 =        178.68 ± 0.06              tg128 =          2.77 ± 0.13 
PAIRE défaut                                        pp512 =        700.41 ± 0.13              tg128 =         25.78 ± 0.00 
PAIRE 84:16 (rapide d'abord)                         pp512 =        937.08 ± 1.09              tg128 =         27.45 ± 0.07 
PAIRE 88:12                                          pp512 =        988.44 ± 0.37              tg128 =         27.75 ± 0.01 
PAIRE 92:8                                           pp512 =       1024.27 ± 0.34              tg128 =         27.88 ± 0.04 
```
