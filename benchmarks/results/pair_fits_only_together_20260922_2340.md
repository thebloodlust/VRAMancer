# Modèle qui ne tient QUE sur la paire — 2026-09-22T23:40:15+02:00

Modèle : /home/jeremie/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K.gguf
Devices : NVIDIA=Vulkan0 AMD=Vulkan1
```
Available devices:
  Vulkan0: NVIDIA GeForce RTX 3090 (24576 MiB, 24098 MiB free)
  Vulkan1: AMD Radeon RX 7900 XT (RADV NAVI31) (20464 MiB, 20415 MiB free)
```

## CPU seul (référence)

```
load_backend: loaded RPC backend from /home/jeremie/.cache/vramancer/bin/b11112-linux-vulkan/llama-b11112/libggml-rpc.so
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 (NVIDIA) | uma: 0 | fp16: 1 | bf16: 1 | fp4: 0 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
ggml_vulkan: 1 = AMD Radeon RX 7900 XT (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
load_backend: loaded Vulkan backend from /home/jeremie/.cache/vramancer/bin/b11112-linux-vulkan/llama-b11112/libggml-vulkan.so
load_backend: loaded CPU backend from /home/jeremie/.cache/vramancer/bin/b11112-linux-vulkan/llama-b11112/libggml-cpu-sse42.so
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q6_K         |  27.29 GiB |    34.66 B | Vulkan     |   0 |   1 |           pp512 |        195.37 ± 1.29 |
| qwen35moe 35B.A3B Q6_K         |  27.29 GiB |    34.66 B | Vulkan     |   0 |   1 |           tg128 |          7.46 ± 0.04 |

build: 4098fdc92 (11112)
```

## 3090 seule — max de couches qui tient

```
load_backend: loaded RPC backend from /home/jeremie/.cache/vramancer/bin/b11112-linux-vulkan/llama-b11112/libggml-rpc.so
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 (NVIDIA) | uma: 0 | fp16: 1 | bf16: 1 | fp4: 0 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
ggml_vulkan: 1 = AMD Radeon RX 7900 XT (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
load_backend: loaded Vulkan backend from /home/jeremie/.cache/vramancer/bin/b11112-linux-vulkan/llama-b11112/libggml-vulkan.so
load_backend: loaded CPU backend from /home/jeremie/.cache/vramancer/bin/b11112-linux-vulkan/llama-b11112/libggml-cpu-sse42.so
| model                          |       size |     params | backend    | ngl |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | ------------ | --------------: | -------------------: |
| qwen35moe 35B.A3B Q6_K         |  27.29 GiB |    34.66 B | Vulkan     |  30 |   1 | Vulkan0      |           pp512 |        506.97 ± 0.93 |
| qwen35moe 35B.A3B Q6_K         |  27.29 GiB |    34.66 B | Vulkan     |  30 |   1 | Vulkan0      |           tg128 |         20.74 ± 2.51 |

build: 4098fdc92 (11112)
```

## 7900 XT seule — max de couches qui tient

```
load_backend: loaded RPC backend from /home/jeremie/.cache/vramancer/bin/b11112-linux-vulkan/llama-b11112/libggml-rpc.so
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 (NVIDIA) | uma: 0 | fp16: 1 | bf16: 1 | fp4: 0 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
ggml_vulkan: 1 = AMD Radeon RX 7900 XT (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
load_backend: loaded Vulkan backend from /home/jeremie/.cache/vramancer/bin/b11112-linux-vulkan/llama-b11112/libggml-vulkan.so
load_backend: loaded CPU backend from /home/jeremie/.cache/vramancer/bin/b11112-linux-vulkan/llama-b11112/libggml-cpu-sse42.so
| model                          |       size |     params | backend    | ngl |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | ------------ | --------------: | -------------------: |
| qwen35moe 35B.A3B Q6_K         |  27.29 GiB |    34.66 B | Vulkan     |  26 |   1 | Vulkan1      |           pp512 |        211.56 ± 3.89 |
| qwen35moe 35B.A3B Q6_K         |  27.29 GiB |    34.66 B | Vulkan     |  26 |   1 | Vulkan1      |           tg128 |         18.03 ± 0.10 |

build: 4098fdc92 (11112)
```

## PAIRE — modèle ENTIER sur les deux cartes

```
load_backend: loaded RPC backend from /home/jeremie/.cache/vramancer/bin/b11112-linux-vulkan/llama-b11112/libggml-rpc.so
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 (NVIDIA) | uma: 0 | fp16: 1 | bf16: 1 | fp4: 0 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
ggml_vulkan: 1 = AMD Radeon RX 7900 XT (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
load_backend: loaded Vulkan backend from /home/jeremie/.cache/vramancer/bin/b11112-linux-vulkan/llama-b11112/libggml-vulkan.so
load_backend: loaded CPU backend from /home/jeremie/.cache/vramancer/bin/b11112-linux-vulkan/llama-b11112/libggml-cpu-sse42.so
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q6_K         |  27.29 GiB |    34.66 B | Vulkan     |  99 |   1 |           pp512 |       2141.22 ± 7.28 |
| qwen35moe 35B.A3B Q6_K         |  27.29 GiB |    34.66 B | Vulkan     |  99 |   1 |           tg128 |         91.68 ± 3.64 |

build: 4098fdc92 (11112)
```

## PAIRE — split 24:20

```
load_backend: loaded RPC backend from /home/jeremie/.cache/vramancer/bin/b11112-linux-vulkan/llama-b11112/libggml-rpc.so
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 (NVIDIA) | uma: 0 | fp16: 1 | bf16: 1 | fp4: 0 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
ggml_vulkan: 1 = AMD Radeon RX 7900 XT (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
load_backend: loaded Vulkan backend from /home/jeremie/.cache/vramancer/bin/b11112-linux-vulkan/llama-b11112/libggml-vulkan.so
load_backend: loaded CPU backend from /home/jeremie/.cache/vramancer/bin/b11112-linux-vulkan/llama-b11112/libggml-cpu-sse42.so
| model                          |       size |     params | backend    | ngl |  fa | ts           |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | ------------ | --------------: | -------------------: |
| qwen35moe 35B.A3B Q6_K         |  27.29 GiB |    34.66 B | Vulkan     |  99 |   1 | 24.00/20.00  |           pp512 |       2138.57 ± 4.41 |
| qwen35moe 35B.A3B Q6_K         |  27.29 GiB |    34.66 B | Vulkan     |  99 |   1 | 24.00/20.00  |           tg128 |         92.74 ± 2.51 |

build: 4098fdc92 (11112)
```
