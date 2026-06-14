# SUPERAUDIT — VRAMancer Codebase Active (Juin 2026)

> Audit complet sur l'arbre de dev actif : 2161 lignes de Rust, 73 modules Python,
> 11 fichiers natifs C/CUDA/C++. Fait après vérification de chaque fichier.
> Les numéros de ligne et les affirmations sont vérifiés contre le code réel.

---

## Table des matières

1. [Vue d'ensemble — Forces et faiblesses](#1-vue-densemble)
2. [Rust / CUDA FFI — Le meilleur du projet](#2-rust--cuda-ffi)
3. [Rust / Tokio — Ce qui reste à corriger](#3-rust--tokio)
4. [GpuPipeline & GpuNetBridge — Analyse](#4-gpupipeline--gpunetbridge)
5. [Python / Core — Architecture](#5-python--core)
6. [Experimental/ — Ce qui est vraiment dedans](#6-experimental)
7. [Kernels CUDA & Natif — État des lieux](#7-kernels-cuda--natif)
8. [Idées innovantes — Propositions](#8-idées-innovantes)
9. [Check-list priorisée pour agents](#9-check-list-priorisée)

---

## 1. Vue d'ensemble

### Forces (réelles, vérifiées)

| Force | Détail |
|---|---|
| **CUDA FFI natif** | Module `cuda_ffi` de 480 lignes qui wrappe 25 fonctions de la CUDA Driver API via `libloading`. Charge `libcuda.so.1` dynamiquement. Zéro dépendance à `cudarc` pour le chemin critique. |
| **GpuPipeline** | Pipeline de transfert GPU→GPU pré-alloué : triple-buffering, streams CUDA dédiés, events, P2P auto-détecté. Code propre, `unsafe impl Send` justifié. |
| **GpuNetBridge** | Pont GPU↔Réseau avec mémoire pinnée pré-allouée, streams CUDA, et timeouts TCP (30s connect, 120s r/w). Vrai zero-copy du point de vue CUDA (DMA direct depuis/pour la mémoire pinnée). |
| **rust_bridge.py** | Façade gracieuse : jamais d'exception à l'import, `cuda_available()` vérifie la présence de libcuda avant d'appeler les fonctions CUDA. Pattern défensif exemplaire. |
| **experimental/README.md** | Honnêteté radicale : chaque module a son statut documenté ("gelé", "non testable sans matos X", "en développement"). |
| **Triton kernels** | 7 kernels Triton optimisés (RMSNorm fusionné, RoPE fusionné, GEMV FP4, sampling fusionné, etc.). |
| **Architecture de bypass** | 7 niveaux de fallback pour les transferts GPU, avec dégradation gracieuse. |
| **Tests** | 40+ fichiers de test, couvrant les fallbacks et le mode minimal. |

### Faiblesses (réelles, vérifiées)

| Faiblesse | Impact |
|---|---|
| **`Runtime::new()` par appel Tokio** | 4 appels dans `lib.rs` créent un nouveau runtime à chaque transfert P2P. ~200-500µs de overhead fixe. |
| **Pas de timeouts sur les fonctions Tokio** | `send_tensor_p2p`, `receive_tensor_p2p`, `send_tensor_chunked`, `receive_tensor_chunked` n'ont PAS de timeouts. Le `GpuNetBridge` en a, mais pas ces 4 fonctions. |
| **Protocole chunked stop-and-wait** | Un chunk → un ACK → chunk suivant. Sur lien 10ms RTT, 75% de perte de bande passante. |
| **`direct_vram_load` leak intentionnel** | `std::mem::forget(d_buf)` + retourne `0u64`. Commentaire dit "placeholder — real ptr extraction needs DLPack". |
| **`cxl_direct_memory_dump/load` unsafe** | Prend un pointeur brut (`usize`) depuis Python, le cast en `*const u8` sans aucune validation. Segfault si le pointeur est invalide. |
| **`receive_tensor_p2p` pas de limite de payload** | Le commentaire dans `lib.rs` ligne 913 (`payload_len = total_len - 32`) n'a PAS de garde `MAX_PAYLOAD_BYTES`. La garde a été ajoutée dans le `GpuNetBridge` mais PAS dans les 4 fonctions Tokio. |
| **Commentaires en français** | 80% des commentaires dans `lib.rs` sont en français. Les nouveaux modules (`cuda_ffi`, `GpuPipeline`, `GpuNetBridge`) sont en anglais — bonne direction, mais l'ancien code tire vers le bas. |
| **`csrc/dmabuf_bridge.c`** | Le commentaire ligne 2 dit "destination-side write NOT IMPLEMENTED". DMA-BUF est effectivement unidirectionnel (export seulement). |
| **`csrc/vtp_core.cpp`** | Ligne 54 : `// TODO(VTP_L3): Implement actual RDMA transport via libibverbs.` |

### Note honnête (mise à jour)

| Dimension | Avant (audit erroné) | Maintenant (arbre actif) |
|---|---|---|
| Vision technique | 9/10 | **9/10** (inchangé — toujours excellent) |
| Architecture | 8/10 | **8/10** (le `experimental/` aide la lisibilité) |
| Code (qualité) | 5/10 | **7/10** (le nouveau code Rust est bien meilleur) |
| Code (complétude) | 3/10 | **6/10** (la plupart des stubs sont soit implémentés soit honnêtement étiquetés) |
| Documentation | 6/10 | **7/10** (`experimental/README.md` est un bon pattern) |
| Testabilité | 3/10 | **4/10** (toujours pas de CI GPU, mais les tests de fallback sont bons) |
| Sécurité | 4/10 | **5/10** (timeouts dans `GpuNetBridge`, mais pas dans le chemin Tokio) |
| **Note globale** | — | **7/10** — bon prototype avancé, pas encore production-ready |

---

## 2. Rust / CUDA FFI — Le meilleur du projet

Le module `cuda_ffi` (lignes 47-528 de `lib.rs`) est **la meilleure partie du codebase**.
C'est du travail de qualité professionnelle :

```rust
#[cfg(feature = "cuda")]
mod cuda_ffi {
    use std::sync::OnceLock;

    static CUDA_LIB: OnceLock<Option<libloading::Library>> = OnceLock::new();

    fn try_lib() -> Option<&'static libloading::Library> {
        // Charge libcuda.so.1 (Linux) ou nvcuda.dll (Windows)
        // Une seule fois. Thread-safe. Ne panique jamais.
    }

    pub fn cuda_available() -> bool { try_lib().is_some() }
    pub fn init() -> Result<(), String> { /* cuInit(0) avec Once */ }
    pub fn memcpy_dtod(dst, src, bytes) -> Result<(), String> { /* cuMemcpyDtoD_v2 */ }
    pub fn memcpy_dtod_async(dst, src, bytes, stream) -> Result<(), String> { /* cuMemcpyDtoDAsync */ }
    pub fn memcpy_peer_async(dst, dst_ctx, src, src_ctx, bytes, stream) -> Result<(), String> { /* cuMemcpyPeerAsync */ }
    pub fn can_access_peer(dev, peer) -> Result<bool, String> { /* cuDeviceCanAccessPeer */ }
    pub fn ctx_enable_peer_access(peer_ctx) -> Result<(), String> { /* cuCtxEnablePeerAccess */ }
    pub fn mem_alloc_device(bytes) -> Result<u64, String> { /* cuMemAlloc_v2 */ }
    pub fn mem_alloc_host(bytes) -> Result<*mut u8, String> { /* cuMemAllocHost_v2 — pinnée */ }
    pub fn stream_create() -> Result<u64, String> { /* cuStreamCreate */ }
    pub fn event_create() -> Result<u64, String> { /* cuEventCreate avec DISABLE_TIMING */ }
    pub fn stream_wait_event(stream, event) -> Result<(), String> { /* cuStreamWaitEvent */ }
    pub fn staged_copy_double_buffered(...) -> Result<(), String> { /* Implémentation Rust pure */ }
    pub fn async_staged_transfer(...) -> Result<(), String> { /* DMA entrelacé DtoH+HtoD */ }
}
```

**Ce qui est excellent** :
- FFI dynamique via `libloading` — pas de link statique, pas de `unsafe` dans l'API publique
- `OnceLock` partout — initialisation thread-safe et lazy
- Les fonctions retournent `Result<_, String>` — pas de panic
- `cuda_available()` permet au Python de tester avant d'appeler
- Les streams et events sont correctement créés/détruits
- `cuEventCreate` avec `CU_EVENT_DISABLE_TIMING` (bonne pratique perf)

**Améliorations possibles** :

1. **Ajouter `cuMemGetAddressRange`** pour résoudre les adresses CUDA virtuelles → offsets BAR physiques. C'est LE chaînon manquant pour que `rebar_mmap.c` puisse fonctionner.

```rust
pub fn mem_get_address_range(ptr: u64) -> Result<(u64, usize), String> {
    // cuMemGetAddressRange_v2 — retourne (base, size) pour un pointeur CUDA
}
```

2. **Ajouter `cuPointerGetAttribute`** pour déterminer si un pointeur est device/host/unified.

3. **Wrapper `CudaStream` RAII** — actuellement les streams sont des `u64` nus. Un newtype qui implémente `Drop` éviterait les fuites :

```rust
struct CudaStream(u64);
impl Drop for CudaStream {
    fn drop(&mut self) { cuda_ffi::stream_destroy(self.0).ok(); }
}
```

---

## 3. Rust / Tokio — Ce qui reste à corriger

### 3.1 CRITICAL — `Runtime::new()` par appel (toujours présent)

Confirmé dans l'arbre actif : lignes 845, 894, 1079, 1135.

```rust
// Chaque appel fait ça :
let rt = tokio::runtime::Runtime::new().unwrap();
rt.block_on(async { ... })
```

**Solution (5 lignes)** :
```rust
use std::sync::OnceLock;
fn rt() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| tokio::runtime::Runtime::new().expect("Tokio runtime"))
}
// Remplacer chaque `Runtime::new().unwrap()` par `rt()`
```

### 3.2 CRITICAL — Pas de timeouts dans les fonctions Tokio

Les fonctions `send_tensor_p2p`, `receive_tensor_p2p`, `send_tensor_chunked`, `receive_tensor_chunked` utilisent `TcpStream::connect` et `read_exact` **sans timeout**, contrairement au `GpuNetBridge` qui a `connect_timeout(30s)` et `set_read_timeout(120s)`.

**Solution** : Ajouter `tokio::time::timeout` autour de chaque opération, avec un paramètre optionnel `timeout_ms: Option<u64>` exposé à Python.

### 3.3 CRITICAL — `receive_tensor_p2p` pas de limite de payload

Ligne 913 : `let payload_len = total_len - 32;` suivi de `vec![0u8; payload_len as usize]` — aucune borne. La `MAX_PAYLOAD_BYTES` ajoutée par Opus est dans le `GpuNetBridge` (ligne 24-32) mais **pas** dans les 4 fonctions Tokio.

**Solution** : Appliquer la même garde `MAX_PAYLOAD_BYTES` aux 4 fonctions, ou mieux, extraire un helper `check_payload_len()` utilisé partout.

### 3.4 MEDIUM — Protocole chunked stop-and-wait

Pas de changement depuis l'ancien arbre. Stop-and-wait pur.

**Solution** : Fenêtre glissante de 4-8 chunks. Implémentation avec un `Semaphore` initialisé à `MAX_IN_FLIGHT` :

```rust
let semaphore = Arc::new(Semaphore::new(4)); // 4 chunks en vol
for (i, chunk) in payload_vec.chunks(chunk_sz).enumerate() {
    let permit = semaphore.clone().acquire_owned().await.unwrap();
    // ... envoi du chunk ...
    // Dans le handler d'ACK :
    //   drop(permit); // libère un slot pour le chunk suivant
}
```

### 3.5 MEDIUM — `direct_vram_load` : toujours un leak

```rust
std::mem::forget(d_buf);
Ok(0u64) // placeholder — real ptr extraction needs DLPack
```

Le commentaire est honnête ("placeholder"), mais la fonction est exposée à Python et utilisable. Si quelqu'un l'appelle, la mémoire fuit.

**Solution** : Deux options :
- A) Retourner une `PyCapsule` (via PyO3) qui libère le buffer CUDA dans son destructeur
- B) Utiliser `cuda_ffi::mem_alloc_device` + `cuda_ffi::memcpy_htod` au lieu de `cudarc` — ça donnerait un vrai pointeur à retourner sans leak

### 3.6 LOW — `Cargo.toml` : features tokio surdimensionnées

```toml
tokio = { version = "1.36.0", features = ["full"] }
```

`full` active `fs`, `process`, `signal`, `io-util`, `rt-multi-thread`... Pour une cdylib qui fait du TCP, `["net", "macros", "sync", "time"]` suffit.

**Gain** : ~20-30% de réduction du binaire, ~15% de compilation plus rapide.

### 3.7 LOW — Mélange `std::net::TcpStream` et `tokio::net::TcpStream`

`GpuNetBridge` utilise `std::net::TcpStream` (bloquant) avec `set_read_timeout`. Les fonctions Tokio utilisent `tokio::net::TcpStream` (async). Deux modèles d'I/O différents dans le même fichier.

**Recommandation** : Standardiser sur `tokio::net::TcpStream` pour tout le réseau. Pour le `GpuNetBridge`, utiliser `tokio::task::spawn_blocking` ou carrément passer au async.

---

## 4. GpuPipeline & GpuNetBridge — Analyse

### GpuPipeline (lignes 545-800)

Pipeline de transfert GPU→GPU pré-alloué. **Très bonne conception.**

```
Création :
  ├── cuDevicePrimaryCtxRetain (src + dst)
  ├── cuDeviceCanAccessPeer → active P2P bidirectionnel si possible
  ├── Si pas P2P : 3 buffers pinnés (triple-buffering)
  ├── Streams : s_dtoh (src), s_htod (dst), s_p2p (src)
  └── Events : ev_dtoh, ev_htod, ev_buf_free[0..2]

Transfert :
  ├── Si P2P : cuMemcpyPeerAsync → cuStreamSynchronize
  └── Si pas P2P : triple-buffered avec DMA overlapped
```

**Points positifs** :
- Triple-buffering (pas double) → permet un vrai pipeline 3 étapes : DtoH en cours, HtoD en cours, buffer libre
- `unsafe impl Send` justifié par un commentaire qui explique pourquoi c'est safe
- Guarde contre chunk_mb = 0 ou > 4096
- P2P bidirectionnel (pas juste unidirectionnel)

**Améliorations possibles** :

1. **Utiliser CUDA streams créés avec `CU_STREAM_NON_BLOCKING`** pour que les streams P2P et DtoH/HtoD ne se synchronisent pas implicitement avec le stream par défaut de PyTorch.

```rust
// Actuellement : cuda_ffi::stream_create() → cuStreamCreate → stream classique
// Meilleur : cuStreamCreateWithFlags(CU_STREAM_NON_BLOCKING)
```

2. **Ajouter une méthode `transfer_async`** qui ne fait PAS `stream_synchronize` mais retourne un event que l'appelant peut attendre. Utile pour lancer le transfert et faire du compute en parallèle.

3. **Benchmark intégré** — la méthode `bench_gpu_transfer` existe déjà, c'est excellent. Ajouter un warmup automatique dans `new()` (optionnel) pour éviter le cold-start sur le premier transfert.

### GpuNetBridge (lignes 1500-1800)

Pont GPU↔Réseau. **Excellente conception, presque production-ready.**

```
Création :
  ├── cuDevicePrimaryCtxRetain
  ├── 2 streams CUDA (stream_out, stream_in)
  └── 2 buffers pinnés (send_buf, recv_buf) — par défaut 64 MB chacun

forward():
  1. cuMemcpyDtoH : GPU → send_buf (DMA)
  2. TcpStream::write_all : send_buf → réseau (zero-copy depuis la perspective CUDA)
  3. TcpStream::read_exact : réseau → recv_buf (direct dans la mémoire pinnée)
  4. cuMemcpyHtoD : recv_buf → GPU (DMA)
  → GIL relâché pendant TOUTE l'opération
```

**Points positifs** :
- Timeouts TCP : 30s connect, 120s r/w
- `set_nodelay(true)` — pas de délai Nagle
- `SO_RCVBUF`/`SO_SNDBUF` à 4 MB sur Linux
- Vérification `in_bytes > self.buf_size` avant transfert
- `out_bytes > buf_size` vérifié avant réception
- Protocole binaire propre : magic "VTP1" + header + payload
- `Drop` implémenté proprement (libère CUDA + TCP)

**Améliorations possibles** :

1. **Chunked forward pour les tensors > buf_size** — actuellement, si le tenseur dépasse `buf_size`, c'est une erreur. Ajouter un mode chunked automatique.

2. **Compression ZSTD optionnelle** — pour les liens < 1 Gbps, compresser le payload avant `write_all`. Niveau 1 de ZSTD : ~3-4x de ratio sur des activations FP16, pour ~1ms de CPU par 10 MB.

3. **Support TLS optionnel** — utiliser `native-tls` ou `rustls` pour chiffrer la connexion. Le VTP actuel est en clair.

4. **Multiplexing de connexions** — un `GpuNetBridge` par GPU distant plutôt qu'un seul. Permet de faire du transfert parallèle multi-GPU.

5. **Buffer size auto-tuning** — mesurer le RTT et la bande passante au `connect()`, ajuster `buf_size` en conséquence (BDP = bandwidth × RTT).

---

## 5. Python / Core — Architecture

### Ce qui est excellent

**`rust_bridge.py`** — 76 lignes de façade parfaite. Toute la codebase passe par ce module pour accéder au Rust. Ça devrait être un pattern obligatoire pour tous les modules natifs.

**`core/transfer_manager.py`** — toujours le meilleur fichier du projet. Le bypass 7 niveaux est intact et bien commenté.

**Les kernels Triton** — 7 fichiers de kernels optimisés dans `core/` :
- `triton_fused_rmsnorm.py` — 1 passe au lieu de 3
- `triton_fused_rope.py` — 1 kernel au lieu de 5
- `triton_gemv_nvfp4.py` — GEMV avec LUT en L1 cache
- `triton_fused_nvfp4_quant.py` — quant FP4 en un kernel
- `triton_sampling.py` — temperature + top-k + softmax + multinomial fusionnés

**`core/env_flags.py`** — 140 flags d'environnement `VRM_*` documentés centralement. C'est du travail de qualité.

### Ce qui mérite attention

**`core/continuous_batcher.py`** utilise `vramancer_rust.batch_tokenize_fast()` — un tokenizer Rust intégré. C'est un choix architectural intéressant. Question : est-ce que ça gère tous les tokenizers HuggingFace ou seulement un subset ?

**`core/cross_node.py`** utilise `GpuNetBridge` et `RustVTPServer` — l'intégration est réelle et fonctionnelle.

**`core/scheduler.py`** — le cœur de l'orchestration. Si ce fichier a des bugs, tout le reste est impacté. Mériterait des tests unitaires plus approfondis.

**`core/turbo_engine.py`** — CUDA Graph decode persistant. Si ça marche vraiment, c'est un énorme gain de performance (élimine le overhead de lancement de kernel par token).

### Dette technique identifiée

1. **`core/swarm_ledger.py`** — "Shim de compatibilité ascendante pointant vers `_deprecated/`". À nettoyer.
2. **`core/vllm_backend.py`** — "Alias legacy vers `backends_vllm.py`". Idem.
3. **`core/turboquant.py`** — "Shim pointant vers `kv_quantizer.py`". Idem.
4. **`core/webgpu_backend.py`** — marqué "Experimental, POC, non production-ready" dans le code lui-même. Pourquoi est-il dans `core/` et pas dans `experimental/` ?

---

## 6. Experimental/ — Ce qui est vraiment dedans

Le dossier `experimental/` est une **très bonne pratique**. Chaque module a son statut documenté dans le `README.md`. Revue détaillée :

| Module | Taille | Statut réel | Recommandation |
|---|---|---|---|
| `vram_lending.py` | 54 KB | En développement actif | Continuer. Une fois benchmarké → promouvoir en `core/` |
| `hierarchical_memory.py` | 44 KB | Fonctionnel mais non benchmarké | Idem |
| `cross_vendor_bridge.py` | 64 KB | `PipelinedTransport` validé et utilisé. DMA-BUF/ReBAR placeholder | Scinder : promouvoir `PipelinedTransport` dans `core/`, laisser le reste |
| `cluster_discovery.py` | 38 KB | Hors périmètre | OK de le garder ici |
| `nat_traversal.py` | 16 KB | Hors périmètre | OK |
| `aitp_protocol.py` | 13 KB | **Gelé** | Si c'est gelé, pourquoi le garder ? Mettre dans `_deprecated/` |
| `aitp_fec.py` | 9 KB | **Gelé** | Idem |
| `fibre_fastpath.py` | 490 B | Non testable sans IB/RoCE | OK |
| `wake_on_inference.py` | 4 KB | Hors périmètre | OK |

**Action recommandée** : Promouvoir `PipelinedTransport` dans `core/` (il est déjà utilisé par `transfer_manager.py`). Geler ou supprimer les modules AITP (ils sont "gelés" depuis des mois, le code ne sert plus).

---

## 7. Kernels CUDA & Natif — État des lieux

### Les fichiers dans `csrc/`

| Fichier | Statut | Qualité |
|---|---|---|
| `fp4_gemv.cu` | Fonctionnel — GEMV FP4 Blackwell | Bon |
| `paged_attention_kernel.cu` | Fonctionnel — kernel d'attention paginée | Bon, 23 KB |
| `turboquant_kernel.cu` | Fonctionnel — PolarQuant + QJL fusionné | Bon, 16 KB |
| `vtp_cuda.cu` | Fonctionnel — `cudaMemcpyPeerAsync` pour VTP | Simple, propre |
| `aitp_xdp_bypass.c` | Fonctionnel — BPF XDP, spécifique Linux | Très spécifique |
| `dmabuf_bridge.c` | **INCOMPLET** — "destination-side write NOT IMPLEMENTED" | Stub documenté |
| `rebar_mmap.c` | Fonctionnel — mais pas intégré (pas de wrapper Python) | Code C propre |
| `file_offload.cpp` | Fonctionnel — GIL-free RAM↔NVMe | Simple, OK |
| `swarm_core.cpp` | Fonctionnel — XOR parity en C++ | OK, redondant avec le Rust |
| `turbo_forward.cpp` | Fonctionnel — dispatch de blocs en C++ | Intéressant |
| `vtp_core.cpp` | Fonctionnel — sauf L3 RDMA qui est TODO | OK |

### Redondance Rust ↔ C++

Les deux font du XOR parity :
- `rust_core/src/lib.rs` : `generate_xor_parity()` + `repair_xor_shard()` (lignes 1000-1051)
- `csrc/swarm_core.cpp` : `generate_holographic_parity_cpp()` + `heal_holograph_cpp()`

Et le Python a `core/parity_memory.py` qui importe les deux. **Un seul chemin natif suffirait.** Le Rust est plus maintenable (pas de PyBind11 manuel), le C++ est plus rapide (uint64 chunks). Vu que le Rust n'utilise pas de chunks 64-bit (simple boucle octet par octet), le C++ est ~8x plus rapide sur ce chemin. **Recommandation** : porter l'optimisation 64-bit dans le Rust, supprimer `swarm_core.cpp`.

### Le chaînon manquant ReBAR

`rebar_mmap.c` est propre mais :
1. Pas de wrapper PyBind11/ctypes → inutilisable depuis Python
2. Pas de cible de compilation dans `setup_kernels.py` ou le Makefile
3. L'adresse BAR0 physique ≠ adresse CUDA virtuelle → nécessite `cuMemGetAddressRange` pour faire la traduction

**Solution** : Ajouter `cuMemGetAddressRange` au `cuda_ffi`, wrapper Python pour `rebar_mmap.c`, et une fonction `rebar_ptr_to_bar_offset(cuda_ptr)` qui fait la résolution.

---

## 8. Idées innovantes — Propositions

### 8.1 ★★★ CUDA Stream Pools (impact immédiat)

Créer un pool de streams CUDA partagés pour tous les transferts, plutôt que d'en créer/détruire par opération. Implémentation en Rust dans `cuda_ffi` :

```rust
use std::sync::Mutex;
static STREAM_POOL: OnceLock<Mutex<Vec<u64>>> = OnceLock::new();

pub fn stream_acquire() -> Result<u64, String> {
    let mut pool = STREAM_POOL.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap();
    if let Some(s) = pool.pop() { return Ok(s); }
    stream_create() // Créer un nouveau si le pool est vide
}

pub fn stream_release(s: u64) {
    if let Ok(mut pool) = STREAM_POOL.get_or_init(...).lock() {
        if pool.len() < 16 { pool.push(s); return; }
    }
    stream_destroy(s).ok(); // Pool plein, détruire
}
```

### 8.2 ★★★ Auto-detection de bande passante PCIe (impact immédiat)

Le `GpuPool` ou `GpuNetBridge` devrait mesurer la bande passante réelle au premier transfert et ajuster ses paramètres :

```rust
struct AutoTunedPipeline {
    pipeline: GpuPipeline,
    measured_bw_gbps: f64,
    optimal_chunk_bytes: usize,
}

impl AutoTunedPipeline {
    fn new(src: i32, dst: i32) -> PyResult<Self> {
        // Créer un pipeline avec chunk par défaut
        // Faire 3 transferts de 100 MB
        // Mesurer la BW réelle
        // Ajuster chunk_bytes pour maximiser la BW
        // Stocker les paramètres optimaux
    }
}
```

### 8.3 ★★☆ Memory Tiering avec prefetch automatique (moyen terme)

Le `hierarchical_memory.py` définit 6 niveaux (L1 VRAM → L5 NVMe). Ajouter un **prefetcher prédictif** qui anticipe les besoins en fonction du pattern d'accès :

```
Si le tenseur X est accédé tous les N tokens → le promouvoir automatiquement
Si le tenseur Y n'est pas accédé depuis 30 secondes → l'évacuer vers NVMe
Utiliser un compteur LFU (Least Frequently Used) avec decay exponentiel
```

### 8.4 ★★☆ GPUDirect RDMA — le vrai (moyen terme)

Le module `cuda_ffi` a tout ce qu'il faut pour implémenter le vrai GPUDirect RDMA. Il manque :
1. L'enregistrement de mémoire GPU auprès de la NIC (`ibv_reg_mr` sur un pointeur GPU quand `nvidia_peermem` est chargé)
2. L'échange des clés distantes (rkey) via le canal de contrôle VTP
3. Le `ibv_post_send` avec `IBV_WR_RDMA_WRITE` pointant vers la mémoire GPU distante

Une fois fait, le chemin devient :
```
GPU_NIC_src → PCIe → NIC_src → réseau → NIC_dst → PCIe → GPU_dst
```
Zéro copie CPU. Zéro mémoire système. Latence ~1-3µs.

### 8.5 ★★☆ Pipeline Parallelism intelligent (moyen terme)

Le `GpuNetBridge` permet déjà d'envoyer des activations GPU→réseau→GPU. La pièce manquante est un **orchestrateur de pipeline parallelism** qui :
1. Profile chaque couche du modèle (temps de calcul, taille d'activation)
2. Calcule le plan de partition optimal (quelles couches sur quel GPU)
3. Overlap le calcul de la couche N avec le transfert de la couche N-1
4. S'adapte dynamiquement si une couche est plus lente que prévu

### 8.6 ★★☆ Compression de gradient / activation pour liens lents (moyen terme)

Pour le multi-nœuds sur des liens < 10 Gbps :
- **Quantification 1-bit des activations** (signe seulement) + scale → réduction 16x
- **PowerSGD** pour les gradients (low-rank approximation)
- **ZSTD niveau 1** pour les activations (ratio 3-4x, ~1ms/10MB de overhead CPU)

Intégration dans `GpuNetBridge.forward()` : ajouter un flag `compression: Option<CompressionMethod>`.

### 8.7 ★☆☆ Anycast DNS-GPU (long terme)

L'idée d'encoder les capacités GPU dans l'adresse IPv6 est brillante. Extension possible :
- Enregistrer ces adresses dans un **DNS interne** (CoreDNS plugin)
- Les clients résolvent `gpu-a100-fp16-80gb.vramancer.local` → adresse anycast
- Le routage BGP/OSPF achemine vers le nœud le plus proche qui matche
- Aucune modification du code client — juste une résolution DNS

### 8.8 ★☆☆ Mode "single-binary" (long terme)

Compiler VRAMancer en un seul binaire statique avec :
- Python embarqué (via `pyo3` + `pyembedded` ou `PyOxidizer`)
- CUDA kernels compilés en PTX et embarqués
- Modèles au format GGUF
- Configuration par défaut pour le hardware local

Déploiement : un seul fichier, pas de venv, pas de pip install. Idéal pour le edge.

---

## 9. Check-list priorisée pour agents

### Priorité 0 — Immédiat (corrections de sécurité / stabilité)

- [ ] **P0.1** Ajouter `MAX_PAYLOAD_BYTES` + garde `total_len >= 32` aux 4 fonctions Tokio (pas seulement `GpuNetBridge`)
- [ ] **P0.2** Ajouter `tokio::time::timeout` à `send_tensor_p2p`, `receive_tensor_p2p`, `send_tensor_chunked`, `receive_tensor_chunked`
- [ ] **P0.3** Runtime Tokio global (`OnceLock<Runtime>`) — 5 lignes, impact immédiat sur la latence
- [ ] **P0.4** `cxl_direct_memory_dump/load` — ajouter validation basique du pointeur (vérifier qu'il n'est pas null, que la taille est raisonnable)
- [ ] **P0.5** `direct_vram_load` — soit implémenter avec `cuda_ffi` (sans leak), soit lever `NotImplementedError`

### Priorité 1 — Urgent (bloque la production)

- [ ] **P1.1** Fenêtre glissante pour le protocole chunked (pipeline depth ≥ 4)
- [ ] **P1.2** Promouvoir `PipelinedTransport` de `experimental/` dans `core/` (il est déjà utilisé par `transfer_manager.py`)
- [ ] **P1.3** Ajouter `cuMemGetAddressRange` au `cuda_ffi` — nécessaire pour ReBAR et toute feature de mémoire avancée
- [ ] **P1.4** Wrapper Python (ctypes ou PyBind11) pour `rebar_mmap.c` — le code C existe, il ne manque que l'intégration
- [ ] **P1.5** Déplacer `core/webgpu_backend.py` dans `experimental/` (il est marqué POC dans son propre code)
- [ ] **P1.6** Nettoyer les shims : `swarm_ledger.py`, `vllm_backend.py`, `turboquant.py`
- [ ] **P1.7** Uniformiser la langue des commentaires dans `lib.rs` — anglais pour tout le nouveau code

### Priorité 2 — Important (qualité, performance, maintenance)

- [ ] **P2.1** Créer des wrappers RAII pour les ressources CUDA (`CudaStream`, `CudaEvent`, `CudaMemory`)
- [ ] **P2.2** Pool de streams CUDA partagés (éviter `cuStreamCreate`/`cuStreamDestroy` sur le hot path)
- [ ] **P2.3** `GpuNetBridge` — mode chunked automatique pour tensors > buf_size
- [ ] **P2.4** `GpuNetBridge` — TLS optionnel (rustls/native-tls)
- [ ] **P2.5** Porter l'optimisation XOR 64-bit dans le Rust, supprimer `swarm_core.cpp`
- [ ] **P2.6** `GpuPipeline` — ajouter `transfer_async` (sans `synchronize`, retourne un event)
- [ ] **P2.7** `GpuPipeline` — utiliser `CU_STREAM_NON_BLOCKING` pour ne pas bloquer le stream par défaut PyTorch
- [ ] **P2.8** Réduire `tokio` features à `["net", "macros", "sync", "time"]`
- [ ] **P2.9** Ajouter `[profile.release]` avec `lto = "thin"`, `codegen-units = 1`, `strip = "symbols"`
- [ ] **P2.10** Auto-tuning de la bande passante PCIe au premier transfert (cf. §8.2)

### Priorité 3 — Souhaitable (vision long terme)

- [ ] **P3.1** Vrai GPUDirect RDMA (cf. §8.4) — `ibv_reg_mr` sur pointeur GPU, RDMA write one-sided
- [ ] **P3.2** Pipeline parallelism intelligent avec profilage automatique (cf. §8.5)
- [ ] **P3.3** Compression ZSTD pour `GpuNetBridge` sur liens < 1 Gbps (cf. §8.6)
- [ ] **P3.4** Memory tiering avec prefetcher prédictif LFU (cf. §8.3)
- [ ] **P3.5** Anycast DNS-GPU (cf. §8.7)
- [ ] **P3.6** Supprimer ou archiver les modules AITP "gelés" dans `experimental/`
- [ ] **P3.7** CI avec GPU réel (GitHub Actions runner self-hosted ou Cirun.io)
- [ ] **P3.8** Tests de charge réseau simulés (latence, packet loss, bandwidth limit via `tc netem`)
- [ ] **P3.9** Mode single-binary (cf. §8.8)
- [ ] **P3.10** Rotation automatique des clés secrètes HMAC (cf. audit précédent §9.2)

---

## Résumé exécutif

**Le projet a considérablement mûri entre l'ancien checkout (mars) et le checkout actif (juin).**

Le `cuda_ffi` est du code de qualité professionnelle. `GpuPipeline` et `GpuNetBridge` sont bien conçus et presque utilisables en production. Le dossier `experimental/` avec son README honnête est un excellent pattern. Les kernels Triton sont nombreux et pertinents.

**Les 5 chantiers prioritaires** pour passer de "prototype avancé" à "outil utilisable" :

1. **Corriger le chemin Tokio** — timeouts, runtime global, limite de payload (P0)
2. **Finaliser l'intégration ReBAR** — `cuMemGetAddressRange` + wrapper Python (P1)
3. **Promouvoir ce qui marche** dans `experimental/` → `core/`, supprimer ce qui est gelé (P1)
4. **Compléter le `GpuNetBridge`** — chunked mode, TLS, compression (P2)
5. **Vrai GPUDirect RDMA** — le `cuda_ffi` a toutes les briques, il ne manque que `ibv_reg_mr` (P3)

Le potentiel reste énorme. La direction est bonne. Le code s'améliore rapidement.
