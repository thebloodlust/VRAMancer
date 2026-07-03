# Questions pour DeepSeek — revue CUDA / Rust / Tokio (Phase 7)

> De : l'architecte « Fable » (via Opus) · À : DeepSeek
> Origine : section 6 de `decision_architecte_7.md`. Contexte : les correctifs P0
> de `hardening/rust-p0-network` sont approuvés ; ces questions visent une revue
> d'approfondissement **avant** la bascule Path 2 (décision A, palier A1).
> Code concerné : `rust_core/src/lib.rs` (~2160 lignes), modules `GpuPipeline`,
> `GpuNetBridge`, `cuda_ffi`, et la couche FFI pyo3.

---

## Q1 — Pointeurs CUDA détenus à travers les `.await`
Dans `GpuPipeline` et `GpuNetBridge`, des `CUdeviceptr` ou des buffers pinned
sont-ils détenus **à travers** un point `.await` ? Les timeouts P0.2
(`tokio::time::timeout`, 30 s connect / 120 s I/O) peuvent **annuler** une tâche
entre l'allocation et la libération. Qui libère alors le device ptr / le pinned
buffer ? → Auditer chaque chemin d'annulation pour fuite VRAM / pinned.

## Q2 — Design RAII (P2.1)
Proposer les types wrappers (`DevicePtr`, `PinnedBuf`, `StreamGuard`) avec leurs
impls `Drop`, et statuer sur `Send`/`Sync` : un `CUstream` est-il jamais partagé
entre threads Tokio sans synchronisation ? Justifier chaque `unsafe impl
Send/Sync` s'il y en a.

## Q3 — `shared_runtime` (P0.3) : block_on imbriqué + GIL
1. Risque de `block_on` imbriqué (panic « cannot start a runtime from within a
   runtime ») si un appel pyo3 arrive depuis un thread **déjà** dans le runtime
   Tokio global (`OnceLock<Runtime>`) ?
2. GIL : les chemins de transfert relâchent-ils bien le GIL (`py.allow_threads`)
   pendant les copies ? Sinon le serveur Python gèle pendant chaque transfert de
   256 MB. (Note : `bench_gpu_transfer` utilise déjà `py.allow_threads` autour
   des `cuda_ffi`, cf. lib.rs — vérifier que **tous** les chemins le font.)

## Q4 — Cycle de vie des buffers pinned (triple-buffering)
Sont-ils alloués **une fois** et réutilisés, ou re-pinnés à chaque appel ?
`cudaHostAlloc`/`cudaHostRegister` est coûteux ; un re-pinning par appel
ruinerait les 25 GB/s mesurés (Partie D) sur les petits tenseurs. → Mesurer le
coût du premier appel vs régime établi.

## Q5 — Annulation et empoisonnement du pipeline
Si un transfert échoue à mi-course (timeout I/O), l'état du `GpuPipeline`
(buffers en vol, events CUDA non signalés) est-il récupérable, ou faut-il
recréer le `GpuPipeline` ? → Proposer un test de chaos.

## Q6 — Vérifier la claim AVX-512 (B.4 / P2.5)
Confirmer avec `cargo asm` (ou `objdump -d`) sur le CPU cible que la boucle XOR
(`iter_mut().zip()`) auto-vectorise réellement en AVX-512 à `-O3`. Si le CPU de
prod ne supporte qu'AVX2, la claim « auto-vectorise en AVX-512 » doit être
reformulée. (CPU desktop = à préciser ; commande possible :
`grep -o 'avx512[a-z]*' /proc/cpuinfo | sort -u`.)

## Q7 — Instrumentation pour A1
Exposer côté Rust des compteurs lisibles depuis Python : octets transférés, temps
DtoH / HtoD / overlap effectif. But : le benchmark A1 (Path 2 vs accelerate) doit
**décomposer** le temps par token entre calcul et transfert, pour expliquer un
éventuel écart de tok/s.

## Q8 — `unwrap()`/`panic` atteignables depuis la FFI pyo3
Passer en revue les `unwrap`/`expect` atteignables depuis pyo3 : chaque panic
doit devenir une exception Python propre, **jamais** un abort (cohérent avec le
refus de `panic="abort"`, P2.9). Lister les sites et proposer la conversion
(`map_err` → `PyValueError`/`PyRuntimeError`).

---

### Note d'Opus (ce que je peux pré-investiguer si tu veux)
Plusieurs de ces questions sont vérifiables directement dans le code sans
DeepSeek : **Q1** (grep des ptr détenus à travers `.await`), **Q3.1** (chercher
un `block_on` réentrant), **Q6** (`cargo asm` + `/proc/cpuinfo` sur le desktop),
**Q8** (audit `unwrap`/`expect` derrière `#[pyfunction]`). Je peux livrer un
pré-rapport sur celles-ci pour que DeepSeek valide/approfondisse plutôt que de
repartir de zéro — dis-moi si tu veux que je le fasse.
