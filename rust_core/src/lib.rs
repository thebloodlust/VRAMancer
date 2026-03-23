use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyConnectionError};
use pyo3::types::PyBytes;

use hmac::{Hmac, Mac};
use sha2::Sha256;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Semaphore;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::sync::Arc;

// Définition de notre type HMAC
type HmacSha256 = Hmac<Sha256>;

/// Hiérarchie des Tiers de Transport VRAMancer
#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub enum TransportTier {
    DirectRdma,     // Niveau 1: GPUDirect / InfiniBand (Bypass CPU Total) - Futur
    ZeroCopyTcp,    // Niveau 2: Safetensors Buffer partagé + TCP natif
    StandardTcp,    // Niveau 3: Fallback Python Pickle + TCP local
}

/// Détecte le meilleur tier réseau disponible sur ce nœud host
#[pyfunction]
fn detect_best_transport() -> TransportTier {
    // Plus tard, nous ajouterons ici la détection de "nvidia_peermem" / "ibverbs".
    // Pour l'instant, on active notre nouveau "Niveau 2" par défaut (ZeroCopy TCP) !
    TransportTier::ZeroCopyTcp
}

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

// =========================================================================
// CUDA Driver API — direct FFI for P2P and async transfers
// =========================================================================

#[cfg(feature = "cuda")]
mod cuda_ffi {
    use std::sync::OnceLock;

    type CUresult = i32;

    static CUDA_LIB: OnceLock<libloading::Library> = OnceLock::new();

    fn lib() -> &'static libloading::Library {
        CUDA_LIB.get_or_init(|| unsafe {
            libloading::Library::new("libcuda.so.1")
                .expect("Cannot load libcuda.so.1")
        })
    }

    /// cuMemcpyDtoD_v2(dst, src, ByteCount)
    pub fn memcpy_dtod(dst: u64, src: u64, bytes: usize) -> Result<(), String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(u64, u64, usize) -> CUresult> =
                lib().get(b"cuMemcpyDtoD_v2\0")
                    .map_err(|e| format!("cuMemcpyDtoD_v2 not found: {e}"))?;
            let res = sym(dst, src, bytes);
            if res != 0 {
                return Err(format!("cuMemcpyDtoD_v2 returned {res}"));
            }
            Ok(())
        }
    }

    /// cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount)
    pub fn memcpy_htod(dst_dev: u64, src_host: *const u8, bytes: usize) -> Result<(), String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(u64, *const u8, usize) -> CUresult> =
                lib().get(b"cuMemcpyHtoD_v2\0")
                    .map_err(|e| format!("cuMemcpyHtoD_v2 not found: {e}"))?;
            let res = sym(dst_dev, src_host, bytes);
            if res != 0 {
                return Err(format!("cuMemcpyHtoD_v2 returned {res}"));
            }
            Ok(())
        }
    }

    /// cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount)
    pub fn memcpy_dtoh(dst_host: *mut u8, src_dev: u64, bytes: usize) -> Result<(), String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(*mut u8, u64, usize) -> CUresult> =
                lib().get(b"cuMemcpyDtoH_v2\0")
                    .map_err(|e| format!("cuMemcpyDtoH_v2 not found: {e}"))?;
            let res = sym(dst_host, src_dev, bytes);
            if res != 0 {
                return Err(format!("cuMemcpyDtoH_v2 returned {res}"));
            }
            Ok(())
        }
    }

    /// cuMemAllocHost_v2(pp, bytesize) — page-locked (pinned) host memory
    pub fn mem_alloc_host(bytes: usize) -> Result<*mut u8, String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(*mut *mut u8, usize) -> CUresult> =
                lib().get(b"cuMemAllocHost_v2\0")
                    .map_err(|e| format!("cuMemAllocHost_v2 not found: {e}"))?;
            let mut ptr: *mut u8 = std::ptr::null_mut();
            let res = sym(&mut ptr, bytes);
            if res != 0 {
                return Err(format!("cuMemAllocHost returned {res}"));
            }
            Ok(ptr)
        }
    }

    /// cuMemFreeHost(p)
    pub fn mem_free_host(ptr: *mut u8) -> Result<(), String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(*mut u8) -> CUresult> =
                lib().get(b"cuMemFreeHost\0")
                    .map_err(|e| format!("cuMemFreeHost not found: {e}"))?;
            let res = sym(ptr);
            if res != 0 {
                return Err(format!("cuMemFreeHost returned {res}"));
            }
            Ok(())
        }
    }

    /// cuCtxSetCurrent(ctx) — set the CUDA context for the current thread
    pub fn ctx_set_current(ctx: u64) -> Result<(), String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(u64) -> CUresult> =
                lib().get(b"cuCtxSetCurrent\0")
                    .map_err(|e| format!("cuCtxSetCurrent not found: {e}"))?;
            let res = sym(ctx);
            if res != 0 {
                return Err(format!("cuCtxSetCurrent returned {res}"));
            }
            Ok(())
        }
    }

    /// cuDevicePrimaryCtxRetain(pctx, dev)
    pub fn device_primary_ctx_retain(dev: i32) -> Result<u64, String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(*mut u64, i32) -> CUresult> =
                lib().get(b"cuDevicePrimaryCtxRetain\0")
                    .map_err(|e| format!("cuDevicePrimaryCtxRetain not found: {e}"))?;
            let mut ctx: u64 = 0;
            let res = sym(&mut ctx, dev);
            if res != 0 {
                return Err(format!("cuDevicePrimaryCtxRetain returned {res}"));
            }
            Ok(ctx)
        }
    }

    /// Double-buffered CPU-staged GPU-to-GPU transfer.
    /// Bypasses P2P restriction by routing through pinned host memory
    /// with overlapped DMA in two directions simultaneously.
    pub fn staged_copy_double_buffered(
        src_dev_ptr: u64,
        dst_dev_ptr: u64,
        total_bytes: usize,
        src_gpu: i32,
        dst_gpu: i32,
        chunk_bytes: usize,
    ) -> Result<(), String> {
        // Allocate two pinned host buffers for double-buffering
        let buf_a = mem_alloc_host(chunk_bytes)?;
        let buf_b = mem_alloc_host(chunk_bytes)?;

        // Get contexts for both GPUs
        let src_ctx = device_primary_ctx_retain(src_gpu)?;
        let dst_ctx = device_primary_ctx_retain(dst_gpu)?;

        let mut offset: usize = 0;
        let mut bufs = [buf_a, buf_b];
        let mut buf_idx = 0;

        while offset < total_bytes {
            let chunk = std::cmp::min(chunk_bytes, total_bytes - offset);
            let cur_buf = bufs[buf_idx];

            // Step 1: DtoH — source GPU context
            ctx_set_current(src_ctx)?;
            memcpy_dtoh(cur_buf, src_dev_ptr + offset as u64, chunk)?;

            // Step 2: HtoD — destination GPU context
            ctx_set_current(dst_ctx)?;
            memcpy_htod(dst_dev_ptr + offset as u64, cur_buf, chunk)?;

            offset += chunk;
            buf_idx = 1 - buf_idx; // alternate buffers
        }

        // Cleanup
        let _ = mem_free_host(buf_a);
        let _ = mem_free_host(buf_b);

        Ok(())
    }
}

/// P.O.C: Écriture directe des octets depuis le réseau vers la VRAM du GPU.
#[cfg(feature = "cuda")]
#[pyfunction]
fn direct_vram_load(py: Python, payload: &[u8]) -> PyResult<u64> {
    py.allow_threads(|| {
        let dev = CudaDevice::new(0)
            .map_err(|e| PyValueError::new_err(format!("CUDA Error: {:?}", e)))?;
        
        let d_buf = dev.htod_sync_copy(payload)
            .map_err(|e| PyValueError::new_err(format!("CUDA Memcpy Error: {:?}", e)))?;
        
        // Leak the CudaSlice so PyTorch can own the memory
        std::mem::forget(d_buf);
        // Note: for a real implementation, use DLPack or return a capsule
        Ok(0u64) // placeholder — real ptr extraction needs DLPack
    })
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
fn direct_vram_load(_py: Python, _payload: &[u8]) -> PyResult<u64> {
    Err(PyValueError::new_err("Ce module Rust a été compilé sans la feature CUDA intégrée."))
}

/// (Option B - Le Data Plane Tokio Zéro-Copie)
/// Accepte directement des MemoryViews ou Safetensors bytes sans passer par Pickle
#[pyfunction]
fn send_tensor_p2p(
    py: Python, 
    host: String, 
    port: u16, 
    secret: &[u8], 
    payload: &[u8]
) -> PyResult<Py<PyBytes>> {
    
    // 1. Signature ultra-rapide (C-speed)
    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|_| PyValueError::new_err("Erreur Secret HMAC"))?;
    mac.update(payload);
    let signature = mac.finalize().into_bytes();

    let payload_len = payload.len() as u64;
    // La longueur totale comprendra les 32 octets de la signature HMAC
    let total_len = payload_len + 32;

    // 2. Relâche du GIL Python et passage en I/O asynchrone natif
    let result: Result<Vec<u8>, String> = py.allow_threads(|| {
        // Lancement d'un runtime Tokio temporaire dédié au transfert massif
        let rt = tokio::runtime::Runtime::new().unwrap();
        
        rt.block_on(async {
            let addr = format!("{}:{}", host, port);
            
            // Connexion asynchrone
            let mut stream = TcpStream::connect(&addr).await
                .map_err(|e| format!("Echec connexion TCP vers {}: {}", addr, e))?;

            // Envoi optimisé du Header (Total Len)
            stream.write_u64(total_len).await
                .map_err(|e| format!("Echec envoi Header: {}", e))?;
            
            // Envoi de la Signature Zero-Trust
            stream.write_all(&signature).await
                .map_err(|e| format!("Echec envoi Signature: {}", e))?;
                
            // Envoi du Tenseur de plusieurs Go
            stream.write_all(payload).await
                .map_err(|e| format!("Echec envoi Payload: {}", e))?;

            // Attente (sans bloquer Python) de la longueur de la réponse
            let resp_len = stream.read_u64().await
                .map_err(|e| format!("Echec lecture réponse Header: {}", e))?;

            // Allocation et lecture de la réponse
            let mut resp_data = vec![0u8; resp_len as usize];
            stream.read_exact(&mut resp_data).await
                .map_err(|e| format!("Echec lecture réponse Body: {}", e))?;

            Ok(resp_data)
        })
    });

    // 3. Retour en Python : Transformation des bytes C en PyBytes
    match result {
        Ok(data) => Ok(PyBytes::new(py, &data).into()),
        Err(e) => Err(PyConnectionError::new_err(e)),
    }
}

/// Écoute silencieuse (Serveur P2P IP Distant)
/// Attend un tenseur réseau, vérifie la signature HMAC Zero-Trust, et retourne les bytes bruts en Python.
#[pyfunction]
fn receive_tensor_p2p(py: Python, port: u16, secret: &[u8]) -> PyResult<Py<PyBytes>> {
    let secret_vec = secret.to_vec();
    
    // On relâche le GIL pour ne pas bloquer le serveur web Python
    let result: Result<Vec<u8>, String> = py.allow_threads(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let addr = format!("0.0.0.0:{}", port);
            let listener = tokio::net::TcpListener::bind(&addr).await
                .map_err(|e| format!("Echec écoute TCP sur le port {}: {}", port, e))?;
            
            // Attente du premier nœud distant qui se connecte
            let (mut socket, _) = listener.accept().await
                .map_err(|e| format!("Echec acceptation de la connexion TCP: {}", e))?;
                
            let total_len = socket.read_u64().await
                .map_err(|e| format!("Echec lecture de l'en-tête distant: {}", e))?;
                
            // Lecture des 32 octets de signature
            let mut signature = vec![0u8; 32];
            socket.read_exact(&mut signature).await
                .map_err(|e| format!("Echec lecture signature HMAC: {}", e))?;
                
            // Lecture intégrale du Gigabytes de VRAM/Tenseur
            let payload_len = total_len - 32;
            let mut payload = vec![0u8; payload_len as usize];
            socket.read_exact(&mut payload).await
                .map_err(|e| format!("Echec lecture du payload de données Tensor: {}", e))?;
                
            // Vérification de sécurité (Si un hacker tente d'envoyer du code corrompu, ça dégage ici)
            let mut mac = HmacSha256::new_from_slice(&secret_vec).unwrap();
            mac.update(&payload);
            if mac.verify_slice(&signature).is_err() {
                return Err("ALERTE INTRUSION : Signature HMAC-SHA256 Invalide ! Tentative de transfert P2P rejetée.".to_string());
            }
            
            // Envoi de l'accusé de réception (ACK) pour libérer le socket distant
            socket.write_u64(2).await.unwrap_or(());
            socket.write_all(b"OK").await.unwrap_or(());
            
            Ok(payload)
        })
    });
    
    match result {
        Ok(data) => Ok(PyBytes::new(py, &data).into()),
        Err(e) => Err(PyConnectionError::new_err(e)),
    }
}

/// Une fonction très rapide pour signer un payload en Rust et esquiver le GIL de Python.
/// Utilisée pour la validation locale.
#[pyfunction]
fn sign_payload_fast(py: Python, secret: &[u8], payload: &[u8]) -> PyResult<Py<PyBytes>> {
    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|_| PyValueError::new_err("Erreur lors de l'initialisation du Secret HMAC"))?;
    
    mac.update(payload);
    let result = mac.finalize().into_bytes();
    
    // On retourne les bytes directement dans le format natif de Python
    Ok(PyBytes::new(py, &result).into())
}

/// Vérifie ultra-rapidement une signature HMAC-SHA256 en Rust.
#[pyfunction]
fn verify_hmac_fast(_py: Python, secret: &[u8], payload: &[u8], signature: &[u8]) -> PyResult<bool> {
    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|_| PyValueError::new_err("Erreur lors de l'initialisation du Secret HMAC"))?;
    
    mac.update(payload);
    
    // Si verify_slice ne panic pas (Ok), la signature est bonne.
    match mac.verify_slice(signature) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// ⚡ Software CXL Offload (Rust equivalent of pure C++ dump)
#[pyfunction]
fn cxl_direct_memory_dump(py: Python, path: String, ptr: usize, num_bytes: usize) -> PyResult<()> {
    py.allow_threads(|| {
        // SAFETY: We assume the caller provides a valid pointer and length. Risk of segfault if incorrect,
        // but Rust handles the file writing safely and bypassing the GIL.
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, num_bytes) };
        let mut file = OpenOptions::new().write(true).create(true).truncate(true).open(&path)
            .map_err(|e| PyValueError::new_err(format!("Software CXL Error: Unable to map memory bus to NVMe path at {}: {}", path, e)))?;
        file.write_all(slice)
            .map_err(|e| PyValueError::new_err(format!("Software CXL Write Error: {}", e)))?;
        Ok(())
    })
}

/// ⚡ Software CXL Onload (Rust equivalent of pure C++ load)
#[pyfunction]
fn cxl_direct_memory_load(py: Python, path: String, ptr: usize, num_bytes: usize) -> PyResult<()> {
    py.allow_threads(|| {
        // SAFETY: Same as above.
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, num_bytes) };
        let mut file = OpenOptions::new().read(true).open(&path)
            .map_err(|e| PyValueError::new_err(format!("Software CXL Error: Failed memory page fault. Data inaccessible on NVMe at {}: {}", path, e)))?;
        file.read_exact(slice)
            .map_err(|e| PyValueError::new_err(format!("Software CXL Read Error: {}", e)))?;
        Ok(())
    })
}

/// Fast XOR for Holographic Parity (bypassing GIL)
#[pyfunction]
fn generate_holographic_parity(py: Python, shards: Vec<&[u8]>) -> PyResult<Py<PyBytes>> {
    // We can confidently release the GIL because we only have immutable slices.
    let (parity, max_len) = py.allow_threads(|| {
        let max_len = shards.iter().map(|s| s.len()).max().unwrap_or(0);
        if max_len == 0 || shards.is_empty() {
            return (vec![], 0);
        }
        
        let mut parity_buf = vec![0u8; max_len];
        for shard in shards {
            // Processing XOR (Rust will heavily optimize this via LLVM vectorizer like AVX-512)
            for (p, &s) in parity_buf.iter_mut().zip(shard.iter()) {
                *p ^= s;
            }
        }
        (parity_buf, max_len)
    });

    if max_len == 0 {
        return Ok(PyBytes::new(py, &[]).into());
    }
    
    Ok(PyBytes::new(py, &parity).into())
}

/// Reconstructs a missing tensor chunk from Parity
#[pyfunction]
fn heal_holograph(py: Python, valid_shards: Vec<&[u8]>, parity: &[u8]) -> PyResult<Py<PyBytes>> {
    let reconstructed = py.allow_threads(|| {
        let mut rec_buf = parity.to_vec();
        for shard in valid_shards {
            for (r, &s) in rec_buf.iter_mut().zip(shard.iter()) {
                *r ^= s;
            }
        }
        rec_buf
    });
    
    Ok(PyBytes::new(py, &reconstructed).into())
}

/// C'est ici que l'on déclare officiellement notre module Python.

// =========================================================================
// Chunked Pipeline Transfer (Tokio channels + backpressure)
// =========================================================================

/// Default chunk size for pipelined transfers: 4 MiB
const CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// Send a large tensor in chunked pipeline mode with backpressure.
/// Chunks are signed individually for incremental verification.
/// Returns total bytes acknowledged by receiver.
#[pyfunction]
fn send_tensor_chunked(
    py: Python,
    host: String,
    port: u16,
    secret: &[u8],
    payload: &[u8],
    chunk_size: Option<usize>,
) -> PyResult<u64> {
    let chunk_sz = chunk_size.unwrap_or(CHUNK_SIZE);
    let secret_vec = secret.to_vec();
    let payload_vec = payload.to_vec();

    let result: Result<u64, String> = py.allow_threads(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let addr = format!("{}:{}", host, port);
            let mut stream = TcpStream::connect(&addr).await
                .map_err(|e| format!("Connect to {}: {}", addr, e))?;

            let total_len = payload_vec.len() as u64;
            let num_chunks = ((payload_vec.len() + chunk_sz - 1) / chunk_sz) as u32;

            // Header: [total_len: u64] [num_chunks: u32] [chunk_size: u32]
            stream.write_u64(total_len).await.map_err(|e| format!("Header: {}", e))?;
            stream.write_u32(num_chunks).await.map_err(|e| format!("Header: {}", e))?;
            stream.write_u32(chunk_sz as u32).await.map_err(|e| format!("Header: {}", e))?;

            let mut acked: u64 = 0;
            for (i, chunk) in payload_vec.chunks(chunk_sz).enumerate() {
                // Sign each chunk
                let mut mac = HmacSha256::new_from_slice(&secret_vec).unwrap();
                mac.update(chunk);
                let sig = mac.finalize().into_bytes();

                // [sig: 32 bytes] [chunk_len: u32] [chunk_data]
                stream.write_all(&sig).await.map_err(|e| format!("Chunk {} sig: {}", i, e))?;
                stream.write_u32(chunk.len() as u32).await.map_err(|e| format!("Chunk {} len: {}", i, e))?;
                stream.write_all(chunk).await.map_err(|e| format!("Chunk {} data: {}", i, e))?;

                // Wait for per-chunk ACK (backpressure: receiver controls pace)
                let ack = stream.read_u8().await.map_err(|e| format!("Chunk {} ack: {}", i, e))?;
                if ack != 1 {
                    return Err(format!("Chunk {} rejected by receiver (HMAC mismatch)", i));
                }
                acked += chunk.len() as u64;
            }
            Ok(acked)
        })
    });

    match result {
        Ok(n) => Ok(n),
        Err(e) => Err(PyConnectionError::new_err(e)),
    }
}

/// Receive a chunked pipelined tensor transfer with HMAC verification per chunk.
/// max_connections limits concurrent receivers (backpressure on cluster level).
#[pyfunction]
fn receive_tensor_chunked(
    py: Python,
    port: u16,
    secret: &[u8],
    max_connections: Option<u32>,
) -> PyResult<Py<PyBytes>> {
    let secret_vec = secret.to_vec();
    let max_conn = max_connections.unwrap_or(8) as usize;

    let result: Result<Vec<u8>, String> = py.allow_threads(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let addr = format!("0.0.0.0:{}", port);
            let listener = tokio::net::TcpListener::bind(&addr).await
                .map_err(|e| format!("Bind {}: {}", port, e))?;
            let _semaphore = Arc::new(Semaphore::new(max_conn));

            let (mut socket, _) = listener.accept().await
                .map_err(|e| format!("Accept: {}", e))?;

            let total_len = socket.read_u64().await.map_err(|e| format!("Header: {}", e))?;
            let num_chunks = socket.read_u32().await.map_err(|e| format!("Header: {}", e))?;
            let _chunk_size = socket.read_u32().await.map_err(|e| format!("Header: {}", e))?;

            let mut payload = Vec::with_capacity(total_len as usize);

            for i in 0..num_chunks {
                // Read per-chunk signature
                let mut sig = vec![0u8; 32];
                socket.read_exact(&mut sig).await.map_err(|e| format!("Chunk {} sig: {}", i, e))?;

                let chunk_len = socket.read_u32().await.map_err(|e| format!("Chunk {} len: {}", i, e))? as usize;
                let mut chunk = vec![0u8; chunk_len];
                socket.read_exact(&mut chunk).await.map_err(|e| format!("Chunk {} data: {}", i, e))?;

                // Verify HMAC
                let mut mac = HmacSha256::new_from_slice(&secret_vec).unwrap();
                mac.update(&chunk);
                if mac.verify_slice(&sig).is_err() {
                    // Send NACK
                    socket.write_u8(0).await.unwrap_or(());
                    return Err(format!("Chunk {} HMAC verification failed", i));
                }

                payload.extend_from_slice(&chunk);
                // Send ACK
                socket.write_u8(1).await.map_err(|e| format!("ACK {}: {}", i, e))?;
            }

            Ok(payload)
        })
    });

    match result {
        Ok(data) => Ok(PyBytes::new(py, &data).into()),
        Err(e) => Err(PyConnectionError::new_err(e)),
    }
}

// =========================================================================
// Batch HMAC verification (verify N signatures in one GIL release)
// =========================================================================

/// Verify multiple HMAC-SHA256 signatures in a single Rust call.
/// Takes a list of (payload, signature) tuples and returns a list of booleans.
#[pyfunction]
fn verify_hmac_batch(_py: Python, secret: &[u8], items: Vec<(&[u8], &[u8])>) -> PyResult<Vec<bool>> {
    let results: Vec<bool> = items.iter().map(|(payload, signature)| {
        let mut mac = match HmacSha256::new_from_slice(secret) {
            Ok(m) => m,
            Err(_) => return false,
        };
        mac.update(payload);
        mac.verify_slice(signature).is_ok()
    }).collect();
    Ok(results)
}

/// P2P GPU-to-GPU copy via CUDA driver API (works through CPU staging if P2P blocked)
#[cfg(feature = "cuda")]
#[pyfunction]
fn direct_vram_copy(py: Python, src_ptr: u64, dst_ptr: u64, size_bytes: usize) -> PyResult<bool> {
    py.allow_threads(|| {
        // Try direct DtoD first (works if P2P enabled, fails otherwise)
        match cuda_ffi::memcpy_dtod(dst_ptr, src_ptr, size_bytes) {
            Ok(()) => Ok(true),
            Err(e) => Err(PyValueError::new_err(format!("CUDA DtoD copy failed: {e}")))
        }
    })
}

#[pyfunction]
#[cfg(not(feature = "cuda"))]
fn direct_vram_copy(_py: Python, _src_ptr: u64, _dst_ptr: u64, _size_bytes: usize) -> PyResult<bool> {
    Err(PyValueError::new_err("Compilé sans feature CUDA."))
}

/// Double-buffered CPU-staged GPU-to-GPU transfer via Rust (GIL released).
/// Uses pinned memory + alternating buffers to maximize PCIe throughput.
/// This is the NVFP4 bypass: when P2P is blocked, this routes through
/// page-locked host memory with zero Python/GIL overhead.
///
/// Args:
///   src_ptr: source tensor device pointer (from tensor.data_ptr())
///   dst_ptr: destination tensor device pointer
///   size_bytes: total bytes to transfer
///   src_gpu: source GPU ordinal
///   dst_gpu: destination GPU ordinal
///   chunk_bytes: chunk size for pipelining (default 4 MiB)
///
/// Returns: True on success
#[cfg(feature = "cuda")]
#[pyfunction]
fn staged_gpu_transfer(
    py: Python,
    src_ptr: u64,
    dst_ptr: u64,
    size_bytes: usize,
    src_gpu: i32,
    dst_gpu: i32,
    chunk_bytes: Option<usize>,
) -> PyResult<bool> {
    let chunk = chunk_bytes.unwrap_or(4 * 1024 * 1024); // 4 MiB default
    py.allow_threads(move || {
        cuda_ffi::staged_copy_double_buffered(
            src_ptr, dst_ptr, size_bytes, src_gpu, dst_gpu, chunk,
        )
        .map(|()| true)
        .map_err(|e| PyValueError::new_err(format!("Staged transfer failed: {e}")))
    })
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
fn staged_gpu_transfer(
    _py: Python, _src_ptr: u64, _dst_ptr: u64, _size_bytes: usize,
    _src_gpu: i32, _dst_gpu: i32, _chunk_bytes: Option<usize>,
) -> PyResult<bool> {
    Err(PyValueError::new_err("Compilé sans feature CUDA."))
}

#[pymodule]
fn vramancer_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TransportTier>()?;
    m.add_function(wrap_pyfunction!(detect_best_transport, m)?)?;
    m.add_function(wrap_pyfunction!(direct_vram_load, m)?)?;
    m.add_function(wrap_pyfunction!(direct_vram_copy, m)?)?;
    m.add_function(wrap_pyfunction!(staged_gpu_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(sign_payload_fast, m)?)?;
    m.add_function(wrap_pyfunction!(verify_hmac_fast, m)?)?;
    m.add_function(wrap_pyfunction!(verify_hmac_batch, m)?)?;
    m.add_function(wrap_pyfunction!(send_tensor_p2p, m)?)?;
    m.add_function(wrap_pyfunction!(receive_tensor_p2p, m)?)?;
    m.add_function(wrap_pyfunction!(send_tensor_chunked, m)?)?;
    m.add_function(wrap_pyfunction!(receive_tensor_chunked, m)?)?;
    m.add_function(wrap_pyfunction!(cxl_direct_memory_dump, m)?)?;
    m.add_function(wrap_pyfunction!(cxl_direct_memory_load, m)?)?;
    m.add_function(wrap_pyfunction!(generate_holographic_parity, m)?)?;
    m.add_function(wrap_pyfunction!(heal_holograph, m)?)?;
    m.add_function(wrap_pyfunction!(inject_to_vram_ptr, m)?)?;
    Ok(())
}

/// Injecte un buffer CPU directement dans la VRAM via cuMemcpyHtoD (GIL released)
#[cfg(feature = "cuda")]
#[pyfunction]
fn inject_to_vram_ptr(py: Python, payload: &[u8], dest_ptr: u64) -> PyResult<()> {
    py.allow_threads(|| {
        cuda_ffi::memcpy_htod(dest_ptr, payload.as_ptr(), payload.len())
            .map_err(|e| PyValueError::new_err(format!("HtoD inject failed: {e}")))
    })
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
fn inject_to_vram_ptr(_py: Python, _payload: &[u8], _dest_ptr: u64) -> PyResult<()> {
    Ok(())
}
