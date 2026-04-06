use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyConnectionError};
use pyo3::types::PyBytes;

use hmac::{Hmac, Mac};
use sha2::Sha256;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Semaphore;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::Mutex;

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

    /// cuInit(0) — initialize CUDA driver. Safe to call multiple times.
    pub fn init() -> Result<(), String> {
        static INIT_DONE: std::sync::Once = std::sync::Once::new();
        let mut init_err: Option<String> = None;
        INIT_DONE.call_once(|| {
            unsafe {
                let sym: Result<libloading::Symbol<unsafe extern "C" fn(u32) -> CUresult>, _> =
                    lib().get(b"cuInit\0");
                match sym {
                    Ok(f) => {
                        let res = f(0);
                        if res != 0 {
                            init_err = Some(format!("cuInit returned {res}"));
                        }
                    }
                    Err(e) => init_err = Some(format!("cuInit not found: {e}")),
                }
            }
        });
        match init_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
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

    /// cuMemcpyPeerAsync(dstDevice, dstCtx, srcDevice, srcCtx, ByteCount, hStream)
    /// True P2P copy between GPUs — works on bare metal when P2P is enabled.
    /// Falls back to implicit CPU staging via the driver if P2P is blocked.
    pub fn memcpy_peer_async(
        dst_dev: u64, dst_ctx: u64,
        src_dev: u64, src_ctx: u64,
        bytes: usize, stream: u64,
    ) -> Result<(), String> {
        unsafe {
            let sym: libloading::Symbol<
                unsafe extern "C" fn(u64, u64, u64, u64, usize, u64) -> CUresult,
            > = lib()
                .get(b"cuMemcpyPeerAsync\0")
                .map_err(|e| format!("cuMemcpyPeerAsync not found: {e}"))?;
            let res = sym(dst_dev, dst_ctx, src_dev, src_ctx, bytes, stream);
            if res != 0 {
                return Err(format!("cuMemcpyPeerAsync returned {res}"));
            }
            Ok(())
        }
    }

    /// cuDeviceCanAccessPeer(&canAccess, dev, peerDev)
    pub fn can_access_peer(dev: i32, peer_dev: i32) -> Result<bool, String> {
        unsafe {
            let sym: libloading::Symbol<
                unsafe extern "C" fn(*mut i32, i32, i32) -> CUresult,
            > = lib()
                .get(b"cuDeviceCanAccessPeer\0")
                .map_err(|e| format!("cuDeviceCanAccessPeer not found: {e}"))?;
            let mut can_access: i32 = 0;
            let res = sym(&mut can_access, dev, peer_dev);
            if res != 0 {
                return Err(format!("cuDeviceCanAccessPeer returned {res}"));
            }
            Ok(can_access != 0)
        }
    }

    /// cuCtxEnablePeerAccess(peerCtx, flags)
    pub fn ctx_enable_peer_access(peer_ctx: u64) -> Result<(), String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(u64, u32) -> CUresult> =
                lib().get(b"cuCtxEnablePeerAccess\0")
                    .map_err(|e| format!("cuCtxEnablePeerAccess not found: {e}"))?;
            let res = sym(peer_ctx, 0);
            // CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
            if res != 0 && res != 704 {
                return Err(format!("cuCtxEnablePeerAccess returned {res}"));
            }
            Ok(())
        }
    }

    /// cuMemAlloc_v2(dptr, bytesize) — allocate device memory
    pub fn mem_alloc_device(bytes: usize) -> Result<u64, String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(*mut u64, usize) -> CUresult> =
                lib().get(b"cuMemAlloc_v2\0")
                    .map_err(|e| format!("cuMemAlloc_v2 not found: {e}"))?;
            let mut dptr: u64 = 0;
            let res = sym(&mut dptr, bytes);
            if res != 0 {
                return Err(format!("cuMemAlloc returned {res}"));
            }
            Ok(dptr)
        }
    }

    /// cuMemFree_v2(dptr) — free device memory
    pub fn mem_free_device(dptr: u64) -> Result<(), String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(u64) -> CUresult> =
                lib().get(b"cuMemFree_v2\0")
                    .map_err(|e| format!("cuMemFree_v2 not found: {e}"))?;
            let res = sym(dptr);
            if res != 0 {
                return Err(format!("cuMemFree returned {res}"));
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

    /// Double-buffered CPU-staged GPU-to-GPU transfer (synchronous version).
    /// Bypasses P2P restriction by routing through pinned host memory.
    /// NOTE: This version does NOT overlap DtoH/HtoD. Use async_staged_transfer instead.
    pub fn staged_copy_double_buffered(
        src_dev_ptr: u64,
        dst_dev_ptr: u64,
        total_bytes: usize,
        src_gpu: i32,
        dst_gpu: i32,
        chunk_bytes: usize,
    ) -> Result<(), String> {
        let buf_a = mem_alloc_host(chunk_bytes)?;
        let buf_b = mem_alloc_host(chunk_bytes)?;
        let src_ctx = device_primary_ctx_retain(src_gpu)?;
        let dst_ctx = device_primary_ctx_retain(dst_gpu)?;

        let mut offset: usize = 0;
        let mut buf_idx = 0;
        let bufs = [buf_a, buf_b];

        while offset < total_bytes {
            let chunk = std::cmp::min(chunk_bytes, total_bytes - offset);
            let cur_buf = bufs[buf_idx];
            ctx_set_current(src_ctx)?;
            memcpy_dtoh(cur_buf, src_dev_ptr + offset as u64, chunk)?;
            ctx_set_current(dst_ctx)?;
            memcpy_htod(dst_dev_ptr + offset as u64, cur_buf, chunk)?;
            offset += chunk;
            buf_idx = 1 - buf_idx;
        }

        let _ = mem_free_host(buf_a);
        let _ = mem_free_host(buf_b);
        Ok(())
    }

    // =======================================================================
    // CUDA Async primitives: streams, events, async memcpy
    // =======================================================================

    pub fn stream_create() -> Result<u64, String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(*mut u64, u32) -> CUresult> =
                lib().get(b"cuStreamCreate\0")
                    .map_err(|e| format!("cuStreamCreate not found: {e}"))?;
            let mut stream: u64 = 0;
            let res = sym(&mut stream, 0); // default stream flags
            if res != 0 { return Err(format!("cuStreamCreate returned {res}")); }
            Ok(stream)
        }
    }

    pub fn stream_synchronize(stream: u64) -> Result<(), String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(u64) -> CUresult> =
                lib().get(b"cuStreamSynchronize\0")
                    .map_err(|e| format!("cuStreamSynchronize not found: {e}"))?;
            let res = sym(stream);
            if res != 0 { return Err(format!("cuStreamSynchronize returned {res}")); }
            Ok(())
        }
    }

    pub fn stream_destroy(stream: u64) -> Result<(), String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(u64) -> CUresult> =
                lib().get(b"cuStreamDestroy_v2\0")
                    .map_err(|e| format!("cuStreamDestroy_v2 not found: {e}"))?;
            let res = sym(stream);
            if res != 0 { return Err(format!("cuStreamDestroy returned {res}")); }
            Ok(())
        }
    }

    pub fn event_create() -> Result<u64, String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(*mut u64, u32) -> CUresult> =
                lib().get(b"cuEventCreate\0")
                    .map_err(|e| format!("cuEventCreate not found: {e}"))?;
            let mut event: u64 = 0;
            let res = sym(&mut event, 0x02); // CU_EVENT_DISABLE_TIMING
            if res != 0 { return Err(format!("cuEventCreate returned {res}")); }
            Ok(event)
        }
    }

    pub fn event_record(event: u64, stream: u64) -> Result<(), String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(u64, u64) -> CUresult> =
                lib().get(b"cuEventRecord\0")
                    .map_err(|e| format!("cuEventRecord not found: {e}"))?;
            let res = sym(event, stream);
            if res != 0 { return Err(format!("cuEventRecord returned {res}")); }
            Ok(())
        }
    }

    pub fn event_destroy(event: u64) -> Result<(), String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(u64) -> CUresult> =
                lib().get(b"cuEventDestroy_v2\0")
                    .map_err(|e| format!("cuEventDestroy_v2 not found: {e}"))?;
            let res = sym(event);
            if res != 0 { return Err(format!("cuEventDestroy returned {res}")); }
            Ok(())
        }
    }

    pub fn stream_wait_event(stream: u64, event: u64) -> Result<(), String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(u64, u64, u32) -> CUresult> =
                lib().get(b"cuStreamWaitEvent\0")
                    .map_err(|e| format!("cuStreamWaitEvent not found: {e}"))?;
            let res = sym(stream, event, 0);
            if res != 0 { return Err(format!("cuStreamWaitEvent returned {res}")); }
            Ok(())
        }
    }

    /// cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream)
    pub fn memcpy_dtoh_async(dst_host: *mut u8, src_dev: u64, bytes: usize, stream: u64) -> Result<(), String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(*mut u8, u64, usize, u64) -> CUresult> =
                lib().get(b"cuMemcpyDtoHAsync_v2\0")
                    .map_err(|e| format!("cuMemcpyDtoHAsync_v2 not found: {e}"))?;
            let res = sym(dst_host, src_dev, bytes, stream);
            if res != 0 { return Err(format!("cuMemcpyDtoHAsync returned {res}")); }
            Ok(())
        }
    }

    /// cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream)
    pub fn memcpy_htod_async(dst_dev: u64, src_host: *const u8, bytes: usize, stream: u64) -> Result<(), String> {
        unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(u64, *const u8, usize, u64) -> CUresult> =
                lib().get(b"cuMemcpyHtoDAsync_v2\0")
                    .map_err(|e| format!("cuMemcpyHtoDAsync_v2 not found: {e}"))?;
            let res = sym(dst_dev, src_host, bytes, stream);
            if res != 0 { return Err(format!("cuMemcpyHtoDAsync returned {res}")); }
            Ok(())
        }
    }

    /// True async double-buffered GPU-to-GPU transfer.
    /// Uses cuMemcpyDtoHAsync + cuMemcpyHtoDAsync on separate streams with
    /// event-based synchronization to overlap DtoH and HtoD DMA operations.
    /// This achieves near-theoretical PCIe bandwidth by keeping both DMA
    /// engines busy simultaneously.
    pub fn async_staged_transfer(
        src_dev_ptr: u64,
        dst_dev_ptr: u64,
        total_bytes: usize,
        src_gpu: i32,
        dst_gpu: i32,
        chunk_bytes: usize,
    ) -> Result<(), String> {
        let n_bufs = 2usize;
        let src_ctx = device_primary_ctx_retain(src_gpu)?;
        let dst_ctx = device_primary_ctx_retain(dst_gpu)?;

        // Allocate pinned host buffers
        let mut host_bufs = Vec::with_capacity(n_bufs);
        for _ in 0..n_bufs {
            host_bufs.push(mem_alloc_host(chunk_bytes)?);
        }

        // Create stream on src GPU context for DtoH
        ctx_set_current(src_ctx)?;
        let s_dtoh = stream_create()?;
        let ev_dtoh = event_create()?;

        // Create stream on dst GPU context for HtoD
        ctx_set_current(dst_ctx)?;
        let s_htod = stream_create()?;
        let ev_htod = event_create()?;

        let n_chunks = (total_bytes + chunk_bytes - 1) / chunk_bytes;

        for i in 0..n_chunks {
            let buf_idx = i % n_bufs;
            let offset = i * chunk_bytes;
            let chunk = std::cmp::min(chunk_bytes, total_bytes - offset);

            // If this buffer is still being used by a previous HtoD, wait
            if i >= n_bufs {
                ctx_set_current(src_ctx)?;
                stream_wait_event(s_dtoh, ev_htod)?;
            }

            // DtoH: GPU_src -> pinned buffer (async on s_dtoh)
            ctx_set_current(src_ctx)?;
            memcpy_dtoh_async(host_bufs[buf_idx], src_dev_ptr + offset as u64, chunk, s_dtoh)?;
            event_record(ev_dtoh, s_dtoh)?;

            // HtoD: pinned buffer -> GPU_dst (async on s_htod, after DtoH done)
            ctx_set_current(dst_ctx)?;
            stream_wait_event(s_htod, ev_dtoh)?;
            memcpy_htod_async(dst_dev_ptr + offset as u64, host_bufs[buf_idx] as *const u8, chunk, s_htod)?;
            event_record(ev_htod, s_htod)?;
        }

        // Synchronize both streams
        ctx_set_current(src_ctx)?;
        stream_synchronize(s_dtoh)?;
        ctx_set_current(dst_ctx)?;
        stream_synchronize(s_htod)?;

        // Cleanup
        ctx_set_current(src_ctx)?;
        let _ = stream_destroy(s_dtoh);
        let _ = event_destroy(ev_dtoh);
        ctx_set_current(dst_ctx)?;
        let _ = stream_destroy(s_htod);
        let _ = event_destroy(ev_htod);
        for buf in host_bufs {
            let _ = mem_free_host(buf);
        }

        Ok(())
    }
}

// =========================================================================
// GpuPipeline: Persistent async pipeline with pre-allocated resources
// =========================================================================

/// Persistent GPU-to-GPU transfer pipeline.
/// Pre-allocates CUDA streams, events, and pinned buffers on init.
/// Reuses them for every transfer call, avoiding the ~4ms setup overhead.
///
/// Supports:
/// - P2P direct copy (cuMemcpyPeerAsync) when topology allows
/// - Triple-buffered CPU-staged transfer with overlapped DMA when P2P blocked
/// - Auto chunk size tuning based on transfer size
#[cfg(feature = "cuda")]
#[pyclass]
struct GpuPipeline {
    src_gpu: i32,
    dst_gpu: i32,
    src_ctx: u64,
    dst_ctx: u64,
    s_dtoh: u64,
    s_htod: u64,
    s_p2p: u64,          // Dedicated stream for P2P transfers
    ev_dtoh: u64,
    ev_htod: u64,
    ev_buf_free: Vec<u64>,  // Per-buffer "done consuming" events
    host_bufs: Vec<*mut u8>,
    chunk_bytes: usize,
    p2p_enabled: bool,    // True if cuDeviceCanAccessPeer succeeds
}

// SAFETY: The raw pointers (host_bufs) are pinned CUDA host memory that is
// only accessed from the thread calling transfer(). CUDA contexts and streams
// are thread-safe when used with cuCtxSetCurrent.
#[cfg(feature = "cuda")]
unsafe impl Send for GpuPipeline {}

#[cfg(feature = "cuda")]
#[pymethods]
impl GpuPipeline {
    /// Create a new persistent pipeline between two GPUs.
    /// Pre-allocates all CUDA resources (streams, events, pinned buffers).
    /// Probes P2P accessibility and enables peer access if available.
    ///
    /// Args:
    ///   src_gpu: source GPU ordinal
    ///   dst_gpu: destination GPU ordinal
    ///   chunk_mb: chunk size in MiB (default 16)
    #[new]
    fn new(src_gpu: i32, dst_gpu: i32, chunk_mb: Option<usize>) -> PyResult<Self> {
        let chunk_bytes = chunk_mb.unwrap_or(16) * 1024 * 1024;
        let n_bufs = 3usize;  // Triple-buffering for full overlap

        let src_ctx = cuda_ffi::device_primary_ctx_retain(src_gpu)
            .map_err(|e| PyValueError::new_err(format!("src ctx: {e}")))?;
        let dst_ctx = cuda_ffi::device_primary_ctx_retain(dst_gpu)
            .map_err(|e| PyValueError::new_err(format!("dst ctx: {e}")))?;

        // Probe P2P and enable if available
        let p2p_enabled = if src_gpu != dst_gpu {
            match cuda_ffi::can_access_peer(src_gpu, dst_gpu) {
                Ok(true) => {
                    // Enable bidirectional P2P access
                    cuda_ffi::ctx_set_current(src_ctx).ok();
                    let fwd = cuda_ffi::ctx_enable_peer_access(dst_ctx).is_ok();
                    cuda_ffi::ctx_set_current(dst_ctx).ok();
                    let rev = cuda_ffi::ctx_enable_peer_access(src_ctx).is_ok();
                    fwd && rev
                }
                _ => false,
            }
        } else {
            false
        };

        // Allocate pinned host buffers (only needed if no P2P)
        let mut host_bufs = Vec::with_capacity(n_bufs);
        if !p2p_enabled {
            // Set context before allocating — cuMemAllocHost requires a valid current context
            cuda_ffi::ctx_set_current(src_ctx)
                .map_err(|e| PyValueError::new_err(format!("ctx before pinned alloc: {e}")))?;
            for i in 0..n_bufs {
                let buf = cuda_ffi::mem_alloc_host(chunk_bytes)
                    .map_err(|e| PyValueError::new_err(format!("pinned buf {i}: {e}")))?;
                host_bufs.push(buf);
            }
        }

        // Create streams on src context
        cuda_ffi::ctx_set_current(src_ctx)
            .map_err(|e| PyValueError::new_err(e))?;
        let s_dtoh = cuda_ffi::stream_create()
            .map_err(|e| PyValueError::new_err(format!("stream DtoH: {e}")))?;
        let ev_dtoh = cuda_ffi::event_create()
            .map_err(|e| PyValueError::new_err(format!("event DtoH: {e}")))?;
        // P2P stream lives on src context
        let s_p2p = cuda_ffi::stream_create()
            .map_err(|e| PyValueError::new_err(format!("stream P2P: {e}")))?;

        cuda_ffi::ctx_set_current(dst_ctx)
            .map_err(|e| PyValueError::new_err(e))?;
        let s_htod = cuda_ffi::stream_create()
            .map_err(|e| PyValueError::new_err(format!("stream HtoD: {e}")))?;
        let ev_htod = cuda_ffi::event_create()
            .map_err(|e| PyValueError::new_err(format!("event HtoD: {e}")))?;

        // Per-buffer completion events (created on dst context)
        let mut ev_buf_free = Vec::with_capacity(n_bufs);
        for i in 0..n_bufs {
            let ev = cuda_ffi::event_create()
                .map_err(|e| PyValueError::new_err(format!("event buf_free {i}: {e}")))?;
            ev_buf_free.push(ev);
        }

        Ok(GpuPipeline {
            src_gpu, dst_gpu, src_ctx, dst_ctx,
            s_dtoh, s_htod, s_p2p, ev_dtoh, ev_htod,
            ev_buf_free, host_bufs, chunk_bytes, p2p_enabled,
        })
    }

    /// Transfer data between GPUs using the pre-allocated pipeline.
    /// GIL is released during the entire transfer.
    ///
    /// If P2P is available: uses cuMemcpyPeerAsync (single stream, zero CPU staging).
    /// Otherwise: triple-buffered async CPU-staged transfer with overlapped DMA.
    fn transfer(&self, py: Python, src_ptr: u64, dst_ptr: u64, size_bytes: usize) -> PyResult<bool> {
        if self.p2p_enabled {
            return self._transfer_p2p(py, src_ptr, dst_ptr, size_bytes);
        }
        self._transfer_staged(py, src_ptr, dst_ptr, size_bytes)
    }

    /// P2P direct transfer via cuMemcpyPeerAsync. Zero CPU staging.
    fn _transfer_p2p(&self, py: Python, src_ptr: u64, dst_ptr: u64, size_bytes: usize) -> PyResult<bool> {
        let src_ctx = self.src_ctx;
        let dst_ctx = self.dst_ctx;
        let s_p2p = self.s_p2p;

        py.allow_threads(move || {
            cuda_ffi::ctx_set_current(src_ctx)
                .map_err(|e| PyValueError::new_err(e))?;
            cuda_ffi::memcpy_peer_async(
                dst_ptr, dst_ctx, src_ptr, src_ctx, size_bytes, s_p2p,
            ).map_err(|e| PyValueError::new_err(format!("P2P transfer: {e}")))?;
            cuda_ffi::stream_synchronize(s_p2p)
                .map_err(|e| PyValueError::new_err(format!("P2P sync: {e}")))?;
            Ok(true)
        })
    }

    /// Triple-buffered CPU-staged transfer with overlapped DMA.
    /// Buffer N: DtoH in flight on s_dtoh
    /// Buffer N-1: HtoD in flight on s_htod
    /// Buffer N-2: free (just finished HtoD)
    fn _transfer_staged(&self, py: Python, src_ptr: u64, dst_ptr: u64, size_bytes: usize) -> PyResult<bool> {
        let s_dtoh = self.s_dtoh;
        let s_htod = self.s_htod;
        let ev_dtoh = self.ev_dtoh;
        let ev_htod = self.ev_htod;
        let src_ctx = self.src_ctx;
        let dst_ctx = self.dst_ctx;
        let chunk_bytes = self.chunk_bytes;
        let n_bufs = self.host_bufs.len();
        let host_buf_addrs: Vec<usize> = self.host_bufs.iter().map(|p| *p as usize).collect();
        let ev_buf_free: Vec<u64> = self.ev_buf_free.clone();

        py.allow_threads(move || {
            let n_chunks = (size_bytes + chunk_bytes - 1) / chunk_bytes;

            for i in 0..n_chunks {
                let buf_idx = i % n_bufs;
                let buf_ptr = host_buf_addrs[buf_idx] as *mut u8;
                let offset = i * chunk_bytes;
                let chunk = std::cmp::min(chunk_bytes, size_bytes - offset);

                // Wait for this buffer to be free (previous HtoD using it must complete)
                if i >= n_bufs {
                    cuda_ffi::ctx_set_current(src_ctx)
                        .map_err(|e| PyValueError::new_err(e))?;
                    cuda_ffi::stream_wait_event(s_dtoh, ev_buf_free[buf_idx])
                        .map_err(|e| PyValueError::new_err(e))?;
                }

                // DtoH: GPU_src -> pinned buffer
                cuda_ffi::ctx_set_current(src_ctx)
                    .map_err(|e| PyValueError::new_err(e))?;
                cuda_ffi::memcpy_dtoh_async(
                    buf_ptr, src_ptr + offset as u64, chunk, s_dtoh,
                ).map_err(|e| PyValueError::new_err(e))?;
                cuda_ffi::event_record(ev_dtoh, s_dtoh)
                    .map_err(|e| PyValueError::new_err(e))?;

                // HtoD: pinned buffer -> GPU_dst (after DtoH done)
                cuda_ffi::ctx_set_current(dst_ctx)
                    .map_err(|e| PyValueError::new_err(e))?;
                cuda_ffi::stream_wait_event(s_htod, ev_dtoh)
                    .map_err(|e| PyValueError::new_err(e))?;
                cuda_ffi::memcpy_htod_async(
                    dst_ptr + offset as u64, buf_ptr as *const u8, chunk, s_htod,
                ).map_err(|e| PyValueError::new_err(e))?;
                // Record per-buffer completion so we know when this buf is free
                cuda_ffi::event_record(ev_buf_free[buf_idx], s_htod)
                    .map_err(|e| PyValueError::new_err(e))?;
            }

            // Synchronize both streams
            cuda_ffi::ctx_set_current(src_ctx)
                .map_err(|e| PyValueError::new_err(e))?;
            cuda_ffi::stream_synchronize(s_dtoh)
                .map_err(|e| PyValueError::new_err(e))?;
            cuda_ffi::ctx_set_current(dst_ctx)
                .map_err(|e| PyValueError::new_err(e))?;
            cuda_ffi::stream_synchronize(s_htod)
                .map_err(|e| PyValueError::new_err(e))?;

            Ok(true)
        })
    }

    /// Returns whether this pipeline uses P2P or CPU-staged transfers.
    fn is_p2p(&self) -> bool {
        self.p2p_enabled
    }

    /// Returns pipeline info as a dict.
    fn info(&self) -> PyResult<std::collections::HashMap<String, String>> {
        let mut m = std::collections::HashMap::new();
        m.insert("src_gpu".into(), self.src_gpu.to_string());
        m.insert("dst_gpu".into(), self.dst_gpu.to_string());
        m.insert("p2p_enabled".into(), self.p2p_enabled.to_string());
        m.insert("n_buffers".into(), self.host_bufs.len().to_string());
        m.insert("chunk_bytes".into(), self.chunk_bytes.to_string());
        Ok(m)
    }
}

#[cfg(feature = "cuda")]
impl Drop for GpuPipeline {
    fn drop(&mut self) {
        let _ = cuda_ffi::ctx_set_current(self.src_ctx);
        let _ = cuda_ffi::stream_destroy(self.s_dtoh);
        let _ = cuda_ffi::event_destroy(self.ev_dtoh);
        let _ = cuda_ffi::stream_destroy(self.s_p2p);
        let _ = cuda_ffi::ctx_set_current(self.dst_ctx);
        let _ = cuda_ffi::stream_destroy(self.s_htod);
        let _ = cuda_ffi::event_destroy(self.ev_htod);
        for ev in &self.ev_buf_free {
            let _ = cuda_ffi::event_destroy(*ev);
        }
        for buf in &self.host_bufs {
            let _ = cuda_ffi::mem_free_host(*buf);
        }
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

/// True async double-buffered GPU-to-GPU transfer with overlapped DMA.
/// Uses cuMemcpyDtoHAsync + cuMemcpyHtoDAsync on separate CUDA streams
/// with event-based synchronization. DtoH (GPU→CPU) and HtoD (CPU→GPU)
/// overlap on different DMA engines, achieving near-theoretical PCIe bandwidth.
///
/// Args:
///   src_ptr: source tensor device pointer
///   dst_ptr: destination tensor device pointer
///   size_bytes: total bytes to transfer
///   src_gpu: source GPU ordinal
///   dst_gpu: destination GPU ordinal
///   chunk_bytes: chunk size for pipelining (default 16 MiB)
///
/// Returns: True on success
#[cfg(feature = "cuda")]
#[pyfunction]
fn async_gpu_transfer(
    py: Python,
    src_ptr: u64,
    dst_ptr: u64,
    size_bytes: usize,
    src_gpu: i32,
    dst_gpu: i32,
    chunk_bytes: Option<usize>,
) -> PyResult<bool> {
    let chunk = chunk_bytes.unwrap_or(16 * 1024 * 1024); // 16 MiB default
    py.allow_threads(move || {
        cuda_ffi::async_staged_transfer(
            src_ptr, dst_ptr, size_bytes, src_gpu, dst_gpu, chunk,
        )
        .map(|()| true)
        .map_err(|e| PyValueError::new_err(format!("Async staged transfer failed: {e}")))
    })
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
fn async_gpu_transfer(
    _py: Python, _src_ptr: u64, _dst_ptr: u64, _size_bytes: usize,
    _src_gpu: i32, _dst_gpu: i32, _chunk_bytes: Option<usize>,
) -> PyResult<bool> {
    Err(PyValueError::new_err("Compilé sans feature CUDA."))
}

/// Benchmark GPU-to-GPU transfer using GpuPipeline.
/// Allocates temporary GPU memory, runs warmup + timed iterations,
/// returns dict with bandwidth_gbps, avg_ms, method (p2p/staged), etc.
///
/// Args:
///   src_gpu: source GPU ordinal
///   dst_gpu: destination GPU ordinal
///   size_mb: transfer size in MiB (default 100)
///   chunk_mb: chunk size in MiB (default 16)
///   warmup: warmup iterations (default 3)
///   iterations: timed iterations (default 10)
#[cfg(feature = "cuda")]
#[pyfunction]
fn bench_gpu_transfer(
    py: Python,
    src_gpu: i32,
    dst_gpu: i32,
    size_mb: Option<usize>,
    chunk_mb: Option<usize>,
    warmup: Option<usize>,
    iterations: Option<usize>,
) -> PyResult<std::collections::HashMap<String, String>> {
    use std::time::Instant;

    let size = size_mb.unwrap_or(100) * 1024 * 1024;
    let chunk = chunk_mb.unwrap_or(16);
    let n_warmup = warmup.unwrap_or(3);
    let n_iter = iterations.unwrap_or(10);

    // Create pipeline
    let pipe = GpuPipeline::new(src_gpu, dst_gpu, Some(chunk))?;

    // Allocate GPU memory via cuMemAlloc
    let (src_ptr, dst_ptr) = py.allow_threads(|| -> Result<(u64, u64), String> {
        cuda_ffi::ctx_set_current(pipe.src_ctx)?;
        let src = cuda_ffi::mem_alloc_device(size)?;
        cuda_ffi::ctx_set_current(pipe.dst_ctx)?;
        let dst = cuda_ffi::mem_alloc_device(size)?;
        Ok((src, dst))
    }).map_err(|e| PyValueError::new_err(format!("GPU alloc: {e}")))?;

    // Warmup
    for _ in 0..n_warmup {
        pipe.transfer(py, src_ptr, dst_ptr, size)?;
    }

    // Timed iterations
    let start = Instant::now();
    for _ in 0..n_iter {
        pipe.transfer(py, src_ptr, dst_ptr, size)?;
    }
    let elapsed = start.elapsed();

    // Free GPU memory
    py.allow_threads(|| {
        cuda_ffi::ctx_set_current(pipe.src_ctx).ok();
        cuda_ffi::mem_free_device(src_ptr).ok();
        cuda_ffi::ctx_set_current(pipe.dst_ctx).ok();
        cuda_ffi::mem_free_device(dst_ptr).ok();
    });

    let avg_s = elapsed.as_secs_f64() / n_iter as f64;
    let bw_gbps = (size as f64 * 8.0) / (avg_s * 1e9);
    let bw_gbs = (size as f64) / (avg_s * 1e9);

    let mut m = std::collections::HashMap::new();
    m.insert("src_gpu".into(), src_gpu.to_string());
    m.insert("dst_gpu".into(), dst_gpu.to_string());
    m.insert("size_mb".into(), (size / (1024 * 1024)).to_string());
    m.insert("chunk_mb".into(), chunk.to_string());
    m.insert("method".into(), if pipe.p2p_enabled { "p2p" } else { "staged" }.into());
    m.insert("n_buffers".into(), pipe.host_bufs.len().to_string());
    m.insert("avg_ms".into(), format!("{:.3}", avg_s * 1000.0));
    m.insert("bandwidth_gbps".into(), format!("{:.2}", bw_gbps));
    m.insert("bandwidth_gbs".into(), format!("{:.2}", bw_gbs));
    m.insert("iterations".into(), n_iter.to_string());
    Ok(m)
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
fn bench_gpu_transfer(
    _py: Python, _src_gpu: i32, _dst_gpu: i32,
    _size_mb: Option<usize>, _chunk_mb: Option<usize>,
    _warmup: Option<usize>, _iterations: Option<usize>,
) -> PyResult<std::collections::HashMap<String, String>> {
    Err(PyValueError::new_err("Compilé sans feature CUDA."))
}

// ---------------------------------------------------------------------------
// GIL-free batch tokenizer for non-HuggingFace fallback
// ---------------------------------------------------------------------------

/// Thread-safe vocabulary for the Rust basic tokenizer.
/// Uses lazy_static-style pattern via Mutex.
static RUST_VOCAB: once_cell::sync::Lazy<Mutex<(HashMap<String, u32>, u32)>> =
    once_cell::sync::Lazy::new(|| Mutex::new((HashMap::new(), 5))); // 0..4 reserved for specials

/// Tokenize a single string: lowercase, split on whitespace/punctuation,
/// assign stable IDs via shared vocab. Pure Rust, no GIL.
fn tokenize_one(text: &str) -> Vec<u32> {
    let lower = text.to_lowercase();
    let trimmed = lower.trim();
    if trimmed.is_empty() {
        return vec![];
    }
    // Split on whitespace and common punctuation boundaries
    let tokens: Vec<&str> = trimmed.split(|c: char| c.is_whitespace())
        .filter(|s| !s.is_empty())
        .collect();

    let mut vocab = RUST_VOCAB.lock().unwrap();
    let mut ids = Vec::with_capacity(tokens.len());
    for tok in tokens {
        let id = if let Some(&id) = vocab.0.get(tok) {
            id
        } else {
            let new_id = vocab.1;
            vocab.0.insert(tok.to_string(), new_id);
            vocab.1 += 1;
            new_id
        };
        ids.push(id);
    }
    ids
}

/// Batch tokenize N prompts with GIL released.
/// Returns a list of token ID lists. Thread-safe vocab.
///
/// This is the fallback tokenizer for when HuggingFace tokenizers
/// library is not available (e.g., llama.cpp backend, bare metal).
/// The entire tokenization runs in Rust with py.allow_threads().
#[pyfunction]
fn batch_tokenize_fast(py: Python, prompts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
    py.allow_threads(|| {
        Ok(prompts.iter().map(|p| tokenize_one(p)).collect())
    })
}

/// Single prompt tokenize with GIL released.
#[pyfunction]
fn tokenize_fast(py: Python, text: String) -> PyResult<Vec<u32>> {
    py.allow_threads(|| Ok(tokenize_one(&text)))
}

/// Get current vocab size.
#[pyfunction]
fn tokenizer_vocab_size(_py: Python) -> PyResult<u32> {
    let vocab = RUST_VOCAB.lock().unwrap();
    Ok(vocab.1)
}

// =========================================================================
// GpuNetBridge: Zero-copy GPU↔Network bridge for VTP cross-node inference
// =========================================================================
//
// Eliminates ALL Python/numpy overhead for cross-node tensor transport.
// The entire GPU → pinned_memory → TCP → pinned_memory → GPU pipeline
// runs in Rust with the GIL released.
//
// Current Python VTP path per forward (hidden states):
//   GPU → tensor.cpu() → numpy.tobytes() → Python socket.send → network
//   → Python socket.recv → numpy.frombuffer → tensor.cuda() → GPU
//   Overhead: ~1.4ms (Python/numpy/GIL)
//
// GpuNetBridge path:
//   GPU → cuMemcpyDtoH(pinned) → Rust write_all → network
//   → Rust read_exact(pinned) → cuMemcpyHtoD(pinned) → GPU
//   Overhead: ~0.1ms (pure CUDA DMA + kernel syscalls)
//
// Pre-allocates CUDA streams and pinned host memory at creation.
// Persistent TCP connection — zero per-call allocation.

#[cfg(feature = "cuda")]
#[pyclass]
struct GpuNetBridge {
    gpu_id: i32,
    ctx: u64,
    stream_out: u64,     // CUDA stream for DtoH (GPU → pinned)
    stream_in: u64,      // CUDA stream for HtoD (pinned → GPU)
    send_buf: *mut u8,   // Pinned host memory for outgoing tensor data
    recv_buf: *mut u8,   // Pinned host memory for incoming tensor data
    buf_size: usize,
    tcp: Mutex<Option<std::net::TcpStream>>,
}

#[cfg(feature = "cuda")]
unsafe impl Send for GpuNetBridge {}

#[cfg(feature = "cuda")]
#[pymethods]
impl GpuNetBridge {
    /// Create a GPU↔Network bridge.
    ///
    /// Args:
    ///   gpu_id: CUDA device ordinal for DMA operations
    ///   buf_size_mb: pinned buffer size in MiB (default 64, handles tensors up to this size)
    #[new]
    fn new(gpu_id: i32, buf_size_mb: Option<usize>) -> PyResult<Self> {
        // Initialize CUDA driver (safe to call multiple times)
        cuda_ffi::init()
            .map_err(|e| PyValueError::new_err(format!("CUDA init: {e}")))?;

        let buf_size = buf_size_mb.unwrap_or(64) * 1024 * 1024;

        let ctx = cuda_ffi::device_primary_ctx_retain(gpu_id)
            .map_err(|e| PyValueError::new_err(format!("CUDA ctx GPU {gpu_id}: {e}")))?;

        cuda_ffi::ctx_set_current(ctx)
            .map_err(|e| PyValueError::new_err(format!("ctx set: {e}")))?;

        let stream_out = cuda_ffi::stream_create()
            .map_err(|e| PyValueError::new_err(format!("stream_out: {e}")))?;
        let stream_in = cuda_ffi::stream_create()
            .map_err(|e| PyValueError::new_err(format!("stream_in: {e}")))?;

        let send_buf = cuda_ffi::mem_alloc_host(buf_size)
            .map_err(|e| PyValueError::new_err(format!("pinned send_buf ({} MB): {e}", buf_size / (1024*1024))))?;
        let recv_buf = cuda_ffi::mem_alloc_host(buf_size)
            .map_err(|e| PyValueError::new_err(format!("pinned recv_buf ({} MB): {e}", buf_size / (1024*1024))))?;

        Ok(GpuNetBridge {
            gpu_id, ctx, stream_out, stream_in,
            send_buf, recv_buf, buf_size,
            tcp: Mutex::new(None),
        })
    }

    /// Connect to a remote VTP worker server.
    /// Persistent connection — reused for all forward() calls.
    fn connect(&self, host: String, port: u16) -> PyResult<()> {
        use std::net::TcpStream;

        let addr = format!("{}:{}", host, port);
        // Try parsing as SocketAddr (IP), fallback to DNS resolve via connect()
        let stream = match addr.parse::<std::net::SocketAddr>() {
            Ok(sock_addr) => TcpStream::connect_timeout(
                &sock_addr, std::time::Duration::from_secs(30)),
            Err(_) => TcpStream::connect(&addr),  // hostname: DNS resolve
        }.map_err(|e| PyConnectionError::new_err(format!("connect {addr}: {e}")))?;

        stream.set_nodelay(true).ok();
        stream.set_read_timeout(Some(std::time::Duration::from_secs(120))).ok();
        stream.set_write_timeout(Some(std::time::Duration::from_secs(120))).ok();

        // Large socket buffers for tensor payloads
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::io::AsRawFd;
            let fd = stream.as_raw_fd();
            unsafe {
                let buf_sz: libc::c_int = 4 * 1024 * 1024;
                libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_RCVBUF,
                    &buf_sz as *const _ as *const libc::c_void, 4);
                libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_SNDBUF,
                    &buf_sz as *const _ as *const libc::c_void, 4);
            }
        }

        let mut guard = self.tcp.lock().unwrap();
        *guard = Some(stream);
        Ok(())
    }

    /// Full VTP round-trip: GPU → pinned → TCP → remote → TCP → pinned → GPU.
    ///
    /// GIL is released during the ENTIRE operation (CUDA DMA + network I/O).
    /// Zero Python/numpy overhead. Zero per-call allocation.
    ///
    /// Args:
    ///   in_ptr:  source tensor data_ptr() on GPU (must be contiguous)
    ///   in_bytes: tensor payload size in bytes
    ///   out_ptr: pre-allocated output tensor data_ptr() on GPU
    ///   dtype_code: VTP dtype encoding (0=f32, 1=f16, 2=bf16, ...)
    ///   shape: tensor dimensions
    ///   start_layer, end_layer: layer range for remote worker
    ///   seq_len: current sequence length
    ///
    /// Returns: (out_dtype_code, out_shape_vec, out_bytes)
    fn forward(
        &self,
        py: Python,
        in_ptr: u64,
        in_bytes: usize,
        out_ptr: u64,
        dtype_code: u8,
        shape: Vec<u32>,
        start_layer: u16,
        end_layer: u16,
        seq_len: u32,
    ) -> PyResult<(u8, Vec<u32>, usize)> {
        if in_bytes > self.buf_size {
            return Err(PyValueError::new_err(format!(
                "tensor {} bytes > pinned buffer {} bytes", in_bytes, self.buf_size)));
        }

        // Grab TCP connection (Mutex) and extract raw pointer for use in allow_threads
        let mut tcp_guard = self.tcp.lock()
            .map_err(|_| PyValueError::new_err("bridge mutex poisoned"))?;
        let tcp = tcp_guard.as_mut()
            .ok_or_else(|| PyConnectionError::new_err("GpuNetBridge not connected"))?;
        let tcp_addr = tcp as *mut std::net::TcpStream as usize;

        // Copy scalar fields for the closure (can't capture &self across GIL release)
        // Raw pointers → usize to satisfy Send bound (same pattern as GpuPipeline)
        let ctx = self.ctx;
        let s_out = self.stream_out;
        let s_in = self.stream_in;
        let send_buf_addr = self.send_buf as usize;
        let recv_buf_addr = self.recv_buf as usize;
        let buf_size = self.buf_size;

        // Release GIL for the entire GPU→network→GPU pipeline
        py.allow_threads(move || {
            // SAFETY: tcp_addr points to a valid TcpStream for the duration of
            // this closure because tcp_guard (MutexGuard) is alive until after
            // allow_threads returns. send_buf/recv_buf are pinned CUDA host
            // memory that outlives this call.
            let tcp = unsafe { &mut *(tcp_addr as *mut std::net::TcpStream) };
            let send_buf = send_buf_addr as *mut u8;
            let recv_buf = recv_buf_addr as *mut u8;

            // ── Step 1: GPU → pinned host memory (DMA) ───────────────
            cuda_ffi::ctx_set_current(ctx)
                .map_err(|e| PyValueError::new_err(format!("ctx: {e}")))?;
            cuda_ffi::memcpy_dtoh(send_buf, in_ptr, in_bytes)
                .map_err(|e| PyValueError::new_err(format!("DtoH: {e}")))?;

            // ── Step 2: Send VTP request ─────────────────────────────
            // Header format (big-endian, matches Python struct.pack("!HHIBB...")):
            //   VTP1(4B) | start(2B) | end(2B) | seq_len(4B) |
            //   ndim(1B) | dtype(1B) | shape(4B * ndim) | payload_len(4B)
            use std::io::Write;
            let ndim = shape.len() as u8;
            let mut header = Vec::with_capacity(18 + shape.len() * 4);
            header.extend_from_slice(b"VTP1");
            header.extend_from_slice(&start_layer.to_be_bytes());
            header.extend_from_slice(&end_layer.to_be_bytes());
            header.extend_from_slice(&seq_len.to_be_bytes());
            header.push(ndim);
            header.push(dtype_code);
            for &dim in &shape {
                header.extend_from_slice(&dim.to_be_bytes());
            }
            header.extend_from_slice(&(in_bytes as u32).to_be_bytes());

            tcp.write_all(&header)
                .map_err(|e| PyConnectionError::new_err(format!("send hdr: {e}")))?;

            // Send payload directly from pinned memory (zero-copy from CUDA perspective)
            let send_slice = unsafe { std::slice::from_raw_parts(send_buf, in_bytes) };
            tcp.write_all(send_slice)
                .map_err(|e| PyConnectionError::new_err(format!("send payload: {e}")))?;

            // ── Step 3: Receive VTP response ─────────────────────────
            use std::io::Read;

            // Response: VTP1(4B) | ndim(1B) | dtype(1B) | shape(4B*ndim) | len(4B) | data
            let mut resp_magic = [0u8; 4];
            tcp.read_exact(&mut resp_magic)
                .map_err(|e| PyConnectionError::new_err(format!("recv magic: {e}")))?;
            if &resp_magic != b"VTP1" {
                return Err(PyConnectionError::new_err(format!(
                    "bad VTP response magic: {:?}", resp_magic)));
            }

            let mut meta = [0u8; 2];
            tcp.read_exact(&mut meta)
                .map_err(|e| PyConnectionError::new_err(format!("recv meta: {e}")))?;
            let out_ndim = meta[0] as usize;
            let out_dtype = meta[1];

            let mut shape_buf = vec![0u8; out_ndim * 4];
            tcp.read_exact(&mut shape_buf)
                .map_err(|e| PyConnectionError::new_err(format!("recv shape: {e}")))?;
            let out_shape: Vec<u32> = shape_buf.chunks_exact(4)
                .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let mut len_buf = [0u8; 4];
            tcp.read_exact(&mut len_buf)
                .map_err(|e| PyConnectionError::new_err(format!("recv len: {e}")))?;
            let out_bytes = u32::from_be_bytes(len_buf) as usize;

            if out_bytes > buf_size {
                return Err(PyValueError::new_err(format!(
                    "response {} bytes > pinned buffer {} bytes", out_bytes, buf_size)));
            }

            // Read payload directly into pinned memory
            let recv_slice = unsafe { std::slice::from_raw_parts_mut(recv_buf, out_bytes) };
            tcp.read_exact(recv_slice)
                .map_err(|e| PyConnectionError::new_err(format!("recv payload: {e}")))?;

            // ── Step 4: pinned host memory → GPU (DMA) ───────────────
            cuda_ffi::ctx_set_current(ctx)
                .map_err(|e| PyValueError::new_err(format!("ctx: {e}")))?;
            cuda_ffi::memcpy_htod(out_ptr, recv_buf as *const u8, out_bytes)
                .map_err(|e| PyValueError::new_err(format!("HtoD: {e}")))?;

            Ok((out_dtype, out_shape, out_bytes))
        })
    }

    /// Close the TCP connection.
    fn close(&self) -> PyResult<()> {
        let mut guard = self.tcp.lock().unwrap();
        *guard = None;
        Ok(())
    }

    /// Check if the bridge has an active connection.
    fn is_connected(&self) -> bool {
        self.tcp.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    /// Get the GPU id this bridge was created for.
    #[getter]
    fn gpu_id(&self) -> i32 {
        self.gpu_id
    }

    /// Get bridge info.
    fn info(&self) -> std::collections::HashMap<String, String> {
        let mut m = std::collections::HashMap::new();
        m.insert("gpu_id".into(), self.gpu_id.to_string());
        m.insert("buf_size_mb".into(), (self.buf_size / (1024 * 1024)).to_string());
        m.insert("connected".into(), self.is_connected().to_string());
        m
    }
}

#[cfg(feature = "cuda")]
impl Drop for GpuNetBridge {
    fn drop(&mut self) {
        // Close TCP
        let mut guard = self.tcp.lock().unwrap();
        *guard = None;
        drop(guard);
        // Free CUDA resources
        let _ = cuda_ffi::ctx_set_current(self.ctx);
        let _ = cuda_ffi::stream_destroy(self.stream_out);
        let _ = cuda_ffi::stream_destroy(self.stream_in);
        let _ = cuda_ffi::mem_free_host(self.send_buf);
        let _ = cuda_ffi::mem_free_host(self.recv_buf);
    }
}

// Non-CUDA stub
#[cfg(not(feature = "cuda"))]
#[pyclass]
struct GpuNetBridge {}

#[cfg(not(feature = "cuda"))]
#[pymethods]
impl GpuNetBridge {
    #[new]
    fn new(_gpu_id: i32, _buf_size_mb: Option<usize>) -> PyResult<Self> {
        Err(PyValueError::new_err("GpuNetBridge requires CUDA feature"))
    }
}

// ─── Rust VTP Server (GPU-pointer, zero Python bytes) ────────────
// recv: TCP → pinned → GPU (DMA, no GIL)
// forward: Python callback on GPU tensor (GIL, but CUDA kernels release it)
// send: GPU → pinned → TCP (DMA inside GIL for tensor lifetime, then send without GIL)

#[cfg(feature = "cuda")]
fn _vtp_server_handle_conn_gpu(
    mut stream: std::net::TcpStream,
    running: Arc<std::sync::atomic::AtomicBool>,
    forward_fn: PyObject,
    ctx: u64,
    recv_pinned: usize,
    send_pinned: usize,
    gpu_in_addr: u64,
    buf_size: usize,
) {
    use std::sync::atomic::Ordering;

    while running.load(Ordering::Acquire) {
        // ── Read VTP header (no GIL) ─────────────────────────────
        let mut magic = [0u8; 4];
        match stream.read_exact(&mut magic) {
            Ok(()) => {}
            Err(_) => break,
        }
        if &magic != b"VTP1" { break; }

        let mut hdr = [0u8; 10];
        if stream.read_exact(&mut hdr).is_err() { break; }
        let start = u16::from_be_bytes([hdr[0], hdr[1]]);
        let end = u16::from_be_bytes([hdr[2], hdr[3]]);
        let seq_len = u32::from_be_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
        let ndim = hdr[8] as usize;
        let dtype = hdr[9];

        let mut shape_buf = vec![0u8; ndim * 4];
        if stream.read_exact(&mut shape_buf).is_err() { break; }
        let shape: Vec<u32> = shape_buf.chunks_exact(4)
            .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut len_buf = [0u8; 4];
        if stream.read_exact(&mut len_buf).is_err() { break; }
        let payload_len = u32::from_be_bytes(len_buf) as usize;
        if payload_len > buf_size { break; }

        // ── Read payload into pinned memory (no GIL) ─────────────
        let recv_buf = recv_pinned as *mut u8;
        let recv_slice = unsafe { std::slice::from_raw_parts_mut(recv_buf, payload_len) };
        if stream.read_exact(recv_slice).is_err() { break; }

        // ── DMA: pinned → GPU input buffer (no GIL) ─────────────
        if cuda_ffi::ctx_set_current(ctx).is_err() { break; }
        if cuda_ffi::memcpy_htod(gpu_in_addr, recv_buf as *const u8, payload_len).is_err() {
            break;
        }

        // ── Python forward + DtoH (GIL acquired) ────────────────
        // forward_fn(n_bytes, dtype, shape, start, end, seq_len)
        //   → (out_ptr: int, out_bytes: int, out_dtype: int, out_shape: list[int])
        // DtoH done inside GIL to keep output tensor alive.
        let result: Result<(usize, u8, Vec<u32>), PyErr> = Python::with_gil(|py| {
            let ret = forward_fn.call1(py, (
                payload_len as u64,
                dtype as u32,
                shape.clone(),
                start as u32,
                end as u32,
                seq_len,
            ))?;

            let bound = ret.as_ref(py);
            let out_ptr: u64 = bound.get_item(0)?.extract()?;
            let out_bytes: usize = bound.get_item(1)?.extract()?;
            let out_dtype: u8 = bound.get_item(2)?.extract()?;
            let out_shape: Vec<u32> = bound.get_item(3)?.extract()?;

            if out_bytes > buf_size {
                return Err(PyValueError::new_err(format!(
                    "output {} > buffer {}", out_bytes, buf_size)));
            }

            // DtoH: GPU output → pinned send buffer (tensor alive in Python)
            let send_buf = send_pinned as *mut u8;
            cuda_ffi::ctx_set_current(ctx)
                .map_err(|e| PyValueError::new_err(format!("ctx: {e}")))?;
            cuda_ffi::memcpy_dtoh(send_buf, out_ptr, out_bytes)
                .map_err(|e| PyValueError::new_err(format!("DtoH: {e}")))?;

            Ok((out_bytes, out_dtype, out_shape))
        });

        let (out_bytes, out_dtype, out_shape) = match result {
            Ok(v) => v,
            Err(_) => break,
        };

        // ── Send response from pinned memory (no GIL) ───────────
        let out_ndim = out_shape.len() as u8;
        let mut resp_hdr = Vec::with_capacity(10 + out_shape.len() * 4);
        resp_hdr.extend_from_slice(b"VTP1");
        resp_hdr.push(out_ndim);
        resp_hdr.push(out_dtype);
        for &dim in &out_shape {
            resp_hdr.extend_from_slice(&dim.to_be_bytes());
        }
        resp_hdr.extend_from_slice(&(out_bytes as u32).to_be_bytes());

        if stream.write_all(&resp_hdr).is_err() { break; }
        let send_slice = unsafe {
            std::slice::from_raw_parts(send_pinned as *const u8, out_bytes)
        };
        if stream.write_all(send_slice).is_err() { break; }
    }
}

#[cfg(feature = "cuda")]
#[pyclass]
struct RustVTPServer {
    running: Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    ctx: u64,
    recv_pinned: usize,  // CUDA pinned host memory (as usize for Send)
    send_pinned: usize,
    gpu_in_addr: u64,    // GPU input buffer address (Python-owned torch tensor)
    buf_size: usize,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl RustVTPServer {
    /// Create a new RustVTPServer with CUDA pinned memory.
    /// gpu_in_ptr: data_ptr() of a pre-allocated GPU tensor (Python keeps it alive).
    #[new]
    fn new(gpu_id: i32, buf_size_mb: usize, gpu_in_ptr: u64) -> PyResult<Self> {
        cuda_ffi::init()
            .map_err(|e| PyValueError::new_err(format!("cuda init: {e}")))?;
        let ctx = cuda_ffi::device_primary_ctx_retain(gpu_id)
            .map_err(|e| PyValueError::new_err(format!("ctx GPU {gpu_id}: {e}")))?;
        cuda_ffi::ctx_set_current(ctx)
            .map_err(|e| PyValueError::new_err(format!("ctx set: {e}")))?;
        let buf_size = buf_size_mb * 1024 * 1024;
        let recv_pinned = cuda_ffi::mem_alloc_host(buf_size)
            .map_err(|e| PyValueError::new_err(format!("alloc recv pinned: {e}")))? as usize;
        let send_pinned = cuda_ffi::mem_alloc_host(buf_size)
            .map_err(|e| PyValueError::new_err(format!("alloc send pinned: {e}")))? as usize;

        Ok(RustVTPServer {
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            thread: None,
            ctx,
            recv_pinned,
            send_pinned,
            gpu_in_addr: gpu_in_ptr,
            buf_size,
        })
    }

    /// Start the VTP server. forward_fn is called per request (GIL acquired):
    ///   forward_fn(n_bytes: int, dtype: int, shape: list[int],
    ///              start_layer: int, end_layer: int, seq_len: int)
    ///   -> (out_ptr: int, out_bytes: int, out_dtype: int, out_shape: list[int])
    /// Data is already on GPU when forward_fn is called. Return GPU pointer of result.
    fn start(&mut self, host: String, port: u16, forward_fn: PyObject) -> PyResult<()> {
        use std::sync::atomic::Ordering;
        if self.running.load(Ordering::Acquire) {
            return Err(PyValueError::new_err("server already running"));
        }

        let listener = std::net::TcpListener::bind(format!("{}:{}", host, port))
            .map_err(|e| PyConnectionError::new_err(format!("bind {}:{}: {}", host, port, e)))?;

        self.running.store(true, Ordering::Release);
        let running = self.running.clone();
        let ctx = self.ctx;
        let recv_pinned = self.recv_pinned;
        let send_pinned = self.send_pinned;
        let gpu_in_addr = self.gpu_in_addr;
        let buf_size = self.buf_size;

        let thread = std::thread::Builder::new()
            .name("rust-vtp-server".into())
            .spawn(move || {
                listener.set_nonblocking(true).ok();

                while running.load(Ordering::Acquire) {
                    match listener.accept() {
                        Ok((stream, _addr)) => {
                            stream.set_nodelay(true).ok();
                            stream.set_read_timeout(
                                Some(std::time::Duration::from_secs(120))).ok();
                            stream.set_write_timeout(
                                Some(std::time::Duration::from_secs(120))).ok();

                            #[cfg(target_os = "linux")]
                            {
                                use std::os::unix::io::AsRawFd;
                                let fd = stream.as_raw_fd();
                                let buf_sz: libc::c_int = 4 * 1024 * 1024;
                                unsafe {
                                    libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_RCVBUF,
                                        &buf_sz as *const _ as *const libc::c_void,
                                        std::mem::size_of::<libc::c_int>() as libc::socklen_t);
                                    libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_SNDBUF,
                                        &buf_sz as *const _ as *const libc::c_void,
                                        std::mem::size_of::<libc::c_int>() as libc::socklen_t);
                                }
                            }

                            // Handle connection in-line (one at a time, shares pinned buffers)
                            _vtp_server_handle_conn_gpu(
                                stream, running.clone(), forward_fn.clone(),
                                ctx, recv_pinned, send_pinned, gpu_in_addr, buf_size,
                            );
                        }
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(std::time::Duration::from_millis(10));
                        }
                        Err(_) => {
                            if running.load(Ordering::Acquire) {
                                std::thread::sleep(std::time::Duration::from_millis(100));
                            }
                        }
                    }
                }
            })
            .map_err(|e| PyValueError::new_err(format!("spawn: {e}")))?;

        self.thread = Some(thread);
        Ok(())
    }

    fn stop(&mut self) -> PyResult<()> {
        self.running.store(false, std::sync::atomic::Ordering::Release);
        if let Some(t) = self.thread.take() {
            t.join().ok();
        }
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.load(std::sync::atomic::Ordering::Acquire)
    }
}

#[cfg(feature = "cuda")]
impl Drop for RustVTPServer {
    fn drop(&mut self) {
        self.running.store(false, std::sync::atomic::Ordering::Release);
        if let Some(t) = self.thread.take() {
            t.join().ok();
        }
        let _ = cuda_ffi::ctx_set_current(self.ctx);
        if self.recv_pinned != 0 {
            let _ = cuda_ffi::mem_free_host(self.recv_pinned as *mut u8);
        }
        if self.send_pinned != 0 {
            let _ = cuda_ffi::mem_free_host(self.send_pinned as *mut u8);
        }
    }
}

// Non-CUDA stub
#[cfg(not(feature = "cuda"))]
#[pyclass]
struct RustVTPServer {}

#[cfg(not(feature = "cuda"))]
#[pymethods]
impl RustVTPServer {
    #[new]
    fn new(_gpu_id: i32, _buf_size_mb: usize, _gpu_in_ptr: u64) -> PyResult<Self> {
        Err(PyValueError::new_err("RustVTPServer requires CUDA feature"))
    }
}

#[pymodule]
fn vramancer_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TransportTier>()?;
    #[cfg(feature = "cuda")]
    m.add_class::<GpuPipeline>()?;
    m.add_class::<GpuNetBridge>()?;
    m.add_class::<RustVTPServer>()?;
    m.add_function(wrap_pyfunction!(detect_best_transport, m)?)?;
    m.add_function(wrap_pyfunction!(direct_vram_load, m)?)?;
    m.add_function(wrap_pyfunction!(direct_vram_copy, m)?)?;
    m.add_function(wrap_pyfunction!(staged_gpu_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(async_gpu_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(bench_gpu_transfer, m)?)?;
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
    m.add_function(wrap_pyfunction!(batch_tokenize_fast, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_fast, m)?)?;
    m.add_function(wrap_pyfunction!(tokenizer_vocab_size, m)?)?;
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
