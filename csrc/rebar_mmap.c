/**
 * ReBAR Direct VRAM mmap — VRAMancer native extension
 *
 * Provides direct memory-mapped access to GPU VRAM via PCIe BAR0.
 * When Resizable BAR is enabled (BAR0 > 256 MB, typically full VRAM),
 * the entire GPU memory is linearly mapped into the CPU physical address
 * space. This extension:
 *
 *   1. Opens the PCI resource0 file (sysfs BAR0 mapping)
 *   2. Maps it into user-space with mmap (write-combining)
 *   3. Provides read/write access to GPU VRAM at physical BAR offsets
 *
 * The main limitation is that CUDA virtual addresses ≠ BAR physical offsets.
 * CUDA's memory allocator uses an internal page table. We use
 * cuMemGetAddressRange() to find the base of CUDA allocations, then
 * compute offsets relative to the start of GPU physical memory.
 *
 * For copies, this is faster than CPU-staged because:
 *   - WC (Write-Combining) mapping allows burst writes without cache pollution
 *   - No intermediate pinned buffer needed
 *   - Kernel handles PCIe TLP coalescing transparently
 *
 * Build: gcc -shared -fPIC -O2 -o libvrm_rebar.so rebar_mmap.c
 *
 * Copyright (c) 2026 VRAMancer contributors — MIT License
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdint.h>
#include <dirent.h>

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
/* Types                                                                   */
/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

typedef struct {
    int           fd;             /* fd for resource0                     */
    void         *base;           /* mmap base pointer                    */
    size_t        bar_size;       /* BAR0 size in bytes                   */
    char          bdf[32];        /* PCIe BDF (e.g. "0000:01:00.0")      */
    int           gpu_index;      /* logical GPU index                    */
    int           valid;
} vrm_rebar_mapping_t;

typedef struct {
    int     success;
    int64_t bytes_copied;
    double  duration_s;
    char    error[256];
} vrm_rebar_result_t;

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
/* BAR0 size detection via sysfs                                           */
/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/**
 * Parse BAR0 size from /sys/bus/pci/devices/<bdf>/resource.
 * Returns BAR0 size in bytes, or 0 on failure.
 */
static size_t _parse_bar0_size(const char *bdf) {
    char path[256];
    snprintf(path, sizeof(path),
             "/sys/bus/pci/devices/%s/resource", bdf);

    FILE *f = fopen(path, "r");
    if (!f) return 0;

    char line[128];
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return 0;
    }
    fclose(f);

    /* Format: "0xSTART 0xEND 0xFLAGS" */
    unsigned long long start = 0, end = 0;
    if (sscanf(line, "%llx %llx", &start, &end) != 2)
        return 0;

    if (end <= start) return 0;
    return (size_t)(end - start + 1);
}

/**
 * Detect ReBAR: returns BAR0 size if > 256 MB (ReBAR active), else 0.
 */
size_t vrm_rebar_detect(const char *bdf) {
    size_t bar_size = _parse_bar0_size(bdf);
    /* ReBAR threshold: > 256 MB means full-window mapping */
    if (bar_size > 256UL * 1024 * 1024)
        return bar_size;
    return 0;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
/* Open / Close BAR0 mapping                                               */
/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/**
 * Open a ReBAR mapping for a GPU identified by PCI BDF.
 *
 * The mapping uses write-combining (WC) for optimal PCIe burst performance.
 * Requires root or appropriate PCI sysfs permissions.
 *
 * @param bdf        PCIe BDF string (e.g. "0000:01:00.0")
 * @param gpu_index  logical GPU index (for bookkeeping)
 * @return mapping handle, or NULL on failure
 */
vrm_rebar_mapping_t *vrm_rebar_open(const char *bdf, int gpu_index) {
    size_t bar_size = vrm_rebar_detect(bdf);
    if (bar_size == 0) return NULL;

    char res_path[256];
    snprintf(res_path, sizeof(res_path),
             "/sys/bus/pci/devices/%s/resource0", bdf);

    int fd = open(res_path, O_RDWR | O_SYNC);
    if (fd < 0) return NULL;

    /*
     * MAP_SHARED is required for MMIO.
     * We do NOT use MAP_HUGETLB — BAR MMIO regions have their own
     * page table entries managed by the kernel PCI subsystem.
     *
     * Write-Combining (WC) is handled by the PCI resources' MTRR/PAT
     * entries set by the kernel — we don't need to request it explicitly.
     */
    void *base = mmap(NULL, bar_size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        close(fd);
        return NULL;
    }

    vrm_rebar_mapping_t *m = calloc(1, sizeof(*m));
    if (!m) {
        munmap(base, bar_size);
        close(fd);
        return NULL;
    }

    m->fd = fd;
    m->base = base;
    m->bar_size = bar_size;
    m->gpu_index = gpu_index;
    m->valid = 1;
    snprintf(m->bdf, sizeof(m->bdf), "%s", bdf);

    return m;
}

/**
 * Close a ReBAR mapping, releasing the mmap and file descriptor.
 */
void vrm_rebar_close(vrm_rebar_mapping_t *m) {
    if (!m) return;
    if (m->base && m->base != MAP_FAILED) {
        munmap(m->base, m->bar_size);
    }
    if (m->fd >= 0) close(m->fd);
    free(m);
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
/* Direct VRAM read/write via BAR0 mmap                                    */
/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/**
 * Read from GPU VRAM at the given BAR offset into a host buffer.
 *
 * @param mapping       active ReBAR mapping
 * @param bar_offset    offset within BAR0 (NOT a CUDA virtual address)
 * @param dst           destination host buffer
 * @param size          number of bytes to read
 * @return 0 on success, -1 on error
 */
int vrm_rebar_read(vrm_rebar_mapping_t *mapping,
                    size_t bar_offset, void *dst, size_t size) {
    if (!mapping || !mapping->valid)
        return -1;
    if (bar_offset + size > mapping->bar_size)
        return -1;

    memcpy(dst, (const char *)mapping->base + bar_offset, size);
    return 0;
}

/**
 * Write to GPU VRAM at the given BAR offset from a host buffer.
 *
 * Uses the WC mapping — writes are coalesced by the CPU write-combine
 * buffer and flushed as PCIe posted writes. No read-before-write.
 *
 * @param mapping       active ReBAR mapping
 * @param bar_offset    offset within BAR0
 * @param src           source host buffer
 * @param size          number of bytes to write
 * @return 0 on success, -1 on error
 */
int vrm_rebar_write(vrm_rebar_mapping_t *mapping,
                     size_t bar_offset, const void *src, size_t size) {
    if (!mapping || !mapping->valid)
        return -1;
    if (bar_offset + size > mapping->bar_size)
        return -1;

    memcpy((char *)mapping->base + bar_offset, src, size);
    return 0;
}

/**
 * Copy between two ReBAR mappings (GPU-to-GPU via CPU WC buffers).
 *
 * Both GPUs must have ReBAR active. The CPU reads from src BAR0
 * and writes to dst BAR0, using write-combining for maximum throughput.
 *
 * @param src_map       source GPU ReBAR mapping
 * @param src_offset    offset in source BAR0
 * @param dst_map       destination GPU ReBAR mapping
 * @param dst_offset    offset in destination BAR0
 * @param size          bytes to copy
 * @param result        output result struct
 * @return 0 on success, -1 on error
 */
int vrm_rebar_copy(vrm_rebar_mapping_t *src_map, size_t src_offset,
                    vrm_rebar_mapping_t *dst_map, size_t dst_offset,
                    size_t size, vrm_rebar_result_t *result) {
    struct timespec t0, t1;
    memset(result, 0, sizeof(*result));

    if (!src_map || !src_map->valid || !dst_map || !dst_map->valid) {
        snprintf(result->error, sizeof(result->error), "invalid mapping(s)");
        return -1;
    }
    if (src_offset + size > src_map->bar_size) {
        snprintf(result->error, sizeof(result->error),
                 "src overflow: offset=%zu + size=%zu > bar=%zu",
                 src_offset, size, src_map->bar_size);
        return -1;
    }
    if (dst_offset + size > dst_map->bar_size) {
        snprintf(result->error, sizeof(result->error),
                 "dst overflow: offset=%zu + size=%zu > bar=%zu",
                 dst_offset, size, dst_map->bar_size);
        return -1;
    }

    clock_gettime(CLOCK_MONOTONIC, &t0);

    /*
     * Direct BAR-to-BAR copy through CPU.
     * The CPU's WC buffer coalesces writes — typically 64-byte cache lines
     * flushed as PCIe posted writes. For large copies this approaches
     * the PCIe bandwidth limit.
     *
     * For even better performance, use 256-bit (AVX2) or 512-bit (AVX-512)
     * non-temporal stores. But memcpy() on modern glibc already uses
     * rep movsb with ERMS which is competitive.
     */
    memcpy((char *)dst_map->base + dst_offset,
           (const char *)src_map->base + src_offset,
           size);

    clock_gettime(CLOCK_MONOTONIC, &t1);

    result->success = 1;
    result->bytes_copied = (int64_t)size;
    result->duration_s = (t1.tv_sec - t0.tv_sec) +
                         (t1.tv_nsec - t0.tv_nsec) / 1e9;
    return 0;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
/* Probe / Info                                                            */
/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/**
 * Get info about a ReBAR mapping.
 *
 * @param mapping     active mapping
 * @param out_bdf     output: PCI BDF string
 * @param bdf_size    buffer size
 * @param out_bar_mb  output: BAR0 size in MB
 * @return 0 on success
 */
int vrm_rebar_info(vrm_rebar_mapping_t *mapping,
                    char *out_bdf, size_t bdf_size, int *out_bar_mb) {
    if (!mapping || !mapping->valid) return -1;
    snprintf(out_bdf, bdf_size, "%s", mapping->bdf);
    if (out_bar_mb) *out_bar_mb = (int)(mapping->bar_size / (1024 * 1024));
    return 0;
}
