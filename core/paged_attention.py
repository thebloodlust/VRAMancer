"""VRAMancer Paged Attention — KV Cache Manager.

Implements block-based KV cache management inspired by vLLM's PagedAttention:
  - KV cache is divided into fixed-size **pages** (blocks of token slots)
  - Each request gets a list of page references (not contiguous memory)
  - Pages are allocated on-demand and freed when requests complete
  - Enables memory sharing (copy-on-write for beam search, prefix caching)

This eliminates the 2 main problems of naive KV cache:
  1. Memory waste from over-allocation (reserved max_seq_len per request)
  2. Memory fragmentation when requests have different lengths

Architecture:
    PagedKVCacheManager
      ├── PagePool           (pre-allocated GPU memory pages)
      ├── PageTable          (per-request virtual → physical page mapping)
      └── PrefixCache        (optional: share prefix pages across requests)

References:
  - vLLM: Efficient Memory Management for LLM Serving with PagedAttention
    (Kwon et al., 2023) — https://arxiv.org/abs/2309.06180
"""

from __future__ import annotations

import os
import math
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from core.logger import LoggerAdapter
    _logger = LoggerAdapter("paged_attn")
except Exception:
    _logger = logging.getLogger("vramancer.paged_attention")
_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

try:
    import torch
    _TORCH = True
except ImportError:
    torch = None  # type: ignore
    _TORCH = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PagedKVConfig:
    """Configuration for the paged KV cache."""

    page_size: int = 16          # tokens per page (vLLM default = 16)
    num_layers: int = 12         # transformer layers (set from model config)
    num_kv_heads: int = 12       # KV attention heads
    head_dim: int = 64           # dimension per head
    max_pages: int = 4096        # total physical pages in the pool
    device: str = "cuda:0"       # primary device
    dtype: str = "float16"       # storage precision
    devices: List[str] = field(default_factory=list)  # all devices for distributed pool
    pages_per_device: Dict[str, int] = field(default_factory=dict)  # per-device page budget
    enable_lending: bool = True  # use VRAMLendingPool for overflow

    @property
    def page_size_bytes(self) -> int:
        """Memory per page: 2 (K+V) * layers * heads * head_dim * page_size * dtype_bytes."""
        dtype_size = 2 if "16" in self.dtype else 4
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.page_size * dtype_size

    @property
    def total_memory_bytes(self) -> int:
        return self.page_size_bytes * self.max_pages

    @classmethod
    def from_model(cls, model: Any, max_pages: int = 4096, device: str = "cuda:0") -> "PagedKVConfig":
        """Auto-detect config from a HuggingFace model."""
        config = getattr(model, 'config', None)
        if config is None:
            return cls(max_pages=max_pages, device=device)

        num_layers = getattr(config, 'num_hidden_layers', 12)
        num_heads = getattr(config, 'num_attention_heads', 12)
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
        hidden_size = getattr(config, 'hidden_size', 768)
        head_dim = hidden_size // num_heads

        return cls(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_pages=max_pages,
            device=device,
        )


# ---------------------------------------------------------------------------
# Physical page
# ---------------------------------------------------------------------------

@dataclass
class PhysicalPage:
    """A single page in the KV cache pool."""

    page_id: int
    ref_count: int = 0          # copy-on-write reference counting
    allocated: bool = False
    last_access: float = field(default_factory=time.time)
    device: str = "cuda:0"      # which GPU this page lives on
    is_borrowed: bool = False   # True if allocated via VRAMLendingPool
    lease_id: Optional[str] = None  # lending lease ID (if borrowed)

    # Actual tensors (allocated lazily or from pool)
    # Shape: [num_layers, 2(K/V), num_kv_heads, page_size, head_dim]
    data: Any = None


# ---------------------------------------------------------------------------
# Page table (per-request virtual→physical mapping)
# ---------------------------------------------------------------------------

@dataclass
class PageTableEntry:
    """Maps a request to its list of physical pages."""

    request_id: str
    pages: List[int] = field(default_factory=list)  # physical page IDs
    num_tokens: int = 0           # tokens written so far
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# PagedKVCacheManager
# ---------------------------------------------------------------------------

class PagedKVCacheManager:
    """Block-based KV cache manager with on-demand allocation.

    Key features:
      - Pages allocated only when tokens are actually generated
      - Copy-on-write for beam search (share prefix, copy on diverge)
      - Prefix caching: identical prefixes reuse the same physical pages
      - LRU eviction when pool is exhausted
    """

    def __init__(self, config: Optional[PagedKVConfig] = None):
        self.config = config or PagedKVConfig()
        self._lock = threading.Lock()

        # Physical page pool
        self._pages: List[PhysicalPage] = [
            PhysicalPage(page_id=i) for i in range(self.config.max_pages)
        ]
        self._free_pages: List[int] = list(range(self.config.max_pages))

        # Per-request page tables
        self._page_tables: Dict[str, PageTableEntry] = {}

        # Prefix cache: hash(token_ids) -> physical page ID
        self._prefix_cache: Dict[int, int] = {}

        # Stats
        self._total_allocations = 0
        self._total_frees = 0
        self._peak_usage = 0
        self._cache_hits = 0
        self._overflow_borrows = 0

        # Multi-GPU pool: device -> tensor pool
        self._gpu_pools: Dict[str, Any] = {}
        self._gpu_pool: Any = None  # Legacy compat (primary pool)
        self._page_device_map: Dict[int, str] = {}  # page_id -> device

        # VRAMLendingPool integration for overflow
        self._lending_pool = None
        self._lending_leases: Dict[str, Any] = {}  # lease_id -> lease

        # Build device list
        if not self.config.devices:
            self.config.devices = [self.config.device]

        # Pre-allocate GPU memory pools
        if _TORCH and not _MINIMAL:
            self._init_multi_gpu_pools()

        # Try to connect to lending pool
        if self.config.enable_lending:
            self._init_lending_pool()

        total_pages = sum(
            self.config.pages_per_device.get(d, self.config.max_pages)
            for d in self.config.devices
        ) if len(self.config.devices) > 1 else self.config.max_pages
        _logger.info(
            "PagedKVCache: %d pages @ %d tok/page = %.1f MB (devices=%s)",
            total_pages,
            self.config.page_size,
            self.config.total_memory_bytes / 1e6,
            self.config.devices,
        )

    def _init_multi_gpu_pools(self) -> None:
        """Pre-allocate page pools across all registered GPUs.

        Distributes pages proportionally based on pages_per_device config,
        or equally if not specified.
        """
        dtype = torch.float16 if "16" in self.config.dtype else torch.float32
        page_start = 0

        for device in self.config.devices:
            device_pages = self.config.pages_per_device.get(
                device, self.config.max_pages
            )
            if device_pages <= 0:
                continue
            try:
                pool = torch.zeros(
                    device_pages,
                    self.config.num_layers,
                    2,  # K and V
                    self.config.num_kv_heads,
                    self.config.page_size,
                    self.config.head_dim,
                    dtype=dtype,
                    device=device,
                )
                self._gpu_pools[device] = pool
                # Map page IDs to this device
                for pid in range(page_start, page_start + device_pages):
                    self._page_device_map[pid] = device
                    if pid < len(self._pages):
                        self._pages[pid].device = device
                page_start += device_pages
                _logger.info(
                    "Pool allocated: %d pages (%.1f MB) on %s",
                    device_pages,
                    pool.nelement() * pool.element_size() / 1e6,
                    device,
                )
            except Exception as e:
                _logger.warning("Pool allocation on %s failed: %s", device, e)

        # Legacy compat: primary pool = first device pool
        if self.config.device in self._gpu_pools:
            self._gpu_pool = self._gpu_pools[self.config.device]
        elif self._gpu_pools:
            self._gpu_pool = next(iter(self._gpu_pools.values()))

    def _init_gpu_pool(self) -> None:
        """Legacy single-GPU pool init (backward compat)."""
        self._init_multi_gpu_pools()

    def _init_lending_pool(self) -> None:
        """Connect to the VRAMLendingPool for overflow pages."""
        try:
            from core.vram_lending import get_lending_pool
            self._lending_pool = get_lending_pool()
            _logger.debug("Connected to VRAMLendingPool for overflow")
        except Exception:
            self._lending_pool = None

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate(self, request_id: str, num_tokens: int = 0) -> PageTableEntry:
        """Allocate page table for a new request.

        Pages are allocated on-demand — initially only enough for num_tokens.
        """
        with self._lock:
            if request_id in self._page_tables:
                return self._page_tables[request_id]

            entry = PageTableEntry(request_id=request_id)

            # Allocate pages for initial tokens
            pages_needed = math.ceil(num_tokens / self.config.page_size) if num_tokens > 0 else 0
            for _ in range(pages_needed):
                page_id = self._alloc_page()
                if page_id is None:
                    _logger.warning("Page pool exhausted for request %s", request_id)
                    break
                entry.pages.append(page_id)

            entry.num_tokens = num_tokens
            self._page_tables[request_id] = entry

            usage = self.config.max_pages - len(self._free_pages)
            self._peak_usage = max(self._peak_usage, usage)

            return entry

    def append_token(self, request_id: str) -> Optional[Tuple[int, int]]:
        """Append one token to a request's KV cache.

        Returns (page_id, slot_index) for where to write the new KV,
        or None if allocation failed.
        """
        with self._lock:
            entry = self._page_tables.get(request_id)
            if entry is None:
                return None

            slot_in_page = entry.num_tokens % self.config.page_size
            page_index = entry.num_tokens // self.config.page_size

            # Need a new page?
            if page_index >= len(entry.pages):
                page_id = self._alloc_page()
                if page_id is None:
                    # Try eviction
                    page_id = self._evict_lru()
                    if page_id is None:
                        return None
                entry.pages.append(page_id)

            physical_page = entry.pages[page_index]
            entry.num_tokens += 1
            self._pages[physical_page].last_access = time.time()

            return (physical_page, slot_in_page)

    def free(self, request_id: str) -> int:
        """Free all pages for a request. Returns number of pages freed."""
        with self._lock:
            entry = self._page_tables.pop(request_id, None)
            if entry is None:
                return 0

            freed = 0
            for page_id in entry.pages:
                page = self._pages[page_id]
                page.ref_count -= 1
                if page.ref_count <= 0:
                    page.allocated = False
                    page.ref_count = 0
                    self._free_pages.append(page_id)
                    freed += 1
                    self._total_frees += 1

            return freed

    def fork(self, src_request_id: str, dst_request_id: str) -> Optional[PageTableEntry]:
        """Copy-on-write fork for beam search.

        The new request shares all existing pages (ref_count++) but
        will copy-on-write when either request appends new tokens.
        """
        with self._lock:
            src = self._page_tables.get(src_request_id)
            if src is None:
                return None

            new_entry = PageTableEntry(
                request_id=dst_request_id,
                pages=list(src.pages),  # share page references
                num_tokens=src.num_tokens,
            )

            # Increment ref counts
            for page_id in new_entry.pages:
                self._pages[page_id].ref_count += 1

            self._page_tables[dst_request_id] = new_entry
            return new_entry

    # ------------------------------------------------------------------
    # Prefix caching
    # ------------------------------------------------------------------

    def try_prefix_cache(
        self,
        request_id: str,
        token_ids: List[int],
    ) -> int:
        """Check if a prefix of token_ids is already cached.

        Returns the number of tokens that were cache hits (reused pages).
        """
        with self._lock:
            entry = self._page_tables.get(request_id)
            if entry is None:
                entry = PageTableEntry(request_id=request_id)
                self._page_tables[request_id] = entry

            hits = 0
            for page_start in range(0, len(token_ids), self.config.page_size):
                page_tokens = tuple(token_ids[page_start:page_start + self.config.page_size])
                if len(page_tokens) < self.config.page_size:
                    break  # incomplete page, can't cache

                page_hash = hash(page_tokens)
                cached_page = self._prefix_cache.get(page_hash)

                if cached_page is not None and self._pages[cached_page].allocated:
                    # Cache hit — reuse the page
                    entry.pages.append(cached_page)
                    self._pages[cached_page].ref_count += 1
                    self._pages[cached_page].last_access = time.time()
                    hits += self.config.page_size
                    self._cache_hits += 1
                else:
                    # Cache miss — allocate new page and register in cache
                    page_id = self._alloc_page()
                    if page_id is None:
                        break
                    entry.pages.append(page_id)
                    self._prefix_cache[page_hash] = page_id
                    hits = 0  # reset — only contiguous prefix counts

            entry.num_tokens = len(entry.pages) * self.config.page_size
            return hits

    # ------------------------------------------------------------------
    # KV read/write (for integration with custom attention)
    # ------------------------------------------------------------------

    def write_kv(
        self,
        request_id: str,
        layer_idx: int,
        page_id: int,
        slot: int,
        key: Any,
        value: Any,
    ) -> None:
        """Write key/value vectors to a specific page slot."""
        if self._gpu_pool is not None:
            self._gpu_pool[page_id, layer_idx, 0, :, slot, :] = key
            self._gpu_pool[page_id, layer_idx, 1, :, slot, :] = value

    def read_kv(
        self,
        request_id: str,
        layer_idx: int,
    ) -> Optional[Tuple[Any, Any]]:
        """Read all KV from a request's pages for a given layer.

        Returns (keys, values) concatenated across all pages.
        """
        if self._gpu_pool is None:
            return None

        with self._lock:
            entry = self._page_tables.get(request_id)
            if entry is None or not entry.pages:
                return None

        page_ids = entry.pages
        # Gather from pool: shape [num_pages, num_kv_heads, page_size, head_dim]
        keys = self._gpu_pool[page_ids, layer_idx, 0]
        values = self._gpu_pool[page_ids, layer_idx, 1]

        # Reshape to [num_kv_heads, total_tokens, head_dim]
        # [num_pages, num_kv_heads, page_size, head_dim] → [num_kv_heads, num_pages*page_size, head_dim]
        num_kv_heads = keys.shape[1]
        keys = keys.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, self.config.head_dim)
        values = values.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, self.config.head_dim)

        # Trim to actual token count
        tokens = entry.num_tokens
        keys = keys[:, :tokens, :]
        values = values[:, :tokens, :]

        return keys, values

    # ------------------------------------------------------------------
    # HuggingFace past_key_values integration
    # ------------------------------------------------------------------

    def from_hf_cache(
        self,
        request_id: str,
        past_key_values: Any,
    ) -> None:
        """Store HuggingFace past_key_values into paged memory.

        past_key_values: tuple of (key, value) per layer.
          key/value shape: [batch, num_kv_heads, seq_len, head_dim]
        """
        if self._gpu_pool is None or past_key_values is None:
            return

        with self._lock:
            entry = self._page_tables.get(request_id)
            if entry is None:
                entry = self.allocate(request_id)

        num_layers = len(past_key_values)
        seq_len = past_key_values[0][0].shape[2]  # [batch, heads, seq, dim]

        # Ensure enough pages allocated
        pages_needed = math.ceil(seq_len / self.config.page_size)
        with self._lock:
            while len(entry.pages) < pages_needed:
                page_id = self._alloc_page()
                if page_id is None:
                    page_id = self._evict_lru()
                    if page_id is None:
                        break
                entry.pages.append(page_id)
            entry.num_tokens = seq_len

        # Write KV into pages
        for layer_idx in range(min(num_layers, self.config.num_layers)):
            k = past_key_values[layer_idx][0][0]  # [heads, seq, dim]
            v = past_key_values[layer_idx][1][0]

            for token_idx in range(seq_len):
                page_index = token_idx // self.config.page_size
                slot = token_idx % self.config.page_size
                if page_index < len(entry.pages):
                    page_id = entry.pages[page_index]
                    self._gpu_pool[page_id, layer_idx, 0, :, slot, :] = k[:, token_idx, :]
                    self._gpu_pool[page_id, layer_idx, 1, :, slot, :] = v[:, token_idx, :]

    def to_hf_cache(self, request_id: str) -> Optional[Any]:
        """Reconstruct HuggingFace past_key_values from paged memory.

        Returns tuple of (key, value) per layer.
          key/value shape: [1, num_kv_heads, seq_len, head_dim]
        Returns None if request not found or GPU pool unavailable.
        """
        if self._gpu_pool is None:
            return None

        with self._lock:
            entry = self._page_tables.get(request_id)
            if entry is None or not entry.pages:
                return None
            page_ids = list(entry.pages)
            num_tokens = entry.num_tokens

        past_key_values = []
        for layer_idx in range(self.config.num_layers):
            # Gather all pages for this layer
            # gpu_pool shape: [page_id, layer, K/V, heads, page_size, dim]
            k_pages = self._gpu_pool[page_ids, layer_idx, 0]  # [N, heads, ps, dim]
            v_pages = self._gpu_pool[page_ids, layer_idx, 1]

            # Reshape: [N, heads, page_size, dim] → [heads, N*page_size, dim]
            heads = k_pages.shape[1]
            k_flat = k_pages.permute(1, 0, 2, 3).reshape(heads, -1, self.config.head_dim)
            v_flat = v_pages.permute(1, 0, 2, 3).reshape(heads, -1, self.config.head_dim)

            # Trim to actual tokens and add batch dim
            k_flat = k_flat[:, :num_tokens, :].unsqueeze(0)  # [1, heads, seq, dim]
            v_flat = v_flat[:, :num_tokens, :].unsqueeze(0)

            past_key_values.append((k_flat, v_flat))

        return tuple(past_key_values)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _alloc_page(self) -> Optional[int]:
        """Allocate a free page. Returns page_id or None.

        If the local pool is exhausted and a VRAMLendingPool is
        available, borrows overflow pages from idle GPUs.
        """
        if self._free_pages:
            page_id = self._free_pages.pop()
            self._pages[page_id].allocated = True
            self._pages[page_id].ref_count = 1
            self._pages[page_id].last_access = time.time()
            self._total_allocations += 1
            return page_id

        # Pool exhausted — try to borrow from lending pool
        return self._borrow_overflow_page()

    def _borrow_overflow_page(self) -> Optional[int]:
        """Borrow a page from VRAMLendingPool when local pool is exhausted."""
        if self._lending_pool is None:
            return None

        try:
            # Determine which GPU to borrow from (not self)
            primary_idx = int(self.config.device.split(":")[-1]) if ":" in self.config.device else 0

            lease = self._lending_pool.borrow(
                borrower_gpu=primary_idx,
                size_bytes=self.config.page_size_bytes,
                purpose="kv_cache_overflow",
                priority=1,
            )
            if lease is None:
                return None

            # Create a new overflow page
            overflow_id = len(self._pages)
            page = PhysicalPage(
                page_id=overflow_id,
                ref_count=1,
                allocated=True,
                last_access=time.time(),
                device=f"cuda:{lease.owner_gpu}",
                is_borrowed=True,
                lease_id=lease.lease_id,
            )
            self._pages.append(page)
            self._lending_leases[lease.lease_id] = lease
            self._overflow_borrows += 1
            self._total_allocations += 1

            _logger.debug(
                "Overflow page %d borrowed from GPU %d (lease %s)",
                overflow_id, lease.owner_gpu, lease.lease_id,
            )
            return overflow_id

        except Exception as e:
            _logger.debug("Overflow borrow failed: %s", e)
            return None

    def _evict_lru(self) -> Optional[int]:
        """Evict the least-recently-used page (ref_count=1, oldest).

        Prefers to evict borrowed (overflow) pages first to return
        memory to lending GPUs.
        """
        candidates = [
            p for p in self._pages
            if p.allocated and p.ref_count <= 1
        ]
        if not candidates:
            return None

        # Prefer evicting borrowed pages first (return to owner)
        borrowed = [p for p in candidates if p.is_borrowed]
        if borrowed:
            victim = min(borrowed, key=lambda p: p.last_access)
        else:
            victim = min(candidates, key=lambda p: p.last_access)

        # Remove from any request's page table
        for entry in self._page_tables.values():
            if victim.page_id in entry.pages:
                entry.pages.remove(victim.page_id)
                break

        # Remove from prefix cache
        to_remove = [k for k, v in self._prefix_cache.items() if v == victim.page_id]
        for k in to_remove:
            del self._prefix_cache[k]

        # Release lending lease if borrowed
        if victim.is_borrowed and victim.lease_id:
            try:
                if self._lending_pool:
                    self._lending_pool.release(victim.lease_id)
                self._lending_leases.pop(victim.lease_id, None)
            except Exception:
                pass
            victim.is_borrowed = False
            victim.lease_id = None

        victim.allocated = False
        victim.ref_count = 0
        self._total_frees += 1
        return victim.page_id

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Cache statistics."""
        used = self.config.max_pages - len(self._free_pages)
        borrowed_pages = sum(1 for p in self._pages if p.is_borrowed and p.allocated)
        per_device = {}
        for p in self._pages:
            if p.allocated:
                per_device[p.device] = per_device.get(p.device, 0) + 1
        return {
            "total_pages": len(self._pages),
            "used_pages": used,
            "free_pages": len(self._free_pages),
            "borrowed_pages": borrowed_pages,
            "overflow_borrows": self._overflow_borrows,
            "utilization": used / max(len(self._pages), 1),
            "peak_usage": self._peak_usage,
            "total_allocations": self._total_allocations,
            "total_frees": self._total_frees,
            "active_requests": len(self._page_tables),
            "prefix_cache_entries": len(self._prefix_cache),
            "prefix_cache_hits": self._cache_hits,
            "page_size_tokens": self.config.page_size,
            "memory_mb": self.config.total_memory_bytes / 1e6,
            "devices": list(self._gpu_pools.keys()) if self._gpu_pools else [self.config.device],
            "pages_per_device": per_device,
            "lending_active": self._lending_pool is not None,
            "active_leases": len(self._lending_leases),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"PagedKVCache(pages={s['used_pages']}/{s['total_pages']}, "
            f"requests={s['active_requests']}, mem={s['memory_mb']:.1f}MB)"
        )


__all__ = [
    "PagedKVCacheManager",
    "PagedKVConfig",
    "PageTableEntry",
    "PhysicalPage",
]
