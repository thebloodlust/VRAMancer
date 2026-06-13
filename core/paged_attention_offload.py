"""Adapter between PagedKVCache and HierarchicalMemoryManager.

Allows the paged KV cache to evict cold pages to the CPU/NVMe tiers and
restore them on demand.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PagedAttentionOffloader:
    """Bridge cold KV pages to the hierarchical memory manager."""

    def __init__(self, kv_manager: Any, hmm: Any):
        self.kv_manager = kv_manager
        self.hmm = hmm
        self._offloaded: Dict[int, str] = {}
        self._evict_count = 0
        self._restore_count = 0

    def evict_cold_pages(self, n: int) -> int:
        if n <= 0:
            return 0
        get_lru = getattr(self.kv_manager, "get_lru_pages", None)
        free_page = getattr(self.kv_manager, "free_page", None)
        if get_lru is None or free_page is None:
            logger.debug("kv_manager has no LRU/free hooks; skipping eviction")
            return 0
        pages = get_lru(n) or []
        evicted = 0
        for page in pages:
            page_id = getattr(page, "id", None)
            tensor = getattr(page, "tensor", None)
            if page_id is None or tensor is None:
                continue
            key = f"kvpage::{page_id}"
            try:
                self.hmm.put(key, tensor.detach().to("cpu"))
                free_page(page_id)
                self._offloaded[page_id] = key
                evicted += 1
                self._evict_count += 1
            except Exception as exc:
                logger.warning("Failed to evict page %s: %s", page_id, exc)
        return evicted

    def restore_page(self, page_id: int) -> Optional[Any]:
        key = self._offloaded.pop(page_id, None)
        if key is None:
            return None
        try:
            tensor = self.hmm.get(key)
            self._restore_count += 1
            return tensor
        except Exception as exc:
            logger.warning("Failed to restore page %s: %s", page_id, exc)
            return None

    def stats(self) -> Dict[str, int]:
        return {
            "evicted_total": self._evict_count,
            "restored_total": self._restore_count,
            "in_offload": len(self._offloaded),
        }


__all__ = ["PagedAttentionOffloader"]
