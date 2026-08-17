"""Dual-pool (HBM + DRAM receive pool) allocator wrapper for Ascend PD decode.

Composition decorator around the existing token_to_kv_pool_allocator: only
`alloc` (the PD prealloc entry, single-pool path since decode-side radix cache
is mutually exclusive with the DRAM pool), `free` (split by pool) and
`available_size` (receive semantics) are intercepted; everything else is
forwarded to the inner allocator untouched via __getattr__, so the inner
allocator never sees globally-encoded DRAM indices in its own state
(merge_and_sort_free semantics stay valid).

Watermark (design D8, derived, no new tunable):
    budget = HBM_free - pending_promote_tokens - num_reserved_decode_tokens
    budget >= need  -> all HBM (single hop, light load)
    budget <  need  -> whole request lands in DRAM (heavy load: HBM is left
                       entirely to running requests and promotes)
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class AscendDramFallbackAllocator:
    def __init__(self, inner, dram_pool, num_reserved_decode_tokens: int):
        self.inner = inner
        self.dram_pool = dram_pool
        self.reserved_tokens = num_reserved_decode_tokens
        # Units: dram_pool.n_hbm_tokens is the HBM PAGE count (NPU paged
        # pools report kv_item_lens as per-PAGE bytes; the PD wire encoding
        # is page-indexed) while inner.size is the allocator's TOKEN count.
        # The allocator must fit inside the registered HBM pages:
        #   inner.size <= n_hbm_tokens * page_size
        # (n_hbm covers the allocator pages plus the pool's spare page).
        # Do NOT "fix" n_hbm to inner.size — that mixes tokens into the page
        # boundary and breaks the prefill sender's DRAM addressing.
        n_hbm_tokens = int(dram_pool.n_hbm_tokens)
        page_size = int(dram_pool.page_size)
        inner_size = int(getattr(inner, "size", 0) or 0)
        if inner_size > n_hbm_tokens * page_size:
            logger.error(
                f"[DRAM] allocator size={inner_size} exceeds HBM page space "
                f"{n_hbm_tokens * page_size} (n_hbm_pages={n_hbm_tokens}, "
                f"page_size={page_size}) — page encoding boundary is wrong"
            )

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        budget = (
            self.inner.available_size()
            - self.dram_pool.allocated_tokens()
            - self.reserved_tokens
        )
        if budget >= need_size:
            logger.info(
                f"[DRAM] wrapper alloc->HBM: need={need_size} budget={budget} "
                f"hbm_free={self.inner.available_size()} "
                f"dram_debt={self.dram_pool.allocated_tokens()}"
            )
            return self.inner.alloc(need_size)
        logger.info(
            f"[DRAM] wrapper alloc watermark tripped: need={need_size} budget={budget} "
            f"hbm_free={self.inner.available_size()} "
            f"dram_free={self.dram_pool.available_size()} "
            f"dram_debt={self.dram_pool.allocated_tokens()}"
        )
        # Watermark tripped: land the whole request in DRAM so the remaining
        # HBM pages are fully reserved for running decode growth and promotes.
        loc = self.dram_pool.alloc(need_size)
        if loc is not None:
            return loc
        # DRAM pool is full too: keep the original single-pool behavior
        # (caller queues the request / fails prealloc as before).
        logger.info("[DRAM] wrapper alloc: DRAM full too, fallback to HBM alloc")
        return self.inner.alloc(need_size)

    def alloc_extend(
        self,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens: int,
        num_new_pages=None,
        **kwargs,
    ):
        # Paged (page_size>1) prealloc path. With the decode radix cache
        # mutually exclusive with the DRAM pool, prefix is always 0 here, so
        # falling back to DRAM is equivalent to allocating fresh pages.
        if int(prefix_lens_cpu[0]) == 0:
            budget = (
                self.inner.available_size()
                - self.dram_pool.allocated_tokens()
                - self.reserved_tokens
            )
            if budget < extend_num_tokens:
                logger.info(
                    f"[DRAM] wrapper alloc_extend watermark: need={extend_num_tokens} "
                    f"budget={budget} hbm_free={self.inner.available_size()}"
                )
                loc = self.dram_pool.alloc(extend_num_tokens)
                if loc is not None:
                    return loc
        return self.inner.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages=num_new_pages,
            **kwargs,
        )

    def free(self, free_index: torch.Tensor) -> None:
        if free_index.numel() == 0:
            return
        # Split by pool so the inner allocator only ever frees pure HBM pages.
        # DRAM synthetic tokens start at n_hbm_tokens * page_size (the HBM
        # allocator's real token space ends below that), so the split
        # threshold must be page-scaled — n_hbm_tokens alone is a PAGE count.
        dram_n = self.dram_pool.free_tokens(free_index)
        if dram_n == free_index.numel():
            logger.info(f"[DRAM] wrapper free: all-DRAM tokens={dram_n}")
            return
        hbm = free_index[
            free_index < self.dram_pool.n_hbm_tokens * self.dram_pool.page_size
        ]
        logger.info(
            f"[DRAM] wrapper free split: total={free_index.numel()} dram={dram_n} hbm={hbm.numel()}"
        )
        if hbm.numel() > 0:
            self.inner.free(hbm)

    def available_size(self) -> int:
        # Growth/general semantics (decode growth admission, retract checks):
        # HBM minus the outstanding promote debt. Deliberately does NOT add
        # the DRAM capacity - running decode growth must be pure HBM.
        return max(
            0, self.inner.available_size() - self.dram_pool.allocated_tokens()
        )

    def available_size_for_prealloc(self) -> int:
        # Receive semantics, only consumed by PD prealloc admission: HBM
        # (debt deducted) plus the DRAM receive capacity, so heavy-load
        # requests can be admitted into the DRAM pool.
        return self.available_size() + self.dram_pool.available_size()

    def alloc_hbm(self, need_size: int):
        """Force-HBM allocation (promote target pages), bypassing the watermark."""
        return self.inner.alloc(need_size)

    def __getattr__(self, name):
        # Forward everything else (page_size, get_kvcache, alloc_decode,
        # alloc_extend, merge_and_sort_free, ...) to the inner allocator
        # unchanged. This keeps the wrapper transparent for paths that must
        # stay pure-HBM, and the inner allocator never sees DRAM indices
        # because free() splits by pool first.
        return getattr(self.inner, name)
