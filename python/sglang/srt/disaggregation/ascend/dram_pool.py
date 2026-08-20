"""Decode-side DRAM receive pool for Ascend PD disaggregation.

The pool is allocated by the memfabric offload component (URMA_POOL scene,
plain HVA + an independent device-visible DVA), then registered to smem_trans
so the remote prefill can write KV cache into it over DEVICE_URMA. smem_trans
itself does not allocate memory on release/1.2, hence the offload allocation.

Layout mirrors the HBM KV pool (per-layer contiguous segments, same item_len),
so prefill uses one addressing formula for both pools:
    addr = layer_base + page_index * item_len

IMPORTANT units: the NPU paged pools ((layer, size//page+1, page_size, ...)
page-major layout) report kv_item_lens as PER-PAGE bytes and the PD transfer
path is page-indexed (kv_to_page_indices). Everything in this class is
therefore PAGE-native:

Global page encoding (same currency the prefill sender uses):
    [0, n_hbm_tokens)                  -> HBM pool page (n_hbm_tokens is the
                                          HBM PAGE count despite the legacy
                                          attribute name, kept for the
                                          pd_extension wire key compatibility)
    [n_hbm_tokens, n_hbm_tokens + size)-> DRAM pool page (size = DRAM pages)

req_to_token stores TOKEN indices, so DRAM-resident tokens are encoded as
synthetic token values:
    token = global_page * page_size + intra_page_offset
These start at n_hbm_tokens * page_size, which is beyond the HBM allocator's
real token space, so they can never collide with live HBM tokens, and
kv_to_page_indices() recovers the global page on the prefill side.
"""

import logging
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

try:
    from memfabric_hybrid import offload as mf_offload

    offload_import_error = None
except ImportError as e:  # pragma: no cover - depends on deployment
    mf_offload = None
    offload_import_error = e

GB = 1 << 30


class AscendDramPool:
    def __init__(
        self,
        npu_id: int,
        pool_size_gb: int,
        page_size: int,
        item_lens: List[int],
        n_hbm_tokens: int,
    ):
        if offload_import_error is not None:
            raise RuntimeError(
                "DRAM receive pool requires memfabric_hybrid with the offload "
                "component, please install it first"
            ) from offload_import_error
        self.npu_id = npu_id
        self.page_size = page_size
        self.item_lens = list(item_lens)
        self.n_hbm_tokens = n_hbm_tokens
        self.device = f"npu:{npu_id}"

        self._alloc_offload_pool(pool_size_gb)
        self._init_free_list()
        logger.info(
            f"[DRAM pool] size={pool_size_gb}GB, page_size={page_size}, "
            f"layers={len(self.item_lens)}, pages={self.size}, "
            f"capacity≈{self.size * self.page_size} tokens, "
            f"base=0x{self.base:x}, dva=0x{self.dva:x}"
        )

    def _alloc_offload_pool(self, pool_size_gb: int) -> None:
        """Allocate the DRAM pool via offload(URMA_POOL) and derive layout."""
        # Keep the layout identical to the HBM pool: per-layer contiguous
        # segments sized size*item_len, so one addressing formula covers both.
        # item_lens are PER-PAGE bytes (NPU page-major pools), so
        # sum(item_lens) is the per-page cost and self.size is a PAGE count.
        per_page_bytes = sum(self.item_lens)
        self.size = (pool_size_gb * GB) // per_page_bytes
        if self.size < 1:
            raise RuntimeError(
                f"DRAM pool too small: {pool_size_gb}GB cannot hold one page "
                f"({per_page_bytes} bytes/page, page_size={self.page_size})"
            )

        cfg = mf_offload.OffloadConfig()
        cfg.device_id = self.npu_id
        cfg.reserve_size = pool_size_gb * GB
        cfg.alloc_size = pool_size_gb * GB
        cfg.world_size = 1
        cfg.rank_id = 0
        cfg.scene = mf_offload.Scene.LOCAL
        cfg.flags = mf_offload.OFFLOAD_FLAG_URMA_POOL
        ret = mf_offload.initialize(cfg)
        if ret != 0:
            raise RuntimeError(f"offload.initialize failed: {ret}")
        nbytes = self.size * per_page_bytes
        self.base = mf_offload.malloc(nbytes, 0)
        if self.base == 0:
            raise RuntimeError(f"offload.malloc failed for {nbytes} bytes")
        self.dva = mf_offload.get_dva(self.base)
        if self.dva == 0:
            raise RuntimeError("offload.get_dva failed")

        self.layer_offsets: List[int] = []
        offset = 0
        for item_len in self.item_lens:
            self.layer_offsets.append(offset)
            offset += self.size * item_len

    def _init_free_list(self) -> None:
        # self.size is a PAGE count (see module docstring).
        self.free_pages = torch.arange(self.size, dtype=torch.int64)

    # ---- capacity bookkeeping ----

    def available_size(self) -> int:
        """Free tokens of the DRAM pool (receive capacity)."""
        return len(self.free_pages) * self.page_size

    def allocated_tokens(self) -> int:
        """DRAM tokens not yet promoted back to HBM (the promote debt)."""
        return self.size * self.page_size - self.available_size()

    # ---- page allocation (page-granular free list, mirrors paged allocator) ----

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Allocate `need_size` tokens, returning synthetic token indices.

        The returned values are page-encoded token indices:
            token = (n_hbm_tokens + local_page) * page_size + intra_page
        so they sort beyond the HBM allocator's real token space (no
        collision with live HBM tokens) and the prefill sender recovers the
        DRAM global page via kv_to_page_indices(). Returns None when the
        pool is short (caller falls back to the HBM allocator).
        """
        num_pages = (need_size + self.page_size - 1) // self.page_size
        if num_pages > len(self.free_pages):
            logger.info(
                f"[DRAM] pool alloc FAILED short: need={need_size} tokens "
                f"({num_pages} pages), free_pages={len(self.free_pages)}"
            )
            return None
        pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        in_page = torch.arange(self.page_size, dtype=torch.int64)
        # (n_hbm + page) * page_size keeps the intra-page offset and makes
        # token // page_size the global page the sender addresses.
        out = (pages[:, None] + self.n_hbm_tokens) * self.page_size + in_page
        out = out.reshape(-1)[:need_size]
        # Allocation log: the single source of truth for "who took what".
        logger.info(
            f"[DRAM] alloc: tokens={need_size} pages={num_pages} "
            f"global_pages=[{int(pages[0]) + self.n_hbm_tokens}.."
            f"{int(pages[-1]) + self.n_hbm_tokens}] "
            f"free_pages->{len(self.free_pages)}"
        )
        return out.to(self.device)

    def free_tokens(self, tokens: torch.Tensor) -> int:
        """Return page-encoded DRAM tokens to the pool. Returns count."""
        # DRAM synthetic tokens start at n_hbm_tokens * page_size (the HBM
        # allocator's real token space ends below that), NOT at n_hbm_tokens
        # (which is a page count).
        dram = tokens[tokens >= self.n_hbm_tokens * self.page_size]
        if dram.numel() == 0:
            return 0
        pages = torch.unique((dram.cpu() // self.page_size) - self.n_hbm_tokens)
        self.free_pages = torch.cat([pages.to(torch.int64), self.free_pages])
        # Release log: the single source of truth for "who returned what".
        logger.info(
            f"[DRAM] free: tokens={dram.numel()} pages={len(pages)} "
            f"free_pages->{len(self.free_pages)}"
        )
        return dram.numel()

    # ---- transfer registration / addressing helpers ----

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """Per-layer (hva, len, item_len), layout-compatible with the HBM pool."""
        ptrs = [self.base + off for off in self.layer_offsets]
        lens = [self.size * item_len for item_len in self.item_lens]
        return ptrs, lens, list(self.item_lens)

    def layer_src_dva(self, layer_id: int, page_offset_in_pool: int) -> int:
        """DVA source address for AIV promote of one page in one layer."""
        return (
            self.dva
            + self.layer_offsets[layer_id]
            + page_offset_in_pool * self.item_lens[layer_id]
        )

    def promote(
        self,
        entries: List[Tuple[int, int, int]],
        device_id: int,
    ) -> int:
        """Batch DRAM(DVA)->HBM copy via the offload sparse_copy AIV kernel.

        entries: (src_dva, dst_hbm_addr, nbytes) triplets gathered per layer.
        """
        if not entries:
            return 0
        total_bytes = sum(e[2] for e in entries)
        dev = torch.device("npu", device_id)
        src_ptrs = torch.tensor([e[0] for e in entries], dtype=torch.int64).to(dev)
        dst_ptrs = torch.tensor([e[1] for e in entries], dtype=torch.int64).to(dev)
        lens = torch.tensor([e[2] for e in entries], dtype=torch.int32).to(dev)
        cnt = torch.tensor(len(entries), dtype=torch.int32).to(dev)
        # The python wrapper takes device tensors + torch.device and derives
        # data_ptr()/device index itself (mf_acc_offload.sparse_copy).
        ret = mf_offload.sparse_copy(src_ptrs, dst_ptrs, lens, cnt, dev)
        if ret != 0:
            logger.error(f"sparse_copy promote failed, ret={ret}, entries={len(entries)}")
        return ret

    def uninitialize(self) -> None:
        if getattr(self, "base", 0):
            mf_offload.free(self.base, 0)
            self.base = 0
            mf_offload.uninitialize()
