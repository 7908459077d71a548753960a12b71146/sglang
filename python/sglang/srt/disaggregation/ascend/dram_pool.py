"""Decode-side DRAM receive pool for Ascend PD disaggregation.

The pool is allocated by the memfabric offload component (URMA_POOL scene,
plain HVA + an independent device-visible DVA), then registered to smem_trans
so the remote prefill can write KV cache into it over DEVICE_URMA. smem_trans
itself does not allocate memory on release/1.2, hence the offload allocation.

Layout mirrors the HBM KV pool (per-layer contiguous segments, same item_len),
so prefill uses one addressing formula for both pools:
    addr = layer_base + token_index * item_len

Token indices are globally encoded by the callers of this pool:
    [0, n_hbm_tokens)             -> HBM pool token index
    [n_hbm_tokens, n_hbm + size)  -> DRAM pool token index (offset from n_hbm)
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
            f"layers={len(self.item_lens)}, num_tokens={self.size}, "
            f"capacity≈{self.size} tokens, base=0x{self.base:x}, dva=0x{self.dva:x}"
        )

    def _alloc_offload_pool(self, pool_size_gb: int) -> None:
        """Allocate the DRAM pool via offload(URMA_POOL) and derive layout."""
        # Keep the layout identical to the HBM pool: per-layer contiguous
        # segments sized size*item_len, so one addressing formula covers both.
        per_token_bytes = sum(self.item_lens)
        self.size = ((pool_size_gb * GB) // per_token_bytes // self.page_size) * self.page_size
        if self.size < self.page_size:
            raise RuntimeError(
                f"DRAM pool too small: {pool_size_gb}GB cannot hold one page "
                f"({per_token_bytes} bytes/token, page_size={self.page_size})"
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
        nbytes = self.size * per_token_bytes
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
        num_pages = self.size // self.page_size
        self.free_pages = torch.arange(num_pages, dtype=torch.int64)

    # ---- capacity bookkeeping ----

    def available_size(self) -> int:
        """Free tokens of the DRAM pool (receive capacity)."""
        return len(self.free_pages) * self.page_size

    def allocated_tokens(self) -> int:
        """DRAM tokens not yet promoted back to HBM (the promote debt)."""
        return self.size - self.available_size()

    # ---- page allocation (page-granular free list, mirrors paged allocator) ----

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Allocate `need_size` tokens, returning globally-encoded indices.

        Returns None when the pool is short (caller falls back to the HBM
        allocator's original behavior).
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
        out = (pages[:, None] * self.page_size + in_page).reshape(-1)
        logger.info(
            f"[DRAM] pool alloc: need={need_size} pages={num_pages} "
            f"global_range=[{int(out[0]) + self.n_hbm_tokens}..{int(out[-1]) + self.n_hbm_tokens}] "
            f"free_pages {len(self.free_pages) + num_pages}->{len(self.free_pages)}"
        )
        return (out + self.n_hbm_tokens).to(self.device)

    def free_tokens(self, tokens: torch.Tensor) -> int:
        """Return globally-encoded DRAM tokens to the pool. Returns count."""
        dram = tokens[tokens >= self.n_hbm_tokens]
        if dram.numel() == 0:
            return 0
        pages = torch.unique(((dram - self.n_hbm_tokens).cpu()) // self.page_size)
        self.free_pages = torch.cat([pages.to(torch.int64), self.free_pages])
        logger.info(
            f"[DRAM] pool free: tokens={dram.numel()} pages={len(pages)} "
            f"free_pages->{len(self.free_pages)} allocated_tokens={self.allocated_tokens()}"
        )
        return dram.numel()

    # ---- transfer registration / addressing helpers ----

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """Per-layer (hva, len, item_len), layout-compatible with the HBM pool."""
        ptrs = [self.base + off for off in self.layer_offsets]
        lens = [self.size * item_len for item_len in self.item_lens]
        return ptrs, lens, list(self.item_lens)

    def layer_src_dva(self, layer_id: int, token_offset_in_pool: int) -> int:
        """DVA source address for AIV promote of one token in one layer."""
        return self.dva + self.layer_offsets[layer_id] + token_offset_in_pool * self.item_lens[layer_id]

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
        logger.info(
            f"[DRAM] sparse_copy: entries={len(entries)} total_bytes={total_bytes} "
            f"dev={device_id}"
        )
        dev = torch.device("npu", device_id)
        src_ptrs = torch.tensor([e[0] for e in entries], dtype=torch.int64).to(dev)
        dst_ptrs = torch.tensor([e[1] for e in entries], dtype=torch.int64).to(dev)
        lens = torch.tensor([e[2] for e in entries], dtype=torch.int32).to(dev)
        cnt = torch.tensor(len(entries), dtype=torch.int32).to(dev)
        ret = mf_offload.sparse_copy(
            src_ptrs.data_ptr(), dst_ptrs.data_ptr(), lens.data_ptr(), cnt.data_ptr(), device_id
        )
        if ret != 0:
            logger.error(f"sparse_copy promote failed, ret={ret}, entries={len(entries)}")
        return ret

    def uninitialize(self) -> None:
        if getattr(self, "base", 0):
            mf_offload.free(self.base, 0)
            self.base = 0
            mf_offload.uninitialize()
