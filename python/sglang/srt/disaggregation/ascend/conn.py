import concurrent.futures
import enum
import logging
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

from sglang.srt.disaggregation.ascend.dram_pool import AscendDramPool
from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.utils.network import get_local_ip_auto

logger = logging.getLogger(__name__)


class AscendStateType(str, enum.Enum):
    """DSV4-on-NPU per-pool PD components, kept out of the cross-hardware
    StateType enum. Sent via the same page-indexed path as SWA."""

    DSV4_SWA = "dsv4_swa"
    DSV4_C4 = "dsv4_c4"
    DSV4_C128 = "dsv4_c128"
    DSV4_INDEXER = "dsv4_indexer"
    DSV4_C4_STATE = "dsv4_c4_state"
    DSV4_C128_STATE = "dsv4_c128_state"


_DSV4_KVCACHE_STATE_TYPES = tuple(AscendStateType)


class AscendKVManager(MooncakeKVManager):
    def __init__(self, args, disaggregation_mode, server_args, is_mla_backend=False):
        # The DRAM receive pool must exist before super().__init__ because the
        # base constructor registers all buffers to the transfer engine and
        # pd_extension must be ready for the ZMQ registration.
        self.dram_pool = self._maybe_create_dram_pool(args, disaggregation_mode, server_args)
        if self.dram_pool is not None:
            ptrs, lens, item_lens = self.dram_pool.get_contiguous_buf_infos()
            # DRAM pool addresses ride the generic optional extension frame.
            # Page encoding (the sender path is page-indexed via
            # kv_to_page_indices and kv_item_lens are per-PAGE bytes):
            # [0, n_hbm_tokens) HBM pages, [n_hbm_tokens, ...) DRAM pages.
            args.pd_extension = {
                "dram_kv_ptrs": ptrs,
                "dram_item_lens": item_lens,
                "n_hbm_tokens": self.dram_pool.n_hbm_tokens,
            }
            logger.info(
                f"[DRAM] manager init: pool created, dram_layers={len(ptrs)} "
                f"n_hbm_pages={self.dram_pool.n_hbm_tokens} "
                f"dram_pages={self.dram_pool.size} "
                f"base=0x{self.dram_pool.base:x} dva=0x{self.dram_pool.dva:x}"
            )
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)

    @staticmethod
    def _maybe_create_dram_pool(args, disaggregation_mode, server_args):
        size_gb = getattr(server_args, "disaggregation_decode_dram_pool_size", 0)
        if size_gb <= 0 or disaggregation_mode != DisaggregationMode.DECODE:
            return None
        if not args.kv_item_lens or not args.kv_data_lens:
            return None
        # item_lens here already includes speculative draft layers (appended
        # by _init_kv_manager), so the DRAM pool mirrors target+draft. Draft
        # pools share the target pool's token indices, hence the boundary is
        # derived from the first (target) layer only. NOTE: NPU paged pools
        # report kv_item_lens as PER-PAGE bytes (page-major layout), so this
        # division yields the HBM PAGE count — the wire encoding boundary.
        # It covers the allocator's token space (size//page pages) plus the
        # pool's spare page, e.g. 103 allocatable + 1 spare = 104.
        n_hbm_tokens = args.kv_data_lens[0] // args.kv_item_lens[0]
        return AscendDramPool(
            npu_id=args.gpu_id,
            pool_size_gb=size_gb,
            page_size=args.page_size,
            item_lens=args.kv_item_lens,
            n_hbm_tokens=n_hbm_tokens,
        )

    def _requires_exact_state_index_match(self, st: StateType) -> bool:
        return (
            super()._requires_exact_state_index_match(st)
            or st in _DSV4_KVCACHE_STATE_TYPES
        )

    def init_engine(self):
        # TransferEngine initialized on ascend.
        local_ip = get_local_ip_auto()
        self.engine = AscendTransferEngine(
            hostname=local_ip,
            npu_id=self.kv_args.gpu_id,
            disaggregation_mode=self.disaggregation_mode,
        )

    def register_buffer_to_engine(self):
        # MemFabric aligns registered buffers to 2 MiB. Register everything in
        # one batch so overlapping aligned ranges from small tensors are merged
        # before they are published to the peer.
        ptrs = list(self.kv_args.kv_data_ptrs)
        lens = list(self.kv_args.kv_data_lens)
        ptrs.extend(self.kv_args.aux_data_ptrs)
        lens.extend(self.kv_args.aux_data_lens)
        for component_ptrs, component_lens in zip(
            self.kv_args.state_data_ptrs or [],
            self.kv_args.state_data_lens or [],
        ):
            ptrs.extend(component_ptrs)
            lens.extend(component_lens)
        if self.dram_pool is not None:
            # Host-memory registration automatically builds the device mapping
            # (DVA) and URMA MR: once registered the pool is remotely writable.
            dram_ptrs, dram_lens, _ = self.dram_pool.get_contiguous_buf_infos()
            ptrs.extend(dram_ptrs)
            lens.extend(dram_lens)
            logger.info(
                f"[DRAM] register: appended {len(dram_ptrs)} dram layers, "
                f"first=0x{dram_ptrs[0]:x} len={dram_lens[0]}"
            )
        # Per-component registration manifest (summary only): component
        # counts map a failing slice index in the memfabric logs back to its
        # origin component. Per-buffer details are demoted to debug; the
        # batch is 2MiB-aligned and merged inside memfabric, so a 1:1
        # sliceIdx->buffer mapping is approximate anyway.
        n_dram = len(dram_ptrs) if self.dram_pool is not None else 0
        logger.info(
            f"[DRAM] register manifest: kv={len(self.kv_args.kv_data_ptrs)} "
            f"aux={len(self.kv_args.aux_data_ptrs)} "
            f"state={[len(p) for p in (self.kv_args.state_data_ptrs or [])]} "
            f"dram={n_dram} "
            f"total={len(ptrs)}"
        )
        for name, plist, llist in (
            ("kv", self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens),
            ("aux", self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens),
        ):
            for i, (p, l) in enumerate(zip(plist, llist)):
                if l < 256 * 1024 * 1024:
                    logger.debug(
                        f"[DRAM] register manifest: {name}[{i}] 0x{int(p):x} len={l}"
                    )
        for j, (component_ptrs, component_lens) in enumerate(
            zip(self.kv_args.state_data_ptrs or [], self.kv_args.state_data_lens or [])
        ):
            for i, (p, l) in enumerate(zip(component_ptrs, component_lens)):
                logger.debug(
                    f"[DRAM] register manifest: state[{j}][{i}] 0x{int(p):x} len={l}"
                )
        if ptrs:
            self.engine.batch_register(ptrs, lens)

    def deregister_buffer_to_engine(self):
        super().deregister_buffer_to_engine()
        if self.dram_pool is not None:
            self.dram_pool.uninitialize()

    def get_mla_kv_ptrs_with_pp(
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int], state_type=None
    ) -> Tuple[List[int], List[int], int]:
        # src_kv_ptrs: k_data, v_data, index_k_data(optional)
        # dst_kv_ptrs: k_data, v_data, index_k_data(optional)
        # state_type is accepted for parity with the common disaggregation path;
        # the NPU kv_buf_groups slicing below is state-type agnostic.
        start_layer = self.kv_args.prefill_start_layer
        kv_buf_groups = getattr(self.kv_args, "kv_buf_groups", 1)
        total_kv_layers = getattr(self.kv_args, "total_kv_layers", 0)
        src_layers = len(src_kv_ptrs) // kv_buf_groups
        # When only speculative-algorithm is enabled for decode
        # the KV has one more layer than prefill.
        # The draft layer needs to be skipped.
        dst_total_layers = (
            min(len(dst_kv_ptrs) // kv_buf_groups, total_kv_layers)
            if total_kv_layers
            else len(dst_kv_ptrs) // kv_buf_groups
        )
        end_layer = start_layer + src_layers
        if src_layers == dst_total_layers:
            sliced_dst_kv_ptrs = dst_kv_ptrs
        else:
            sliced_dst_kv_ptrs = []
            for i in range(kv_buf_groups):
                layer_offset = i * dst_total_layers
                sliced_dst_kv_ptrs.extend(
                    dst_kv_ptrs[layer_offset + start_layer : layer_offset + end_layer]
                )
        layers_current_pp_stage = len(src_kv_ptrs)
        return src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage

    def _get_dram_dst_info(self, mooncake_session_id: str):
        """Decode-side DRAM pool (per-layer ptrs, global-encoding boundary).

        Fetched from the registration table instead of a new send_kvcache
        parameter, so the shared mooncake transfer_worker stays unchanged.
        """
        info = self.decode_kv_args_table.get(mooncake_session_id)
        ext = getattr(info, "pd_extension", None) if info is not None else None
        if not ext:
            logger.info(f"[DRAM] send_kvcache: no pd_extension for session {mooncake_session_id}")
            return None, None
        logger.info(
            f"[DRAM] send_kvcache: ext found, dram_layers={len(ext['dram_kv_ptrs'])} "
            f"n_hbm_tokens={ext['n_hbm_tokens']} session={mooncake_session_id}"
        )
        return ext["dram_kv_ptrs"], ext["n_hbm_tokens"]

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        dst_layer_ids: Optional[List[int]] = None,
        dst_device_kv_indices: Optional[npt.NDArray[np.int32]] = None,
        dst_kv_item_len: Optional[int] = None,
        dst_attn_tp_size: Optional[int] = None,
    ):
        if dst_device_kv_indices is not None:
            raise NotImplementedError(
                "Ascend PD transfer does not support HiSparse "
                "destination device KV indices"
            )
        self._validate_envelope_kv_layout(
            dst_kv_ptrs, dst_kv_item_len, dst_attn_tp_size
        )
        dram_dst_ptrs, n_hbm = self._get_dram_dst_info(mooncake_session_id)
        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )
        if dram_dst_ptrs is not None and n_hbm is not None:
            # One summary line per request for DRAM-pool writes (the per-block
            # details below are demoted to debug to keep this observable).
            try:
                total_dst = sum(len(bl) for bl in dst_kv_blocks)
                dram_dst = sum(
                    int((np.asarray(bl) >= n_hbm).sum()) for bl in dst_kv_blocks
                )
                if dram_dst > 0:
                    logger.info(
                        f"[DRAM] send_kvcache: writing {dram_dst}/{total_dst} dst "
                        f"tokens to DRAM pool (n_hbm={n_hbm}), "
                        f"session={mooncake_session_id}"
                    )
            except Exception:
                pass

        if self.pp_size > 1:
            if self.is_mla_backend:
                src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage = (
                    self.get_mla_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
                )
                sliced_dram_ptrs = (
                    self.get_mla_kv_ptrs_with_pp(None, dram_dst_ptrs)[1]
                    if dram_dst_ptrs
                    else [None] * layers_current_pp_stage
                )
                layers_params = [
                    (
                        src_kv_ptrs[layer_id],
                        sliced_dst_kv_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layer_id],
                        sliced_dram_ptrs[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
            else:
                (
                    src_k_ptrs,
                    src_v_ptrs,
                    dst_k_ptrs,
                    dst_v_ptrs,
                    layers_current_pp_stage,
                ) = self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
                if dram_dst_ptrs:
                    _, _, dram_k_ptrs, dram_v_ptrs, _ = self.get_mha_kv_ptrs_with_pp(
                        None, dram_dst_ptrs
                    )
                else:
                    dram_k_ptrs = dram_v_ptrs = [None] * layers_current_pp_stage

                layers_params = [
                    (
                        src_k_ptrs[layer_id],
                        dst_k_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layer_id],
                        dram_k_ptrs[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ] + [
                    (
                        src_v_ptrs[layer_id],
                        dst_v_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layers_current_pp_stage + layer_id],
                        dram_v_ptrs[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
        else:
            num_layers = len(self.kv_args.kv_data_ptrs)
            layers_params = [
                (
                    self.kv_args.kv_data_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layer_id],
                    dram_dst_ptrs[layer_id] if dram_dst_ptrs else None,
                )
                for layer_id in range(num_layers)
            ]

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int, dram_ptr: Optional[int]
        ) -> List[Tuple[int, int, int]]:
            # dst_kv_indices are globally encoded: [0, n_hbm) addresses the
            # HBM pool via dst_ptr, [n_hbm, ...) the DRAM pool via dram_ptr.
            # A contiguous block may straddle the boundary and is split so
            # every emitted block uses exactly one base pointer.
            transfer_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                first, last = int(decode_index[0]), int(decode_index[-1])
                # NOTE: parenthesized on purpose — the chained form
                # `first < n_hbm == (last < n_hbm)` means
                # `(first < n_hbm) and (n_hbm == (last < n_hbm))`, which is
                # always False and would send every block (including pure
                # HBM ones) into the split branch below, crashing with
                # IndexError on decode_index[mid].
                same_pool = (first < n_hbm) == (last < n_hbm)
                if dram_ptr is None or same_pool:
                    base, off = (dst_ptr, 0) if first < n_hbm else (dram_ptr, n_hbm)
                    if first >= n_hbm:
                        logger.debug(
                            f"[DRAM] xfer block: DRAM dst=0x{base + (first - off) * item_len:x} "
                            f"len={item_len * len(prefill_index)} idx=[{first}..{last}]"
                        )
                    transfer_blocks.append(
                        (
                            src_ptr + int(prefill_index[0]) * item_len,
                            base + (first - off) * item_len,
                            item_len * len(prefill_index),
                        )
                    )
                    continue
                mid = int(np.searchsorted(decode_index, n_hbm))
                logger.debug(
                    f"[DRAM] xfer block split at n_hbm={n_hbm}: "
                    f"hbm_part={mid} dram_part={len(prefill_index) - mid} idx=[{first}..{last}]"
                )
                transfer_blocks.append(
                    (
                        src_ptr + int(prefill_index[0]) * item_len,
                        dst_ptr + first * item_len,
                        item_len * mid,
                    )
                )
                d1 = int(decode_index[mid])
                transfer_blocks.append(
                    (
                        src_ptr + int(prefill_index[mid]) * item_len,
                        dram_ptr + (d1 - n_hbm) * item_len,
                        item_len * (len(prefill_index) - mid),
                    )
                )
            return transfer_blocks

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int, dram_ptr) -> int:
            transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len, dram_ptr)
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int, Optional[int]]]) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len, dram_ptr in layers_params:
                transfer_blocks.extend(
                    set_transfer_blocks(src_ptr, dst_ptr, item_len, dram_ptr)
                )
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(
                    process_layer,
                    src_ptr,
                    dst_ptr,
                    item_len,
                    dram_ptr,
                )
                for (src_ptr, dst_ptr, item_len, dram_ptr) in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            return process_layers(layers_params)

        return 0

    @staticmethod
    def _co_segments(a, b):
        """Cut aligned index arrays into segments contiguous in both, so each
        (src, dst) pair of a promote entry is addressable by one base+len."""
        cuts = [0]
        for i in range(1, len(a)):
            if a[i] != a[i - 1] + 1 or b[i] != b[i - 1] + 1:
                cuts.append(i)
        cuts.append(len(a))
        return [(cuts[k], cuts[k + 1] - cuts[k]) for k in range(len(cuts) - 1)]

    def promote_dram_pages(
        self, req, req_to_token_pool, hbm_kv_pool, allocator, draft_kv_pool=None
    ) -> bool:
        """Synchronously lift a request's DRAM-resident KV pages back to HBM
        (offload sparse_copy AIV kernel, DVA -> HBM), rewiring req_to_token.

        Called at transfer-commit time, before the request becomes schedulable
        (NPU attention reads HBM only). Returns False when HBM is temporarily
        short: the caller keeps the request in the transfer queue and retries
        after the scheduler retracts running requests.

        Units: the NPU paged pools are page-major and report kv_item_lens as
        PER-PAGE bytes; the wire encoding (prefill sender) is page-indexed.
        req_to_token holds token indices, where DRAM-resident tokens are the
        synthetic page-encoded values produced by AscendDramPool.alloc:
            token = (n_hbm + local_page) * page_size + intra_page
        This function therefore works on PAGES end to end: it derives DRAM
        pages from the synthetic tokens, allocates whole HBM pages and copies
        per (page-run, layer) with per-page item_lens.

        Speculative decoding (EAGLE/MTP): draft layers share the target pool's
        token indices, so they are mirrored in the same DRAM pool and must be
        promoted together with the target layers.
        """
        if self.dram_pool is None:
            return True
        # n_hbm is the HBM PAGE count (legacy attribute name); DRAM synthetic
        # tokens start at n_hbm * page_size, beyond the HBM allocator's real
        # token space, so the mask below can never catch a live HBM token.
        n_hbm = self.dram_pool.n_hbm_tokens
        page_size = int(self.dram_pool.page_size)
        length = getattr(req, "kv_committed_len", 0)
        if length <= 0:
            return True
        row = req_to_token_pool.req_to_token[req.req_pool_idx][:length]
        dram_mask = row >= n_hbm * page_size
        num = int(dram_mask.sum().item())
        if num == 0:
            return True
        import time as _time

        t0 = _time.time()
        dram_tokens = row[dram_mask].clone()
        dram_np = dram_tokens.cpu().numpy()
        # Global DRAM pages (ascending, unique) -> local pool page indices.
        gpages = np.unique(dram_np // page_size)
        lpages = gpages - n_hbm
        num_pages = len(lpages)
        # HBM allocation is page-granular: the paged allocator floors a
        # sub-page request to zero pages and returns an EMPTY tensor (not
        # None), which would slip past the None guard. Allocate whole pages;
        # the page tail is recovered page-granularly at request free time
        # (free() uniquifies by page).
        alloc_tokens = num_pages * page_size
        hbm_loc = allocator.alloc_hbm(alloc_tokens)  # force-HBM, bypasses the watermark
        if hbm_loc is None or hbm_loc.numel() < alloc_tokens:
            logger.info(
                f"[DRAM] promote deferred (HBM short): rid={req.rid}, dram_pages={num_pages}"
            )
            return False
        hbm_np = hbm_loc.cpu().numpy()
        if int(hbm_np[0]) % page_size:
            logger.error(
                f"[DRAM] promote HBM alloc not page-aligned: first={int(hbm_np[0])} "
                f"page_size={page_size} — paged allocator invariant broken"
            )
        # Page ids of the allocated HBM run (paged allocator returns
        # page-aligned contiguous tokens).
        hbm_pages = hbm_np[::page_size] // page_size
        hbm_ptrs, _, item_lens = hbm_kv_pool.get_contiguous_buf_infos()
        num_target_layers = len(hbm_ptrs)
        if draft_kv_pool is not None:
            # Draft layers are appended after the target layers, matching the
            # registration order (and the DRAM pool layout) in _init_kv_manager.
            draft_ptrs, _, draft_item_lens = draft_kv_pool.get_contiguous_buf_infos()
            hbm_ptrs = list(hbm_ptrs) + list(draft_ptrs)
            item_lens = list(item_lens) + list(draft_item_lens)
        logger.info(
            f"[DRAM] promote start: rid={req.rid} dram_tokens={num} "
            f"dram_pages={num_pages} dram_page=[{int(lpages[0])}..{int(lpages[-1])}] "
            f"hbm_page=[{int(hbm_pages[0])}..{int(hbm_pages[-1])}] "
            f"target_layers={num_target_layers} total_layers={len(item_lens)}"
        )
        entries = []
        bad_entries = []
        # Per-layer HBM capacity (PAGES) from the registered buffer geometry
        # (kv_data_lens // kv_item_lens with per-page item_lens = page count).
        hbm_caps = [
            l // il if il else 0
            for l, il in zip(self.kv_args.kv_data_lens, self.kv_args.kv_item_lens)
        ]
        for start, cnt in self._co_segments(lpages, hbm_pages):
            for layer_id, item_len in enumerate(item_lens):
                src = self.dram_pool.layer_src_dva(layer_id, int(lpages[start]))
                dst = hbm_ptrs[layer_id] + int(hbm_pages[start]) * item_len
                entries.append((src, dst, cnt * item_len))
                # Range validation in PAGE units: DRAM side within pool pages,
                # HBM side within this layer's buffer. The plog fault "MTE DDR
                # address out of range" (errcode 95) points at an OOB entry.
                if int(lpages[start]) < 0 or int(lpages[start]) + cnt > self.dram_pool.size:
                    bad_entries.append(("dram", layer_id, int(lpages[start]), cnt))
                if layer_id < len(hbm_caps) and (
                    int(hbm_pages[start]) + cnt > hbm_caps[layer_id]
                ):
                    bad_entries.append(("hbm", layer_id, int(hbm_pages[start]), cnt))
        if bad_entries:
            logger.error(
                f"[DRAM] promote RANGE VIOLATION: n_hbm_pages={n_hbm} "
                f"dram_pages={self.dram_pool.size} hbm_caps(pages) min/max="
                f"{min(hbm_caps)}/{max(hbm_caps)} "
                f"hbm_page=[{int(hbm_pages.min())}..{int(hbm_pages.max())}] "
                f"bad(first 5)={bad_entries[:5]}"
            )
        # Encoding-boundary sanity in matched units: the allocator's TOKEN
        # space must fit inside the registered HBM page buffers
        # (n_hbm pages, one of which is the pool's spare page).
        real_pool = getattr(getattr(allocator, "inner", allocator), "size", None)
        if real_pool is not None and int(real_pool) > n_hbm * page_size:
            logger.error(
                f"[DRAM] ENCODING MISMATCH: allocator tokens={real_pool} exceed "
                f"HBM page space={n_hbm * page_size} (n_hbm_pages={n_hbm}, "
                f"page_size={page_size}) — global page encoding is wrong"
            )
        # Isolation probe: sync BEFORE the AIV copy so a device fault here
        # exonerates sparse_copy (the fault would come from concurrently
        # running forward kernels, e.g. deepep/attention).
        torch.npu.synchronize()
        logger.info(
            f"[DRAM] promote pre-sync ok: rid={req.rid} committed={length} "
            f"dram={num} pages={num_pages} n_hbm_pages={n_hbm}"
        )
        ret = self.dram_pool.promote(entries, self.kv_args.gpu_id)
        if ret != 0:
            allocator.free(hbm_loc)
            return False
        # The AIV copy must land before the DRAM pages are recycled, otherwise
        # a newly-allocated transfer could race with the in-flight copy.
        torch.npu.synchronize()
        # Rewire req_to_token: k-th DRAM page -> k-th HBM page, keeping the
        # intra-page token offset. searchsorted maps each synthetic token's
        # global page back to its position in the unique ascending gpages.
        k = np.searchsorted(gpages, dram_np // page_size)
        new_tokens = hbm_pages[k] * page_size + (dram_np % page_size)
        # req_to_token is int32 while allocator indices are int64; aclnn's
        # index_put requires both sides to match or it fails with 161002
        # (AclNN_Parameter_Error dtype mismatch).
        row[dram_mask] = (
            torch.from_numpy(new_tokens.astype(np.int32)).to(row.device)
        )
        self.dram_pool.free_tokens(dram_tokens)
        total_bytes = sum(e[2] for e in entries)
        logger.info(
            f"[DRAM] promote done: rid={req.rid} tokens={num} pages={num_pages} "
            f"entries={len(entries)} bytes={total_bytes / 1e6:.1f}MB "
            f"elapsed={(_time.time() - t0) * 1e3:.1f}ms"
        )
        return True

    def _is_generic_kvcache_state_type(self, st) -> bool:
        # DSV4 per-pool components also use the page-indexed send path.
        return (
            super()._is_generic_kvcache_state_type(st)
            or st in _DSV4_KVCACHE_STATE_TYPES
        )


class AscendKVSender(MooncakeKVSender):
    pass


class AscendKVReceiver(MooncakeKVReceiver):
    pass


class AscendKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
