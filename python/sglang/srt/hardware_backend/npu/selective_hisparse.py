"""Selective HiSparse: offload selected DSA layers' KV Cache from HBM to Host DRAM.

This module implements the coordinator, host pool, ACL callback, and validation
for the GLM-5.2 NPU selective HiSparse feature.  See design document
``NPU_GLM52_SELECTIVE_HISPARSE_DESIGN_V2.md`` for the full specification.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import torch

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECORD_BYTES = 656
VERIFY_WIDTH = 6
DEFAULT_TOPK = 2048
DSA_KV_QUANT_TILE_SIZE = 128

# ---------------------------------------------------------------------------
# ACL Host Callback (mirrors sglang-kv-offload-simpleImpel/host_callback.py)
# ---------------------------------------------------------------------------

_REPORTERS: dict[int, "NPUSelectiveACLCallback"] = {}


def _get_stream_ptr(stream) -> int:
    for attr in ("npu_stream", "stream_ptr", "cuda_stream"):
        if hasattr(stream, attr):
            value = getattr(stream, attr)
            value = value() if callable(value) else value
            return int(value)
    raise RuntimeError("cannot get raw NPU stream ptr")


class NPUSelectiveACLCallback:
    """ACL host-callback thread for async H2D under NPUGraph.

    A daemon thread per device loops on ``acl.rt.process_report(100)`` so that
    ``sparse_copy`` H2D async copies work correctly inside captured
    NPU graphs.
    """

    def __init__(self, device_index: int):
        self.device_index = device_index
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.thread_id: Optional[int] = None
        self.streams: set[int] = set()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.ready.wait()

    def _loop(self):
        torch.npu.set_device(self.device_index)
        try:
            import acl

            acl.rt.set_device(self.device_index)
        except Exception:
            pass
        self.thread_id = threading.current_thread().ident
        self.ready.set()
        while not self.stop.is_set():
            try:
                import acl

                acl.rt.process_report(100)
            except Exception:
                break

    def subscribe(self, stream):
        stream_ptr = _get_stream_ptr(stream)
        if stream_ptr in self.streams:
            return
        try:
            import acl

            ret = acl.rt.subscribe_report(self.thread_id, stream_ptr)
            if ret != 0:
                raise RuntimeError(
                    f"acl.rt.subscribe_report failed, ret={ret}, "
                    f"thread_id={self.thread_id}, stream_ptr={stream_ptr}"
                )
        except ImportError:
            logger.warning(
                "acl module not available; ACL host callback disabled. "
                "Graph-mode H2D may not work."
            )
        self.streams.add(stream_ptr)

    def close(self):
        for stream_ptr in list(self.streams):
            try:
                import acl

                acl.rt.unsubscribe_report(self.thread_id, stream_ptr)
            except Exception:
                pass
        self.streams.clear()
        self.stop.set()
        self.thread.join(timeout=1)


def register_npu_selective_callback_stream(stream, device):
    """Register *stream* with the per-device ACL callback thread."""
    device_index = device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    reporter = _REPORTERS.get(device_index)
    if reporter is None:
        reporter = NPUSelectiveACLCallback(device_index)
        _REPORTERS[device_index] = reporter
    reporter.subscribe(stream)


# ---------------------------------------------------------------------------
# SelectiveHostKVPool — memfabric URMA registered Host memory
# ---------------------------------------------------------------------------


class SelectiveHostKVPool:
    """memfabric URMA registered Host memory for selected layer KV.

    Each selected layer gets a contiguous ``[num_slots, record_bytes]`` region
    in Host DRAM.  The region is registered with memfabric's offload URMA pool
    so that it has both a Host Virtual Address (HVA) and a Device-Visible
    Address (DVA).  The DVA allows NPU DMA to read/write Host memory directly
    via ``mf_offload.sparse_copy`` (DVA scatter-copy).
    """

    def __init__(
        self,
        layer_ids: Sequence[int],
        num_slots: int,
        record_bytes: int,
        npu_id: int,
    ):
        self.layer_ids = sorted(layer_ids)
        self.num_slots = num_slots
        self.record_bytes = record_bytes
        self.npu_id = npu_id
        self.device = f"npu:{npu_id}"

        per_layer_bytes = num_slots * record_bytes
        total_bytes = per_layer_bytes * len(self.layer_ids)

        self.base_hva: int = 0
        self.dva: int = 0
        self.layer_offsets: dict[int, int] = {}

        self._init_memfabric(total_bytes, per_layer_bytes)
        self._init_sentinel()

        # Build CPU tensor views for fallback indexed copy
        self._host_tensor_views: dict[int, torch.Tensor] = {}
        for lid in self.layer_ids:
            self._host_tensor_views[lid] = self._create_host_tensor_view(lid)

    def _init_memfabric(self, total_bytes: int, per_layer_bytes: int):
        try:
            from memfabric_hybrid import offload as mf_offload
        except ImportError:
            logger.warning(
                "memfabric_hybrid.offload not available; falling back to "
                "ctypes shared memory for SelectiveHostKVPool."
            )
            self._init_ctypes_fallback(total_bytes, per_layer_bytes)
            return

        cfg = mf_offload.OffloadConfig()
        cfg.device_id = self.npu_id
        cfg.reserve_size = total_bytes
        cfg.alloc_size = total_bytes
        cfg.world_size = 1
        cfg.rank_id = 0
        cfg.scene = mf_offload.Scene.LOCAL
        cfg.flags = mf_offload.OFFLOAD_FLAG_URMA_POOL
        ret = mf_offload.initialize(cfg)
        if ret != 0:
            raise RuntimeError(f"mf_offload.initialize failed: ret={ret}")

        self.base_hva = mf_offload.malloc(total_bytes, 0)
        if not isinstance(self.base_hva, int) or self.base_hva == 0:
            raise RuntimeError(
                f"mf_offload.malloc failed for {total_bytes} bytes "
                f"(ret={self.base_hva!r})"
            )
        self.dva = mf_offload.get_dva(self.base_hva)
        if self.dva == 0:
            raise RuntimeError("mf_offload.get_dva failed")

        for i, lid in enumerate(self.layer_ids):
            self.layer_offsets[lid] = i * per_layer_bytes

        logger.info(
            f"SelectiveHostKVPool: allocated {total_bytes / 1e9:.2f} GB "
            f"Host DRAM for {len(self.layer_ids)} selected layers, "
            f"{self.num_slots} slots/layer, DVA=0x{self.dva:x}"
        )

    def _init_ctypes_fallback(self, total_bytes: int, per_layer_bytes: int):
        """Fallback when memfabric is unavailable (single-node dev/test)."""
        self._ctypes_buf = (ctypes.c_uint8 * total_bytes)()
        self.base_hva = ctypes.addressof(self._ctypes_buf)
        self.dva = self.base_hva
        for i, lid in enumerate(self.layer_ids):
            self.layer_offsets[lid] = i * per_layer_bytes
        logger.info(
            f"SelectiveHostKVPool (ctypes fallback): allocated "
            f"{total_bytes / 1e9:.2f} GB for {len(self.layer_ids)} layers"
        )

    def _init_sentinel(self):
        """Zero-initialize the sentinel record (slot 0 of first layer)."""
        if not self.layer_ids:
            return
        first = self.layer_ids[0]
        offset = self.layer_offsets[first]
        buf = (ctypes.c_uint8 * self.record_bytes).from_address(
            self.base_hva + offset
        )
        for i in range(self.record_bytes):
            buf[i] = 0

    @property
    def host_sentinel_loc(self) -> int:
        """The slot index used as sentinel (always 0, the extra page row)."""
        return 0

    def layer_hva(self, layer_id: int) -> int:
        return self.base_hva + self.layer_offsets[layer_id]

    def layer_dva(self, layer_id: int) -> int:
        return self.dva + self.layer_offsets[layer_id]

    def layer_bytes(self, layer_id: int) -> int:
        return self.num_slots * self.record_bytes

    def _create_host_tensor_view(self, layer_id: int) -> torch.Tensor:
        """Wrap memfabric HVA as a CPU uint8 tensor for fallback indexed copy."""
        hva = self.layer_hva(layer_id)
        nbytes = self.num_slots * self.record_bytes
        buf = (ctypes.c_uint8 * nbytes).from_address(hva)
        return torch.frombuffer(buf, dtype=torch.uint8).reshape(
            self.num_slots, self.record_bytes
        )

    def get_host_tensor(self, layer_id: int) -> torch.Tensor:
        return self._host_tensor_views[layer_id]

    def get_contiguous_buf_infos(
        self, layer_ids: Sequence[int]
    ) -> tuple[list[int], list[int], list[int]]:
        """Return (ptrs, lens, item_lens) for the given selected layers."""
        ptrs: list[int] = []
        lens: list[int] = []
        item_lens: list[int] = []
        for lid in layer_ids:
            ptrs.append(self.layer_hva(lid))
            lens.append(self.layer_bytes(lid))
            item_lens.append(self.record_bytes)
        return ptrs, lens, item_lens


# ---------------------------------------------------------------------------
# SelectedPrefetchState
# ---------------------------------------------------------------------------


@dataclass
class SelectedPrefetchState:
    """State carried from anchor prefetch to selected-layer attention."""

    selected_layer_id: int
    real_batch: int
    real_tokens: int
    gather_locs: torch.Tensor  # [T, K] int64 — Host locs for H2D
    valid_mask: torch.Tensor  # [T, K] bool — historical entries
    current_source_row: torch.Tensor  # [T, K] int64 — current row or -1
    h2d_done: Optional[object]  # torch.npu.Event (eager only)


# ---------------------------------------------------------------------------
# NPUSelectiveHiSparseCoordinator
# ---------------------------------------------------------------------------


class NPUSelectiveHiSparseCoordinator:
    """Orchestrates H2D prefetch, current-patch, unpack+SFA, and D2H backup
    for all selected layers.

    Lifecycle per verify round:
    1. anchor layer (L-3) computes Top-K → ``maybe_start_prefetch()``
    2. layers L-3+1, L-3+2 execute normally (coverage window)
    3. selected layer L: ``set_kv_buffer()`` → ``publish_new_packed_kv()``
    4. selected layer L: ``run_selected_attention()`` → wait H2D, patch,
       unpack, SFA
    5. backup stream: D2H new KV → Host
    """

    def __init__(
        self,
        pool: "SelectiveHostKVPool",
        req_to_token_pool: "ReqToTokenPool",
        selected_layer_ids: Sequence[int],
        local_batch_capacity: int,
        verify_width: int = VERIFY_WIDTH,
        topk: int = DEFAULT_TOPK,
        record_bytes: int = RECORD_BYTES,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
    ):
        self.pool = pool
        self.req_to_token_pool = req_to_token_pool
        self.selected_layer_ids: frozenset[int] = frozenset(selected_layer_ids)
        self.selected_layer_ids_sorted: tuple[int, ...] = tuple(
            sorted(selected_layer_ids)
        )
        self.local_batch_capacity = local_batch_capacity
        self.verify_width = verify_width
        self.topk = topk
        self.record_bytes = record_bytes
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.device = torch.device(pool.device)

        # anchor → selected mapping (anchor = selected - 3)
        self._anchor_to_selected: dict[int, int] = {}
        for sel in self.selected_layer_ids_sorted:
            anchor = sel - 3
            if anchor >= 0:
                self._anchor_to_selected[anchor] = sel

        # Capacities
        self.tcap = local_batch_capacity * verify_width  # max tokens
        self.rcap = self.tcap * topk  # max records

        # Streams and events
        self.prefetch_stream = torch.npu.Stream(device=self.device)
        self.backup_stream = torch.npu.Stream(device=self.device)
        self._initial_event = torch.npu.Event()
        self._initial_event.record()

        self.staging_free_event: torch.npu.Event = self._initial_event
        self.backup_done_event: dict[int, torch.npu.Event] = {}
        self._eager_async_pending = False
        # E7 (C6 probe): capture-time state for deferred D2H launch —
        # holds (layer_id, N) of the selected layer whose D2H kernel has
        # been captured-later than its scratch copy_ (see
        # publish_new_packed_kv). Python-only, consumed during capture.
        self._pending_d2h: Optional[tuple] = None

        # Device staging buffers (allocated once)
        self._alloc_staging_buffers()

        # Active prefetch state
        self.active_prefetch: Optional[SelectedPrefetchState] = None

        # Latest D2H event across ALL selected layers (eager mode only):
        # new_packed_scratch is shared, so overwriting it must wait for the
        # latest in-flight D2H read. backup_stream is single-stream
        # serialized, so the latest event subsumes all earlier ones.
        self._last_backup_event: Optional[torch.npu.Event] = None

        # Graph mode state
        self._graph_mode = False
        self._selective_real_tokens = torch.tensor(
            0, dtype=torch.int32, device=self.device
        )

        logger.info(
            f"NPUSelectiveHiSparseCoordinator: selected={self.selected_layer_ids_sorted}, "
            f"anchors={sorted(self._anchor_to_selected.keys())}, "
            f"Bcap={local_batch_capacity}, Tcap={self.tcap}, Rcap={self.rcap}"
        )

    # === public properties ===

    def is_selected(self, layer_id: int) -> bool:
        return layer_id in self.selected_layer_ids

    @property
    def anchor_to_selected(self) -> dict[int, int]:
        return self._anchor_to_selected

    @property
    def num_selected(self) -> int:
        return len(self.selected_layer_ids)

    # === staging allocation ===

    def _alloc_staging_buffers(self):
        """Pre-allocate all Device staging/workspace buffers."""
        T = self.tcap
        K = self.topk
        R = self.record_bytes

        # Staging ping-pong: slices alternating by the layer's position in
        # the selected list (~193MB each at Tcap=144, fixed, layer count
        # independent) — same shape as the R9 workspace ping-pong below.
        # Safety: adjacent selected layers are 4 apart (5, 9, 13, ...) and
        # use adjacent sets; a set is re-written only 4*n_slices model
        # layers later. E10 (C-set probe): 19L dose curve is linear
        # (~1.25pt/layer) with all python timing levers dead — remaining
        # suspect is a per-layer resource; this raises isolation from 2
        # (rewrite distance 8 layers, open window at 19L queue depth) to
        # env-configurable. pool_configurator reads the same env for bias.
        _staging_slices = int(
            os.getenv("SGLANG_SELECTIVE_STAGING_SLICES", "2")
        )
        self._staging_slices = _staging_slices
        self.packed_staging_all = torch.zeros(
            _staging_slices,
            T,
            K,
            R,
            dtype=torch.uint8,
            device=self.device,
        )
        # R9-final (ping-pong): the unpack/workspace family was the last
        # cross-layer shared resource on the compute path — layer L's SFA
        # (compute queue) read unpack_k_* while layer L+4's unpack chain
        # (torch ops) already overwrote them, one window per adjacent layer
        # pair per replay (linear dose-response, confirmed by the 4-layer
        # A/B: shared 0.88 vs per-layer 0.92).
        # Production shape: TWO sets alternating by selected-layer parity
        # (buffer = ws[_si % 2]) instead of one per layer. Safety: adjacent
        # selected layers are 4 apart (5, 9, 13, ...), so when layer L's SFA
        # still reads set p, the earliest writer is layer L+4 which uses set
        # p^1 — read and write never share a set. Memory: +510MB total,
        # independent of layer count (vs 510MB/layer for the debug R9).
        # E11 (C-set probe, env-gated): same isolation question as staging
        # (E10, innocent). 19L linear dose curve with all timing levers
        # dead points at a per-layer resource; this raises the workspace
        # isolation from 2 (rewrite distance 8 model layers) to
        # env-configurable. pool_configurator reads the same env for bias.
        self._n_ws = int(
            os.getenv("SGLANG_SELECTIVE_UNPACK_WS", "2")
        )
        n_ws = self._n_ws
        self.unpack_k_nope_bf16_all = torch.zeros(
            n_ws, T, K, self.kv_lora_rank, dtype=torch.bfloat16, device=self.device
        )
        self.unpack_k_rope_bf16_all = torch.zeros(
            n_ws, T, K, self.qk_rope_head_dim, dtype=torch.bfloat16, device=self.device
        )
        # Fixed-address scratch holding the current forward's newly packed
        # KV: publish_new_packed_kv copies into it and the selected layer's
        # current-patch reads from it. Replaces the Python-held
        # ``_published_packed`` reference to a graph-pool tensor with a
        # stable pre-allocated buffer (already counted in the pool
        # configurator's selective fixed bias).
        # PER-LAYER (debug R5): a single shared scratch forces layer L+4's
        # overwrite to race layer L's in-flight D2H read inside one graph
        # replay — the wait_flag gate blocks the compute queue only and can
        # be bypassed by a vector-queue overwrite op. One [T, R] slice per
        # selected layer removes the cross-layer window entirely; the extra
        # memory is num_selected * T * R bytes (trivial vs packed_staging).
        # Layer index resolved in Python at capture time (bakes a fixed view
        # into the graph).
        self.new_packed_scratch_all = torch.zeros(
            len(self.selected_layer_ids_sorted),
            T,
            R,
            dtype=torch.uint8,
            device=self.device,
        )
        self._layer_scratch_index: dict[int, int] = {
            lid: i for i, lid in enumerate(self.selected_layer_ids_sorted)
        }
        # Contiguous component buffers for safe Cast (avoid non-contiguous FP8→BF16)
        # R9: per-layer slices of the same family.
        TK = T * K
        self.fp8_nope_buf_all = torch.zeros(
            n_ws, TK, self.kv_lora_rank, dtype=torch.float8_e4m3fn, device=self.device
        )
        self.scales_buf_all = torch.zeros(
            n_ws, TK, self.kv_lora_rank // 128, dtype=torch.float32, device=self.device
        )
        # R10: per-layer loc-plan buffers. build_loc_plan rewrites these two
        # for EVERY selected layer while the PREVIOUS layer's H2D copy /
        # current-patch / SFA still consume its plan (gather_locs feed the
        # DMA descriptors, current_source_row feeds the patch). This was the
        # remaining cross-layer window after R9: 7-layer per-layer-workspace
        # still scored 0.86 (expected ~0.90 if R9 were the only cause).
        # int64 [T, K] x2 per layer is ~4.8MB/layer — cheap, full per-layer.
        _n_lp = len(self.selected_layer_ids_sorted)
        self.host_locs_buf_all = torch.zeros(
            _n_lp, T, K, dtype=torch.int64, device=self.device
        )
        self.current_source_row_buf_all = torch.full(
            (_n_lp, T, K), -1, dtype=torch.int64, device=self.device
        )
        self.sparse_indices_buf_all = torch.zeros(
            n_ws, T, 1, 1, K, dtype=torch.int32, device=self.device
        )
        self.actual_seq_lens_kv_buf_all = torch.ones(
            n_ws, T, dtype=torch.int32, device=self.device
        )
        self.actual_seq_lens_q_buf_all = torch.ones(
            n_ws, T, dtype=torch.int32, device=self.device
        )
        self.arange_k_buf = torch.arange(
            K, dtype=torch.int32, device=self.device
        )
        self.arange_token_buf = torch.arange(
            T, dtype=torch.int32, device=self.device
        )

        # Pre-allocated pointer arrays for mf_offload.sparse_copy.
        # H2D and D2H run on separate streams (prefetch_stream / backup_stream)
        # and MUST have independent pointer buffers to avoid races when both
        # are in flight simultaneously.
        # PER-LAYER (debug R6): every selected layer used to write the SAME
        # descriptor buffers (h2d_src_ptrs[:N] = ... torch ops) while the
        # previous layer's captured sparse_copy kernel may still be reading
        # them in-flight — a per-layer cross-layer window that scales with
        # layer count (matches the dose-response: 1-2 layers clean, 10
        # layers 0.80, 19 layers 0.72). One descriptor set per layer removes
        # the window; layer index resolved in Python at capture time.
        n_sel = len(self.selected_layer_ids_sorted)
        Rmax = T * K
        # H2D buffers (max entries = Tcap * K)
        self.h2d_src_ptrs_all = torch.zeros(
            n_sel, Rmax, dtype=torch.int64, device=self.device
        )
        self.h2d_dst_ptrs_all = torch.zeros(
            n_sel, Rmax, dtype=torch.int64, device=self.device
        )
        self.h2d_lens_all = torch.zeros(
            n_sel, Rmax, dtype=torch.int32, device=self.device
        )
        self.h2d_cnt = torch.zeros(
            (), dtype=torch.int32, device=self.device
        )
        # D2H buffers (max entries = Tcap)
        self.d2h_src_ptrs_all = torch.zeros(
            n_sel, T, dtype=torch.int64, device=self.device
        )
        self.d2h_dst_ptrs_all = torch.zeros(
            n_sel, T, dtype=torch.int64, device=self.device
        )
        self.d2h_lens_all = torch.zeros(
            n_sel, T, dtype=torch.int32, device=self.device
        )
        self.d2h_cnt = torch.zeros(
            (), dtype=torch.int32, device=self.device
        )
        # Pre-compute sequential HBM offsets for H2D dst_ptrs (constant).
        # R8: per-layer — each layer's H2D lands in its OWN staging slice.
        # (A/B mode: all rows point at slice 0 — exact shared-buffer aliasing.)
        Rmax = T * K
        _ar = torch.arange(Rmax, device=self.device, dtype=torch.int64) * R
        self._h2d_dst_ptrs_preset_all = torch.stack(
            [
                self.packed_staging_all[i % _staging_slices]
                .view(-1)
                .data_ptr()
                + _ar
                for i in range(len(self.selected_layer_ids_sorted))
            ]
        )

        # Completion-flag protocol for graph-mode DMA ordering (see
        # mf_offload.sparse_copy_notify / wait_flag). Monotonic-count
        # variant (no host participation per replay — magic bumps would be
        # Python and frozen at capture): every sparse_copy_notify bumps
        # flag[0] by exactly blockDim (64) once per AIV core, and the
        # captured code pairs each submission with a captured
        # ``expect[0] += 64`` op. wait_flag spins until
        # count == expect; since count only grows and each submission
        # matches one expect increment, a stale (earlier-round) count can
        # never satisfy a later expect. Idle replays (cnt=0) still bump:
        # the copy kernel notifies unconditionally, so the pairing stays
        # 1:1. Flag buffers are int32[16]: the kernel's atomic add is a
        # 32B-aligned DataCopy over 8 lanes (lane0=+1, rest 0) and a full
        # 64B cacheline keeps dcci spin-reads isolated from neighbors.
        self.h2d_flag = torch.zeros(16, dtype=torch.int32, device=self.device)
        self.h2d_expect = torch.zeros(16, dtype=torch.int32, device=self.device)
        self.d2h_flag = torch.zeros(16, dtype=torch.int32, device=self.device)
        self.d2h_expect = torch.zeros(16, dtype=torch.int32, device=self.device)
        # AIV block count used by the copy kernel (sparse_copy launches 64).
        self._dma_flag_block_num = 64

        persistent_buffers = (
            self.packed_staging_all,
            self._h2d_dst_ptrs_preset_all,
            self.unpack_k_nope_bf16_all,
            self.unpack_k_rope_bf16_all,
            self.new_packed_scratch_all,
            self.fp8_nope_buf_all,
            self.scales_buf_all,
            self.host_locs_buf_all,
            self.current_source_row_buf_all,
            self.sparse_indices_buf_all,
            self.actual_seq_lens_kv_buf_all,
            self.actual_seq_lens_q_buf_all,
            self.arange_k_buf,
            self.arange_token_buf,
            self.h2d_src_ptrs_all,
            self.h2d_dst_ptrs_all,
            self.h2d_lens_all,
            self.h2d_cnt,
            self.d2h_src_ptrs_all,
            self.d2h_dst_ptrs_all,
            self.d2h_lens_all,
            self.d2h_cnt,
            self.h2d_flag,
            self.h2d_expect,
            self.d2h_flag,
            self.d2h_expect,
        )
        total_mb = sum(buf.nbytes for buf in persistent_buffers) / (1024 * 1024)
        logger.info(
            f"SelectiveHiSparse staging: {total_mb:.1f} MiB total "
            f"(Tcap={T}, K={K}, R={R})"
        )

    # === loc plan ===

    def _graph_real_token_mask(self, num_tokens: int) -> torch.Tensor:
        """Return the replay-time real-token mask for a captured shape.

        ``num_tokens`` is the static graph token count.  The scalar on the
        right-hand side is updated immediately before every replay, so the
        resulting mask remains dynamic inside the captured graph.
        """
        return self.arange_token_buf[:num_tokens] < self._selective_real_tokens

    def build_loc_plan(
        self,
        topk_indices: torch.Tensor,
        forward_batch: "ForwardBatch",
        selected_layer_id: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert Top-K position indices to Host gather locations.

        Returns:
            gather_locs: [T, K] int64 — Host locs (or sentinel for invalid/current)
            valid_mask: [T, K] bool — True for valid historical entries
            current_source_row: [T, K] int64 — row index into new_packed for
                current-position hits, or -1 for historical entries

        R10: with selected_layer_id the two persistent scratch buffers are
        this layer's OWN slices (cross-layer plan overwrite eliminated).
        Without it (legacy callers) the shared slice 0 is used.
        """
        B = forward_batch.batch_size
        T = B * self.verify_width
        K = self.topk
        if selected_layer_id is not None:
            _lp = self._layer_scratch_index[selected_layer_id]
            host_locs_buf = self.host_locs_buf_all[_lp, :T]
            current_source_row_buf = self.current_source_row_buf_all[_lp, :T]
        else:
            host_locs_buf = self.host_locs_buf_all[0, :T]
            current_source_row_buf = self.current_source_row_buf_all[0, :T]

        topk = topk_indices[:T].reshape(T, K).to(torch.int64)

        # Per-token request mapping
        positions = forward_batch.positions[:T].reshape(B, self.verify_width)
        row_req = torch.arange(B, device=topk.device).repeat_interleave(
            self.verify_width
        )
        req_rows = forward_batch.req_pool_indices[row_req]  # [T]

        # Compute first position of each request's current batch
        positions_mat = positions.reshape(B, self.verify_width)  # [B, V]
        req_first_pos = positions_mat[:, 0]  # [B]
        req_first_pos_exp = req_first_pos.repeat_interleave(
            self.verify_width
        )  # [T]

        # Valid range check
        max_pos = self.req_to_token_pool.req_to_token.shape[1]
        in_range = (topk >= 0) & (topk < max_pos)
        pos_expanded = positions.reshape(T).unsqueeze(1)  # [T, 1]
        causal_ok = in_range & (topk <= pos_expanded)  # [T, K]
        if self._graph_mode:
            # Graph batches are padded to a capture bucket.  Padded rows must
            # never read Host KV or become valid SFA rows.  Capture itself sets
            # real_tokens=0, which also makes all Host DMA a true no-op.
            real_token_mask = self._graph_real_token_mask(T)
            causal_ok = causal_ok & real_token_mask.unsqueeze(1)

        # Detect ALL current-batch positions (self + cross-token within
        # the same request's current verify batch)
        is_current = causal_ok & (
            topk >= req_first_pos_exp.unsqueeze(1)
        )  # [T, K]

        # For current-batch entries, compute row in new_packed_scratch:
        # row = request_index * V + offset_within_batch
        offset_within_batch = topk - req_first_pos_exp.unsqueeze(1)  # [T, K]
        current_row_idx = (
            row_req.unsqueeze(1) * self.verify_width + offset_within_batch
        )  # [T, K]

        # current_source_row: for current-batch hits, row index into
        # new_packed; for historical/invalid → -1
        current_source_row_buf.fill_(-1)
        current_rows = torch.where(
            is_current, current_row_idx, current_source_row_buf
        )  # [T, K]

        # Host locs for historical entries
        safe_pos = topk.clamp(min=0, max=max_pos - 1)
        req_to_token = self.req_to_token_pool.req_to_token
        hist_locs = req_to_token[
            req_rows.unsqueeze(1), safe_pos
        ]  # [T, K]

        # gather_locs: historical → hist_locs, current/invalid → sentinel
        sentinel = self.pool.host_sentinel_loc
        host_locs_buf.fill_(sentinel)
        gather_locs = torch.where(
            causal_ok & ~is_current,
            hist_locs,
            host_locs_buf,
        )

        # valid_mask: True for valid historical entries (not current, not invalid)
        valid_mask = causal_ok & ~is_current  # [T, K]

        return gather_locs, valid_mask, current_rows

    # === anchor prefetch ===

    def maybe_start_prefetch(
        self,
        anchor_layer_id: int,
        topk_indices: torch.Tensor,
        forward_batch: "ForwardBatch",
    ):
        """Called from anchor layer after Top-K is ready.

        If *anchor_layer_id* maps to a selected layer, enqueue H2D prefetch
        on the prefetch stream.  Does NOT block the main stream.
        """
        selected = self._anchor_to_selected.get(anchor_layer_id)
        if selected is None:
            return
        if forward_batch.forward_mode.is_idle():
            return

        B = forward_batch.batch_size
        T = B * self.verify_width

        if T > self.tcap:
            raise RuntimeError(
                f"Selective HiSparse: real tokens {T} > capacity {self.tcap}. "
                f"Increase --max-running_requests or decrease verify width."
            )

        # Eager mode keeps the device-side real-token count in sync (harmless
        # bookkeeping; the eager DMA path submits all N entries with the
        # valid mask and does not consume this scalar). Graph mode never
        # writes it here — replay reads the value filled by
        # prepare_graph_replay (capture sees 0 → DMA no-op).
        if not self._graph_mode:
            self._selective_real_tokens.fill_(T)

        # Wait for previous staging/scratch/backup to complete.
        # During graph capture, skip cross-iteration event waits — the graph
        # replay enforces ordering implicitly and NPU graph requires every
        # wait_event to have a matching record_event within the capture.
        if not self._graph_mode:
            self.prefetch_stream.wait_event(self.staging_free_event)
            prev_backup = self.backup_done_event.get(selected, self._initial_event)
            self.prefetch_stream.wait_event(prev_backup)

        # Build loc plan (R10: this layer's own scratch slices)
        locs, valid, current_rows = self.build_loc_plan(
            topk_indices, forward_batch, selected_layer_id=selected
        )

        # Eager H2D uses a side stream and needs an explicit hand-off.  Graph
        # capture deliberately keeps all work on the capture stream, where
        # stream order already supplies the dependency.  Recording graph-owned
        # events here would leak them into later eager execution.
        loc_plan_ready = (
            None
            if self._graph_mode
            else torch.npu.current_stream().record_event()
        )

        if not self._graph_mode and logger.isEnabledFor(logging.INFO):
            n_hist = valid.sum().item()
            n_curr = (current_rows >= 0).sum().item()
            n_invalid = (~(valid | (current_rows >= 0))).sum().item()
            loc_min = locs[valid].min().item() if n_hist > 0 else -1
            loc_max = locs[valid].max().item() if n_hist > 0 else -1
            K = self.topk
            logger.info(
                f"[H2D] anchor={anchor_layer_id}→sel={selected} "
                f"B={B} T={T} K={K} N={T*K} "
                f"hist={n_hist} curr={n_curr} invalid={n_invalid} "
                f"loc_range=[{loc_min},{loc_max}]"
            )

        # During graph capture, run on the capture stream (no side streams).
        # During normal execution, use prefetch_stream for H2D/compute overlap.
        h2d_stream = (
            torch.npu.current_stream()
            if self._graph_mode
            else self.prefetch_stream
        )

        with torch.npu.stream(h2d_stream):
            # Wait for loc plan computation (on main stream) to complete
            # before reading locs/valid/current_rows on this stream.
            if loc_plan_ready is not None:
                h2d_stream.wait_event(loc_plan_ready)

            staging_flat = self.packed_staging_all[
                self._layer_scratch_index[selected] % self._staging_slices
            ].view(-1, self.record_bytes)

            valid_flat = valid.reshape(-1)
            N = T * self.topk

            try:
                from memfabric_hybrid import offload as mf_offload

                base_dva = self.pool.layer_dva(selected)

                # R6: this layer's own descriptor slice — no other layer
                # writes it, so the previous layer's in-flight copy cannot
                # read descriptors being overwritten.
                _si = self._layer_scratch_index[selected]
                h2d_src = self.h2d_src_ptrs_all[_si]
                h2d_dst = self.h2d_dst_ptrs_all[_si]
                h2d_lens = self.h2d_lens_all[_si]
                h2d_src[:N] = (
                    base_dva
                    + locs.reshape(-1).to(torch.int64) * self.record_bytes
                )
                # R8: dst into THIS layer's staging slice.
                h2d_dst[:N] = self._h2d_dst_ptrs_preset_all[_si, :N]
                # Padded rows never reach here with valid entries: graph
                # mode gates causal_ok upstream (build_loc_plan) and the
                # replay-updated h2d_cnt limits submitted entries to the
                # real rows; eager mode submits all N with the valid mask.
                h2d_lens[:N] = torch.where(
                    valid_flat, self.record_bytes, 0
                ).to(torch.int32)
                if not self._graph_mode:
                    self.h2d_cnt.fill_(N)

                # Graph mode: use the completion-flag protocol so the
                # in-graph consumer (current-patch + SFA) can wait for the
                # AIV DMA via an in-graph wait_flag kernel — the stream
                # alone does not order AIV completion before AICore ops
                # inside an NPU graph. The expect increment below is a
                # captured op that re-executes every replay, pairing 1:1
                # with the copy kernel's unconditional 64-core notify bump.
                # Eager mode keeps the plain call (its consumer is gated by
                # the h2d_done event instead).
                # B3 (notify-exclusion probe): with the H2D wait skipped
                # (SKIP_H2D_WAIT=1) and the D2H wait deleted (R11), NOBODY
                # consumes the flags — yet every sparse_copy_notify kernel
                # still runs its notify tail (per-core SetAtomicAdd +
                # S_MTE3 event pair + PipeBarrier<PIPE_MTE3>): at 19L that
                # is 38 kernels x 64 cores = 2432 atomic cacheline writes
                # per replay, present in EVERY experiment so far and never
                # singly excluded. SGLANG_SELECTIVE_PLAIN_COPY=1 switches
                # graph mode to the plain notify-free sparse_copy (the
                # kernel skips NotifyDone entirely when notify == nullptr).
                _plain_copy = (
                    self._graph_mode
                    and os.getenv("SGLANG_SELECTIVE_PLAIN_COPY", "0") == "1"
                )
                if self._graph_mode and not _plain_copy:
                    ret = mf_offload.sparse_copy_notify(
                        h2d_src[:N],
                        h2d_dst[:N],
                        h2d_lens[:N],
                        self.h2d_cnt,
                        self.h2d_flag,
                        self.device,
                    )
                    self.h2d_expect[0].add_(self._dma_flag_block_num)
                else:
                    ret = mf_offload.sparse_copy(
                        h2d_src[:N],
                        h2d_dst[:N],
                        h2d_lens[:N],
                        self.h2d_cnt,
                        self.device,
                    )
                if ret != 0:
                    raise RuntimeError(
                        f"sparse_copy H2D failed: ret={ret}"
                    )
                if not self._graph_mode and logger.isEnabledFor(logging.INFO):
                    n_h2d = valid_flat.sum().item()
                    n_curr = (current_rows >= 0).reshape(-1).sum().item()
                    logger.info(
                        f"[H2D] sparse_copy ret={ret} "
                        f"h2d={n_h2d} curr={n_curr} "
                        f"total_valid={n_h2d + n_curr}/{N} "
                        f"dva=0x{base_dva:x}"
                    )
            except ImportError:
                host_tensor = self.pool.get_host_tensor(selected)
                src = host_tensor[locs.reshape(-1).cpu()]
                staging_flat[:N].copy_(src.to(self.device))

            if self._graph_mode:
                h2d_done = None
            else:
                h2d_done = h2d_stream.record_event()

        self.active_prefetch = SelectedPrefetchState(
            selected_layer_id=selected,
            real_batch=B,
            real_tokens=T,
            gather_locs=locs,
            valid_mask=valid,
            current_source_row=current_rows,
            h2d_done=h2d_done,
        )

    # === new KV publish ===

    def publish_new_packed_kv(
        self,
        layer_id: int,
        logical_locs: torch.Tensor,
        packed_kv: torch.Tensor,
    ):
        """Receive newly packed KV from ``set_kv_buffer()`` and schedule D2H.

        Called on the main stream after packing.  The D2H copy runs on the
        backup stream.
        """
        # Wait for previous D2H of this layer to complete before reusing the
        # d2h pointer buffers. Additionally, new_packed_scratch is shared
        # across all selected layers, so overwriting it must wait for the
        # latest in-flight D2H read of ANY layer (the latest backup event
        # subsumes earlier ones because backup_stream is serialized).
        # During graph capture, skip cross-iteration event waits.
        if not self._graph_mode:
            prev_ev = self.backup_done_event.get(layer_id)
            if prev_ev is not None:
                torch.npu.current_stream().wait_event(prev_ev)
            if self._last_backup_event is not None:
                torch.npu.current_stream().wait_event(self._last_backup_event)
        # R11 (graph mode: NO D2H wait). The old wait_flag here paired each
        # wait with the copy that came AFTER it (waits precede their copies
        # on the D2H path), so wait_k could only pass after copy_k's
        # predecessor finished — off by one. Under multi-layer captures this
        # stalled each layer's publish for a full D2H completion (the
        # seconds-long stalls at 19 layers) and, combined with capture
        # warmup, let stale passes corrupt ordering (0.70). It is also
        # unnecessary now: R5 gives every layer its own new_packed_scratch
        # slice, so no cross-layer scratch overwrite exists to gate, and the
        # MIX copy shares the compute queue with the scratch writes (FIFO).

        T = packed_kv.shape[0]

        # W3: land the new packed KV in the fixed-address scratch so the
        # current-patch consumer (and the D2H DMA below) read a stable
        # pre-allocated buffer instead of a Python-held graph-pool tensor.
        # Same-stream ordering guarantees the copy lands before both
        # consumers. R5: per-layer slice (layer_id resolved at capture time);
        # no other layer ever writes this buffer, so the D2H read below and
        # this write cannot race a different layer's overwrite.
        scratch = self.new_packed_scratch_all[
            self._layer_scratch_index[layer_id]
        ]
        scratch[:T].copy_(packed_kv.view(torch.uint8))

        # E7 (C6 probe, env-gated, graph mode only): flush the PREVIOUS
        # selected layer's D2H now. Its scratch copy_ was issued one full
        # selected-cycle (4 model layers of compute + attention) earlier on
        # the VECTOR queue; launching its MIX D2H here puts enough stream
        # distance that the copy_ has completed even without cross-queue
        # ordering — closing the stale-scratch read window that the E3/E4
        # sleep matrix localized to 19 layers.
        _defer_d2h = (
            self._graph_mode
            and os.getenv("SGLANG_SELECTIVE_DEFER_D2H", "0") == "1"
        )
        if _defer_d2h and self._pending_d2h is not None:
            _p_layer, _p_n = self._pending_d2h
            self._pending_d2h = None
            self._launch_deferred_d2h(_p_layer, _p_n)

        # C6 probe (env-gated, graph mode only): the copy above is a
        # VECTOR-queue torch op while the D2H sparse_copy_notify below runs
        # on the COMPUTE queue (MIX) — same stream, but cross-queue
        # completion is unordered, so the D2H can read a one-step-stale
        # scratch and back up poisoned KV to the host pool. At 19 layers
        # the vector queue is deep enough to open this window (7L closed:
        # sleep probe harmless at 7L, -10pt at 19L). Eager mode gates this
        # via pack_ready below; this records an event after the copy and
        # waits it before the D2H launch so the captured graph carries a
        # cross-queue dependency edge (record+wait both inside the capture
        # — the legal pattern; R11's error was recording outside it).
        _d2h_gate_ev = None
        if self._graph_mode and os.getenv(
            "SGLANG_SELECTIVE_GATE_D2H_EVENT", "0"
        ) == "1":
            _d2h_gate_ev = torch.npu.current_stream().record_event()

        # Eager mode syncs the device-side real-token count (harmless
        # bookkeeping, see maybe_start_prefetch). Graph mode reads the
        # replay-filled value (capture sees 0 → DMA no-op).
        if not self._graph_mode:
            self._selective_real_tokens.fill_(T)

        if not self._graph_mode and logger.isEnabledFor(logging.INFO):
            locs_cpu = logical_locs[:T].cpu()
            logger.info(
                f"[D2H] layer={layer_id} T={T} "
                f"locs=[{locs_cpu.min().item()},{locs_cpu.max().item()}] "
                f"packed_shape={list(packed_kv.shape)} "
                f"packed_dtype={packed_kv.dtype}"
            )

        # Record AFTER the W3 scratch copy so the eager D2H stream waits for
        # it; graph capture keeps everything on the capture stream (stream
        # order supplies the dependency) and must not record graph-owned
        # events.
        pack_ready = (
            None
            if self._graph_mode
            else torch.npu.current_stream().record_event()
        )

        # During graph capture, run on the capture stream (no side streams).
        d2h_stream = (
            torch.npu.current_stream()
            if self._graph_mode
            else self.backup_stream
        )

        with torch.npu.stream(d2h_stream):
            if pack_ready is not None:
                d2h_stream.wait_event(pack_ready)
            # C6 probe: wait the post-copy event (graph mode; d2h_stream IS
            # the current/capture stream there) so the MIX D2H cannot pass
            # the vector-queue copy_.
            if _d2h_gate_ev is not None:
                torch.npu.current_stream().wait_event(_d2h_gate_ev)

            N = T

            try:
                from memfabric_hybrid import offload as mf_offload

                # W3: DMA reads from the fixed-address scratch (R5: this
                # layer's own slice — see publish comment above).
                _si = self._layer_scratch_index[layer_id]
                base_hbm = self.new_packed_scratch_all[_si].data_ptr()
                base_dva = self.pool.layer_dva(layer_id)
                # R6: this layer's own descriptor slice (see H2D comment).
                d2h_src = self.d2h_src_ptrs_all[_si]
                d2h_dst = self.d2h_dst_ptrs_all[_si]
                d2h_lens = self.d2h_lens_all[_si]

                d2h_src[:N] = (
                    base_hbm
                    + torch.arange(
                        N, device=self.device, dtype=torch.int64
                    ) * self.record_bytes
                )
                if self._graph_mode:
                    real_token_mask = self._graph_real_token_mask(N)
                    safe_logical_locs = torch.where(
                        real_token_mask,
                        logical_locs[:N].to(torch.int64),
                        torch.zeros_like(logical_locs[:N], dtype=torch.int64),
                    )
                else:
                    real_token_mask = None
                    safe_logical_locs = logical_locs[:N].to(torch.int64)

                d2h_dst[:N] = (
                    base_dva
                    + safe_logical_locs * self.record_bytes
                )
                if real_token_mask is not None:
                    d2h_lens[:N] = torch.where(
                        real_token_mask,
                        self.record_bytes,
                        0,
                    ).to(torch.int32)
                else:
                    d2h_lens[:N] = self.record_bytes
                    self.d2h_cnt.fill_(N)

                if self._graph_mode:
                    # B3: env re-read here — _plain_copy from
                    # maybe_start_prefetch is out of scope in this method.
                    _plain_copy_d2h = (
                        os.getenv("SGLANG_SELECTIVE_PLAIN_COPY", "0") == "1"
                    )
                    _is_last_selected = (
                        _si == len(self.selected_layer_ids_sorted) - 1
                    )
                    if _defer_d2h and not _is_last_selected:
                        # E7: hold the launch; flushed at the NEXT selected
                        # layer's publish. The last selected layer has no
                        # next publish inside this capture — launch now
                        # (1/19 layers stays exposed; acceptable for the
                        # probe read).
                        self._pending_d2h = (layer_id, N)
                        ret = 0
                    elif _plain_copy_d2h:
                        # B3: notify-free plain copy (see H2D comment).
                        ret = mf_offload.sparse_copy(
                            d2h_src[:N],
                            d2h_dst[:N],
                            d2h_lens[:N],
                            self.d2h_cnt,
                            self.device,
                        )
                    else:
                        ret = mf_offload.sparse_copy_notify(
                            d2h_src[:N],
                            d2h_dst[:N],
                            d2h_lens[:N],
                            self.d2h_cnt,
                            self.d2h_flag,
                            self.device,
                        )
                        self.d2h_expect[0].add_(self._dma_flag_block_num)
                else:
                    ret = mf_offload.sparse_copy(
                        d2h_src[:N],
                        d2h_dst[:N],
                        d2h_lens[:N],
                        self.d2h_cnt,
                        self.device,
                    )
                if ret != 0:
                    raise RuntimeError(
                        f"sparse_copy D2H failed: ret={ret}"
                    )
            except ImportError:
                host_tensor = self.pool.get_host_tensor(layer_id)
                host_tensor[logical_locs[:T].cpu()] = packed_kv.to("cpu")

            if not self._graph_mode:
                self.backup_done_event[layer_id] = d2h_stream.record_event()
                self._eager_async_pending = True
                # W3: track the latest D2H for the shared-scratch
                # (new_packed_scratch) overwrite guard in the next publish
                # of ANY layer.
                self._last_backup_event = self.backup_done_event[layer_id]

    # === selected-layer attention ===

    def _launch_deferred_d2h(self, layer_id: int, N: int):
        """E7: launch the held D2H of a previously published layer.

        Descriptor buffers (per-layer, already filled at that layer's
        publish) are read here; only the MIX kernel launch (and its
        expect bump) moves later in the captured stream order.
        """
        from memfabric_hybrid import offload as mf_offload

        _si = self._layer_scratch_index[layer_id]
        d2h_src = self.d2h_src_ptrs_all[_si]
        d2h_dst = self.d2h_dst_ptrs_all[_si]
        d2h_lens = self.d2h_lens_all[_si]
        ret = mf_offload.sparse_copy_notify(
            d2h_src[:N],
            d2h_dst[:N],
            d2h_lens[:N],
            self.d2h_cnt,
            self.d2h_flag,
            self.device,
        )
        self.d2h_expect[0].add_(self._dma_flag_block_num)
        if ret != 0:
            raise RuntimeError(f"sparse_copy D2H (deferred) failed: ret={ret}")

    def run_selected_attention(
        self,
        layer_id: int,
        layer,  # RadixAttention
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        topk_indices: Optional[torch.Tensor],
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        """Execute the full selected-layer attention pipeline.

        1. Wait for H2D prefetch to complete
        2. Patch current-position KV into staging
        3. Unpack 656B → BF16
        4. Call SFA BSND
        5. Record staging-free event
        """
        from sglang.srt.hardware_backend.npu.attention.selective_sparse_attention import (
            selective_sparse_attention,
        )

        st = self.active_prefetch
        if st is None:
            raise RuntimeError(
                f"Selective HiSparse layer {layer_id} has no active prefetch"
            )
        if st.selected_layer_id != layer_id:
            raise RuntimeError(
                "Selective HiSparse prefetch/attention mismatch: "
                f"prefetched layer {st.selected_layer_id}, executing layer {layer_id}. "
                "Selected layers must use non-overlapping anchor windows."
            )

        T = st.real_tokens
        K = self.topk

        # 1. Wait for H2D
        if st.h2d_done is not None:
            torch.npu.current_stream().wait_event(st.h2d_done)
            # DEBUG (exp F): eager-side staging checksum right after the H2D
            # wait and before the current-patch overwrite — this is the pure
            # H2D-landed content. Compare against the graph-side dump from
            # decode_cuda_graph_runner.execute(). Enable with
            # SGLANG_SELECTIVE_DUMP_STAGING=1.
            if os.getenv("SGLANG_SELECTIVE_DUMP_STAGING", "0") == "1":
                _n = st.real_tokens * K
                _flat = self.packed_staging_all[
                    self._layer_scratch_index[layer_id] % self._staging_slices
                ].view(-1)[: _n * self.record_bytes]
                logger.info(
                    f"[STAGING-DUMP eager] layer={layer_id} "
                    f"N={_n} sum={_flat.sum(dtype=torch.int64).item()} "
                    f"nonzero={int((_flat != 0).sum().item())}"
                )
        elif self._graph_mode:
            # Graph mode: no event to wait on. The captured H2D DMA (AIV)
            # is not ordered before this consumer (AICore) by the stream,
            # so gate on the completion flag instead: wait_flag spins
            # (device-side, single AIV core) until the copy kernel's 64
            # cores all finished THIS round's magic. This op is captured
            # into the graph between the DMA and the patch/SFA reads.
            # R13 (A/B): skip the in-graph H2D wait entirely. In the
            # all-MIX world the compute-queue FIFO already orders the copy
            # kernel before patch/SFA; the wait is documented redundant
            # insurance — but its device-side spin stalls the compute
            # queue, and the stall window grows with layer count (11-19
            # layers lost ~1pt/layer). If skipping restores precision, the
            # spin itself was the enabler. Env-gated single-variable probe:
            # SGLANG_SELECTIVE_SKIP_H2D_WAIT=1
            if os.getenv("SGLANG_SELECTIVE_SKIP_H2D_WAIT", "0") != "1":
                try:
                    from memfabric_hybrid import offload as mf_offload

                    ret = mf_offload.wait_flag(
                        self.h2d_flag,
                        self.h2d_expect,
                        self.device,
                    )
                    if ret != 0:
                        raise RuntimeError(f"wait_flag H2D failed: ret={ret}")
                except ImportError:
                    pass  # fallback: staging copy path, no flag protocol

        # 2. Current KV patch (graph-safe: no boolean indexing / nonzero)
        # W3: read from the fixed-address scratch written by
        # publish_new_packed_kv (same-stream ordering on both the eager
        # main stream and the graph capture stream). R5: this layer's own
        # slice — captured view, stable across replays.
        packed = self.new_packed_scratch_all[
            self._layer_scratch_index[layer_id]
        ][:T]  # [T, 656] uint8
        current_rows = st.current_source_row[:T]  # [T, K]
        mask = (current_rows >= 0).reshape(-1)  # [T*K]
        safe_src = current_rows.reshape(-1).clamp(min=0)  # [T*K]

        # R8: this layer's OWN staging slice — patch and SFA both touch it;
        # no other layer's H2D ever writes it.
        staging = self.packed_staging_all[
            self._layer_scratch_index[layer_id] % self._staging_slices
        ]
        staging_flat = staging.view(-1, self.record_bytes)
        N = T * K
        src_data = packed[safe_src]  # [T*K, 656]
        staging_flat[:N] = torch.where(
            mask.unsqueeze(1),
            src_data,
            staging_flat[:N],
        )

        # 3+4. Unpack + SFA
        # valid_mask for SFA = historical valid OR current-position valid
        all_valid_mask = (
            st.valid_mask[:T] | (st.current_source_row[:T] >= 0)
        )

        if not self._graph_mode and logger.isEnabledFor(logging.INFO):
            n_curr = mask.sum().item()
            n_hist = st.valid_mask[:T].sum().item()
            n_all_valid = all_valid_mask.sum().item()
            staging_nonzero = (
                staging_flat[:N].sum(dim=1) > 0
            ).sum().item()
            logger.info(
                f"[ATTN] layer={layer_id} T={T} K={K} "
                f"hist={n_hist} curr={n_curr} all_valid={n_all_valid} "
                f"staging_nonzero={staging_nonzero}/{N}"
            )

        # R9-final: ping-pong workspace set — parity of the layer's position
        # in the selected list (see _alloc_staging_buffers). Adjacent
        # selected layers are 4 apart, so a set is reused only 4 layers
        # later, after its reader's SFA has long retired.
        _ws = self._layer_scratch_index[layer_id] % self._n_ws
        out = selective_sparse_attention(
            q_nope=q_nope[:T],
            q_rope=q_rope[:T],
            packed_staging=staging[:T],
            valid_mask=all_valid_mask,
            scale=layer.scaling,
            unpack_k_nope_bf16=self.unpack_k_nope_bf16_all[_ws],
            unpack_k_rope_bf16=self.unpack_k_rope_bf16_all[_ws],
            sparse_indices_buf=self.sparse_indices_buf_all[_ws],
            actual_seq_lens_kv_buf=self.actual_seq_lens_kv_buf_all[_ws],
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            actual_seq_lens_q_buf=self.actual_seq_lens_q_buf_all[_ws],
            arange_k_buf=self.arange_k_buf,
            fp8_nope_buf=self.fp8_nope_buf_all[_ws],
            scales_buf=self.scales_buf_all[_ws],
            graph_mode=self._graph_mode,
        )

        # 5. Release staging
        if not self._graph_mode:
            self.staging_free_event = torch.npu.current_stream().record_event()
        self.active_prefetch = None

        # 6. Pad back to full DP shape (no-op during graph capture/replay)
        num_total = q_nope.shape[0]
        if T < num_total:
            out_padded = out.new_zeros(
                num_total, out.shape[1]
            )
            out_padded[:T] = out
            return out_padded
        return out

    # === MTP verify result ===

    def on_verify_result(
        self,
        req_pool_indices: torch.Tensor,
        verify_cache_locs: torch.Tensor,
        old_seq_lens: torch.Tensor,
        accept_lens: torch.Tensor,
        accept_index: torch.Tensor,
    ):
        """Called after eagle_sample returns. Records debug metrics."""
        B = accept_lens.shape[0]
        logger.debug(
            f"Selective HiSparse verify result: B={B}, "
            f"accept_lens={accept_lens.tolist()}"
        )

    # === drain ===

    def drain_all(self):
        """Wait for all pending async D2H writes to complete."""
        if self.backup_done_event:
            self.backup_stream.synchronize()

    def wait_for_pending_backup(self, layer_ids: Sequence[int]):
        """Wait for D2H backup of specific layers."""
        for lid in layer_ids:
            ev = self.backup_done_event.get(lid)
            if ev is not None:
                torch.npu.current_stream().wait_event(ev)

    # === graph support ===

    def prepare_eager_forward(self):
        """Bridge preceding main-stream graph work into eager side streams."""
        self._graph_mode = False
        # An eager prefetch waits on staging_free_event.  Recording it here
        # makes that wait cover a preceding graph replay, whose internal DMA
        # and staging operations are otherwise invisible to Python events.
        self.staging_free_event = torch.npu.current_stream().record_event()

    def _bridge_eager_to_graph(self):
        """Make the main stream wait for outstanding eager side-stream DMA."""
        if not self._eager_async_pending:
            return
        current_stream = torch.npu.current_stream()
        current_stream.wait_event(self.staging_free_event)
        for event in self.backup_done_event.values():
            current_stream.wait_event(event)
        self._eager_async_pending = False
        # Eager DMA used the flagless path, so the monotonic counters stay
        # paired — but a stream-level wait_event does NOT prove the AIV
        # copy kernel has fully landed (the very ordering hole the flag
        # protocol exists to close). The first in-graph D2H-scratch wait
        # would therefore pass while an eager D2H still reads the shared
        # scratch. A full sync here is safe: eager fallback is rare (batch
        # exceeding the graph tier), so this is off the hot path.
        torch.npu.synchronize()

    def prepare_graph_capture(
        self,
        capture_bs: int,
        capture_tokens: int,
    ):
        """Called before graph capture to set static state."""
        self._bridge_eager_to_graph()
        expected_tokens = capture_bs * self.verify_width
        if capture_tokens != expected_tokens:
            raise RuntimeError(
                "Selective HiSparse requires a dense fixed-width verify graph: "
                f"capture_tokens={capture_tokens}, capture_bs={capture_bs}, "
                f"verify_width={self.verify_width}."
            )
        if capture_tokens > self.tcap:
            raise RuntimeError(
                f"Selective HiSparse capture tokens {capture_tokens} exceed "
                f"staging capacity {self.tcap}."
            )
        self._graph_mode = True
        # Captured DMA count/masks read this scalar.  Zero makes warmup/capture
        # avoid touching Host KV; prepare_graph_replay updates it dynamically.
        self._selective_real_tokens.fill_(0)
        self.h2d_cnt.fill_(0)
        self.d2h_cnt.fill_(0)
        # Drain any eager-side in-flight DMA and reset the monotonic
        # completion counters to a consistent zero state: eager copies use
        # the plain (flagless) path, so their completions are invisible to
        # the flag counters. Warmup + capture runs then bump count and
        # expect in lockstep (every notify pairs with one expect += 64).
        torch.npu.synchronize()
        self.h2d_flag.fill_(0)
        self.h2d_expect.fill_(0)
        self.d2h_flag.fill_(0)
        self.d2h_expect.fill_(0)
        logger.info(
            f"SelectiveHiSparse graph capture: bs={capture_bs}, "
            f"tokens={capture_tokens}"
        )

    def prepare_graph_replay(
        self,
        real_batch: int,
        graph_batch: int,
        is_idle: bool,
        real_num_tokens: Optional[torch.Tensor] = None,
    ):
        """Called before graph replay to update real token count."""
        self._bridge_eager_to_graph()
        # Subscribe the replay-issuing stream to the ACL callback thread. The
        # original subscription covers only the capture stream, but graph
        # replay executes on the issuing (main) stream — in-graph sparse_copy
        # async DMA completion reports fired there would never be processed
        # by acl.rt.process_report, so H2D data may not land before the
        # in-graph SFA read. Idempotent (subscribe() skips known streams).
        # Previously gated by SGLANG_SELECTIVE_SUBSCRIBE_REPLAY_STREAM=1;
        # promoted to unconditional after it proved a necessary fix.
        self.register_callback_stream(torch.npu.current_stream(self.device))
        real_tokens = 0 if is_idle else real_batch * self.verify_width
        graph_tokens = graph_batch * self.verify_width
        if real_tokens > graph_tokens:
            raise RuntimeError(
                f"Selective HiSparse real tokens {real_tokens} exceed graph "
                f"capacity {graph_tokens}."
            )
        if graph_tokens > self.tcap:
            raise RuntimeError(
                f"Selective HiSparse graph tokens {graph_tokens} exceed staging "
                f"capacity {self.tcap}."
            )
        if is_idle:
            self._selective_real_tokens.fill_(0)
            self.h2d_cnt.fill_(0)
            self.d2h_cnt.fill_(0)
        elif real_num_tokens is not None:
            # DP/attention padding can make real_batch * width larger than the
            # number of local tokens.  Reuse the runner's already-localized
            # scalar instead of treating those DP padding rows as real.
            real_num_tokens_scalar = real_num_tokens.reshape(()).clamp(
                min=0, max=graph_tokens
            )
            self._selective_real_tokens.copy_(real_num_tokens_scalar)
            self.h2d_cnt.copy_(real_num_tokens_scalar * self.topk)
            self.d2h_cnt.copy_(real_num_tokens_scalar)
        else:
            self._selective_real_tokens.fill_(real_tokens)
            self.h2d_cnt.fill_(real_tokens * self.topk)
            self.d2h_cnt.fill_(real_tokens)
        # DEBUG (graph-mode precision): zero the in-graph D2H backup count so
        # the captured backup sparse_copy submits nothing at replay. If
        # precision recovers, the backup path is corrupting the Host pool
        # (wrong dst pointers / stale logical locs at replay) rather than the
        # H2D prefetch being late. Enable with SGLANG_SELECTIVE_DISABLE_D2H=1.
        if os.getenv("SGLANG_SELECTIVE_DISABLE_D2H", "0") == "1":
            self.d2h_cnt.fill_(0)

        # R7 (flag health probe): read the monotonic counters BEFORE this
        # replay. Invariant: h2d_flag[0] == h2d_expect[0] (every captured
        # notify is paired with one captured expect add_) — checked after a
        # device sync so the read reflects all PREVIOUS replays' kernels.
        # A mismatch (flag < expect: a wait passed early / notify lost;
        # flag > expect: stray notify) means the flag protocol itself is
        # broken at this layer count, which would invalidate every wait
        # downstream and produce exactly the dose-response precision loss.
        # Read-only probe: the sync makes each replay slightly slower but
        # changes no semantics. Enable with SGLANG_SELECTIVE_CHECK_FLAGS=1.
        if os.getenv("SGLANG_SELECTIVE_CHECK_FLAGS", "0") == "1":
            self._dbg_replay_round = getattr(self, "_dbg_replay_round", 0) + 1
            if self._dbg_replay_round % 100 == 1:  # sample to bound overhead
                torch.npu.synchronize()
                h2d_f = int(self.h2d_flag[0].item())
                h2d_e = int(self.h2d_expect[0].item())
                d2h_f = int(self.d2h_flag[0].item())
                d2h_e = int(self.d2h_expect[0].item())
                ok = (h2d_f == h2d_e) and (d2h_f == d2h_e)
                logger.info(
                    f"[FLAG-CHECK] round={self._dbg_replay_round} "
                    f"h2d flag/expect={h2d_f}/{h2d_e} "
                    f"d2h flag/expect={d2h_f}/{d2h_e} "
                    f"{'OK' if ok else '*** MISMATCH ***'}"
                )

    def finish_graph_capture(self):
        """Restore eager coordinator semantics after one graph is captured.

        Replays execute the already-recorded graph and do not call coordinator
        Python methods.  Leaving this flag set would make any eager fallback
        skip side-stream synchronization and reuse stale replay token counts.
        """
        self._graph_mode = False

    def register_callback_stream(self, stream):
        """Register ACL host callback for the given stream."""
        register_npu_selective_callback_stream(stream, self.device)

    @property
    def graph_mode(self) -> bool:
        return self._graph_mode

    @property
    def selective_real_tokens(self) -> torch.Tensor:
        return self._selective_real_tokens


def validate_npu_selective_hisparse_config(
    server_args: "ServerArgs",
    model_config: "ModelConfig",
) -> tuple[int, ...]:
    """Return the sorted tuple of selected layer IDs."""
    layer_ids_raw = server_args.npu_selective_hisparse_layer_ids
    selected = sorted(set(int(x) for x in layer_ids_raw))
    num_hidden_layers = model_config.num_hidden_layers
    out_of_range = [
        layer_id
        for layer_id in selected
        if layer_id < 0 or layer_id >= num_hidden_layers
    ]
    if out_of_range:
        raise ValueError(
            "Selective HiSparse layer IDs must be in model range "
            f"[0, {num_hidden_layers}), got {out_of_range}."
        )

    from sglang.srt.configs.model_config import resolve_dsa_last_shared_layer_ids

    candidates = set(resolve_dsa_last_shared_layer_ids(model_config.hf_text_config))
    unsupported = [layer_id for layer_id in selected if layer_id not in candidates]
    if unsupported:
        raise ValueError(
            "Selective HiSparse only supports the last layer of each shared "
            f"DSA index group; unsupported={unsupported}, "
            f"candidates={sorted(candidates)}."
        )
    logger.info(
        f"Selective HiSparse: selected_layers={selected}"
    )
    return tuple(selected)
