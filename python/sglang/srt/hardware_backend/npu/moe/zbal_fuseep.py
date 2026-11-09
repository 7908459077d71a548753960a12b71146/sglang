"""ZBAL dispatch_ffn_combine (FuseEP) MoE path in pool-only symmetric space: expert weights and the
resident workspace live in the zbal pool, every other tensor stays on the native allocator.
Pool sizing needs ``SGLANG_ZBAL_DFC_MOE_LAYERS``; weight layouts live in ``zbal.zbal_moe_repack``.
The zbal-side imports (zbal / Buffer / zbal_moe_repack) must stay lazy inside functions"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.runtime_context import get_parallel, get_server_args

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import TopKOutput

logger = logging.getLogger(__name__)

# Alignment of the total pool size
_POOL_ALIGN = 2 * 1024 * 1024

# gmm1/gmm2 weights required by zbal FuseEP
_FUSE_EP_PREFIXES = frozenset({"w13", "w2"})


class ZbalFuseEpState:
    """Process-wide FuseEP singleton owning the lazily created symmetric pool and resident Buffer."""

    _instance = None

    def __init__(self) -> None:
        # Cached on first read: forward() asks once per layer per step and the env is fixed for
        # the process lifetime.
        self.enabled: bool | None = None
        self.pool_inited = False
        self.buffer = None
        # Expert shape the pool was sized from; every MoE layer must match it.
        self.pool_shape_key: tuple | None = None
        # Per-forward prefill token bound frozen with the pool sizing: the pool, the workspace and
        # the runtime guard all read this snapshot.
        self.max_prefill_tokens: int | None = None

    @classmethod
    def instance(cls) -> "ZbalFuseEpState":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def use_fuseep(self) -> bool:
        """FuseEP switch: pool-only mode only when explicitly enabled (torch allocator untouched)."""
        if self.enabled is None:
            self.enabled = envs.SGLANG_ZBAL_ENABLE_DFC.get()
        return self.enabled

    def ensure_pool(self, layer: FusedMoE) -> None:
        """Create the symmetric pool and its resident Buffer once, before the first allocation."""
        if self.pool_inited:
            return
        init_pool(layer)
        self.pool_inited = True
        self.buffer = _create_buffer(layer)


def use_zbal_fuseep() -> bool:
    """FuseEP switch: pool-only mode only when explicitly enabled (torch allocator untouched)."""
    return ZbalFuseEpState.instance().use_fuseep()


def _prefill_ceiling() -> int:
    """Per-forward prefill ceiling sglang sizes its own buffers with (0 when chunking is off)."""
    from sglang.srt.arg_groups.overrides import max_prefill_buffer_tokens

    return int(max_prefill_buffer_tokens(get_server_args()))


def _max_prefill_tokens() -> int:
    """Per-forward prefill token bound: the chunk ceiling, or ``--max-prefill-tokens`` without it."""
    return _prefill_ceiling() or int(get_server_args().max_prefill_tokens)


def _bound_tokens() -> int:
    """Frozen prefill bound shared by pool sizing, workspace sizing and the runtime guard."""
    state = ZbalFuseEpState.instance()
    if state.max_prefill_tokens is None:
        state.max_prefill_tokens = _max_prefill_tokens()
    return state.max_prefill_tokens


def _weights_key(layer: FusedMoE) -> tuple:
    """Shape key of one layer's expert weights: the pool is sized from one shape and every MoE
    layer must match it."""
    return (
        layer.num_local_experts,
        layer.hidden_size,
        layer.intermediate_size_per_partition,
    )


def _pool_bytes(layer: FusedMoE) -> int:
    """Pool size = per-layer max-over-ABIs bytes × ``SGLANG_ZBAL_DFC_MOE_LAYERS`` + workspace."""
    moe_layers = envs.SGLANG_ZBAL_DFC_MOE_LAYERS.get()
    if moe_layers <= 0:
        logger.error(
            "ZBAL FuseEP: SGLANG_ZBAL_DFC_MOE_LAYERS is unset or non-positive (%s); set it to "
            "the number of MoE layers the model builds (MTP/nextn included)",
            moe_layers,
        )
        raise ValueError("ZBAL FuseEP requires SGLANG_ZBAL_DFC_MOE_LAYERS")

    # Fail fast: without the chunk gate the scheduler admits a lone request above
    # --max-prefill-tokens up to the context length, so no pool/workspace size is an upper bound
    # and the failure would only surface on the first long request.
    if _prefill_ceiling() <= 0:
        logger.error(
            "ZBAL FuseEP: chunked prefill is off (--chunked-prefill-size=%s), so one request may "
            "exceed --max-prefill-tokens up to the context length and the pool has no fixed "
            "per-forward bound to size from",
            get_server_args().chunked_prefill_size,
        )
        raise ValueError("ZBAL FuseEP requires --chunked-prefill-size > 0")

    bound = _bound_tokens()

    from zbal import zbal_moe_repack

    per_layer = zbal_moe_repack.moe_layer_weight_bytes_upper_bound(
        num_local_experts=layer.num_local_experts,
        hidden=layer.hidden_size,
        ffn=layer.intermediate_size_per_partition,
    )
    state = ZbalFuseEpState.instance()
    # The budget assumes every MoE layer (MTP included) shares this shape.
    state.pool_shape_key = _weights_key(layer)
    weights = per_layer * moe_layers
    workspace = _workspace_bytes(layer)
    total = weights + workspace
    pool = (total + _POOL_ALIGN - 1) // _POOL_ALIGN * _POOL_ALIGN
    logger.info(
        "ZBAL FuseEP pool size: moe_layers=%s prefill_tokens=%s per_layer=%.3f GiB "
        "weights=%.3f GiB workspace=%.3f GiB pool=%.3f GiB",
        moe_layers,
        bound,
        per_layer / 2**30,
        weights / 2**30,
        workspace / 2**30,
        pool / 2**30,
    )
    return pool


def init_pool(layer: FusedMoE) -> None:
    """Initialize the pool-only symmetric pool before weights exist; the world group must already
    be published because the pool is sized per world rank."""
    import zbal

    if dist.is_available() and dist.is_initialized():
        world_size, rank_id = dist.get_world_size(), dist.get_rank()
    else:
        parallel = get_parallel()
        try:
            world_size = parallel.world_group.world_size
            rank_id = parallel.world_group.rank_in_group
        except (RuntimeError, ValueError):
            logger.error(
                "ZBAL FuseEP: world group is not published while the pool is being created, so "
                "world_size/rank_id are unavailable; pool init cannot guess them from tp"
            )
            raise

    device_id = torch.npu.current_device()
    pool_bytes = _pool_bytes(layer)
    ip_port = envs.SGLANG_ZBAL_BOOTSTRAP_URL.get() or ""
    logger.info(
        "ZBAL FuseEP pool-only init: world=%s device=%s rank=%s pool=%.3f GiB bootstrap=%s",
        world_size,
        device_id,
        rank_id,
        pool_bytes / 2**30,
        ip_port or "<default>",
    )
    ok = zbal.zbal_init_pool(
        int(world_size), int(device_id), int(rank_id), pool_bytes, ip_port=ip_port
    )
    if not ok:
        logger.error(
            "ZBAL FuseEP pool-only init failed: world=%s rank=%s pool_bytes=%s",
            world_size,
            rank_id,
            pool_bytes,
        )
        raise RuntimeError("zbal_init_pool failed; check pool size / bootstrap url")


@contextmanager
def _pool_scope():
    """Torch allocations in this scope go to the symmetric pool; only wrap peer-visible tensors."""
    import zbal # type: ignore

    with torch.npu.use_mem_pool(zbal.get_weight_mem_pool()):
        yield


def pool_scope(layer: FusedMoE):
    """Exposed to ``FusedMoE.create_weights`` (caller checked ``use_zbal_fuseep()``) so expert
    weights are born in the pool peers prefetch from."""
    ZbalFuseEpState.instance().ensure_pool(layer)
    return _pool_scope()


def prepare_weights(layer: FusedMoE, prefix: str) -> None:
    """Lay expert weights out for the FuseEP ABI (fp8: ND payload + padded e8m0 scale; int8: zN
    payload + per-channel int64 scale)."""
    if prefix not in _FUSE_EP_PREFIXES:
        logger.error(
            "ZBAL FuseEP: layer_id=%s unsupported weight prefix %r (the ABI expects %s)",
            getattr(layer, "layer_id", "?"),
            prefix,
            ", ".join(sorted(_FUSE_EP_PREFIXES)),
        )
        raise ValueError(f"ZBAL FuseEP: unsupported weight prefix {prefix!r}")

    # The pool is sized from one expert shape; heterogeneous layers (e.g. an MTP layer with a
    # different expert set) are not supported, so refuse them instead of mis-sizing the pool.
    key = _weights_key(layer)
    pool_key = ZbalFuseEpState.instance().pool_shape_key
    if pool_key is not None and key != pool_key:
        logger.error(
            "ZBAL FuseEP: layer_id=%s expert shape %s differs from the pool shape %s; "
            "all MoE layers (MTP/nextn included) must share one shape",
            getattr(layer, "layer_id", "?"),
            key,
            pool_key,
        )
        raise ValueError("ZBAL FuseEP requires identical expert shapes across MoE layers")

    # FuseEP weights cover routed experts only: shared experts must take their own path, fusing
    # them in breaks the E dimension the ABI expects.
    fused_shared = getattr(layer, "num_fused_shared_experts", 0)
    if fused_shared:
        logger.error(
            "ZBAL FuseEP: layer_id=%s has %s fused shared experts; launch with "
            "--disable-shared-experts-fusion",
            getattr(layer, "layer_id", "?"),
            fused_shared,
        )
        raise ValueError("ZBAL FuseEP: shared-expert fusion must be disabled")
    if layer.num_local_experts != layer.num_experts // layer.moe_ep_size:
        logger.error(
            "ZBAL FuseEP: layer_id=%s local_experts=%s but num_experts/ep=%s/%s",
            getattr(layer, "layer_id", "?"),
            layer.num_local_experts,
            layer.num_experts,
            layer.moe_ep_size,
        )
        raise ValueError("ZBAL FuseEP requires contiguous experts without redundancy")

    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        if layer.moe_ep_size != world:
            logger.error(
                "ZBAL FuseEP: layer_id=%s ep_size=%s must span world=%s (FuseEP exchange is WORLD-wide)",
                getattr(layer, "layer_id", "?"),
                layer.moe_ep_size,
                world,
            )
            raise ValueError("ZBAL FuseEP requires moe_ep_size == world_size")

    scale_attr = f"{prefix}_weight_scale"
    weight = getattr(layer, f"{prefix}_weight", None)
    scale = getattr(layer, scale_attr, None)
    if weight is None or scale is None:
        logger.error(
            "ZBAL FuseEP: layer_id=%s %s weights missing (weight=%s %s=%s)",
            getattr(layer, "layer_id", "?"),
            prefix,
            weight is not None,
            scale_attr,
            scale is not None,
        )
        raise ValueError(f"ZBAL FuseEP: {prefix} weight/scale not found")

    # Layout/packing is fully delegated to zbal's repack: this only picks the ABI and puts
    # the allocations inside the symmetric-pool scope.
    from zbal import zbal_moe_repack

    layout = zbal_moe_repack.infer_layout(weight)
    # Both weights and scale must live in the symmetric pool: FuseEP prefetches weights through
    # peers' GVAs, and addresses outside the pool are invisible to peers.
    with _pool_scope():
        repacked = zbal_moe_repack.repack_expert_weights(
            weight,
            scale,
            layout=layout,
            offset=getattr(layer, f"{prefix}_weight_offset", None),
            label=f"layer_id={getattr(layer, 'layer_id', '?')} {prefix}",
        )

    setattr(
        layer,
        f"{prefix}_weight",
        torch.nn.Parameter(repacked.payload, requires_grad=False),
    )
    setattr(
        layer,
        f"{prefix}_weight_scale",
        torch.nn.Parameter(repacked.scale, requires_grad=False),
    )
    processed = getattr(layer, "_zbal_fuseep_prefixes", set())
    layer._zbal_fuseep_prefixes = processed | {prefix}
    logger.info(
        "ZBAL FuseEP: layer_id=%s %s layout=%s payload=%s scale=%s",
        getattr(layer, "layer_id", "?"),
        prefix,
        layout,
        tuple(repacked.payload.shape),
        tuple(repacked.scale.shape),
    )


def _workspace_shape(layer: FusedMoE) -> tuple[int, ...]:
    """Inputs of the workspace query, with maxOutputSize at the worst case ``bound × EP × topk``."""
    bound = _bound_tokens()
    return (
        bound,
        layer.hidden_size,
        2 * layer.intermediate_size_per_partition,
        layer.num_local_experts,
        layer.moe_ep_size,
        layer.top_k,
        bound * layer.moe_ep_size * layer.top_k,
    )


def _workspace_bytes(layer: FusedMoE) -> int:
    """Resident workspace bytes sized from the prefill budget, which also bounds every decode
    launch (decode reserves its own, smaller, row count at launch)."""
    from zbal import Buffer

    bs, h, ffn_dim, nle, ep, topk, max_output_size = _workspace_shape(layer)
    return Buffer.get_dispatch_ffn_combine_workspace_size(
        bs,
        h,
        ffn_dim,
        nle,
        ep,
        topk,
        max_output_size=max_output_size,
        mode="prefill",
    )


def _create_buffer(layer: FusedMoE):
    """Create the FuseEP Buffer and reserve the resident workspace (called once by ``ensure_pool``)."""
    from zbal import Buffer

    group = get_parallel().moe_ep_group.device_group
    buf = Buffer(group, num_nvl_bytes=0, num_rdma_bytes=0, low_latency_mode=False)
    ws_bytes = _workspace_bytes(layer)
    buf.reserve_dfc_workspace(ws_bytes)
    return buf


def forward(layer: FusedMoE, hidden_states: torch.Tensor, topk_output: TopKOutput) -> torch.Tensor:
    """Single-layer MoE forward: dispatch + GMM1 + SwiGLU + GMM2 + combine all inside the op."""
    if getattr(layer, "_zbal_fuseep_prefixes", None) != _FUSE_EP_PREFIXES:
        logger.error(
            "ZBAL FuseEP: layer_id=%s weights not prepared (%s); expected repacked expert weights",
            getattr(layer, "layer_id", "?"),
            getattr(layer, "_zbal_fuseep_prefixes", None),
        )
        raise ValueError("ZBAL FuseEP: expert weights were not laid out for FuseEP")

    tokens = hidden_states.shape[0]
    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights

    # Scenario comes from the rank-synchronous extend flag, never from the token count: idle ranks
    # of a prefill step still launch prefill, and MLP sync refreshes the flag once per step.
    mode = "prefill" if get_is_extend_in_batch() else "decode"
    bound = _bound_tokens()
    if tokens > bound:
        logger.error(
            "ZBAL FuseEP: layer_id=%s local tokens=%s exceeds the reserved prefill bound %s "
            "(--chunked-prefill-size / --max-prefill-tokens); the workspace is sized from it",
            getattr(layer, "layer_id", "?"),
            tokens,
            bound,
        )
        raise ValueError("ZBAL FuseEP: local token count exceeds the reserved prefill bound")

    if tokens == 0:
        hidden_states = hidden_states.new_zeros((1, layer.hidden_size))
        topk_ids = torch.zeros((1, layer.top_k), dtype=torch.int32, device=hidden_states.device)
        topk_weights = torch.zeros((1, layer.top_k), dtype=torch.float32, device=hidden_states.device)

    rows = max(tokens, 1)
    max_output_size = (bound if mode == "prefill" else rows) * layer.moe_ep_size * layer.top_k

    out, _ = ZbalFuseEpState.instance().buffer.dispatch_ffn_combine(
        x=hidden_states.contiguous(),
        topk_idx=topk_ids.to(torch.int32).contiguous(),
        weight1=layer.w13_weight,
        scale1=layer.w13_weight_scale,
        weight2=layer.w2_weight,
        scale2=layer.w2_weight_scale,
        topk_weights=topk_weights.float().contiguous(),
        max_output_size=max_output_size,
        num_experts=layer.num_experts,
        quant_mode=1,
        mode=mode,
    )
    return out[:tokens]
