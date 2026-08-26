# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run the model with NPU graph and torch.compile.

NPUGraphRunner is a thin subclass of DecodeCudaGraphRunner: the
factory returns NPUCudaGraphBackend for NPU devices, so all
capture/replay mechanics live in the backend. This class adds:
  - NPU-specific patch_model monkey-patch for the decode-Full +
    torch.compile path.
  - Profile context override (NPU profiler emits to disk, not in-mem).
  - Replay override that issues an async NPUGraph.update for
    seq_lens before replay (skipped for deepseek-nsa).
  - Smaller cache_loc dtype (int32 instead of int64).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np
import torch

from sglang.srt.configs.model_config import (
    AttentionArch,
    is_deepseek_dsa,
    is_deepseek_v4,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.utils import (
    empty_context,
    get_bool_env_var,
    get_compiler_backend,
    is_npu,
)

is_npu = is_npu()

if is_npu:
    import torch_npu
    from torch_npu.profiler import ProfilerActivity, profile

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors


@contextmanager
def patch_model_npu(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    if enable_compile:
        backend = get_compiler_backend("npugraph_ex")
        yield torch.compile(
            torch.no_grad()(model.forward),
            fullgraph=True,
            dynamic=False,
            backend=backend,
        )
    else:
        yield model.forward


class NPUGraphRunner(DecodeCudaGraphRunner):
    """A NPUGraphRunner runs the forward pass of a model with NPU graph and torch.compile."""

    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        attn_backend=None,
        speculative_num_steps: Optional[int] = None,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        # NPU patch_model override: monkey-patch torch_compile_decoration's
        # patch_model with the NPU-specific version.
        from sglang.srt.compilation import torch_compile_decoration

        torch_compile_decoration.patch_model = patch_model_npu
        super().__init__(
            model_runner,
            attn_backend=attn_backend,
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        self.update_attr_name = None
        self.update_attr_type = None
        self.model_runner = model_runner
        self._init_arch_map()
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        architectures = (
            vars(model_runner.model_config.hf_config).get("architectures") or []
        )
        self.if_use_v2 = any(
            arch
            in ("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM", "Step3p5ForCausalLM")
            for arch in architectures
        ) or (
            model_runner.model_config.attention_arch == AttentionArch.MLA
            and model_runner.kv_cache_dtype == torch.float8_e4m3fn
        )

    def _init_arch_map(self):
        if self.is_dllm:
            self.attr_name: Dict[str, str] = {
                AttentionArch.MLA: "actual_seq_lengths_kv",
                AttentionArch.MHA: "actual_seq_lengths_kv",
                "TARGET_VERIFY": "actual_seq_kvlen",
            }
        else:
            self.attr_name: Dict[str, str] = {
                AttentionArch.MLA: "actual_seq_lengths_kv",
                AttentionArch.MHA: "context_lens",
                "TARGET_VERIFY": "actual_seq_kvlen",
            }
        self.attr_type: Dict[str, Union[list, torch.Tensor]] = {
            AttentionArch.MLA: [],
            AttentionArch.MHA: torch.Tensor(),
            "TARGET_VERIFY": [],
        }

    def _create_device_graph(self):
        return torch.npu.NPUGraph()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        if self.enable_torch_compile:
            skip_guard_context = torch.compiler.set_stance(skip_guard_eval_unsafe=True)
        else:
            skip_guard_context = empty_context()

        with (
            skip_guard_context,
            torch.npu.graph(
                graph,
                pool=pool,
                stream=stream,
                auto_dispatch_capture=True,
            ),
        ):
            out = run_once_fn()
        return out

    def _get_update_attr_name(self):
        if self.if_use_v2:
            return self.attr_name["TARGET_VERIFY"]
        return self.attr_name[AttentionArch.MLA]

    def _get_update_attr_type(self):
        if self.if_use_v2:
            return self.attr_type["TARGET_VERIFY"]
        return self.attr_type[AttentionArch.MLA]

    def _update_inputs(self, seq_lens):
        if isinstance(self.update_attr_type, torch.Tensor):
            seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))

        self.graphs[self.bs].update(
            cpu_update_input=[{self.update_attr_name: seq_lens}]
        )

    def _cache_loc_dtype(self):
        return torch.int32

    def _init_profile_context_and_memory_record(self):
        output_dir = os.path.join(
            os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp"), "graph_capture_profile"
        )
        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Profiling starts for graph capture for NPU. Traces will be saved to: {output_dir}"
        )
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        )
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                output_dir, async_mode=True
            ),
            experimental_config=experimental_config,
        )
        return profile_context

    def _post_process_after_profile(self, prof_context):
        # for NPU, profile data will be saved to disk for further analysis.
        pass

    def execute(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        if forward_batch.needs_forward_metadata_init():
            self.load_batch(forward_batch, pp_proxy_tensors)
        else:
            # In speculative decoding, these two fields are still needed.
            self.buffers.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.buffers.positions[: self.raw_num_token].copy_(forward_batch.positions)
            if (
                self.model_runner.spec_algorithm.is_dflash()
                and self.model_runner.is_draft_worker
                and forward_batch.input_embeds is not None
            ):
                self.buffers.input_embeds[: self.raw_num_token].copy_(
                    forward_batch.input_embeds
                )
            if (
                envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get()
                and forward_batch.mrope_positions is not None
            ):
                self.buffers.mrope_positions[:, : self.raw_num_token].copy_(
                    forward_batch.mrope_positions
                )

        graph_key = self._make_graph_key(self.bs)

        if not (
            is_deepseek_dsa(self.model_runner.model_config.hf_config)
            or is_deepseek_v4(self.model_runner.model_config.hf_config)
        ):
            if forward_batch.forward_mode.is_target_verify():
                seq_lens_cpu = forward_batch.seq_lens.cpu() + self.captured_req_width
                seq_lens = seq_lens_cpu.tolist() + [0] * (self.bs - self.raw_bs)
            else:
                seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (
                    self.bs - self.raw_bs
                )
            output = self.backend.replay_with_input_update(
                graph_key,
                seq_lens=seq_lens,
                attr_name=self._get_update_attr_name(),
                attr_type=self._get_update_attr_type(),
            )
        else:
            output = self.backend.replay(graph_key, forward_batch)

        # DEBUG (graph-mode precision, NPU override): NOTE this method fully
        # overrides DecodeCudaGraphRunner.execute — probes there never run.
        # SGLANG_SELECTIVE_DEBUG_SYNC=1  -> device sync after replay.
        # SGLANG_SELECTIVE_DEBUG_SLEEP_MS=<n> -> wall-clock delay (covers
        #   host-report-driven fire-and-forget DMA that sync cannot see).
        # SGLANG_SELECTIVE_DUMP_STAGING=1 -> staging checksum after replay.
        _dbg = os.getenv("SGLANG_SELECTIVE_DEBUG_SYNC", "0") == "1"
        _sleep_ms = os.getenv("SGLANG_SELECTIVE_DEBUG_SLEEP_MS", "0")
        _dump = os.getenv("SGLANG_SELECTIVE_DUMP_STAGING", "0") == "1"
        if _dbg or _sleep_ms != "0" or _dump:
            torch.npu.synchronize() if _dbg else None
            if _sleep_ms != "0":
                time.sleep(int(_sleep_ms) / 1000.0)
            if _dump and not forward_batch.forward_mode.is_idle():
                _sel_coord = getattr(
                    self.model_runner, "npu_selective_hisparse_coordinator", None
                )
                if _sel_coord is not None:
                    _n = int(_sel_coord.h2d_cnt.item())
                    _flat = _sel_coord.packed_staging_all.view(-1)[
                        : _n * _sel_coord.record_bytes
                    ]
                    logger.info(
                        "[STAGING-DUMP graph] key=%s N=%d sum=%d nonzero=%d",
                        graph_key.size,
                        _n,
                        _flat.sum(dtype=torch.int64).item(),
                        int((_flat != 0).sum().item()),
                    )

        # D1 content-diff dump: after each non-idle replay, snapshot the
        # per-layer debug capture buffers (their captured copies executed
        # inside this replay; .cpu() in dump_diff_snapshot syncs the
        # stream, freezing the values this replay actually produced).
        if (
            os.getenv("SGLANG_SELECTIVE_DIFF_DUMP", "0") == "1"
            and not forward_batch.forward_mode.is_idle()
        ):
            _sel_coord = getattr(
                self.model_runner, "npu_selective_hisparse_coordinator", None
            )
            if _sel_coord is not None and getattr(
                _sel_coord, "_dbg_dump", False
            ):
                _sel_coord._dbg_replay_step += 1
                if _sel_coord._dbg_replay_step <= _sel_coord._dbg_max_steps:
                    _sel_coord.dump_diff_snapshot(
                        _sel_coord._dbg_replay_step, self.raw_num_token
                    )

        if isinstance(output, LogitsProcessorOutput):
            if self.is_dllm:
                next_token_logits = None
                full_logits = (
                    output.full_logits[: self.raw_num_token]
                    if output.full_logits is not None
                    else None
                )
            else:
                full_logits = None
                next_token_logits = (
                    output.next_token_logits[: self.raw_num_token]
                    if output.next_token_logits is not None
                    else None
                )
            return LogitsProcessorOutput(
                next_token_logits=next_token_logits,
                full_logits=full_logits,
                hidden_states=(
                    output.hidden_states[: self.raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
            )
        else:
            assert isinstance(output, PPProxyTensors)
            return PPProxyTensors({k: v[: self.bs] for k, v in output.tensors.items()})
