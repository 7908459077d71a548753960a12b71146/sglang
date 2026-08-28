# NPU Selective HiSparse 图模式精度问题 — 调试记录与最终结论

> 交接文档。问题：P/D 分离 + Selective HiSparse（19 个 selected 层 KV 驻留 Host DRAM）部署 GLM-5.2 时，D 节点 decode 开启 NPU graph 后精度随 selected 层数线性劣化（7 层 0.90-0.96 → 19 层 0.74-0.82；eager 锚点 0.94）。
>
> **最终状态（2026-08-27）：CASE CLOSED，全部收尾完成。** 根因 = RoPE cos/sin 图捕获冻结（第三个根因，见 §4）。修复后 19 层 50 题 **0.90**，层数线性劣化机制消除。前两个根因（kernel 队列类型、ACL 回调订阅，见 §3）是更早阶段 0.7→0.9 的真实修复，仍然有效。completion-flag 协议三步退役（§5）全部完成并逐一复验 0.90。
>
> **补注（2026-08-28）**：第二轮 diff-dump 探针扩展已落地（见 §7），作为后续任何精度回归的定位工具备用。

## 1. 背景与架构

### 运行时 KV 数据流（图模式下每步 replay 内）

```
anchor 层(L-3) topk 计算
    → build_loc_plan: topk 位置 → Host pool 逻辑 loc
    → H2D: sparse_copy 把 topk KV 从 Host DRAM 预取到 HBM packed_staging   [MIX DMA]
    → selected 层(L) set_kv_buffer → publish_new_packed_kv:
        · new_packed_scratch.copy_(packed_kv)                               [torch 向量算子]
        · D2H: sparse_copy 把新 KV 备份到 Host DRAM                          [MIX DMA]
    → run_selected_attention:
        · current-patch: 用 scratch 内容覆盖 staging 的 current 行           [torch 向量算子]
        · unpack 656B → BF16 → SFA 稀疏注意力                                [计算算子]
```

关键文件：
- sglang: `python/sglang/srt/hardware_backend/npu/selective_hisparse.py`（coordinator，DMA/patch/SFA 编排）
- sglang: `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`（**RoPE 根因所在**）
- sglang: `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`（NPU 实际走的 execute override）
- sglang: `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py`（set_kv_buffer → pack → publish）
- memfabric: `src/acc_offload/csrc/operators/acc_offload_sparse_copy.h/.cpp`（sparse_copy AIV kernel）

### 硬件/调度事实

| kernel 类型 | 调度队列 | 图 replay 行为 |
|---|---|---|
| `KERNEL_TYPE_AIV_ONLY` | 独立向量核队列 | 与计算队列**并行推进、无完成排序** |
| `KERNEL_TYPE_MIX_AIV_1_0` | 计算核队列 | 与计算算子 **FIFO 串行** |

- AIV kernel 内部 MTE 异步完成依赖 ACL host 回调线程处理 `acl.rt.process_report`；订阅按流（`acl.rt.subscribe_report`）。
- **NPU graph capture 流程**（`npu_cudagraph_backend.capture_one`）：先对**同一个 forward_batch** 跑 2 次 warmup，再正式 capture —— 这是根因 3 的触发条件。

## 2. 时间线总览

| 阶段 | 实验 | 结果 | 结论 |
|---|---|---|---|
| 早期 | eager 基线 | 0.94 正常 | 逻辑本身正确，问题图模式专属 |
| 早期 | exp F staging checksum | 两侧逐层吻合 | DMA 数据落地正确 → 时序问题 |
| 根因 1 | sparse_copy/wait_flag 改 `MIX_AIV_1_0` | 0.70 → 0.78 | copy 与 SFA 同队列 FIFO 有序 |
| 根因 2 | 订阅 replay 流到 ACL 回调线程 | 0.78 → 0.90 | MTE 完成报告的 host 侧处理 |
| 层数扫描 | 7L/10L/19L | 0.90-0.96 / 0.88 / 0.74-0.82 | **7-9 层阈值 + 线性劣化 ~1.25pt/层** |
| 排除法 R5-R13 | per-layer staging/scratch/loc-plan、ping-pong、删 D2H wait、H2D wait A/B、MTE3_S 回退 | 全部波段内 no-op | python 可及的所有杠杆均非毒源 |
| 排除法 B/C/E/C8 | notify tail、plain copy、spec 维度（draft=1/3/6） | 全部波段内 no-op | flag 协议、spec 维度排除 |
| **D1 定位** | 内容级 diff dump（eager vs graph 逐字段） | kin 位级一致 / **kinv 差 65%** / pos 一致 | **锁定 RoPE**（见 §4） |
| **修复** | 捕获期强制重算 cos/sin | 19 层 **0.90** | 案件关闭 |
| 收尾 step-1 | graph D2H 切 plain sparse_copy | 0.90 | notify tail 无消费者，可删（修复后复验） |
| 收尾 step-2 | graph H2D 切 plain + 删 wait_flag/flag 缓冲 | 0.90 | H2D wait 与 flag 协议 python 侧证伪 |
| 收尾 step-3 | memfabric 删 sparse_copy_notify/wait_flag 算子及 notify 尾巴，重编部署 | 0.90 | flag 机制整体退役收官，纯 MIX plain copy 即最终形态 |

**排除法全程的重要方法论**：gsm8k 50 题单次运行有 ±3pt 散布，判读必须用波段；前 15 轮"no-op"结论中有部分是在 RoPE bug 主导噪声下测的，修复后需复验（step-1/2 即此目的）。

## 3. 根因 1/2（早期已修复，0.7 → 0.9）

### 根因 1：kernel 队列类型错误
`sparse_copy` 原为 `KERNEL_TYPE_AIV_ONLY`：图 replay 时 copy 在向量队列、patch/SFA 在计算队列并行推进 → SFA 读到未落地的 DMA 数据。
**修复**：copy 与 wait_flag 改 `KERNEL_TYPE_MIX_AIV_1_0`（zbal 范式）。文件：`acc_offload_sparse_copy.cpp`。

### 根因 2：ACL 回调未订阅 replay 流
MTE 异步完成报告按流路由；原实现只订阅 capture 流，replay 在主流执行从未被订阅 → 图内 sparse_copy 的完成报告无人处理。
**修复**：`prepare_graph_replay` 中无条件订阅当前流（幂等）——此订阅在 flag 协议退役后仍保留（plain copy 的 DMA 完成同样依赖它）。

## 4. 根因 3（最终根因）：RoPE cos/sin 图捕获冻结

### 现象约束（15 轮排除法收敛出的指纹）
- 纯层数效应：7-9 层阈值 + 线性 1.25pt/层
- eager 完全干净（跨 run 位级确定）
- 所有同步机制（flag/wait/notify/sleep/sync）全部 no-op

### 定位过程（D1 内容级 diff，逐字段二分）
工具：`selective_hisparse.py` 的 DIFF_DUMP 埋点（`SGLANG_SELECTIVE_DIFF_DUMP=1`）+ `hisparse_diff_compare.py`（repo 根目录）。warmup 步（所有 DP rank 内容相同）即可定罪。

| 步骤 | 证据 | 排除/锁定 |
|---|---|---|
| ① | q margin 6.5e-08、locs/valid/stg_pre 精确匹配 | 上游模型状态、H2D 数据、patch 计划全部排除 |
| ② | pub == pkg（run 内比较，两侧 0/6） | stale scratch 读取排除 |
| ③ | eager vs eager2 跨 run 全字段位级 MATCH | 量化算子非确定性排除 |
| ④ | **kin（k_nope，不旋转）0/6 bad vs kinv（k_rope，过 RoPE）6/6 bad rel 0.65** | 锁定 RoPE——同一投影输出的两个切片，没旋转的位级一致 |
| ⑤ | pos（forward_batch.positions）匹配 | positions buffer 本身排除 → 毒在 RoPE 读 positions 的方式 |

### 根因机制
`_apply_dsa_interleave_half_rope`（`SGLANG_NPU_USE_MLAPO=1` 启用，DSA 路径）把 cos/sin latch 到 `forward_batch.npu_dsa_interleave_half_rope_cache` 供层间复用：

1. capture 流程先跑 2 次 warmup（同一个 forward_batch）→ 第 1 次 warmup 用**假 positions** 算出 cos/sin 并 latch
2. capture run 走缓存分支 → **`index_select(positions)` 计算链从未被录进图**
3. 每次 replay：positions buffer 正确更新（pos 捕获证实），但 cos/sin 永远是 warmup 时刻的角度 → q_pe/k_pe 全部用错角度旋转
4. 每个 selected layer 每步都把错角度 k_rope 写进 KV/host 池 → 毒按层数线性累积

完美解释全部观测：kin 位级一致（不旋转）、kinv 差 65%（角度错）、pos 匹配（buffer 没问题）、线性劣化（每层注入同剂量毒）、eager 干净（每步新 ForwardBatch 重算）。

### 修复（一处改动，覆盖所有图）
`deepseek_v2_attention_mla_npu.py` `_apply_dsa_interleave_half_rope`：

```python
_capturing = False
try:
    _capturing = torch.npu.is_current_stream_capturing()
except (AttributeError, RuntimeError):
    pass
if rope_cache is None or _capturing:
    m.rotary_emb.get_cos_sin_with_position(positions)
    ...  # 重算并 latch
```

捕获期强制重算 → 计算链被录进图 → replay 用真实 positions 重算。eager latch 语义不变。

### 验证
- warmup 步 dump：kinv/pub/stg_post 0/6 bad，out max rel diff 2.1e-07（全字段位级一致）
- 19 层 50 题：**0.90**（修复前 0.74-0.82；npu graph True 全图生效）
- 剩余 4pt = graph/eager 常规数值差（7 层时代即有同款 gap）

### 通用教训（同类隐患模式）
任何「首次计算后 latch 到 forward_batch 供后续层复用」的 positions 依赖缓存，在 NPU capture 流程（warmup×2 + capture 共用同一 forward_batch）下都会把计算链挡在图外。排查清单：
- `npu_dsa_interleave_half_rope_cache`（已修）
- `npu_mlaprolog_runtime_cache`（LongCat，layer0 无条件刷新，语义正确）
- `rotary_emb.sin_cos_cache`（start_layer 条件刷新，同型风险，建议复核）

## 5. 收尾清理（flag 机制退役，全部完成）

RoPE 修复后，completion-flag 协议（`sparse_copy_notify` + `wait_flag`）按 A/B 步进退役（每步 50 题验证波段保持，三步均为 **0.90**）：

- **step-1（完成，0.90）**：graph D2H 切 plain `sparse_copy`（R11 删 D2H wait 后 notify tail 无消费者，纯开销）
- **step-2（完成，0.90）**：graph H2D 切 plain `sparse_copy` + 删 `wait_flag` 调用 + 删 4 个 flag/expect 缓冲。依据：copy 与消费者同在计算队列 FIFO 有序；ACL 流订阅保留
- **step-3（完成，0.90，重编部署后复验）**：memfabric 仓库全链删除 `sparse_copy_notify`/`wait_flag`——kernel 内 notify 尾巴与 `OffloadWaitFlagKernel` 类、`OffloadOpsWaitFlag`/`AccOffloadWaitFlag` launch 链、entry 层 `WaitSparseCopyDone`、C API `offload_sparse_copy_notify`/`offload_wait_flag`、pybind 与 python 导出（共 14 处，涉及 `acc_offload_sparse_copy.h/.cpp`、`acc_offload_operators.h`、launch 三件套、entry 四件套、`acc_offload.cpp`、`acc_offload.h`、`pymf_acc_offload.cpp`、`mf_acc_offload.py`、`__init__.py`）。`offload_sparse_copy` 吸收原 notify 实现为唯一入口；`KERNEL_TYPE_MIX_AIV_1_0`（根因 1 修复）保留

最终形态：图内外 H2D/D2H 均为 plain MIX `sparse_copy`，无任何 flag/wait 机制；数据通路排序完全依赖计算队列 FIFO + ACL 流订阅。

debug 探针清理（已完成）：`SGLANG_SELECTIVE_PLAIN_COPY/DEFER_D2H/GATE_D2H_EVENT/SKIP_H2D_WAIT/DUMP_STAGING/DISABLE_D2H/CHECK_FLAGS/DEBUG_SYNC/DEBUG_SLEEP_MS` 及 `_pending_d2h`/`_launch_deferred_d2h` 等全部删除。**保留**：DIFF_DUMP 全套（`SGLANG_SELECTIVE_DIFF_DUMP/DUMP_DIR/DUMP_MAX_STEPS`、`_dbg_*` 缓冲、`dump_diff_snapshot`、`debug_capture_kin`、npu_graph_runner replay 触发）、`D_EAGER`/`MAX_RUNNING_REQ`（eager 对照 dump 工作流）、`STAGING_SLICES`/`UNPACK_WS`（ping-pong 设计参数）、eager 诊断日志、`hisparse_diff_compare.py`。

## 6. 当前代码状态与验证配置

### sglang 侧最终形态
- RoPE 捕获期重算（根因 3 修复）：`deepseek_v2_attention_mla_npu.py`
- 图模式 H2D/D2H 均 plain `sparse_copy`（flag 协议已退役），ACL 流订阅无条件保留：`selective_hisparse.py`
- per-layer staging/scratch/loc-plan + ping-pong workspace（R5/R9/R10，防跨层竞争）：`selective_hisparse.py`
- idle 早退先于 forward_metadata 访问：`dsa_npu_indexer.py`
- DIFF_DUMP 埋点与比对工具：`selective_hisparse.py` / `memory_pool_npu.py` / `npu_graph_runner.py` / `hisparse_diff_compare.py`

### memfabric 侧最终形态
- `sparse_copy` 为唯一拷贝入口（`KERNEL_TYPE_MIX_AIV_1_0`），kernel 无 notify 尾巴；`sparse_copy_notify`/`wait_flag` 全链移除（详见 §5 step-3）

### 验证配置速查
```bash
# D 节点启动（19 层标准规格，repo 根目录）
D_MEM_FRACTION=0.915 \
SELECTIVE_LAYER_IDS="5 9 13 17 21 25 29 33 37 41 45 49 53 57 61 65 69 73 77" \
bash pd_disaggregation_dram_offload.sh

# bench（路由节点）
python -m sglang.test.few_shot_gsm8k --host http://141.61.49.198 --port 6688 \
    --num-questions 50 --num-shots 5 --data-path /home/r00648901/GSM8K.jsonl

# diff-dump 对照（需要时）
# eager 侧: D_EAGER=1 MAX_RUNNING_REQ=24 + SGLANG_SELECTIVE_DIFF_DUMP=1 SGLANG_SELECTIVE_DUMP_DIR=<dir>
# graph 侧: 标准启动 + 同款 dump env；比对: python hisparse_diff_compare.py --eager-dir <e> --graph-dir <g>
#   工具自动取两边共同 dev（同 DP rank 同请求流）；[VERDICT] 段自动裁决量化输入/输出

# 已知配置坑
# 1) --disable-cuda-graph 时保留 --cuda-graph-bs，否则 fixed bias 按默认 bs 列表 max=512 计算到 14GB
# 2) eager 满载 token 数超 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128 复发 161002
# 3) eager 19 层需 MAX_RUNNING_REQ=24 压内存
```

### 精度矩阵（RoPE 修复后）
| 配置 | 精度 |
|---|---|
| eager 19 层 | 0.94（锚点） |
| graph 19 层（修复前） | 0.74-0.82（线性劣化） |
| graph 19 层（修复后） | **0.90** |
| graph 19 层（step-1：D2H plain） | 0.90 |
| graph 19 层（step-2：H2D plain + 删 wait/flag） | 0.90 |
| **graph 19 层（step-3：memfabric 算子删除，最终形态）** | **0.90** |

判读规则：±3pt 为噪声波段；关键结论需复测。

## 7. 第二轮 dump 点扩展（2026-08-28，待跑）

针对此前剩余嫌疑（torch 向量算子与 MIX copy 的跨队列竞态）及代码走读新发现的暗箱，在 D1 diff-dump 上扩展了 6 类探针。开关不变：`SGLANG_SELECTIVE_DIFF_DUMP=1`（+ 可选 `SGLANG_SELECTIVE_DUMP_DIR`、`SGLANG_SELECTIVE_DUMP_MAX_STEPS`），eager/graph 各跑一次同数据集，逐字段对齐比对。

### 新增探针清单

| 字段 | 采集点 | 针对嫌疑 |
|---|---|---|
| `d2h_locs` [L,T] | `publish_new_packed_kv`（D2H 写址 logical_locs，graph 下 captured op） | Host pool 污染——写址本身错位（与 `locs`（H2D 读址）闭环） |
| `d2h_rb` [L,T] | `dump_diff_snapshot` 内 Host pool 按本轮 d2h_locs 回读（eager 先 drain） | Host pool 污染——写址对但内容旧/错（D2H 与下一轮 scratch overwrite 竞态的铁证） |
| `qrope` [L,T] | `run_selected_attention`（q_rope rowsum，此前只采 q_nope） | RoPE/cos-sin recompute（最近 `fix cos/sin recompute` 提交的活跃嫌疑） |
| `unpk`/`unpkr` [L,T] | `selective_sparse_attention` 反量化完成后、SFA 前（captured op） | 拆分「unpack 链 vs SFA kernel」：stg_post 吻合但 out 偏离时的中间盲区 |
| 标量 `real_tokens`/`h2d_cnt`/`d2h_cnt` | `dump_diff_snapshot` post-replay 读取 | runner 传入 real_num_tokens 错误 → DMA 提交错误条目数 |
| `accept_stepNNNN.json` | `on_verify_result`（每 verify 轮 accept_lens） | 最快找到 eager/graph 输出首个发散 step（与 snapshot step 编号 1:1） |

### 判读决策树（graph vs eager 逐字段比对）

```
accept_lens 首个发散 step = S
 └─ step S 起逐层看：
    pos/kin/kinv 偏 → 上游 KV 投影/RoPE 输入已错（图基础设施级）
    kin/kinv 吻合 + pub 偏 → fp8 quant 在图内产出不同（captured-replay bug）
    pub 吻合 + pkg 偏 → patch 读到旧 scratch（跨队列竞态实锤，疑点①）
    locs/d2h_locs 偏 → topk 或 req_to_token 映射在图内错位
    d2h_locs 吻合 + d2h_rb 偏 → D2H 读到被覆盖的 scratch（Host pool 污染实锤，疑点②）
    q 偏 / qrope 偏 → Q 投影 / RoPE 分支（另一路独立于 KV 链路）
    stg_pre 偏 → H2D 数据未落地即被消费 或 Host 读址内容已错（结合 d2h_rb 判定）
    stg_pre/stg_post 吻合 + unpk 偏 → unpack 反量化链图内行为不同
    全部吻合但 out 偏 → SFA kernel 自身（npu_sparse_flash_attention 图 replay 问题）
```

注意事项：
- `stg_pre/stg_post/pkg/pub` 等 captured op 冻结的是**本轮 replay 消费时刻**的值（非 replay 后已自愈值），这是 D1 的核心机制，新探针沿用。
- `d2h_rb` 依赖「stream sync ⇒ in-graph MIX D2H 对 Host 写可见」这一假设；若 graph 侧 d2h_rb 出现 eager 侧没有的偶发噪声，先怀疑该假设本身（本身就是有价值的发现）。
- dump 只在前 `SGLANG_SELECTIVE_DUMP_MAX_STEPS`（默认 20）个非 idle verify 轮生效；比对时两侧用同数据集同并发（max-concurrency 1）。

### 首轮 dump 比对结论（2026-08-28，20 steps，bs1）

- **pair 0（step 1）全字段吻合**：pos/kin/kinv/pub/pkg/locs/stg_pre/stg_post/out 全 match（out 最大相对差 2.1e-07）→ **selected 层 KV 链路（H2D 预取、current-patch、unpack、SFA）在 step 1 完全无辜**。
- **eager step 2 起无法对齐**（no matching graph step remains）：step 1→2 之间输出已发散，或两次跑调度错位。发散位置在探针未覆盖的区域：SFA 下游（logits/sampler/verify 树）、非 selected 层、或 draft 侧。
- 已排除（step 1 范围内）：跨队列竞态（§4 疑点①②）、fp8 quant 图内偏差、Host pool 污染、KV 侧 RoPE 偏差。

### 第二轮探针：verify logits 指纹（2026-08-28，已埋点待跑）

`on_verify_result` 增加可选 `logits_output` 参数（eagle_worker_common 调用点传入），json 文件名带 dev（`accept_dev{N}_step{S}.json`，防 dp8 多 rank 同目录互相覆盖），内容从只记 accept_lens 扩展为：

```json
{"step": 1, "accept_lens": [...],
 "logit_argmax": [...],   // next_token_logits 逐行 argmax（贪心 token id）
 "logit_rowsum": [...]}   // next_token_logits 逐行 float32 rowsum（数值指纹）
```

`hisparse_diff_compare.py` 已加 `=== verify-round fingerprint ===` 段：json 是同号 step 直接比对（不走 .pt 的 q 对齐，两侧各自从 1 计数），在 .pt 对齐失败时仍能判定。判读（以首个发散 step 为准）：
- `logit_rowsum` step 1 即偏 → **target 前向在图内发散**（非 selected 层 / lm_head / logits processor）→ 下一轮加 hidden-state 逐层（或隔层采样）rowsum 二分探针
- rowsum 吻合但 `logit_argmax`/accept_lens 偏 → 采样器 / RNG 状态 / verify 树处理发散，模型前向无辜
- step 1 全吻合但 step 2 输入指纹偏 → draft 侧（NEXTN draft forward）或 scheduler 状态（seq_lens/out_cache_loc 更新）发散 → 加 draft 侧指纹探针
- 全部吻合 → 纯调度错位（两次跑请求/时序不同），固定数据重跑对齐

### 第二轮 dump 比对结论（2026-08-28，bs 默认 8 16）

- **首个发散点钉死：verify round step 3 的 target 前向**（logits rowsum rel 0.77、argmax 4/6 偏）；step 1-2 指纹完全吻合（rowsum rel 0）。
- step 3 同时 `.pt` 对齐断在 L5 的 q 上 → 发散在**进入 layer 5 之前**（layer 0-4 非 selected 层计算、或其 resident KV 被 step 1-2 写坏）或 step 3 输入本身已偏。
- step 9 起 token 数 6→18（更多请求加入），其后全崩（含 accept_lens）——下游级联，非独立事件。
- **已排除**：selected 层全链路（step 1-2 全字段吻合）、跨迭代竞态（exp R2 级别，因 step 1-2 输出一致）。

### 第三轮探针：target 前向逐层二分（2026-08-28，已埋点待跑）

| 字段（state_dev{N}_step{S}.pt） | 采集点 | 判定 |
|---|---|---|
| `in_ids`/`in_pos` [T] | 模型 forward 入口（embedding 后；graph 下 captured） | step 3 输入即偏 → draft/scheduler 侧，target 无辜 |
| `hidden` [L+1,T] | 逐 decoder 层输出 rowsum（row 0=embedding，i+1=层 i；captured） | 首个发散 row = 首个发散层（embedding/中间层/全过→lm_head） |
| `rkv`/`rloc` [L+1,T] | `set_kv_buffer` resident 路径（packed rowsum + 写址；captured） | **step<3 的 rkv/rloc 偏 → resident KV 被 graph 写坏（毒源实锤）**；吻合 → live 计算/attn metadata 偏 |

改动文件：`selective_hisparse.py`（buffer+方法+state 落盘）、`model_runner.py`（传 num_hidden_layers）、`deepseek_v2.py`（forward 入口 + 层循环两个 hook，`forward_batch.npu_selective_hisparse_coordinator` 获取）、`memory_pool_npu.py`（resident 写捕获）、`hisparse_diff_compare.py`（`=== target-forward bisect ===` 段 + `[BISECT VERDICT]`）。

### 第三轮 dump 比对结论（2026-08-28）

- **step 1-2 resident-KV 写完全吻合**（rkv/rloc 逐层 match）→ **resident KV 毒化排除**。
- **step 3 起 verify 输入 token id 即不同**（positions 相同、树形一致）→ **target 前向无辜**（不同输入当然不同输出），毒在 **NEXTN draft 链或 verify→draft 交接**。
- 至此 target 侧全链路（selected 层、resident 层 KV 写、逐层 hidden、logits）在 step 1-2 全部证实干净。

### 第四轮探针：root-vs-draft 二分 + draft 链 bracket（2026-08-28，已埋点待跑）

**无需重跑即可先做的判定**（analyzer 已扩展，对现有 dump 重跑比对即出）：首个发散 step 打印 in_ids 逐 token 对比。topk=1 时 verify 输入是链式：`in_ids[0]` = 上一轮**接受 token**（上轮 logits/accept_lens 已证相同，理应一致），`in_ids[1:]` = 本轮 draft 新提案。

- `in_ids[0]` 偏 → **eagle_sample/accept_index compaction bug**（logits 相同却接受了不同 token）
- `in_ids[0]` 同、后面偏 → **draft 链发散** → 看新 draft 字段：
  - `din_hidden` 偏 → 交接进 draft 的 hidden 态已错（上一轮 verify 的 hidden/采样态）
  - `din_hidden` 同 + `dout_toks` 偏 → **draft 前向本身在图内发散**（draft attention/KV/metadata）
  - `dout_toks` 同但 in_ids 偏 → `build_eagle_verify_input` 树拼装 bug

### 第四轮 dump 比对结论（2026-08-28，旧 dump 重析）

**root-vs-draft 判定已出**（无需重跑）：step 3 的 in_ids 逐 token 对比——root（上轮接受 token）**吻合**（382=382），slot[1][2] 吻合，**slot[3] 偏离（eager=557，graph=0）**，slot[4][5] 吻合。

- **eagle_sample/accept_index 无辜**（接受了相同 token）。
- **draft 链第 3 个提案位发散**，且 graph 侧提案为 token 0（疑似 logits 退化/被清零的征兆）。
- 待 draft 字段（din/dout/dstep）跑出后定：交接态偏 vs draft 前向偏 vs 树拼装偏。

### 第五轮 dump 比对结论（2026-08-28，旧 dump 重析）

- **NaN 归属定案**：graph 侧 dstep_logits/dstep_hidden 各 4 个 NaN（= 全部 4 个已写 step，每步 1 个），eager 侧全 0；`din_hidden` 双侧无 NaN 仅数值不同。
- **pre-div step 1 verify 全净**（含 final-hidden row）→ verify 侧至此彻底洗清。
- slot[1]（链初始提案，来自上轮交接）吻合 + 链内 4 步输出全崩为 0 → **draft 前向从 step 0 就 NaN**。
- 综合：handoff hidden（graph pool 张量）在 draft replay 期间内容已变（post-read 无 NaN 但数值不同 → 同一地址在 replay 过程中被写过）。**pool aliasing 假设当前最优**：hisparse 在 verify 图内的大量临时分配（build_loc_plan 的 where/arange、patch 的 gather、unpack 链、sort 等，每 replay 数百次）加剧 graph pool churn，verify 输出 hidden 在仍被 draft 引用时被覆写。

### 第五轮补充判定（analyzer 已扩展）

- pre-div 轮 hidden 全行 + in_ids 对比（已上，本轮"match (final-hidden row ok)"即出自它）。
- NaN/Inf 归属表（已上）。
- `din_topk`（链初始提案输入）vs `din_hidden`：**din_topk 同 + din_hidden 偏 = hidden 张量专属污染（pool aliasing 签名）**；两者都偏 = 交接态整体污染。

### 第五轮重析结论（2026-08-28）

**`din_topk` 吻合 + `din_hidden` 偏 → aliasing 签名命中**（hidden 张量专属污染）。链条结构理清：verify → **draft-extend**（batch 2，独立 graph）→ 下一轮 draft 链；下一轮 din 是 draft-extend 输出经 fancy-index gather 的**新张量**，其脏只能来自 gather 源（draft-extend 图输出 pool 张量）或更上游 verify 输出在 draft-extend 读它时已脏。

### 第五轮 b：dext_out 探针 + clone 缓解实验（已埋点待跑）

- **探针**：`hout_live`（verify 输出，sample 后活读）、`dext_out`（draft-extend 输出，gather 前活读）、`din`（draft 执行前读）——analyzer 在首发散轮打印三元组链并判定污染窗口：
  - `hout_live(d-1)` 坏 → verify 输出在 sample 读时已脏
  - `hout_live(d-1)` 净 + `dext_out(d)` 坏 → 窗口 = verify sample 后到 draft-extend gather 前（含 draft-extend replay 自身写）
  - `dext_out(d)` 净 + `din(d)` 坏 → gather/select_index 路径
  - 全净但 draft 仍 NaN → 毒不在 hidden 链（draft KV/metadata）
### 第六轮 dump 比对结论（2026-08-28，CLONE=1 跑）

- **clone 把 hidden 交接链完全洗净**：din 在 step 2-9 全 ok（此前 dinBAD），三元组 hout_live(d-1)/dext_out(d)/din(d) max rel 全 0。
- **但 draft 链仍从 step 0 NaN**（graph 侧 dstep nan=4 不变，提案仍退化为 0）→ **NaN 源在 draft 前向自身的其他输入**（draft KV / attention metadata / out_cache_loc），不在 hidden 交接。hidden-aliasing 作为 draft NaN 解释**降级**。
- **eager 与 graph 精度同降 0.2**（模式无关回归；本轮跑法 = DIFF_DUMP=1 + CLONE=1）。来源待隔离（见矩阵）。
- **重新定性**：垃圾 draft 提案本身不应伤贪心精度（verify 只接受匹配 token；accept_len=1 时输出仍由 root 行 target argmax 决定）→ 0.90↔0.94 差距与双降更可能是**共享状态污染**（out_cache_loc 分配错 → KV 落位错 → draft 读垃圾 NaN、target 读错位 KV 劣化）。draft NaN 可能是同毒异症。

### 第六轮 b：跑法 A 结果（2026-08-28，DIFF_DUMP=1、CLONE=0）

- **精度：eager 0.92（≈0.94 噪声带内）、graph 0.90（不变）→ diff-dump 探针基本无损**；此前 -0.2 双降主嫌收敛到 CLONE（B/C 跑法可最终确认，优先级降低）。
- **dloc MATCH** → draft KV 写址排除。
- 无 clone 本轮 din 却干净（上轮无 clone 时 dinBAD）→ **din 污染是布局/时序敏感的边缘现象，非稳定根因**。
- 不变核心：graph draft 链 step 0 起 NaN、提案退化为 0；draft 已查输入（token/hidden/topk/写址/positions）全净。
- 剩余嫌疑（round-7 目标）：**draft KV 历史内容**（写址对内容错——垃圾 fp8 解码出 inf/NaN 完全符合症状）与 **draft attention metadata（seq_lens）**。

### Round-7 dump 比对结论（2026-08-28）

- **dkvh MATCH（draft KV 历史内容净）、dseql MATCH（seq_lens 净）** → draft 链全部输入（token/hidden/topk/写址/KV 内容/seq_lens）证明干净，graph 侧仍 step 0 NaN → **毒在 draft 模型前向内部的计算路径**。
- 本轮精度 eager 0.90 / graph 0.92 互换，且 eager 侧提案 token 跨 run 变动 → **eager 存在核级非确定性**（±2pt 波动为其表现）；graph 的 NaN 崩塌远超噪声、每轮复现，判读不受影响。
- accuracy 跑法 A 已证 dump 探针无损；CLONE 主嫌未最终确认（B/C 可选）。

### Round-8 重采判读（2026-08-28，真实行修复后）— NaN 定位到 lm_head 段

**决定性事实**：dm 表（draft 模型内部 11 点）**graph 侧 NaN 全部为 0**——embedding、图内交接 hidden 读、rot、eh_proj、attention、MLP、final norm 全程无 NaN；而 dstep_logits/dstep_hidden **graph 侧每步 NaN**。

=> **NaN 产生在 `dm_out`（模型输出，喂 lm_head 的 hidden）与 `next_token_logits` 之间——即 logits_processor/lm_head 段**。dm 表其余大数（ids/pos/emb/eh/attn/mlp/out 的 1e5~1e9 rel）是 step0 logits NaN → argmax 垃圾 token → 逐步级联的下游污染（含 prevraw，step1+ 读的 hidden 已是 NaN 后的 dstep_hidden），非根因。

另证：真实行修复后 round 1 恢复吻合（此前"round1 即发散"是首采请求流未对齐的假象），发散仍在 round 2。

### Round-9 重采判读（2026-08-28）— 权重排除，NaN 写入点收窄到 runner 尾部

- **`lmw` = 0（逐位一致）→ lm_head 权重/scale buffer 被踩排除**。
- `lmin` rel 与 `out` 完全相同（同一张量两次捕获，1.397e+07 为级联污染值）。
- **核心矛盾对**：`lmout`（processor 出口，graph **NaN=0**）vs `dstep_logits`（draft_forward 内，graph **NaN=4**）——同一 `next_token_logits` 字段两次图内捕获，中间只隔 runner 尾部代码。**NaN 是在这两位置之间被写入共享 logits buffer 的**（`next_token_logits_buffer` 是 graph 共享固定地址 buffer），或 lmout/dstep 之间发生了缓冲区替换（runner 对 logits 做了 gather/copy 重建）。
- 其余键（ids 3192 / pos 0.2 / prevraw 4.0 / emb 9.3 / eh 1.6 / attn 9.2 / mlp 3.9e6 / out 1.4e7）均为 step0-NaN → 垃圾 token → 级联放大。

### Round-9 b 重跑判读（2026-08-28）— step-0 表暴露 dm 切片索引缺陷

step-0 链值：`out=-63.9 | 7.37`、`lmin=37.5 | 0.0`——out 与 lmin 本应是同一张量却双双对不上 → **inner/outer 双 begin 计数器的基线在 eager（每轮经 draft() 重置）与 graph（capture 时基线任意）之间错位，dm 全表跨 run 比对不可信**（含之前各轮的 dm 数值，仅 NaN 计数与 dstep_* 可靠）。

**仍然坚实的可靠结论**（dstep_* 用循环变量 i 切片）：graph 链全部 4 个已执行步（0-3）logits 均 NaN、提案 token=0，而链的全部输入（token seed / hidden / KV 内容 / seq_lens / 写址）已证干净。lm_head 权重 lmw 因索引缺陷本轮无效，待重验。

### Round-10（已埋点，必须重采）：显式链 step 索引

- `draft_forward` 循环内 `debug_draft_mark_step(i)`（runner forward 前），链尾 `mark_step(-1)`（防 draft-extend 写链切片）；
- `deepseek_nextn` inner/outer 全部改读 `debug_draft_current_step()`（begin 计数器退役）；
- 效果：dm 各键切片 = 真实链 step，eager/graph 对齐，dump 表首次完全可信。

判读（重采后）：按管线序 ids/pos → prevraw/prev → emb → eh → attn/mlp → out → lmin → lmw → lmout，**graph 侧首个 NaN/发散键 = 毒点**；若直到 lmw 全净而 lmout NaN → processor 内部 matmul/DP-gather；若 attn/mlp 先坏 → draft decoder 内部（其 KV 读已证净，怀疑 attention metadata 图内路径）。

### Round-8 全量探针 + 采集降本（2026-08-28，已埋点）

**目标：一跑定案。** 剩余全部可疑点一次埋全（`deepseek_nextn.forward` 内 11 个 dm 键，按管线顺序）：

| 键 | 内容 | 坏时指向 |
|---|---|---|
| ids/pos | draft 图静态输入 buffer 的图内读 | 图输入绑定错 |
| bt | forward_batch.block_tables 图内读 | attention KV 索引源错 |
| topk | DSA indexer 种子（from_mtp_carry） | indexer 读错位置 |
| prevraw | spec_info.hidden 图内原始读（rot 前） | in-replay aliasing（pre-draft 读净 ≠ 图内读净） |
| prev | rot matmul 后 | rot matmul |
| emb | embedding 后 | embedding 查表 |
| eh | eh_proj 后 | enorm/hnorm/eh_proj |
| attn/mlp | decoder 子模块输出（forward hook，烘焙切片） | attention / FFN(MoE) |
| out | final norm 后 | norm/lm_head 前 |

实现要点：coordinator 全局注册表 `_COORDINATORS`；`debug_draft_model_begin()` 记录 `_dbg_dm_cur` 供层 hook 用（避免 off-by-one）；draft-extend 前向按 spec_info 类型跳过（链独占 slice 0-3）。

**采集降本**：`SGLANG_SELECTIVE_DUMP_MAX_STEPS` 默认 20→**6**（首发散总在 step 2-3）；host-pool 回读改 `SGLANG_SELECTIVE_DUMP_READBACK=1` 选开（毒化假设已退役，回读+同步是采集时间大头）；**快速采集 = 单请求 64 token 短输出替代 50 题 gsm8k**（命令在脚本尾部）。

判定：analyzer dm 表按管线顺序输出（eager/graph NaN 计数 + max rel），**graph 侧首个 NaN/发散键 = 毒点**。

### 启动命令规则

**每次试验后必须刷新 `pd_disaggregation_dram_offload.sh` 末尾的启动命令**（含当次 dump 目录/开关/比对命令）。当前尾部 = round-8 快采重采（graph12/eager12 + curl 单请求，真实行比对已修）。

### 第六轮探针（round-6，已埋点）

`dloc`：draft() 执行前捕获 `forward_batch.out_cache_loc`（draft KV 写址，精确比对）。verdict 链输出新增 dloc MATCH/DIFFER——DIFFER 即共享状态污染实锤（draft KV 落位错），同时是精度差距的机制候选。

### 第五轮探针（已埋点待跑）

1. `hout_live`：on_verify_result（eagle_sample 后）活读 `logits_output.hidden_states` rowsum。**run 内**与 in-graph frozen `hidden[-1]` 对照：不一致 → replay 后被覆写（aliasing 实锤）；跨 run 在 pre-div 轮对照。
2. `din_*` 改为 draft 执行**前**读取（真 handoff 语义）；输出 `dout_*` 仍在执行后。
3. analyzer：pre-div 轮现在对比 in_ids + hidden 全行（含 final row）；verdict 增加 NaN/Inf 归属表（eager vs graph 侧各自计数）。

判读：`hout_live(k) == hidden[-1](k)` 但 `din(k+1)` 偏 → 覆写发生在 verify 返回与 draft 读之间（pool aliasing 窗口）；`hout_live(k) != hidden[-1](k)` → 覆写发生在 replay 结束到 sample 之间；`din(k+1) == hout_live(k)` 且都偏 → 上一轮 verify 输出本体就偏。
