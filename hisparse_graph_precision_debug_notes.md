# NPU Selective HiSparse 图模式精度问题 — 调试记录与最终结论

> 交接文档。问题：P/D 分离 + Selective HiSparse（19 个 selected 层 KV 驻留 Host DRAM）部署 GLM-5.2 时，D 节点 decode 开启 NPU graph 后精度随 selected 层数线性劣化（7 层 0.90-0.96 → 19 层 0.74-0.82；eager 锚点 0.94）。
>
> **最终状态（2026-08-27）：CASE CLOSED，全部收尾完成。** 根因 = RoPE cos/sin 图捕获冻结（第三个根因，见 §4）。修复后 19 层 50 题 **0.90**，层数线性劣化机制消除。前两个根因（kernel 队列类型、ACL 回调订阅，见 §3）是更早阶段 0.7→0.9 的真实修复，仍然有效。completion-flag 协议三步退役（§5）全部完成并逐一复验 0.90。
>
> **补注（2026-08-28）**：第二轮 diff-dump 探针扩展已落地（见 §7），作为后续任何精度回归的定位工具备用。
>
> **二次战役收尾（2026-08-29，见 §7-§10）**：dump 比对暴露独立于 hi-sparse 的 **draft 链图内 NaN**（A5 优化合入引入，性能 bug 非精度 bug）。已修复其一：attention metadata 张量别名链式污染（§7.2）；其余定案：draft 链 DSA attention kernel（`npu_kv_quant_sparse_flash_attention`）在链步 0 六项输入位级全同下图内输出 NaN，证据包齐，升级 CANN/算子侧（§7.2 定案块、§10.1）。精度结论：两模式无系统差（0.88~0.96 噪声带），精度复核需 200+ 题。方法论教训与调试基建沉淀于 §7.4/§8。

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

debug 探针清理（已完成）：`SGLANG_SELECTIVE_PLAIN_COPY/DEFER_D2H/GATE_D2H_EVENT/SKIP_H2D_WAIT/DUMP_STAGING/DISABLE_D2H/CHECK_FLAGS/DEBUG_SYNC/DEBUG_SLEEP_MS` 及 `_pending_d2h`/`_launch_deferred_d2h` 等全部删除。**保留**：`STAGING_SLICES`/`UNPACK_WS`（ping-pong 设计参数）。
**（2026-08-29 追记）DIFF_DUMP 全套已彻底删除**：`SGLANG_SELECTIVE_DIFF_DUMP/DUMP_DIR/DUMP_MAX_STEPS/DUMP_READBACK` 开关、coordinator 全部 `_dbg_*` 缓冲与 `debug_*` 方法、`dump_diff_snapshot`、ascend_backend 的 am_*/attn_raw 探针与 `DBG-META` 打印、on_verify_result 及其调用点、deepseek_nextn/deepseek_v2/eagle_worker_v2 埋点、npu_graph_runner replay 触发、`D_EAGER` 开关、`hisparse_diff_compare.py` 工具，全部移除（Round-24 定案后证据已固定，探针使命结束）。eager24/graph24 的 dump 文件仍在机器上，CANN 证据包不依赖已删代码。

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

## 7. 最终结论（2026-08-29 整理）

### 7.1 三个独立事实（原始症状的分解）

原始症状"graph 精度 0.90 vs eager 0.94"经 17 轮定位，实为三个独立事实：

1. **精度无系统性差距**。gsm8k 50 题（1 题 = 2pt），今日全部跑（两模式 × 各种开关组合）落在 **0.88~0.96 交叠带**；且已证 target 前向对相同输入逐位一致、accept_len=1 时提交 token 由 target argmax 决定 → 两模式文本序列应一致，分数波动为运行间噪声（含 eager 自身核级非确定性）。**精度复核需 200+ 题**。
2. **确定性 bug 一个：draft 链图内 NaN（性能 bug 非精度 bug，已部分修复）**。graph 下 draft 链 attention 读到的 KV 长度是 capture 烘焙常数（四步全 15，正确值 6/7/8/9），越界读未初始化 KV → NaN → 提案全退化 token 0 → 接受率塌 1 → spec 加速失效。根因 = **张量别名链式污染**（见 7.2），已修复；修复后仍有 NaN 定案诞生于 DSA attention kernel 内部——链步 0 输入位级全同、图内输出 NaN（见 7.2 定案块）。
3. **归属：与 hi-sparse 无关**（见 7.3）。hi-sparse 排查建的 diff-dump 第一次深入 draft 链内部，才让这个潜伏 bug 显形。

### 7.2 draft 链 NaN 根因链

- **已修复（真根因）——张量别名链式污染**：`_init_cuda_graph_metadata` 的 `metadata.seq_lens = seq_lens` 直接别名 draft runner 的静态 `buffers.seq_lens`；四个 per-step 后端共享同一 forward_batch → 同一 buffer。replay 前逐后端刷新（读 buffer、加本步偏移、写回原 buffer）变成链式累加：backend0 写 5+off0 → backend1 在其上加 off1 → … → backend3 最后写 5+Σoff = **15**；四步的图内 attention 全绑定该 buffer → 全读 15 → 越界读未初始化 KV → NaN。
  **修复**：`metadata.seq_lens = seq_lens.clone()`（每个后端独立 buffer）。附带消除对 runner 输入 buffer 的原地污染（decode 打印 row0 曾随轮次漂移 6→17）。
  **修复生效证据**：`am_kvlen` 修复后两侧逐位一致（[6,7,8,9]）；`am_seqlens` per-backend 独立刷新值正确。
- **定案（2026-08-29 Round-24）——「eager 与 graph 输入完全相同、输出不同」的确切位置：draft 链链步 0 的 DSA 稀疏注意力 kernel `npu_kv_quant_sparse_flash_attention`（fp8 DSA，`quant_scale_repo_mode=1`）的图内 replay**。调用点：`ascend_backend.py` `forward_sparse`（`block_table=self.forward_metadata.block_tables` 的调用处），NEXTN draft 层 self-attention，被 eagle draft NPU graph 捕获。eager24 vs graph24 链步 0 全部显式输入位级全同、输出不同：

  | 输入（captured-op 探针） | eager | graph |
  |---|---|---|
  | am_q（q_nope） | 587.8189697265625 | 同 |
  | am_qpe（q_rope） | 459.941650390625 | 同 |
  | am_tik（sparse_indices） | -2033 | 同 |
  | am_kvlen（actual_seq_lengths_kv） | 6 | 同 |
  | am_bt（block_table） | 1 | 同 |
  | am_pgsum（kernel 经页表实际读到的 KV 页字节） | 628294 | 同 |

  输出：eager `attn_raw` = **-136.17**（正常量级，R16 时代同款），graph = **NaN@[0,1,2,3] 全部四链步**，跨 round-23/24 确定性复现 →「captured-replay kernel 对完全一致的输入输出 NaN」定案，升级 CANN/算子侧（证据包见 §10.1）。

  **证据边界（引用本结论时必须附带）**：
  1. 「输入一致」只在**链步 0** 严格成立；链步 1-3 的 am_pgsum 发散（graph 恒 ~-100K/步递减 634523→535069→432124，eager 递增 +7.5K/步）是链步 0 NaN 的下游级联（NaN hidden → quant scale 异常 → 量化 KV 写 0），非独立写路径污染，不构成独立证据。
  2. graph 链步 1-3 的 am_q/am_qpe 探针读出**精确 0**，而其输入 prevraw/eh 为 NaN（NaN 过 matmul 应得 NaN）——图内 per-step 探针存储伪影，不得作为输入证据；链步 0 的 am_q 位级匹配才是有效比对。
  3. 附注（入 CANN 包）：am_seqlens graph[6,7,8,9] vs eager[5,5,5,5]，为 graph 把 per-step offset 写进 seq_lens 设备张量、eager 只进 cpu_int 的元数据语义差；kernel 实际消费 kvlen（两侧一致），无涉本结论。

  **排查收口路径**：R10 定位毒点 = draft self-attention → R14 修张量别名链式污染（`metadata.seq_lens = seq_lens.clone()`，am_kvlen 修正为 6/7/8/9）→ R15 attn_raw 直捕确认 NaN 诞生于 kernel 内部 → R16 验证 q_pe/sparse_indices 一致 → R17/Round-23/Round-24 补齐最后盲区 am_pgsum（页字节和）。探针插曲：round-22 首版 `t[0,0]` 0-dim 算子链被 NPU auto-dispatch 图捕获拒绝（graph 启动即崩，改 `[:1,:1]` 切片全程 ≥1-D + dump 门控零开销后修复）；round-23 的 eager22 参照 dm 异常（attn_raw inf/1.37e37/nan、out rowsum=0、am_q 步间位级重复）为该轮 0-dim 探针代码/坏运行状态的一次性伪影，round-24 干净参照复现 -136 后排除；另有一次误将 eager24 与自身比对得出「全 match、NaN 非确定」的结论，已撤回。

### 7.3 归属：该 bug 与 hi-sparse 无关

- **Git 考古**：NaN 涉及的全部路径（`npu_kv_quant_sparse_flash_attention` 调用、`quant_scale_repo_mode=1`、A5 `draft_replay_pack_npu`、`eagle_draft_npu_graph_runner`）由 **08-15 `94daad003b`（Glm52 a5 opt sync）** 与 **08-19 `1aad5706b9`（glm 5 optim pp cp）** 引入；hi-sparse 主提交（08-24 `b1dcd5c74a`）仅 ascend_backend 36 行 selected 层路由，未触碰上述路径。
- **机制上不可能是精度差距成因**：bug 效果 = 垃圾提案 → 接受率塌 1 → 性能损失；贪心精度由 target argmax 决定。
- **运行时铁证（可选）**：E3 warmup 对比（`SELECTIVE_LAYER_IDS=""` 关 hisparse），NaN 仍在即终证（命令见脚本尾部）。

### 7.4 方法论教训（本轮沉淀）

1. **captured op 探针是图内取证的唯一可信手段**：host 侧读数（pre-draft 读、runner 尾部读）可能读到"replay 后已自愈/已改写"的值；只有图内 captured op 冻结消费时刻的值。
2. **探针切片索引必须用业务真值**（链步循环变量 `i`），不要用 host 计数器——计数器基线在 eager 与 capture 间会错位（round-9 b 教训）。
3. **行宽/桶填充差异是聚合探针的头号伪影源**：跨模式比对必须只比"真实行"（dchain_real），整行页表求和类探针会把 padding 项计入（round-17 教训）。
4. **NaN 类问题优先做"归属表"**（哪一侧、哪些 step、哪个字段先出现 NaN），再做输入二分——比逐个猜嫌疑快得多。
5. **垃圾提案不影响贪心精度**（verify 只接受匹配 token）：draft 侧 NaN/垃圾是性能问题；精度问题必须先证 target 前向对相同输入一致。

## 8. 调试基建与输出规则

> **（2026-08-29）本节所述工具链已随探针全套删除**（见 §5 追记），以下内容仅作方法论存档。

### 8.1 diff-dump 开关与产物

- 总开关：`SGLANG_SELECTIVE_DIFF_DUMP=1`；`SGLANG_SELECTIVE_DUMP_DIR`（默认 /root/hisparse_dump）；`SGLANG_SELECTIVE_DUMP_MAX_STEPS`（默认 **6**，首发散在 step 2-3，够用）；`SGLANG_SELECTIVE_DUMP_READBACK=1`（Host pool 回读，默认关——毒化假设已退役，且回读+同步是采集时间大头）。
- 产物：
  - `{eager,graph}_dev{N}_step{S}.pt` — selected 层逐字段快照（locs/valid/stg_pre/stg_post/q/qrope/out/pkg/crow/allv/pub/kin/kinv/pos/d2h_locs/unpk/unpkr + 标量 real_tokens/h2d_cnt/d2h_cnt）
  - `state_dev{N}_step{S}.pt` — target 前向二分 + draft 链全套（in_ids/in_pos/hidden/rkv/rloc/din_*/dout_*/dstep_*/hout_live/dext_out/dloc/dkvh/dseql/dm_*）
  - `accept_dev{N}_step{S}.json` — accept_lens + logits 指纹（logit_argmax/logit_rowsum）
- 比对：`python hisparse_diff_compare.py --eager-dir <e> --graph-dir <g>`。输出段顺序：verify-round fingerprint（含 NaN 归属表、handoff 链）→ target-forward bisect（含 dm 表、step-0 链值）→ per-pair divergence → first divergence。
- 判读总原则：**dm/dstep 等直捕字段以显式链步索引为准（mark_step），module hook 切片曾有错位（attn [0,2,3]），不一致时以直捕为准**；rowsum 相等是强证据但非逐位相等，整型字段才是精确比对。

### 8.2 采集规则（成本经验）

1. **warmup 即出 dump**：不需要发请求、不需要跑精度——起两个服务，dummy 请求驱动的 verify 轮自动产生全部 dump。
2. `MAX_STEPS=6` 足够：首个发散轮总在 step 2-3，其后全是级联。
3. 两侧必须同一 prompt/请求流，否则 round 1 即假性发散（round-8 首采教训）。
4. dump 开关对精度无损已实证（隔离矩阵跑法 A：eager 0.92 / graph 0.90）。
5. `SGLANG_SELECTIVE_CLONE_HANDOFF`（handoff hidden clone 实验）已证明与 draft NaN 无关且曾伴随精度双降，**不要再默认开**。

### 8.3 启动命令维护规则（用户 2026-08-28 确立）

**每次试验后必须刷新 `pd_disaggregation_dram_offload.sh` 末尾的「启动命令速查」节**：含当次 dump 目录、全部开关变量、eager 对照命令与比对命令。当前尾部 = Round-26（CANN/算子侧升级，证据包定稿；附可选 gsm8k 基线与缓解实验方向）。

### 8.4 判读决策树（selected 层快照，graph vs eager 逐字段）

```
accept_lens 首个发散 step = S
 └─ step S 起逐层看：
    pos/kin/kinv 偏 → 上游 KV 投影/RoPE 输入已错（图基础设施级）
    kin/kinv 吻合 + pub 偏 → fp8 quant 在图内产出不同（captured-replay bug）
    pub 吻合 + pkg 偏 → patch 读到旧 scratch（跨队列竞态实锤）
    locs/d2h_locs 偏 → topk 或 req_to_token 映射在图内错位
    d2h_locs 吻合 + d2h_rb 偏 → D2H 读到被覆盖的 scratch（Host pool 污染实锤）
    q 偏 / qrope 偏 → Q 投影 / RoPE 分支
    stg_pre 偏 → H2D 数据未落地即被消费 或 Host 读址内容已错
    stg_pre/stg_post 吻合 + unpk 偏 → unpack 反量化链图内行为不同
    全部吻合但 out 偏 → SFA kernel 自身
```

draft 链 dm 表判读：按管线序 `ids/pos → prevraw/prev → emb → eh → am_q/am_qpe/am_tik/am_kvlen/am_seqlens/am_bt/am_pgsum → attn → mlp → out → lmin → lmw → lmout`，**graph 侧首个 NaN/发散键 = 毒点**；attn hook 切片有错位前科，与直捕（am_*）冲突时以直捕为准。

## 9. 排查时间线（17 轮详录，按序）

| 轮 | 手段 | 关键结论 |
|---|---|---|
| 首轮 | selected 层 diff-dump（bs1） | pair0 全字段吻合 → selected 层 KV 链路无辜；step2 起发散 |
| R2 | verify logits 指纹 | 首发散 = verify round 3 的 target 前向；step1-2 指纹全同 |
| R3 | target 逐层二分 + resident KV 写指纹 | resident KV 写净；verify **输入 token id** 偏 → 毒在 draft 侧 |
| R4 | root-vs-draft in_ids 对比 | root（上轮接受 token）吻合 → eagle_sample 无辜；draft 提案坏（graph 侧退化为 0） |
| R5 | NaN 归属表 | graph 侧 draft logits/hidden 全 NaN（eager 全净）→ draft 前向从 step0 就 NaN；din 异常引出 pool-aliasing 假设（后被推翻） |
| R6 | clone 实验 + 双降隔离 | clone 洗净 hidden 链但 NaN 仍在 → 毒在 draft 前向内部；双降主嫌=CLONE（后证探针无损、CLONE 弃用） |
| R6b | 隔离矩阵跑法 A | dump 探针精度无损；dloc（draft KV 写址）净 |
| R7 | dkvh/dseql | draft KV 历史内容、seq_lens 净 → 毒在 draft 模型前向内部 |
| R8 | draft 模型子块探针（首版） | dm 索引缺陷暴露（begin 计数器基线错位）→ 真实行修复；"lm_head 段"结论由 R9-b 修正 |
| R9 | 权重 lmw | lm_head 权重逐位一致（排除）；step-0 表暴露索引缺陷 |
| R10 | mark_step 显式索引 | dm 表首次可信：**attn@0 NaN，输入全净** → 毒在 draft self-attention；nan@steps 此前误读为 MLP 新毒点，R15 纠正为 attn[0] 确实 NaN |
| R11 | warmup-only 发现 + am_kvlen/am_q | dump 全部来自 warmup（采集成本崩塌）；graph kvlen = 常数 15 |
| R12 | 修复尝试 1（actual_seq_lengths_kv 刷新） | 未命中；am_seqlens 与 am_kvlen 不同 → baked 张量独立存在 |
| R13 | DBG-META host 打印 | input==written 恒成立 → 别名实锤 |
| R14 | **真根因修复：clone 切断别名** | 机制 = 四后端共享 buffer 链式累加，末步值 15 被全体继承；附带污染 runner 输入 buffer |
| R14b | 修复验证 | am_kvlen 修正✓；attn 仍 NaN（当时误判为"MLP 新毒点"，R15 nan@steps 纠正） |
| R15 | am_bt + attn_raw | block_table 逐位一致；**attn_raw NaN@[0-3]** → NaN 诞生于 kernel 内部；hook 错位确认 |
| R16 | am_tik + am_qpe | sparse_indices 一致（indexer 排除）；q_pe step0 一致（459.94）→ **kernel 全部显式输入一致仍 NaN** |
| R17 | am_pgsum（页字节和） | 首版整行求和有聚合伪影已修正；**待重采闭环** → 一致仍 NaN 即 CANN 升级 |

（早期轮次 R1-R4 同时产出了 selected 层的第二轮 dump 扩展字段与 verify 指纹基建，见 §7/§8；flag 机制退役、RoPE 根因等更早历史见 §4-§6。）

## 10. 遗留事项

1. **CANN/算子侧升级（Round-24 定案，证据包齐）**：kernel = `npu_kv_quant_sparse_flash_attention`（fp8 DSA，`quant_scale_repo_mode=1`）。证据：链步 0 全部输入位级一致（am_q 587.82 / am_qpe 459.94 / am_tik -2033 / am_kvlen 6 / am_bt 1 / am_pgsum 628294）+ graph attn_raw NaN@[0-3] vs eager -136，且跨 round-23/24 确定性复现；eager 参照干净。包内容：dm 表 + step-0 链值（eager24/graph24）+ §7.2。临时缓解方向（§10.2）：draft 链 eager attention 或 non-quant kernel 路径。
2. **临时缓解**（若走 CANN 路径）：该 kernel 仅 DSA 草稿链使用 → 实验 draft 链 eager attention 或 non-quant kernel 路径，先兑现 spec 加速。
3. **gsm8k 200 题**精度对齐复核（50 题噪声 ±2-4pt，已证两模式交叠 0.88~0.96）。
4. **accept_len 恢复观察**：修复前恒 [1]；恢复 >1 的轮次占比即 spec 加速兑现程度。（Round-24 真实比对中 graph 提案仍退化 token 0，NaN 未缓解前不会恢复；待 CANN 缓解/缓解实验落地后复测。）
5. **清理（2026-08-29 完成）**：`SGLANG_SELECTIVE_CLONE_HANDOFF` 已删；全套 diff-dump 探针、`DBG-META` 打印、`D_EAGER` 开关、`hisparse_diff_compare.py` 已全部删除（见 §5 追记）。
6. **可选（已失操作载体，存档）**：E3 warmup 铁证（SELECTIVE_LAYER_IDS=""）——该开关保留在启动脚本中（设为空串即关闭 hisparse）。
