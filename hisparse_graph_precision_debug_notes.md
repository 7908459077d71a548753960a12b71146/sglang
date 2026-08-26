# NPU Selective HiSparse 图模式精度问题 — 调试记录与当前状态

> 交接文档。问题：P/D 分离 + Selective HiSparse（19 个 selected 层 KV 驻留 Host DRAM）部署 GLM-5.2 时，D 节点 decode 开启 NPU graph 后数据集准确率 ~0.7（期望 0.94，eager 模式正常）。
>
> 当前状态（2026-08-25）：**已从 0.7 修复到 0.9**，距期望 0.94 还差 4 个点，剩余问题定位中。

## 1. 背景与架构

### 运行时 KV 数据流（图模式下每步 replay 内）

```
anchor 层(L-3) topk 计算
    → build_loc_plan: topk 位置 → Host pool 逻辑 loc
    → H2D: sparse_copy 把 topk KV 从 Host DRAM 预取到 HBM packed_staging   [AIV DMA]
    → selected 层(L) set_kv_buffer → publish_new_packed_kv:
        · new_packed_scratch.copy_(packed_kv)                               [torch 向量算子]
        · D2H: sparse_copy 把新 KV 备份到 Host DRAM                          [AIV DMA]
    → run_selected_attention:
        · wait H2D → current-patch: 用 scratch 内容覆盖 staging 的 current 行 [torch 向量算子]
        · unpack 656B → BF16 → SFA 稀疏注意力                                [计算算子]
```

关键文件：
- sglang: `python/sglang/srt/hardware_backend/npu/selective_hisparse.py`（coordinator，全部 DMA/等待逻辑）
- sglang: `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`（graph capture/replay 集成）
- sglang: `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`（**NPU 实际走的 execute override**）
- memfabric: `src/acc_offload/csrc/operators/acc_offload_sparse_copy.h/.cpp`（sparse_copy AIV kernel）
- memfabric: `src/acc_offload/csrc/launch/`（RunOpApiV2 下发链路）

### 已确认的硬件/调度事实（本次调试的核心认知）

**NPU 的 kernel 任务类型决定调度队列**：

| kernel 类型 | 调度队列 | 图 replay 行为 |
|---|---|---|
| `KERNEL_TYPE_AIV_ONLY` | 独立向量核队列 | 与计算队列**并行推进、无完成排序** |
| `KERNEL_TYPE_MIX_AIV_1_0` | 计算核队列 | 与计算算子 **FIFO 串行** |

- 图 replay 时 AIV_ONLY kernel 与计算核算子跨队列并行，stream 只保证发射顺序，不保证完成顺序 → 图内消费者读到未落地的 DMA 数据。
- zbal 系（`app/zbal/...`）在 decode 图内与 MoE 计算算子正确共存，全部用 `MIX_AIV_1_0`（launch 语法相同，仅任务类型不同）——这是现网已验证的先例。
- AIV kernel 内部 MTE 队列（TQueBind Alloc/EnQue/DeQue）的异步完成依赖 **ACL host 回调线程**处理 `acl.rt.process_report`；订阅是**按流**的（`acl.rt.subscribe_report`）。

## 2. 实验与结论时间线

所有图模式实验统一 `--cuda-graph-bs 1`、单请求/数据集（max-concurrency 1）。

| # | 实验 | 结果 | 结论 |
|---|---|---|---|
| 0 | eager（`--disable-cuda-graph`） | 精度正常 | 逻辑（W3 scratch、门控、patch、unpack）本身正确；问题图模式专属 |
| A | 订阅 replay 流到 ACL 回调线程（`SGLANG_SELECTIVE_SUBSCRIBE_REPLAY_STREAM=1`） | 略有改善 | 回调处理与数据落地相关，但非唯一问题 |
| B | replay 后 `torch.npu.synchronize()` | 无效 | ~~排除跨迭代竞态~~ **结论作废**：探针加错了文件（见下） |
| C | replay 后 sleep 200ms | 无效 | ~~数据是错的不是晚的~~ **结论作废**：同上 |
| E | 图内 D2H 备份置零（`SGLANG_SELECTIVE_DISABLE_D2H=1`） | 准确率崩到 0.16 | D2H 备份是主数据路径（decode 前 N 步 KV 必须备份进 Host pool 供后续 H2D 读），不是错误源 |
| F | staging checksum dump（eager vs graph） | **两侧 sum/nonzero 逐层逐 step 精确吻合** | DMA 数据最终全部正确落地 → 问题 100% 是图内**消费时序**（SFA 读早于 DMA 落地），不是数据错误 |
| G | flag 协议 v1：copy 尾部原子计数 + AIV wait_flag 自旋 | 精度不变（0.7） | **修复无效原因**：wait_flag 也是 AIV_ONLY，与 copy 同在向量队列，FIFO 使自旋恒真；真正要 gate 的 SFA 在计算队列，不受影响 |
| H | copy/wait_flag 改 `KERNEL_TYPE_MIX_AIV_1_0` | 0.7 → 0.78 | 队列修复部分起效（copy 与 SFA 同队列 FIFO 有序） |
| I | H 基础上再开实验 A 的订阅 | **0.78 → 0.9** | 订阅 replay 流是独立必要条件（MTE 异步完成的 host 侧处理） |

**重要教训（B/C 作废原因）**：NPU 的 decode verify 走 `NPUGraphRunner.execute()`（`npu_graph_runner.py`），完全 override 父类 `DecodeCudaGraphRunner.execute()`（`decode_cuda_graph_runner.py`）。探针加在父类是死代码。改 NPU 侧 runner 逻辑必须改 `npu_graph_runner.py`。

## 3. 最终确认的根因（两个叠加）

### 根因 1：kernel 队列类型错误（已修复 → 0.78）

`sparse_copy` 原为 `KERNEL_TYPE_AIV_ONLY`：图 replay 时 copy 在向量队列、patch/SFA 在计算队列，并行推进 → SFA 读 staging 时 H2D DMA 未落地 → 读到上一步旧数据。随生成长错误累积（表现：长输出先对后错、陷入复读循环）。
**修复**：copy 与 wait_flag 改 `KERNEL_TYPE_MIX_AIV_1_0`（zbal 范式），与计算算子同队列 FIFO。文件：`acc_offload_sparse_copy.cpp`。

### 根因 2：ACL 回调未订阅 replay 流（已修复 → 0.9）

AIV kernel 的 MTE 异步搬运完成依赖 host 侧 `acl.rt.process_report` 被处理；订阅按流路由。原实现只订阅 capture 流，replay 在主流执行、从未被订阅 → 图内 sparse_copy 的 MTE 完成报告无人处理。
**修复**：`prepare_graph_replay` 中订阅当前流（目前仍由 `SGLANG_SELECTIVE_SUBSCRIBE_REPLAY_STREAM=1` 门控，**待固化为无条件**）。

## 4. 当前剩余问题（0.9 → 0.94，待定位）

已排除：数据错误（exp F dump 证明落地数据正确）、D2H 备份写坏、跨迭代竞态。

### 主要嫌疑：torch 向量算子与 MIX copy 的跨队列竞态（根因修复引入的）

改 MIX 后 copy 挪到计算队列，但 patch/scratch 的 torch 算子（`torch.where`、`copy_`）若调度在向量队列，原 AIV_ONLY 时代同队列 FIFO 保证的两个有序关系被破坏：

| 操作对 | AIV_ONLY 时代（同向量队列，有序） | MIX 时代（可能竞态） |
|---|---|---|
| H2D copy ↔ current-patch（读 staging） | ✓ | 竞态 |
| D2H copy ↔ 下层 `new_packed_scratch.copy_`（写 scratch） | ✓ | 竞态（D2H 备份了"下一层"数据 → Host pool 污染） |

### 下一步计划（按序）

1. **校准基线**：同数据集跑 eager 准确率，确认 0.94 是 eager 实测值（若 eager 也 ~0.9，剩余差距与图模式无关）
2. **固化订阅**：去掉 `SGLANG_SELECTIVE_SUBSCRIBE_REPLAY_STREAM` env 门控，`prepare_graph_replay` 无条件订阅（幂等）
3. **dump 比对**：在 0.9 配置上重跑 exp F（`SGLANG_SELECTIVE_DUMP_STAGING=1`），找 sum 开始偏离 eager 轨迹的 step/层，定位到 H2D 或 D2H 哪条链路
4. **若确认跨队列竞态**：架构改为「copy 保持 AIV_ONLY（回向量队列，与 torch 向量算子同队列 FIFO）+ wait_flag 改 MIX（计算队列自旋等 copy 完成再放行 SFA）」——flag 协议的原始设计意图；注意 expect 的 `add_` 是 torch 算子（队列不确定），需改为 kernel 内部轮次计数
5. **多 bs 回归 + 性能**：`--cuda-graph-bs 8 16` 回归；MIX copy 与计算抢核的性能代价评估（必要时双模式：图内 MIX/图外 AIV_ONLY 或减 blockDim）

## 5. 已实现的代码改动清单

### memfabric（需重编 `libmf_hybm_accoffload.so` + 重装 python 包）

| 文件 | 改动 |
|---|---|
| `src/acc_offload/csrc/operators/acc_offload_sparse_copy.h` | ① `OffloadSparseCopyKernel` 增 `notify` 参数 + `NotifyDone()`（TBuf + S_MTE3 硬事件，32B 对齐 8-lane 原子加，lane0=+1）；② 新增 `OffloadWaitFlagKernel`（单核 dcci 自旋 `flag[0]==expect[0] && flag[2]==expect[1]`，超时 assert） |
| `src/acc_offload/csrc/operators/acc_offload_sparse_copy.cpp` | 两个 kernel 均改 `KERNEL_TYPE_MIX_AIV_1_0`（**核心修复**）；`OffloadOpsWaitFlag`（blockDim=1） |
| `src/acc_offload/csrc/operators/acc_offload_operators.h` | 签名加 `notifyPtr`；声明 `OffloadOpsWaitFlag` |
| `src/acc_offload/csrc/launch/acc_offload_operators_launch.cpp` | `AccOffloadSparseCopy` 透传 notify + A5 上 blockDim=64；新增 `AccOffloadWaitFlag`（RunOpApiV2 `"acc_wait_flag"`） |
| `src/acc_offload/csrc/launch/acc_offload_launch.h/.cpp` | dlopen 符号表增 `AccOffloadWaitFlag`，函数指针类型更新 |
| `src/acc_offload/csrc/acc_offload_entry.h`、两个 entry `.h/.cpp`、`acc_offload_entry_manager.h/.cpp` | `SparseCopy` 增 notifyPtr；新增 `WaitFlag` |
| `src/acc_offload/include/host/acc_offload.h`、`src/acc_offload/csrc/acc_offload.cpp` | C API：`offload_sparse_copy_notify`（旧入口转调、notify=0 兼容）、`offload_wait_flag` |
| `src/acc_offload/csrc/python_wrapper/pymf_acc_offload.cpp` | pybind：`sparse_copy_notify` / `wait_flag` |
| `src/smem/python/memfabric_hybrid/memfabric_hybrid/mf_acc_offload.py`、`__init__.py` | python wrapper（torch.device → `.index`、`.data_ptr()`）+ 导出 `offload.sparse_copy_notify` / `offload.wait_flag` |

### sglang（`python/sglang/srt/hardware_backend/npu/selective_hisparse.py`）

- **flag buffers**（`_alloc_staging_buffers`）：`h2d_flag/h2d_expect/d2h_flag/d2h_expect` 各 int32[16]（64B 独占 cacheline）；`_dma_flag_block_num = 64`
- **单调计数协议**（magic 方案已废弃——magic bump 是 Python 值会被 capture 固化）：每次 `sparse_copy_notify`（kernel 无条件 +64）配对一个 **captured op** `expect[0].add_(64)`；`wait_flag` 自旋 `count == expect`；count 只增 + 1:1 配对 → 陈旧 flag 不可能通过
- **H2D 提交**（`maybe_start_prefetch`）：图模式走 `sparse_copy_notify` + `expect += 64`
- **H2D 消费**（`run_selected_attention`）：图模式在 patch/SFA 前 `wait_flag`
- **D2H 提交**（`publish_new_packed_kv`）：图模式走 `sparse_copy_notify` + `expect += 64`
- **D2H 消费**（`publish` 开头）：覆盖共享 scratch 前 `wait_flag`（首轮 count==expect==0 直过）
- **capture 前**（`prepare_graph_capture`）：`npu.synchronize()` + 四个 flag/expect 清零
- **eager→graph 桥接**（`_bridge_eager_to_graph`）：有 eager DMA 在飞时补 `npu.synchronize()`（eager 路径无 flag，防首轮假通过）
- **eager 路径完全未动**（原 event 机制）
- 调试开关（env，默认关）：`SGLANG_SELECTIVE_DUMP_STAGING`（staging checksum，eager 侧打每层、graph 侧在 `npu_graph_runner.py` execute 里打每 replay）、`SGLANG_SELECTIVE_SUBSCRIBE_REPLAY_STREAM`、`SGLANG_SELECTIVE_DEBUG_SYNC`、`SGLANG_SELECTIVE_DEBUG_SLEEP_MS`、`SGLANG_SELECTIVE_DISABLE_D2H`

### 注意：MIX 类型下 flag 协议目前是冗余保险

copy 与 wait_flag 同在计算队列后 FIFO 已保证有序，wait 恒瞬过（零自旋开销）。若第 4 步把 copy 改回 AIV_ONLY，flag 协议才真正承担 gating 职责。

## 6. 验证配置速查

```bash
# 当前 0.9 的配置（D 节点）
export SGLANG_SELECTIVE_SUBSCRIBE_REPLAY_STREAM=1
# 启动参数（其余同 pd_disaggregation_dram_offload.sh 默认）
--cuda-graph-bs 1
# eager 基线
--cuda-graph-bs 8 16 --disable-cuda-graph   # 保留 bs 参数防 fixed bias 膨胀到 14GB（bug：disable_cuda_graph 时 bias 仍按默认 bs 列表 512 计算，见下）

# 已知配置坑
# 1) --disable-cuda-graph 时若删掉 --cuda-graph-bs，pool_configurator 的
#    _compute_selective_fixed_bias 取默认 bs 列表 max=512 → bias 14GB → 启动失败
#    （fix：disable_cuda_graph 时不该用 graph bs cap bcap；尚欠代码修复）
# 2) eager 满载时 token 数超过 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
#    会复发 161002（当前 max-concurrency 1 不受影响）
# 3) D_MEM_FRACTION=0.92 与图模式配对可过；eager 激活峰值更高需下调
```
