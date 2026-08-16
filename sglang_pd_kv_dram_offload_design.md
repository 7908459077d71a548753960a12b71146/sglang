# sglang PD 分离 KV cache DRAM offload 修改设计

> 目标：PD 分离（NPU/Ascend，A5 平台）下，Prefill 节点写 KV cache 到 Decode 节点时，
> 目标地址支持 **HBM 地址 + DRAM 地址** 两类；Decode 启动/注册时把 HBM+DRAM 两类地址
> 都传输给 Prefill；Decode 节点的 KV cache 管理（页分配/释放/提升）兼容 HBM+DRAM 双池。
> 全部基于 memfabric（smem_trans DEVICE_URMA + offload URMA_POOL + sparse_copy）实现。

---

## 1. 背景

### 1.1 已就绪的 memfabric 能力（release/1.2，commit 1ea9da46）

| 能力 | 接口 | 说明 |
|---|---|---|
| URMA 跨机引擎 | `TransferEngine.initialize(store_url, session_id, role, npu_id, TransDataOpType.DEVICE_URMA)` | HBM/DRAM 统一走 DEVICE_URMA |
| 混合内存注册 | `engine.batch_register_memory(addrs, lens)` | HBM（DEVICE 内存）与 DRAM（HOST 内存，HVA）均可注册；HOST 内存注册时自动 `HalHostRegister` 建立 DVA + URMA MR，"注册即可达" |
| 远端直写 | `engine.batch_transfer_sync_write(dst_session, srcs, dsts, lens)` | 一次批量调用可混合 HBM/DRAM 目标地址 |
| DRAM 池分配 | `offload.initialize(cfg)` + `offload.malloc/free` | URMA_POOL 模式（`OFFLOAD_FLAG_URMA_POOL`），reserve/alloc 强制 GB 对齐；smem_trans 不负责分配（release/1.2 约束），分配后注册给 trans |
| DVA 查询 | `offload.get_dva(hva)` | 返回 device 可见 DVA，作为 AIV 提升的源地址 |
| DRAM→HBM 提升 | `offload.sparse_copy(srcPtrs, dstPtrs, lens, cnt, device)` | AIV 算子，任意条目顺序、无 K/V 等分假设 |

平台约束：DEVICE_URMA 仅支持 ASCEND_950（A5）；跨机要求两节点 EID 可达（UBC_CTP）；
构建需启用 URMA 选项（`--build_hcom`，必要时 `--build_hcom_ub`）。

### 1.2 sglang 现状（基线分支 ifmn/npu/glm-5-optim）

sglang 已有 ascend PD 传输后端（`--disaggregation-transfer-backend ascend`），核心链路：

1. **Bootstrap（一次性注册）**：Decode 每个 rank 通过 ZMQ 把本 rank KV 池裸地址
   （`kv_data_ptrs/aux_data_ptrs/state_data_ptrs`）发给 Prefill，缓存在
   `KVArgsRegisterInfo`（`disaggregation/mooncake/conn.py:125-192`）。
   发送点：`MooncakeKVReceiver._register_kv_args()`（`mooncake/conn.py:2368-2450`）。
2. **每请求元数据**：Decode 预分配 HBM 目标页（`disaggregation/decode.py:1536-1674`
   `_pre_alloc`），把页索引 `dst_kv_indices` 经 ZMQ 发给 Prefill（`send_metadata`，
   `mooncake/conn.py:2452-2512`）。
3. **数据面**：Prefill `AscendKVManager.send_kvcache`（`disaggregation/ascend/conn.py:103-220`）
   把页索引组成 `(src_addr, dst_addr, length)` 块，经
   `engine.batch_transfer_sync_write` 写入 Decode **HBM**。
4. **注册**：`AscendKVManager.register_buffer_to_engine`（`ascend/conn.py:54-69`）把
   kv+aux+state 合并一次 `batch_register`（MemFabric 2MiB 对齐合并）。
5. **协议**：`AscendTransferEngine`（`ascend/transfer_engine.py`）仅支持
   `sdma`/`device_rdma`（L95-105），**无 device_urma**。

### 1.3 差距（Gap）分析

| # | 差距 | 影响 |
|---|---|---|
| G1 | 传输引擎不支持 DEVICE_URMA | 跨机（跨节点）PD 传输不可用；且 DRAM 直写依赖 URMA |
| G2 | Decode 无 DRAM 接收池 | KV 只能落 HBM，HBM 不足时请求被 retract/阻塞 |
| G3 | 注册链路只传 HBM 地址 | Prefill 不知道 DRAM 池地址，无法写入 |
| G4 | 页分配只在 HBM allocator | 无法表达"部分页落 DRAM" |
| G5 | 数据面寻址只按 HBM 池基址 | 无法对 DRAM 页寻址 |
| G6 | Decode attention 只读 HBM paged tensor（`ascend_backend.py:1655` 等） | DRAM 页必须先提升（promote）回 HBM 才能参与计算 |
| G7 | 请求释放/退出路径无 DRAM 归还 | 会泄漏 DRAM 页 |

---

## 2. 总体设计

### 2.1 架构与数据流

```text
Prefill 节点                                     Decode 节点
─────────────────                               ─────────────────
AscendTransferEngine                            AscendTransferEngine (DEVICE_URMA)
  src HBM KV 池(batch_register)                   ① HBM KV 池 (torch.zeros, 现有)
                                                  ② DRAM 接收池 (offload URMA_POOL 新增)
                                                     ↪ batch_register(HVA) 自动建 DVA+MR
send_kvcache:                                    pop_preallocated:
  按"全局页号"选池:                                HBM allocator 优先, 不足页落 DRAM 池
  [0, N_hbm)      → hbm_ptr[l] + idx*item_len      dst_kv_indices(全局编码) --ZMQ--> Prefill
  [N_hbm, N_tot)  → dram_ptr[l] + (idx-N_hbm)*item_len
  batch_transfer_sync_write(混合 dst)  ──URMA──►  HBM 页(直达) + DRAM 页(远端直写)
                                                  pop_transferred → commit:
                                                  DRAM 页 promote: sparse_copy(DVA→HBM)
                                                  ↪ 更新 req_to_token, 释放 DRAM 页
                                                  → 进入调度, attention 只读 HBM
```

### 2.2 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| D1 DRAM 分配方 | sglang Decode 侧调用 memfabric `offload` 组件 | release/1.2 起 smem_trans 无分配接口；offload URMA_POOL 池天然 device 可见（AIV 可直读 DVA） |
| D2 DRAM 页定位 | "**接收期临时落点**"：DRAM 页仅在传输窗口存在，commit 时同步 promote 回 HBM | NPU attention 无 DRAM 直读能力（G6/hisparse fail-loud）；同步 promote 语义简单、正确性易验证；异步化留作后续优化 |
| D3 页归属表达 | **全局页号编码**：`dst_kv_indices` 中 `[0,N_hbm)` 为 HBM 页、`[N_hbm,N_tot)` 为 DRAM 页 | 复用现有 `send_metadata`/`TransferInfo`/`group_concurrent_contiguous` 全链路，ZMQ 每请求消息结构零改动；仅在寻址处分流 |
| D4 DRAM 池布局 | 与 HBM KV 池**同构镜像**（per-layer 连续，item_len 一致） | Prefill 侧寻址公式 `base + idx*item_len` 两池统一；MLA `kv_buf_groups` 多 buffer 组同样按组镜像 |
| D5 aux/state 落点 | 首版**仍走 HBM**（不落 DRAM） | aux（首 token 元数据）与 state 池很小，DRAM 化收益低、改动面大 |
| D6 与 hicache 关系 | 首版与 `--disaggregation-decode-enable-offload-kvcache` **互斥**（fail-loud） | 两者都管理 decode 侧 KV 与 host 内存的交互，叠加语义未定义；后续再评估组合 |
| D7 DRAM 池容量配置【已确认】 | 按 **GB 绝对值**配置（`--disaggregation-decode-dram-pool-size`，默认 0=禁用）；初始化完成即打印换算结果：`GB 数 → 页数 → 可容纳 token 数` | 直传 offload reserve/alloc（本身 GB 对齐）无换算损失；DRAM 为共享资源（OS/其他进程），绝对值便于整机规划与 fail-loud 校验；打印换算弥补"能缓多少请求"语义不直观的短板。跨模型压测迁移需求出现时再考虑 ratio 别名（先例：hicache_ratio/hicache_size 双参数并存） |
| D8 接收落池策略【已确认】 | **自动水位，零新增参数**：`可直写预算 = HBM空闲页 − pending_promote_pages − num_reserved_decode_tokens//page_size`（后两者均为运行时已知量/已有参数）；`落 HBM 页数 = min(缺口, max(0, 预算))` | 水位本质是"HBM 剩余先还 promote 债 + 留给 running 增长"，可完全派生，无需用户调参；效果自适应三段：轻载全落 HBM（单跳）、满负荷自动全落 DRAM（HBM 完全留给 running/提升，接收不与之竞争）、中间按预算部分落 |
| D9 侵入性边界【已确认】 | **公共文件只加"默认关闭时零行为变化"的通用机制与 no-op 钩子；DRAM 全部逻辑收拢在 `disaggregation/ascend/` 内**（新文件 + 子类覆盖 + allocator wrapper 组合，不复制公共代码） | DRAM 是 ascend/memfabric 专属能力，不应焊进跨后端共享的 `mooncake/conn.py`/`decode.py` 核心路径；隔离保证：`dram_pool_size=0`（默认）时所有后端、所有路径与现状行为完全一致 |
| D10 投机推理（EAGLE/MTP）适配【已确认】 | **draft 层随主池一并 DRAM 化**：draft 层 ptrs 在 `_init_kv_manager` 中 append 于主池之后且**共享同一 token 索引与同一 allocator**（eagle_worker_v2 `alloc_memory_pool` 证实），故 ①DRAM 池布局天然含 draft 层（item_lens 全量镜像）；②promote 时 draft 层与主池层一起提升；③allocator wrapper 通过**有界 BFS** 替换 scheduler→target/draft worker→model_runner 全链引用，draft 侧 free 同样分流 | 共享索引使 draft 层与主池层在传输寻址、全局编码、提升上完全同构，无需独立 DRAM 池；draft 用裸 allocator free 会把 DRAM 编码索引污染进 draft 池 free list，必须全链替换 |

**已确认决策（2026-08-16）**：D2 promote 首版在 commit 内**同步执行**；D7 DRAM 池大小按
**GB 方式**配置并打印换算日志；D8 接收落池用自动水位（无新增水位参数，尽量少参数原则）；
D9 公共文件零 DRAM 专属逻辑（低侵入原则）；D10 支持投机推理（draft 层随主池 DRAM 化，
解除与 speculative 的互斥）。

---

## 3. 分模块修改方案

### M1 传输引擎支持 DEVICE_URMA

**文件**：`python/sglang/srt/disaggregation/ascend/transfer_engine.py`

| 位置 | 修改 | 原因 |
|---|---|---|
| `_get_transfer_protocol`（L95-105） | `allowed_protocols` 增加 `"device_urma"` | 当前非法值直接回退 sdma（仅 warning），跨机用户配置会被静默吞掉 |
| `initialize`（L58-84） | 增加 `device_urma` 分支 → `TransDataOpType.DEVICE_URMA` | G1：跨机 PD + DRAM 直写均依赖 URMA |

### M2 新增 Decode DRAM 接收池 `AscendDramPool`

**新文件**：`python/sglang/srt/disaggregation/ascend/dram_pool.py`

职责与接口（仿照 examples/trans_offload 的用法）：

```python
class AscendDramPool:
    def __init__(self, npu_id, num_pages, page_size, layer_layout):
        # layer_layout 来自 HBM 池 get_contiguous_buf_infos() 的 (ptrs, lens, item_lens)
        # 池总大小 = sum(lens) 向上对齐 GB(offload reserve/alloc 强制 GB 对齐)
        # offload.initialize(URMA_POOL) + offload.malloc
    def get_contiguous_buf_infos(self) -> (ptrs, lens, item_lens):  # per-layer HVA, 与 HBM 池同构
    def get_dva(self) -> int                    # offload.get_dva(base), promote 源基址
    def alloc_pages(self, n) -> Tensor[int64]   # 全局页号 [N_hbm, N_tot), free-list 管理
    def free_pages(self, global_pages)          # 归还(仅接受 DRAM 段页号)
    def num_free_pages(self) -> int
    def addr_of(self, layer_id, global_page) -> (hva, offset)  # 寻址换算
    def uninitialize(self)                      # offload.free + offload.uninitialize
```

初始化完成即打印换算日志（D7 已确认）：

```text
[DRAM pool] size=8GB, page_size=1, layers=62, num_pages=8192, capacity≈8192 tokens
```

**原因**：G2。页粒度 free-list 参考 `PagedTokenToKVPoolAllocator`（`mem_cache/allocator/paged.py:105`）
的 `free_pages/release_pages` 双张量思路，但独立实现（不侵入通用 allocator 类层次）。
GB 数与可缓存的 token/请求数之间隔着层数与 item_len（随模型变化），日志换算弥补
容量语义不直观的短板，便于压测参数选择与问题定位。

### M3 DRAM 地址注册：通用可选扩展帧（公共文件只加机制）

| 文件 | 位置 | 修改 | 原因 |
|---|---|---|---|
| `disaggregation/base/conn.py` | `KVArgs`（L38-87） | 新增 **1 个通用可选字段** `pd_extension: Optional[dict]`（默认 None） | 通用容器而非 dram 专用字段：ascend 填内容（`dram_kv_ptrs/dram_item_lens/n_hbm_pages`），其他后端零感知 |
| `disaggregation/mooncake/conn.py` | `_register_kv_args`（L2368-2450） | 尾部追加 **1 帧**：`pd_extension` 序列化（json），None 时发空帧（帧数恒定，~5 行通用机制） | 机制与内容分离：公共文件只知"有可选扩展"，不知 DRAM 为何物 |
| `disaggregation/mooncake/conn.py` | `KVArgsRegisterInfo.from_zmq`（L149） | 可选解析尾帧存 `pd_extension`（~4 行，旧版本对端发的消息无此帧时为 None） | 兼容：按索引读固定帧，多余尾帧旧端不受影响 |
| `disaggregation/ascend/conn.py` | `AscendKVReceiver` / `AscendKVManager` | 填充/解析 `pd_extension` 内容；`register_buffer_to_engine` 把 DRAM 池 HVA 追加进同一次 `batch_register` | DRAM 专属逻辑留在 ascend；HOST 内存注册自动建 DVA + URMA MR，与 HBM 一次发布 |
| `disaggregation/decode.py` | `DecodePreallocQueue.__init__` | **仅一个安装点**：`_maybe_install_dram_allocator()` 在 kv_manager 创建后调用，DRAM 池在 ascend manager 构造内分配；安装时以**有界 BFS**（scheduler→tp/target/draft worker→model_runner→tree_cache，深度 4）替换所有裸 allocator 引用（D10：spec worker 的 draft 侧 free 也必须分流） | decode.py 不出现任何 DRAM 数据结构 |

实施注意（已核实）：`mooncake/conn.py` 的 ZMQ 接收线程**无帧数 assert**（按索引读帧），
尾部追加帧对旧版本对端安全；共享 `MooncakeKVReceiver._register_kv_args` 的后端仅
mooncake（GPU 主力）与 ascend（NIXL/Mori 各自独立实现，不受影响，详见 §5.5）。

### M4 双池页分配：allocator wrapper（`_pre_alloc` 本体零改动）

**新文件**：`disaggregation/ascend/allocator_wrapper.py` — `AscendDramFallbackAllocator`（组合装饰器，不继承、不修改任何现有 allocator 类）

```python
class AscendDramFallbackAllocator:          # 包装 decode 现有 token_to_kv_pool_allocator
    def alloc/alloc_extend/alloc_decode(...):   # D8 水位: HBM 部分透传内层, 溢出页取自 dram_pool
        #   可直写预算 = 内层.available_pages() − pending_promote_pages − reserved_pages
        #   返回值 = HBM页号 ++ (dram_pool页号 + N_hbm)   # 全局页号编码, 对调用方透明
    def free(...):                              # 按全局页号分流: <N_hbm 透传内层, >=N_hbm 归还 dram_pool
    def available_size():                       # = 内层available − pending_promote需求(DRAM页非可用容量, 是缓冲)
    def alloc_hbm_for_promote(n):               # promote 专用: 强制 HBM, 不落 DRAM
    pending_promote_pages: int                  # 统计量, 供水位与预算共用
```

| 效果 | 说明 |
|---|---|
| `_pre_alloc` / `pop_preallocated` **零改动** | 现有 `allocator.alloc(...)` 调用自动获得双池语义；`send_metadata` 发的全局页号即 `dst_kv_indices`（D3 消息结构不变） |
| `_allocatable_token_budgets` **零改动** | 预算读 `available_size()`，wrapper 口径已扣 pending_promote 需求，现有判断自动正确 |
| 请求释放/abort 路径**零改动**（原 M8 大部分消失） | 现有 `allocator.free(...)` 经 wrapper 自动分流两池 |

窗口期安全性：落 DRAM 的请求在 transfer queue 中不会被 attention 读取（`pop_transferred`
成功前不可调度），`req_to_token` 中临时存放全局页号是安全的。

### M5 Decode promote：单 no-op 钩子点 + ascend 内实现（含投机推理 draft 层）

**decode.py 仅一个钩子**（`pop_transferred` Success 分支，`_commit_transfer_to_req` 之前），
**duck typing 写法，decode.py 不 import ascend**（避免循环依赖与后端耦合）：

```python
mgr = getattr(prealloc_queue, "kv_manager", None)
if hasattr(mgr, "promote_dram_pages"):
    if not mgr.promote_dram_pages(req, req_to_token_pool, hbm_pool, allocator,
                                  draft_kv_pool=prealloc_queue.draft_token_to_kv_pool):
        continue    # HBM 暂缺: 留在 transfer queue, 等 retract 腾页后重试
```

实现（`ascend/conn.py` + `dram_pool.py`，commit 内同步执行，D2 已确认）：

```python
def promote_dram_pages(req, req_to_token_pool, hbm_kv_pool, allocator, draft_kv_pool=None):
    dram_tokens, dram_pages = 挑出 req_to_token 中 >= N_hbm 的槽位与页号
    if len(dram_tokens) == 0: return True
    hbm_pages = allocator.alloc_hbm(len(dram_tokens))  # 强制 HBM; None→return False 留队重试
    hbm_ptrs, _, item_lens = hbm_kv_pool.get_contiguous_buf_infos()
    if draft_kv_pool is not None:      # D10: draft 层与主池层一起提升(共享索引)
        hbm_ptrs += draft_ptrs; item_lens += draft_item_lens
    # 按连续 token 段 × 每层 组成条目, 一次 sparse_copy 调用:
    #   src = dram_dva + layer_offset + (dram_token-N_hbm)*item_len
    #   dst = hbm_ptr[layer] + hbm_token*item_len
    offload.sparse_copy(srcPtrs, dstPtrs, lens, cnt, device)
    torch.npu.synchronize()            # AIV 拷贝落定后才能回收 DRAM 页(防与新传输竞态)
    req_to_token[req, dram_tokens] = hbm_pages          # 就地替换
    dram_pool.free_pages(dram_pages)
```

**原因**：G6。sparse_copy 源用 DVA、目标用 HBM 设备地址（与 examples/trans_offload 验证过的
用法一致）；提升后 attention 路径零改动。promote 必须发生在调度器线程的请求状态机上，
故 decode.py 保留这一个钩子点是必要的最小侵入。HBM 暂缺返回 False 留队，与 retract
（由 wrapper `available_size` 的 `inner−debt` 口径自然触发）形成闭环。

### M6 Prefill 数据面混合寻址（`mooncake/conn.py` transfer_worker 零改动）

**文件**：仅 `disaggregation/ascend/conn.py`（`send_kvcache` L103-220 覆盖方法内）

| 修改 | 原因 |
|---|---|
| DRAM 地址**不经函数参数传递**：`send_kvcache` 内部自查 `self.decode_kv_args_table[session_id].pd_extension` 取 `dram_kv_ptrs / n_hbm_pages`（该表是 `MooncakeKVManager` 现有属性，子类可直接访问）；无 extension 时走原路径 | 公共 transfer_worker 按基类签名调用，签名不变即零改动 |
| `group_concurrent_contiguous` 之后、`set_transfer_blocks` 内按分界 `N_hbm` 把块拆为 HBM 段与 DRAM 段，分别用 `dst_ptr` / `dram_ptr + (idx-N_hbm)*item_len` 寻址，合并进同一次 `batch_transfer_sync_write` | memfabric 批量写支持混合目标地址（已注册即可达），无需分两次调用 |
| 连续块跨界（同块内含两池页）处理：按池边界再切分 | 保证每块单一池，寻址公式成立 |

### M7 配置项与 fail-loud 校验

**文件**：`python/sglang/srt/server_args.py`（L3060-3170 PD 段）+
`python/sglang/srt/arg_groups/pd_disaggregation_hook.py`

| 项 | 值 | 校验 |
|---|---|---|
| 新增 `--disaggregation-decode-dram-pool-size`（GB，默认 0=禁用）【已确认 D7：GB 方式】 | DRAM 接收池大小；初始化打印换算日志（GB → 页数 → token 数，见 M2） | >0 时要求：decode 模式 + `disaggregation_transfer_backend=ascend` + 未开 `disaggregation_decode_enable_offload_kvcache`（D6 互斥）+ 未开 `disaggregation_decode_enable_radix_cache` + 未开 hisparse；**支持投机推理（D10，无互斥）**，否则 raise |
| `ASCEND_MF_TRANSFER_PROTOCOL=device_urma` | 环境变量（现有机制扩展，M1） | dram pool >0 时强制校验为 device_urma（sdma 同机内无意义、device_rdma 不支持 HOST 内存直写） |
| `MEMFABRIC_HYBRID_EXTEND_LIB_PATH` | 环境变量，指向 `libmf_hybm_accoffload.so` | offload 组件加载依赖，文档说明 |

### M8 生命周期（请求级释放已被 M4 wrapper 覆盖）

| 文件 | 位置 | 修改 | 原因 |
|---|---|---|---|
| （无） | 请求正常完成 / abort / 传输失败路径 | **零改动**：所有释放均走 `allocator.free(...)`，wrapper 按全局页号自动分流两池（M4） | G7 由 wrapper 一次性解决，异常路径不泄漏 |
| `disaggregation/ascend/conn.py` | `AscendKVManager` 清理路径（engine destroy 前后） | `dram_pool.uninitialize()`（offload.free + uninitialize） | 干净退出，避免 HCOM/URMA 残留资源（历史经验：残留导致重启失败 ret:15）；挂在 ascend 类内，decode.py 不感知 |

---

## 4. 端到端时序

```text
[启动]
Decode : offload.initialize(URMA_POOL) → malloc(GB 对齐) → AscendDramPool(打印换算日志)
       → TransferEngine.initialize(..., DEVICE_URMA)
       → batch_register(HBM kv + aux + state + DRAM池HVA)   # HOST 内存自动建 DVA+MR
       → 安装 AscendDramFallbackAllocator(decode.py 唯一安装点)
       → ZMQ 注册: kv_data_ptrs + pd_extension(可选尾帧: dram ptrs/N_hbm) → Prefill
Prefill: engine init + 本侧 src 池 batch_register
       → 收注册帧缓存 KVArgsRegisterInfo(含 pd_extension)

[每请求]
Decode : _pre_alloc(零改动): allocator.alloc(...) 经 wrapper 按水位落池 → 全局页号 → req_to_token
       → send_metadata(dst_kv_indices=全局页号, 消息结构不变)
Prefill: transfer_worker(零改动) → AscendKVManager.send_kvcache(自查 pd_extension):
         块按 N_hbm 分界 → HBM/DRAM 混合寻址
       → batch_transfer_sync_write(混合 dst) --DEVICE_URMA--> Decode HBM + DRAM 直写
       → status(ZMQ) → Decode
Decode : pop_transferred → commit(2行钩子) → promote: sparse_copy(DVA→HBM)
       → req_to_token 就地替换 → 释放 DRAM 页 → 请求可调度(attention 只读 HBM, 零改动)

[结束]
请求完成/abort: allocator.free(零改动) 经 wrapper 自动分流两池
进程退出: AscendKVManager 清理 → dram_pool.uninitialize + engine.destroy
```

## 5. GPU / 共享路径影响分析

后端继承关系（已核实）：`Ascend*` → `Mooncake*` → `Common*`；`Nixl*`、`Mori*` → `Common*`
（各自独立实现，不经过 Mooncake 层）。逐改动点对 GPU 路径的影响：

| 改动点 | GPU（mooncake 主力后端）影响 | NIXL/Mori | 说明 |
|---|---|---|---|
| `KVArgs.pd_extension` 字段（base/conn.py） | 构造时不填 → None，无人读取 → **零行为影响** | 同左 | 纯可选数据字段，无 schema 校验 |
| ZMQ 注册消息尾帧（mooncake/conn.py） | `pd_extension=None` → 多发 1 个**空帧**；新代码 prefill 按长度可选解析，旧代码 prefill 无帧数 assert（已核实）按索引读 → 忽略 → **正常** | **不受影响**（不走 Mooncake 层的收发实现） | 唯一真正的 GPU 可见变化：线上多 1 个空 ZMQ 帧，新旧混部均兼容 |
| decode.py 两个钩子（安装点 + promote） | 条件恒 False（GPU 的 kv_manager 无 `promote_dram_pages` 属性、backend≠ascend）→ **死分支**，开销为一次 hasattr/条件判断 | 同左 | duck typing 写法，decode.py 无 ascend import |
| `send_kvcache` 混合寻址 | 只改 `AscendKVManager` 覆盖版，mooncake 基类方法**不动** → **数据面零影响** | 不受影响 | — |
| 新参数 + hook 校验 | 默认 0，GPU 不配置不触发；误配置（非 ascend 后端设 >0）→ raise（fail-loud 正确行为） | 同左 | — |
| allocator / memory_pool / attention / transfer_worker / send_metadata / `_pre_alloc` 本体 | **零修改** | 零修改 | wrapper 为外包装组合，不改任何类层次 |

结论：GPU 路径的全部可感知变化 = **ZMQ 注册消息多 1 个空帧**（兼容已核实）+ 每请求
commit 处一次恒 False 的 hasattr；数据面、调度核心路径、内存管理层均零修改。
NIXL/Mori 后端因不经过被修改的 Mooncake 层，完全不受影响。

## 6. 限制与风险

| 项 | 说明 | 缓解 |
|---|---|---|
| 平台 | 仅 A5(950)；A3 上 DEVICE_URMA OpenDevice 直接失败 | memfabric fail-loud 兜底 + 文档注明 |
| promote 同步开销 | commit 内同步 sparse_copy（D2 已确认），长序列大页时增加调度延迟 | 首版接受；后续按需异步化（预提升/流水） |
| HBM 仍为瓶颈 | DRAM 只扩"接收容量"，运行态 KV 仍需 HBM；promote 需要瞬时 HBM 页 | wrapper `available_size` 口径已扣 pending_promote 需求 + 现有 retract 语义 |
| 双重写放大 | 落 DRAM 的页经历"跨机写 DRAM + promote 搬 HBM"两跳 | 仅在 HBM 不足时触发；常态全走 HBM 单跳 |
| wrapper 与 allocator 接口面 | wrapper 只拦截 alloc/alloc_extend/free/available_size/alloc_hbm，其余经 `__getattr__` 转发内层（DRAM 页不参与内层 `merge_and_sort_free` 排序）；实施时以 decode.py 实际调用集为准做接口核对 | — |
| 投机推理（D10） | draft 层与主池共享 token 索引/allocator（eagle_worker_v2 `alloc_memory_pool`）：draft 层 DRAM 寻址/promote 与主池同构；draft 侧 free 若走裸 allocator 会污染 | promote 拼接 draft 层条目；安装点 BFS 替换 scheduler→target/draft worker→model_runner 全链 allocator 引用 |
| spec worker 引用替换时序 | DecodePreallocQueue 创建晚于 worker init，spec worker 内部已绑定裸引用 | BFS 安装点按"引用身份相等"替换（`is inner`），覆盖构造期绑定的所有副本 |
| decode 侧 radix cache | 全局页号与 radix tree 页语义不兼容 | M7 与 `disaggregation_decode_enable_radix_cache` 互斥校验（该特性本身实验性） |
| MLA kv_buf_groups | DRAM 池需按 buffer 组镜像切分 | M2 布局同构设计天然支持；首版实测 MHA+MLA 各一 |
| 非 URMA 协议 | sdma/device_rdma 下 dram pool 禁用 | M7 强制校验 |
| 帧兼容 | 旧版 Prefill + 新版 Decode 混部 | 尾部追加帧 + from_zmq 可选解析（M3），旧端忽略 |

## 7. 测试计划

1. **单测（NPU CI 可跑部分）**：全局页号编码/解码；`AscendDramPool` free-list 分配释放一致性；
   wrapper alloc/free 分流与水位公式；`pd_extension` 序列化/反序列化（含 None 路径）。
2. **双机 e2e（A5×2）**：
   - HBM-only（回归）：dram pool size=0，所有路径与现网 bit-exact（验证 D9 隔离性）；
   - GPU 回归：mooncake 后端跑通 PD e2e（验证 §5 空帧/钩子死分支零影响）；
   - DRAM 溢出：压小 `mem_fraction_static` 使 HBM 不足，验证落 DRAM → promote → 推理正确
     （对比 HBM-only 输出一致）；
   - **投机推理 + DRAM（D10）**：`--speculative-algorithm`（MTP/EAGLE）+ dram pool 组合，
     验证 draft 层 DRAM 直写/提升后与 HBM-only spec 输出一致；
   - 带宽：复用 examples/trans_offload 已验证的分档 NPU event 计时方法观测各段带宽。
3. **异常路径**：传输超时 abort 后 DRAM 页归还（wrapper free 分流）；进程重启无 HCOM 资源残留。

## 8. 修改文件清单（按侵入度分层）

**ascend 专属（DRAM 全部逻辑所在，可自由修改）**：

| 文件 | 类型 |
|---|---|
| `python/sglang/srt/disaggregation/ascend/transfer_engine.py` | 修改（M1，协议枚举） |
| `python/sglang/srt/disaggregation/ascend/dram_pool.py` | **新增**（M2/M5：池 + promote 实现） |
| `python/sglang/srt/disaggregation/ascend/allocator_wrapper.py` | **新增**（M4：双池分配 + 分流释放） |
| `python/sglang/srt/disaggregation/ascend/conn.py` | 修改（M3/M6/M8：extension 填充解析、混合寻址、清理） |

**公共文件（只加"默认关闭零行为变化"的机制/钩子）**：

| 文件 | 修改量 | 内容 |
|---|---|---|
| `disaggregation/base/conn.py` | ~3 行 | `KVArgs.pd_extension` 可选字段（M3） |
| `disaggregation/mooncake/conn.py` | ~10 行 | 可选扩展帧机制（发/收，None 零行为，M3） |
| `disaggregation/decode.py` | ~6 行 | wrapper 安装点（M4）+ promote 钩子（M5），非 ascend 时 no-op |
| `server_args.py` / `arg_groups/pd_disaggregation_hook.py` | 常规扩展 | 1 个新参数 + 互斥校验（M7） |

公共文件合计 ~20 行，且不含任何 DRAM 数据结构或寻址逻辑；
`dram_pool_size=0`（默认）时全部为死分支，任何后端行为与现状一致。

memfabric 侧：无新增改动（能力已在 commit 1ea9da46 就绪）。
