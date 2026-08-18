# sglang PD 分离 KV cache DRAM offload 修改设计

> 目标：PD 分离（NPU/Ascend，A5 平台）下，Prefill 节点写 KV cache 到 Decode 节点时，
> 目标地址支持 **HBM 地址 + DRAM 地址** 两类；Decode 启动/注册时把 HBM+DRAM 两类地址
> 都传输给 Prefill；Decode 节点的 KV cache 管理（页分配/释放/提升）兼容 HBM+DRAM 双池。
> 全部基于 memfabric（smem_trans DEVICE_URMA + offload URMA_POOL + sparse_copy）实现。
>
> **修订（2026-08-18）**：按联调实测修正全局编码为**页本位**（D3/M2/M4/M5 重写，
> 修正 §9-B7 页/token 单位混用）；补充 memfabric 侧联调修复（§8）；新增 §9 联调问题
> 与修复记录（B/M/O 三系列）。

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
| D3 页归属表达【实测修正】 | **全局页号编码（页本位）**。关键单位约定（调试踩坑确认，详见 §9）：NPU 分页池为页主序布局 `(layer, size//page+1, page_size, ...)`，`kv_item_lens` 是**每页字节数**（非每 token）；PD 传输链路全程**页索引**（`kv_to_page_indices`）。因此 `n_hbm_tokens = kv_data_lens[0]//kv_item_lens[0]` 是 HBM **页数**（如 103 可分配页+1 备用页=104，对应 allocator token 空间 13184=103×128）。DRAM 页在 `req_to_token` 中以**合成 token** 表达：`token=(n_hbm+local_page)*page_size+intra_page`（≥n_hbm×page_size，落在 allocator 真实 token 空间之外，不与活跃 HBM token 冲突；P 侧 `token//page_size` 即还原全局页号） | 复用现有 `send_metadata`/`TransferInfo`/`group_concurrent_contiguous` 全链路，ZMQ 每请求消息结构零改动；仅在寻址处分流。**严禁把 token 数当页数用**（曾致 MTE errcode 95/507035 越界写，§9 B7） |
| D4 DRAM 池布局 | 与 HBM KV 池**同构镜像**（per-layer 连续；item_len=每页字节，段长=页数×每页字节） | Prefill 侧寻址公式 `base + global_page*item_len` 两池统一；MLA `kv_buf_groups` 多 buffer 组同样按组镜像 |
| D5 aux/state 落点 | 首版**仍走 HBM**（不落 DRAM） | aux（首 token 元数据）与 state 池很小，DRAM 化收益低、改动面大 |
| D6 与 hicache 关系 | 首版与 `--disaggregation-decode-enable-offload-kvcache` **互斥**（fail-loud） | 两者都管理 decode 侧 KV 与 host 内存的交互，叠加语义未定义；后续再评估组合 |
| D7 DRAM 池容量配置【已确认】 | 按 **GB 绝对值**配置（`--disaggregation-decode-dram-pool-size`，默认 0=禁用）；初始化完成即打印换算结果：`GB 数 → 页数 → 可容纳 token 数` | 直传 offload reserve/alloc（本身 GB 对齐）无换算损失；DRAM 为共享资源（OS/其他进程），绝对值便于整机规划与 fail-loud 校验；打印换算弥补"能缓多少请求"语义不直观的短板。跨模型压测迁移需求出现时再考虑 ratio 别名（先例：hicache_ratio/hicache_size 双参数并存） |
| D8 接收落池策略【已确认】 | **自动水位，零新增参数**（token 单位）：`预算(tokens) = inner.available_size() − dram_pool.allocated_tokens() − num_reserved_decode_tokens`；`预算 ≥ 需求 → 全落 HBM`；`预算 < 需求 → 整请求落 DRAM`；DRAM 也满 → 回退 HBM 原语义（排队/预分配失败如常） | 水位本质是"HBM 剩余先还 promote 债 + 留给 running 增长"，可完全派生，无需用户调参；整请求落 DRAM 避免 HBM/DRAM 混排请求的 promote 碎片化 |
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

职责与接口（最终实现，**页本位**，见 D3 单位约定）：

```python
class AscendDramPool:
    def __init__(self, npu_id, pool_size_gb, page_size, item_lens, n_hbm_tokens):
        # item_lens = HBM 池 get_contiguous_buf_infos() 的每页字节（页主序布局）
        # self.size = 池页数 = GB*2^30 // sum(item_lens)   # 页数, 非token数!
        # offload.initialize(URMA_POOL, Scene.LOCAL) + offload.malloc(size*sum(item_lens))
        # layer_offsets[l] = l 号层段基址偏移;  dva = offload.get_dva(base)
    def alloc(self, need_tokens) -> Tensor       # 页粒度 free-list;
        # 返回合成 token: (n_hbm + local_page)*page_size + intra  (排序后天然落在
        # HBM allocator token 空间之外; P 侧 kv_to_page_indices 还原全局页号)
    def free_tokens(self, tokens) -> int          # 阈值 >= n_hbm*page_size 判 DRAM;
        # 页 = token//page_size − n_hbm, 归还 free-list
    def available_size(self) -> int               # = 空闲页数 * page_size (token口径)
    def allocated_tokens(self) -> int             # promote 债 (token口径)
    def get_contiguous_buf_infos(self)            # per-layer (HVA, 页数*item_len, item_len)
    def layer_src_dva(self, layer_id, page_off)   # promote 源: dva + offset + page_off*item_len
    def promote(self, entries, device_id)         # sparse_copy(srcPtrs, dstPtrs, lens, cnt, dev)
    def uninitialize(self)                        # offload.free + offload.uninitialize
```

初始化完成即打印换算日志（D7 已确认）：

```text
[DRAM pool] size=8GB, page_size=128, layers=234, pages=4866, capacity≈622848 tokens, base=0x..., dva=0x...
[DRAM] manager init: pool created, dram_layers=234 n_hbm_pages=104 dram_pages=4866 base=0x... dva=0x...
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
    def alloc(self, need_size):                 # D8 水位(token): 预算=inner.available −
        #   dram债 − reserved; 够→透传内层; 不够→整请求落 dram_pool.alloc;
        #   DRAM也满→回退 inner.alloc(原语义)
    def alloc_extend(...):                      # prefix=0 的分页预分配同水位逻辑
    def free(self, free_index):                 # 按池分流: dram_pool.free_tokens 先收;
        #   阈值 n_hbm_tokens*page_size (页数×页大小! §9 B7), < 阈值透传内层
    def available_size(self):                   # = max(0, inner.available − dram债)
    def available_size_for_prealloc(self):      # = 上式 + dram_pool.available (接收口径)
    def alloc_hbm(self, need_size):             # promote 专用: 强制 HBM 透传 inner.alloc
    def __getattr__(self, name):                # 其余(page_size/get_kvcache/...)转发内层
```

构造时单位自检（fail-loud，不修改边界）：`inner.size ≤ n_hbm_tokens × page_size`
必须成立（n_hbm 含 allocator 页 + 池备用页）；曾出现把 token 数误写进页边界的
错误"修复"（`n_hbm_tokens = inner.size`），已撤销（§9 B7）。

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

实现（`ascend/conn.py` + `dram_pool.py`，commit 内同步执行，D2 已确认，**全程页本位**）：

```python
def promote_dram_pages(req, req_to_token_pool, hbm_kv_pool, allocator, draft_kv_pool=None):
    n_hbm = dram_pool.n_hbm_tokens          # HBM 页数(legacy属性名,勿混淆)
    page_size = dram_pool.page_size
    row = req_to_token[req][:committed_len]
    dram_mask = row >= n_hbm * page_size    # 阈值必须页换算(§9 B7)
    if 无DRAM token: return True
    dram_np = row[dram_mask].cpu().numpy()
    gpages = unique(dram_np // page_size); lpages = gpages - n_hbm   # 全局页→局部页
    hbm_loc = allocator.alloc_hbm(len(lpages) * page_size)  # 整页分配(空张量守卫,§9 B4)
    hbm_pages = hbm_loc[::page_size] // page_size           # token运行→页号
    if draft_kv_pool: hbm_ptrs += draft_ptrs; item_lens += draft_item_lens  # D10
    for (页段 start,cnt) in _co_segments(lpages, hbm_pages):  # 双数组共连续切分
        for layer_id, item_len in enumerate(item_lens):
            src = dram_pool.layer_src_dva(layer_id, lpages[start])       # DVA
            dst = hbm_ptrs[layer_id] + hbm_pages[start] * item_len       # 页寻址!
            entries.append((src, dst, cnt * item_len))
    torch.npu.synchronize()                # 隔离探针: 先于AIV拷贝同步
    offload.sparse_copy(srcPtrs, dstPtrs, lens, cnt, dev)
    torch.npu.synchronize()                # AIV 拷贝落定后才能回收 DRAM 页(防与新传输竞态)
    k = searchsorted(gpages, dram_np // page_size)         # 每token的DRAM页序号
    row[dram_mask] = hbm_pages[k]*page_size + dram_np % page_size  # 保页内偏移, int32(§9 B5)
    dram_pool.free_tokens(dram_tokens)     # 合成token按页归还
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
Decode : _pre_alloc(零改动): allocator.alloc(...) 经 wrapper 按水位落池
       → HBM token(allocator原生) 或 DRAM 合成 token((n_hbm+page)*page_size+intra) → req_to_token
       → send_metadata(dst_kv_indices, 消息结构不变)
Prefill: transfer_worker(零改动) → AscendKVManager.send_kvcache(自查 pd_extension):
         kv_to_page_indices 取页号 → 按 n_hbm 分界 → HBM/DRAM 混合寻址(页×每页字节)
       → batch_transfer_sync_write(混合 dst) --DEVICE_URMA--> Decode HBM + DRAM 直写
       → status(ZMQ) → Decode
Decode : pop_transferred → commit(2行钩子) → promote: 页本位 sparse_copy(DVA→HBM)
       → req_to_token 就地替换(保页内偏移) → 释放 DRAM 页 → 请求可调度(attention 只读 HBM, 零改动)

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
| wrapper 与 allocator 接口面 | wrapper 拦截 alloc/alloc_extend/free/available_size/available_size_for_prealloc/alloc_hbm，其余经 `__getattr__` 转发内层（DRAM 页不参与内层 `merge_and_sort_free` 排序）；free 分流阈值必须页换算 `n_hbm*page_size`（§9 B7/B8） | — |
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

**memfabric 侧（联调中修复，仓库 `mem-pool/memfabric_hybrid` 分支 br_trans_offload）**：

| 文件 | 修改 | 对应问题（§9） |
|---|---|---|
| `src/hybm/csrc/mm/hybm_vmm_based_segment.cpp` | 修复 `hybm_mem_type memType` 重复声明编译错误；RegisterMemory 失败（HalMemExport ret 8）时 **ExportInner 降级**为仅本地注册（保留 MR/VA 供 URMA 传输）；Export(slice) 失败回退 legacy 空名字格式，避免注册交换失败 | M1/M2/M3 |
| `src/acc_offload/csrc/operators/acc_offload_sparse_copy.h` | 多条目拷贝从"拼包单次队列"改为**逐条目独立队列往返**，规避向量核 507035 | M4 |

---

## 9. 联调问题与修复记录（2026-08-17 ~ 08-18）

按发现顺序编号；B 系列为 sglang 侧，M 系列为 memfabric 侧，O 系列为运维/环境。
**B7（页/token 单位混用）是全局编码的核心修正**，D3/M2/M4/M5 均按其结论重写。

### B 系列：sglang 侧

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| B1 | `AttributeError: 'Scheduler' object has no attribute 'token_to_kv_pool'`（decode.py `pop_transferred`） | D 节点 scheduler 未暴露 token_to_kv_pool 属性（P 侧才有） | scheduler.py 中补 `token_to_kv_pool` 属性定义 |
| B2 | P 侧 `send_kvcache` 块拆分 IndexError（`decode_index[mid]`） | 链式比较 `first < n_hbm == last < n_hbm` 在 Python 中语义为 `(first<n_hbm) and (n_hbm==(last<n_hbm))`，恒 False → 所有块（含纯 HBM 块）都进 split 分支 | 加括号显式分组：`same_pool = (first < n_hbm) == (last < n_hbm)` |
| B3 | P 节点启动 `RuntimeError: basic_string::_S_construct null not valid` | `MF_CONFIG_STORE_URL`/`ASCEND_MF_STORE_URL` 未设置，None 传入 C++ binding 构造 std::string | transfer_engine.py 先取环境变量（双名字回退），缺失即 fail-loud 报明确错误 |
| B4 | D 节点 promote `IndexError: list index out of range` | 分页 allocator 对不足一页的请求（如 4 token）向下取整为 0 页，返回**空张量**（非 None）绕过守卫 | HBM 分配按页向上取整（`ceil(num/page)*page`）+ `numel()` 守卫 |
| B5 | `RuntimeError ... error code 161002`（aclnnIndexPutImpl） | `hbm_loc` 是 int64，`req_to_token` 是 int32，aclnn index_put 双侧 dtype 必须一致 | 回写前显式 `.to(torch.int32)`（后统一经 numpy int32） |
| B6 | `npuSynchronizeDevice ... device error type 3, error code 507035`（向量核错误） | sparse_copy 多条目"拼包"单次队列下发触发向量核异常 | memfabric 侧改逐条目独立队列往返（M4），sglang 侧保留 |
| **B7** | **promote RANGE VIOLATION / ENCODING MISMATCH / MTE "DDR address out of range" errcode 95→507035** | **页/token 单位混用**：①NPU 分页池页主序布局，`kv_item_lens` 是**每页字节**（234 层合计 14,057,472 B/页，÷128=109,824 B/token），传输链路页索引；②`n_hbm=104` 是 HBM **页数**（103×128=13184 token+备用页），被当 token 数用；③DRAM 合成 token 旧编码 `page*128+intra+104` 数值落在 HBM 页号区间 → P 侧把 DRAM 页写进 HBM 池（数据错位），D 侧从 DRAM 假页提升垃圾数据；④`dram_mask = row >= 104` 把合法 HBM token 104..13183 误判为 DRAM → 用垃圾覆盖好 KV；⑤promote `dst = hbm_ptr + token_idx*每页字节` 越界约 7GB | **全链路改页本位**（本次重写）：`dram_pool.size`=页数（容量恢复 128 倍）；合成 token `(n_hbm+page)*page_size+intra`（≥n_hbm×page_size 不与 HBM token 冲突）；mask/free 阈值页换算；promote 按页段 co-segment、`dst = hbm_ptr + hbm_page*item_len`、行回写保页内偏移；校验探针页单位（`inner.size ≤ n_hbm*page_size`） |
| B8 | 中间错误"修复"：`fixing n_hbm_tokens: 104 → 13184` 后 P 侧寻址仍错 | 把 allocator 的 token 数写进页边界（单位混入），且该修改**不同步 P 侧 pd_extension**（P 仍按 104 分界） | 撤销该修改；保留 layer0 推导的页数边界 + wrapper 构造时单位自检（fail-loud） |
| B9 | warmup 后 P/D 卡住（核利用率 0，无 `register_kv_args: received pd_extension`） | 见 O1/O2/O3 组合 | 逐一排除后恢复 |
| B10 | transformers 包刷屏 `[transformers] Accessing forward_npu ...` | transformers 日志级别低 | 设 transformers logger ERROR |

### M 系列：memfabric 侧

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| M1 | 编译错误 `redeclaration of hybm_mem_type memType`（hybm_vmm_based_segment.cpp:367） | 同作用域重复声明 | 去重 |
| M2 | `HalMemExport ... ret: 8` → `RegisterMemory failed ret: 12/c`，注册失败 | `expandable_segments:True` 下 VMM 分配的小 KV 层被 batch_register 合并成大 slice，物理不连续 → HalMemExport 失败 | 失败时 **ExportInner 降级**为仅本地注册（保留 MR/VA，URMA 传输不受影响；fabric 句柄交换路径不再必需） |
| M3 | `Export: Assert pos != slices_.end, input slice(idx:0) not exist` → `ExportSliceExchangeInfo failed: -2` | 合并 slice 后按原 idx 查找失败 | Export(slice) 失败回退 legacy 空名字格式 |
| M4 | sparse_copy 拼包路径 507035 向量核错误（同 B6） | 多条目单次队列下发异常 | 逐条目独立队列往返 |
| M5 | P→D DRAM 直写失败：`BatchCopyByAutoGroup failed to auto infer copy direction, dest=0x1c000`（HBM-only 健康探测正常，落 DRAM 的请求必现） | smem 写路径 dst 经 remoteSlices_ 映射（TransformAddr）：`mapped = slice的GVA + slice内偏移`（跨节点 URMA 走 **GVA/fabric 寻址**，非 DVA）。VMM 降级导出的 `gvaOffset = slice->gva_ − 段GVA基址`，而 host slice 的 `slice->gva_` 从未赋值（=0）→ offset 无符号回绕 → P 侧重构 gva = P基址−D基址 ≈ 0（两节点 GVA 布局相同）→ 映射值塌缩为 `0 + 0x1c000` → 地址分类失败 | VMM 两条空名字导出路径（host slice / 降级 slice）：`gvaOffset` 优先取 VA 管理器中该 host 分配的**真实 GVA**（`FindAllocByVa(...,HVM_DVA)` 记录含 gva/deviceVa/hostVa），无记录且 `slice->gva_==0` 时 fail-loud 报错 |
| M6 | 二次回归：`TransformAddr remote address 0x127a95c64000(=DRAM池基址) is invalid (before first slice)`，ret=-2000 | 错误"修复"：`Local2GlobalMap` 是**降序 map**（`std::greater`），降序上 `lower_bound` 本就是包含式查找（首个 key ≤ dst）；改成升序习语 `prev(upper_bound)` 反而选中 dst 上方相邻 slice，且当 dst 为数值最大的注册地址（DRAM 池基址 > 所有设备 VA slice）时 `upper_bound==begin()` 必然误报 | 还原 `lower_bound` 原实现，注释标明降序语义防止再次误改 |
| M7 | ~~e2e 回归：decode 返回地址 0x34→0x10~~ **误报澄清（08-18）**：历史记录确认 A5 上 `HalMemAlloc`（conn-based DRAM 段 `MEM_HOST\|MEM_TYPE_DDR`）返回的 DRAM 池地址就是 **0x10 前缀**，e2e 行为本身无回归；此前 e2e 失败是 M5 首版错误方向（DVA 导入）+ M6 错误查找叠加所致，随两者修正/还原后 **P-D 传输已恢复**。正确机制：decode 打印的 0x10 HVA 作为 P 侧写目标与 slice map 键，TransformAddr 命中后翻译为对端重构 GVA（0x34 段），URMA 按 GVA 跨节点寻址 | 保留 M5 修正（gvaOffset 取真实 GVA）与 M6 还原（lower_bound）；Import 语义维持原状（返回 gva_、空 name slice vAddress_=nullptr）。遗留无害噪音见 M8 |
| M8 | 传输恢复后残留打印：D 侧 `HybmDevUserLegacySegment NOT SUPPORT Mmap`；P/D 侧 `Unable to RemoveOneVaInfo: address not found va=0x10…` | ① legacy 段不支持 Mmap 是设计内行为，调用方（MemEntityDefault::Mmap，e013e65f）显式容忍 `BM_NOT_SUPPORTED`，仅日志级别误用 ERROR；② RemoveOneVaInfo 失败发生在注销/释放收尾：batch_register 将相邻 2MB 对齐区间合并为单 slice 注册，而释放按原始 buffer 粒度逐个走，首次释放已删合并范围记录，后续兄弟 buffer 幂等删除扑空（host 注册走 gva-only 插入亦有 key 差异） | 当前用法（启动注册一次、退出注销一次）**无功能影响**；长期注意：VA manager 可能残留 stale 记录，若未来出现进程内反复 register/deregister 需先清理；节点重启维持冷重启 + 清 `/dev/shm` 习惯（O1）。可选清理：Mmap 日志降级 WARN、RemoveOneVaInfo 幂等删除降级 WARN，或注销侧按注册合并粒度释放 |

### O 系列：运维/环境经验

| # | 现象 | 根因/处置 |
|---|---|---|
| O1 | `HcommThreadAlloc` 返回码 15，重启失败 | HCOM/URMA 残留资源 | 冷重启 P/D 节点 + 清理 `/dev/shm` 残留文件 |
| O2 | P 节点 `MEM_FRACTION=0.91+` 触发 AIV kernel 507035 | prefill 激活 + deepep buffer 挤占 HBM 不足 | P 侧降到 0.85；调试期 D 压到 0.7800（HBM 池仅 ~128 token，强制请求落 DRAM 观察全流程） |
| O3 | P 节点 DeepEP 纯机内 normal dispatch 路径挂死/崩溃 | NPU 构建 deepep 稳定性问题 | P 侧用默认 MoE a2a backend（none） |
| O4 | 启动顺序要求 | memfabric config store 由 P rank0 创建 | **P 先启动且健康后再启 D**；P/D config.json 层数必须一致 |
| O5 | device_urma 必需环境 | VMM 内存走 retain handle + fabric share handle | `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` + `MF_HYBM_USE_VMM_SEGMENT=1` + `MF_CONFIG_STORE_URL`；调试期 `ASCEND_LAUNCH_BLOCKING=1` |

### 验证日志关键字（B7 修复后）

- 正常：`[DRAM pool] ... pages=`（页数，8GB 池约 4866）；`pool alloc: ... token_range=` 起点 ≥ n_hbm×page_size（如 13312）；`promote start: ... hbm_page=[..]` ∈ [0, 104)；`send_kvcache: writing N/N dst tokens to DRAM pool`
- 应消失：`promote RANGE VIOLATION` / `ENCODING MISMATCH` / `fixing n_hbm_tokens` / `DDR address out of range` / `507035`
