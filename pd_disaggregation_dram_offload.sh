#!/bin/bash
# ============================================================================
# PD 分离 + Decode KV cache DRAM offload 启动脚本（A5 / ASCEND_950，memfabric URMA）
#
# 拓扑: Prefill 单机 8 卡 (141.61.49.198) + Decode 单机 8 卡 (141.61.49.195)
#
# 用法（无需参数, 按 hostname -I 自动识别本机角色）:
#   Prefill 节点 (141.61.49.198): bash pd_disaggregation_dram_offload.sh
#   Decode  节点 (141.61.49.195): bash pd_disaggregation_dram_offload.sh
#
# DRAM offload (当前分支): Selective HiSparse, 由 --npu-selective-hisparse-layer-ids
#   控制 selected 层 KV 驻留 Host DRAM (池大小自动推导, 无需手动指定)
#   要求 ASCEND_MF_TRANSFER_PROTOCOL=device_urma (跨机 URMA, 仅 A5)
# ============================================================================

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export SELECTIVE_LAYER_IDS=${SELECTIVE_LAYER_IDS-"5 9 13 17 21 25 29 33 37 41 45 49 53 57 61 65 69 73 77"}
# [精度调试] cuda-graph-bs 可用环境变量覆盖, 免改脚本:
#   CUDA_GRAPH_BS="1"                (对照旧 bs1 实验)
#   CUDA_GRAPH_BS="8 16"             (默认, 对照 0.76)
CUDA_GRAPH_BS=${CUDA_GRAPH_BS-"8 16"}

MODEL_PATH=${MODEL_PATH:-/home/weights/GLM-5.2-W8A8C8-mxfp8}
# P/D 解耦: P(prefill) 8192-token 大 batch 需要大量激活内存, fraction 过高会在
# MoE dispatch(AIV kernel) 内存耗尽 -> 507035 向量核异常; D(decode) batch 小,
# 可用高 fraction 换更大 HBM KV 池
P_MEM_FRACTION=${P_MEM_FRACTION:-0.85}
# 调试用: 压缩 D 的 HBM KV 池到 ~128 token(1页), 令所有请求落 DRAM 以观察
# 写池+提升流程。换算(128GB 卡, 实测 0.105MB/token): 斜率≈124.8万 token/1.0,
# 零KV基线 f0≈0.7799, 目标128tok -> f≈0.7800; 每 ±0.0005 ≈ ±620 tok,
# 以启动日志 "#tokens:" 为准微调(为0则+0.0005)。恢复正常运行改回 0.91。
D_MEM_FRACTION=${D_MEM_FRACTION:-0.915}

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 注意: URMA(DEVICE_URMA) 注册路径下禁止 expandable_segments —— VMM 显存无法被
# RtIpcSetMemoryName IPC 命名(507899), batch_register 会失败; e2e(trans_offload)验证过常规堆 OK
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MF_HYBM_USE_VMM_SEGMENT=1
export STREAMS_PER_DEVICE=32
export ASCEND_LAUNCH_BLOCKING=0

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

# deepep/HCCL 关键环境（对齐已验证可跑通的 base.sh）
# deepep low_latency 单 rank dispatch buffer 容量, 必须 >= max(cuda-graph-bs) * num_draft_tokens
# (NEXTN verify: 16*6=96, 留余量取128; offload4.sh 为 1 4 6 -> 6*6=36 压线通过)
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export DEEPEP_NORMAL_LONG_SEQ_ROUND=64
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=68
export ACL_DEVICE_SYNC_TIMEOUT=60
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MLAPO=1

export PYTHONPATH=`pwd`/python:$PYTHONPATH

# 日志双路输出: sglang serve 的 stdout/stderr 同时打屏 + 落文件 (tee)
# (memfabric C 层日志走 stderr 一并捕获; 文件名带角色/时间戳便于区分)
LOG_DIR=${LOG_DIR:-`pwd`/logs/}
mkdir -p "$LOG_DIR"
LOG_TS=$(date +%Y%m%d_%H%M%S)

# --------------------- PD 拓扑: P 单机 8 卡 + D 单机 8 卡 ---------------------
P_IP=('141.61.49.198')
D_IP=('141.61.49.195')

# 跨机 URMA 传输 + DRAM 池远端直写（P/D 两侧均需, 仅 A5 支持）
export ASCEND_MF_TRANSFER_PROTOCOL=device_urma
# session store, P/D 两侧均可达（挂在 Prefill 节点）
# 注意: Python 层 transfer_engine.py 读 ASCEND_MF_STORE_URL, C++ 层读
# MF_CONFIG_STORE_URL —— 两个名字必须同时导出且指向同一地址,
# 否则 store_url=None 传入 C++ 报 basic_string::_S_construct null not valid
export ASCEND_MF_STORE_URL="tcp://141.61.49.198:24669"
export MF_CONFIG_STORE_URL="$ASCEND_MF_STORE_URL"
# offload 组件依赖库（如已装到默认路径可不设）
# export MEMFABRIC_HYBRID_EXTEND_LIB_PATH=/path/to/libmf_hybm_accoffload.so

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

# HCCL/Gloo 网口: 手动填入本机承载 P/D 通信的网卡名（ifconfig / ip addr 查看）
export HCCL_SOCKET_IFNAME=enp35s0f2
export GLOO_SOCKET_IFNAME=flannel.1

for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        P_LOG="$LOG_DIR/prefill_${LOG_TS}.log"
        echo "Prefill -> ${P_IP[$i]}  (log: $P_LOG)"

        # P 为单机 8 卡: deep_ep normal 的 pure intranode dispatch 路径在当前
        # deep_ep NPU 构建上会挂死/向量核异常(507035), 已用单机非 PD 部署复现;
        # 单机无需 a2a, MoE 走标准 TP 路径(base.sh 16 rank 走 internode 故未踩中)
        sglang serve \
            --model-loader-extra-config '{"enable_multithread_load": true}' \
            --disaggregation-mode prefill --disaggregation-transfer-backend ascend \
            --disaggregation-bootstrap-port $((8998+$i)) \
            --model-path $MODEL_PATH \
            --tokenizer-path $MODEL_PATH \
            --trust-remote-code \
            --attention-backend ascend \
            --device npu \
            --quantization modelslim \
            --dtype bfloat16 \
            --kv-cache-dtype fp8_e4m3 \
            --tp-size 8 \
            --mem-fraction-static $P_MEM_FRACTION \
            --chunked-prefill-size 8192 \
            --max-running-requests 256 \
            --host 0.0.0.0 \
            --port 31000 \
            --watchdog-timeout 9000 \
            --disable-cuda-graph \
            2>&1 | tee "$P_LOG"

        exit 1
    fi
done

for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        D_LOG="$LOG_DIR/decode_${LOG_TS}.log"
        echo "Decode -> ${D_IP[$i]}  (log: $D_LOG)"

        # deepep HCCL buffer 需求随 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK 线性放大:
        # 128 容量时需 825MB (报错日志含 NEEDED_HCCL_BUFFSIZE 算式), 留余量取 850
        export DEEPEP_HCCL_BUFFSIZE=850

        # [精度调试 R3] 单层(layer 5) + 单请求 + cuda-graph-bs 8 16:
        # 相比 0.76 基线(19层)仅减少 selected 层数 → 每步 DMA/向量算子数降 19 倍,
        # pad 行数不变(7 行)。判读:
        #   大幅恢复(≥0.9) → 图内时序竞态(竞争量驱动) → 转 memfabric 架构修复
        #   仍 ~0.76       → pad 行确定性毒化 → 查 pad 路径(SFA zeroing/empty_rows)
        # [已定案] R1: 单请求 bs8=0.76(多请求无关, N 截断正常)
        #         E3: 无 hisparse bs8=0.94(graph 基础设施干净)
        #         R2: replay 后 sync 无效(跨 replay 竞态排除)
        # [精度调试 E3] 无 hisparse 对照: SELECTIVE_LAYER_IDS="" 启动可完全关闭
        # selective hisparse (不传该参数, KV 全 resident HBM), 用于判别
        # graph bs8 精度问题是否 hisparse 特有。判读:
        #   0.94 → hisparse pad 路径实锤; 仍 0.7x → 分支基础图设施问题
        if [[ -n "$SELECTIVE_LAYER_IDS" ]]; then
            HISPARSE_ARGS="--npu-selective-hisparse-layer-ids ${SELECTIVE_LAYER_IDS}"
        else
            HISPARSE_ARGS=""
            echo "[E3] SELECTIVE_LAYER_IDS empty -> selective hisparse DISABLED"
        fi

        # [D1 diff-dump] eager 对照开关: D_EAGER=1 加 --disable-cuda-graph(保留显式
        # --cuda-graph-bs 供 bias 计算)。eager 19层需配 MAX_RUNNING_REQ=24 压缩
        # bcap/tcap 以过内存关。
        if [[ "${D_EAGER-"0"}" == "1" ]]; then
            GRAPH_ARGS="--disable-cuda-graph --cuda-graph-bs ${CUDA_GRAPH_BS}"
        else
            GRAPH_ARGS="--cuda-graph-bs ${CUDA_GRAPH_BS}"
        fi

        sglang serve \
            --model-loader-extra-config '{"enable_multithread_load": true}' \
            --disaggregation-mode decode --disaggregation-transfer-backend ascend \
            --model-path $MODEL_PATH \
            --tokenizer-path $MODEL_PATH \
            --trust-remote-code \
            --attention-backend ascend \
            --device npu \
            --quantization modelslim \
            --dtype bfloat16 \
            --kv-cache-dtype fp8_e4m3 \
            --tp-size 8 \
            --dp 8 \
            --enable-dp-attention \
            --moe-dense-tp-size 1 \
            --mem-fraction-static $D_MEM_FRACTION \
            --chunked-prefill-size 8192 \
            ${GRAPH_ARGS} \
            --speculative-algorithm NEXTN \
            --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
            --max-running-requests ${MAX_RUNNING_REQ-192} \
            --host 0.0.0.0 \
            --port 31000 \
            --moe-a2a-backend deepep \
            --deepep-mode low_latency \
            --watchdog-timeout 9000 \
            --num-reserved-decode-tokens 2048 \
            ${HISPARSE_ARGS} \
            --disaggregation-decode-polling-interval 2 \
            2>&1 | tee "$D_LOG"

        exit 1
    fi
done

# --------------- router + 压测（本机非 P/D 节点时才会走到这里）----------------
# python -m sglang_router.launch_router \
#     --pd-disaggregation --policy cache_aware \
#     --prefill http://141.61.49.198:31000 8998 \
#     --decode http://141.61.49.195:31000 \
#     --host 0.0.0.0 --port 6688

curl --location 'http://141.61.49.198:31000/flush_cache' --header 'Content-Type: application/json'
# python -m sglang.bench_serving \
#     --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
#     --dataset-name random \
#     --backend sglang \
#     --host 141.61.49.195 \
#     --port 6688 \
#     --max-concurrency 1 \
#     --random-input-len 8000 \
#     --random-output-len 1000 \
#     --num-prompts 1 \
#     --disable-ignore-eos \
#     --random-range-ratio 1 \
#     --warmup-request 0

# python3 -m sglang.bench_serving \
#     --dataset-name generated-shared-prefix \
#     --backend sglang --host 141.61.49.195 \
#     --port 6688 \
#     --max-concurrency 1 \
#     --gsp-num-groups 1 \
#     --gsp-prompts-per-group 1 \
#     --gsp-system-prompt-len 127620 \
#     --gsp-question-len 1280 \
#     --gsp-output-len 1000 \
#     --warmup-requests 4

# ============================================================================
#
# ---------- Prefill 节点 (141.61.49.198) ----------
# bash pd_disaggregation_dram_offload.sh
#
# ---------- Decode 节点 (141.61.49.195) ----------
# graph:
# SGLANG_SELECTIVE_DIFF_DUMP=1 SGLANG_SELECTIVE_DUMP_DIR=/root/hisparse_dump/graph5 SGLANG_SELECTIVE_DUMP_MAX_STEPS=20 bash pd_disaggregation_dram_offload.sh
#
# eager:
# SGLANG_SELECTIVE_DIFF_DUMP=1 SGLANG_SELECTIVE_DUMP_DIR=/root/hisparse_dump/eager5 SGLANG_SELECTIVE_DUMP_MAX_STEPS=20 D_EAGER=1 MAX_RUNNING_REQ=24 bash pd_disaggregation_dram_offload.sh
