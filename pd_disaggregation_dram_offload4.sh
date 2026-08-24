#!/bin/bash
# ============================================================================
# PD 分离 + Selective HiSparse 启动脚本（A5 / ASCEND_950，memfabric URMA）
#
# 拓扑: Prefill 2 机 16 卡 + Decode 2 机 16 卡
#   P: 141.61.94.103, 141.61.94.107
#   D: 141.61.94.111, 141.61.94.139
#
# 并行策略 (对照 run_8node_pd.sh):
#   P: TP16, DP16 → attn_tp=1, EP16   (与生产单组完全一致)
#   D: TP16, DP16 → attn_tp=1, EP16   (生产 DP32/EP32, 此处缩减一半)
#
# 用法（无需参数, 按 hostname -I 自动识别本机角色）:
#   P 节点: bash pd_selective_hisparse_4node.sh
#   D 节点: bash pd_selective_hisparse_4node.sh
# ============================================================================

# cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

# ===== Cleanup =====
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING

pkill -9 python 2>/dev/null || true
pkill -9 sglang 2>/dev/null || true

source /usr/local/memfabric_hybrid/set_env.sh

# ===== Environment =====
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export SGLANG_LOG_LEVEL=debug
CODE_PATH=${CODE_PATH:-/home/t00882532/Code/hisparse}
export PYTHONPATH=${CODE_PATH}/python:$PYTHONPATH

export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=68
export HCCL_OP_EXPANSION_MODE=AIV
export ACL_DEVICE_SYNC_TIMEOUT=60

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

# [FIA]
export ASCEND_USE_FIA=1

# [MLAPO]
export SGLANG_NPU_USE_MLAPO=1

# [DEEPEP]
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=72

# [MTP]
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export TRANSFORMERS_VERBOSITY=error

# [多机]
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export GLOO_SOCKET_IFNAME=data0.173

# A5 PD分离
export MF_HYBM_USE_VMM_SEGMENT=1
export ASCEND_MF_TRANSFER_PROTOCOL="device_urma"

unset HCCL_IF_IP 2>/dev/null || true
unset HCCL_SOCKET_FAMILY 2>/dev/null || true
unset RANK_TABLE_FILE 2>/dev/null || true

# ===== Model Config =====
MODEL_PATH=${MODEL_PATH:-/mnt/share/w00936111/GLM-5.2-W8A8C8-mxfp8}
SERVED_MODEL_NAME=glm52

# Selective HiSparse: 选中的卸载层 ID
# 全部 19 层: 5 9 13 17 21 25 29 33 37 41 45 49 53 57 61 65 69 73 77
# 首版测试建议先用少量层, 验证通过后逐步扩展:
#   单层:  SELECTIVE_LAYER_IDS="5"
#   4 层:  SELECTIVE_LAYER_IDS="5 21 49 77"
#   全部:  SELECTIVE_LAYER_IDS="5 9 13 17 21 25 29 33 37 41 45 49 53 57 61 65 69 73 77"
#SELECTIVE_LAYER_IDS=${SELECTIVE_LAYER_IDS:-"5"}
SELECTIVE_LAYER_IDS=${SELECTIVE_LAYER_IDS:-"5 9 13 17 21 25 29 33 37 41 45 49 53 57 61 65 69 73 77"}
# ===== Cluster Config ===========================================
# 4 机: P 2 节点 × 8 卡 + D 2 节点 × 8 卡
#   P: TP16, DP16 → attn_tp = 16/16 = 1, EP16
#   D: TP16, DP16 → attn_tp = 16/16 = 1, EP16
# ================================================================
NUM_NPUS_PER_NODE=8

# Prefill 节点 (2 nodes)
P_IPS=(
  "141.61.94.103"
  "141.61.94.107"
)
P_IFS=(
  "enp35s0f2"
  "enp35s0f2"
)

# Decode 节点 (2 nodes)
D_IPS=(
  "141.61.94.111"
  "141.61.94.139"
)
D_IFS=(
  "enp35s0f2"
  "enp35s0f2"
)

P_NNODES=${#P_IPS[@]}
D_NNODES=${#D_IPS[@]}

P_TP_SIZE=$(( P_NNODES * NUM_NPUS_PER_NODE ))   # 16
D_TP_SIZE=$(( D_NNODES * NUM_NPUS_PER_NODE ))   # 16
P_DP_SIZE=16
D_DP_SIZE=16

P_MASTER="${P_IPS[0]}"
D_MASTER="${D_IPS[0]}"
P_DIST_INIT="${P_MASTER}:5567"
D_DIST_INIT="${D_MASTER}:5568"

# session store, 挂在 Prefill 首节点
export ASCEND_MF_STORE_URL="tcp://${P_MASTER}:31001"

# ===== Auto-detect node by matching local IPs ==================
LOCAL_HOST1=$(hostname -I | awk '{print $1}')
LOCAL_HOST2=$(hostname -I | awk '{print $2}')

# ===== Launch Prefill nodes ====================================
for i in "${!P_IPS[@]}"; do
  if [[ "$LOCAL_HOST1" == "${P_IPS[$i]}" || "$LOCAL_HOST2" == "${P_IPS[$i]}" ]]; then
    export HCCL_SOCKET_IFNAME="${P_IFS[$i]}"

    echo "========================================"
    echo "Launching GLM5.2 Prefill node ${i}"
    echo "node-rank       : ${i}"
    echo "local IPs       : ${LOCAL_HOST1} ${LOCAL_HOST2}"
    echo "dist-init-addr  : ${P_DIST_INIT}"
    echo "nnodes          : ${P_NNODES}"
    echo "tp-size         : ${P_TP_SIZE}"
    echo "dp-size         : ${P_DP_SIZE}"
    echo "HCCL interface  : ${HCCL_SOCKET_IFNAME}"
    echo "GLOO interface  : ${GLOO_SOCKET_IFNAME}"
    echo "========================================"
    export DEEPEP_HCCL_BUFFSIZE=2048

    python3 -m sglang.launch_server \
      --model-path ${MODEL_PATH} \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host 0.0.0.0 \
      --port 30000 \
      --nnodes ${P_NNODES} \
      --node-rank ${i} \
      --dist-init-addr ${P_DIST_INIT} \
      --tp-size ${P_TP_SIZE} \
      --dp ${P_DP_SIZE} \
      --enable-dp-attention \
      --enable-dp-lm-head \
      --load-balance-method round_robin \
      --disaggregation-mode prefill \
      --disaggregation-transfer-backend ascend \
      --disaggregation-bootstrap-port 8998 \
      --trust-remote-code \
      --attention-backend ascend \
      --device npu \
      --watchdog-timeout 9000 \
      --max-running-requests 256 \
      --mem-fraction-static 0.86 \
      --tokenizer-worker-num 12 \
      --enable-dynamic-batch-tokenizer \
      --quantization modelslim \
      --max-prefill-tokens 20480000 \
      --chunked-prefill-size 16384 \
      --kv-cache-dtype "fp8_e4m3" \
      --moe-a2a-backend deepep \
      --deepep-mode auto \
      --disable-cuda-graph \
      --enable-metrics

    exit 1
  fi
done

# ===== Launch Decode nodes (Selective HiSparse) ================
for i in "${!D_IPS[@]}"; do
  if [[ "$LOCAL_HOST1" == "${D_IPS[$i]}" || "$LOCAL_HOST2" == "${D_IPS[$i]}" ]]; then
    export HCCL_SOCKET_IFNAME="${D_IFS[$i]}"

    echo "========================================"
    echo "Launching GLM5.2 Decode node ${i} (Selective HiSparse)"
    echo "node-rank       : ${i}"
    echo "local IPs       : ${LOCAL_HOST1} ${LOCAL_HOST2}"
    echo "dist-init-addr  : ${D_DIST_INIT}"
    echo "nnodes          : ${D_NNODES}"
    echo "tp-size         : ${D_TP_SIZE}"
    echo "dp-size         : ${D_DP_SIZE}"
    echo "selective layers: ${SELECTIVE_LAYER_IDS}"
    echo "HCCL interface  : ${HCCL_SOCKET_IFNAME}"
    echo "GLOO interface  : ${GLOO_SOCKET_IFNAME}"
    echo "========================================"
    export DEEPEP_HCCL_BUFFSIZE=1024
    export HCCL_BUFFSIZE=70
    export SGLANG_LM_HEAD_TP=4
    export SGLANG_ATTN_O_TP_SIZE=4
    export SGLANG_NPU_PROFILING=1

    python3 -m sglang.launch_server \
      --model-path ${MODEL_PATH} \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host 0.0.0.0 \
      --port 30000 \
      --nnodes ${D_NNODES} \
      --node-rank ${i} \
      --dist-init-addr ${D_DIST_INIT} \
      --tp-size ${D_TP_SIZE} \
      --dp ${D_DP_SIZE} \
      --enable-dp-attention \
      --moe-dense-tp-size 1 \
      --load-balance-method round_robin \
      --disaggregation-mode decode \
      --disaggregation-transfer-backend ascend \
      --trust-remote-code \
      --attention-backend ascend \
      --device npu \
      --disable-radix-cache \
      --disable-chunked-prefix-cache \
      --watchdog-timeout 9000 \
      --max-running-requests 64 \
      --mem-fraction-static 0.92 \
      --quantization modelslim \
      --kv-cache-dtype "fp8_e4m3" \
      --moe-a2a-backend deepep \
      --deepep-mode auto \
      --context-len 133120 \
      --tokenizer-worker-num 8 \
      --speculative-algorithm NEXTN \
      --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
      --speculative-draft-kv-cache-dtype bf16 \
      --enable-metrics \
      --npu-selective-hisparse-layer-ids ${SELECTIVE_LAYER_IDS} \
      --cuda-graph-bs 1 4 6

    exit 1
  fi
done

echo "ERROR: local IPs [${LOCAL_HOST1} ${LOCAL_HOST2}] not found in P_IPS=[${P_IPS[*]}] or D_IPS=[${D_IPS[*]}]"
exit 1

#       --disable-cuda-graph \

# ===== Router (在独立节点或任一 P/D 节点上手动执行) ============
# python -m sglang_router.launch_router \
#     --pd-disaggregation --policy cache_aware \
#     --prefill http://141.61.94.103:30000 \
#     --decode http://141.61.94.111:30000 \
#     --host 0.0.0.0 --port 6677
