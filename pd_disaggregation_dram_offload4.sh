#!/bin/bash
# cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000


# ===== Cleanup =====
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING

pkill -9 python  2>/dev/null || true
pkill -9 sglang 2>/dev/null || true
pkill -9 VLLM   2>/dev/null || true

# ===== Environment =====
export PYTHONPATH=/mnt/share/chenxu/codes/sglang/python:$PYTHONPATH


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
# export SGLANG_NPU_GLM_NEXTN_BF16_KV_CACHE=1

# [DEEPEP]
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=36

# [Prefill Delay]
#export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
#export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200

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

export ASCEND_MF_STORE_URL="tcp://141.61.94.143:40187"
#export ASCEND_MF_LOG_LEVEL=1

unset HCCL_IF_IP 2>/dev/null || true
unset HCCL_SOCKET_FAMILY 2>/dev/null || true
unset RANK_TABLE_FILE 2>/dev/null || true

# ===== Model Config =====
#MODEL_PATH_P=/home/cx/glm5_2_weight
MODEL_PATH=/mnt/share/w00936111/GLM-5.2-W8A8C8-mxfp8
SERVED_MODEL_NAME=glm52

# ===== Cluster Config ===========================================
# 每组 2 节点, 每节点 8 NPU => 16 NPU / 组
# 并行策略: DP4 x TP4 = 16 (attention/dense), EP16 (MoE all-to-all)
# ================================================================

# Prefill 节点 (2 nodes)
P_IPS=(
  "141.61.94.143"
  "141.61.94.147"
)
P_IFS=(
  "enp35s0f2"
  "enp35s0f2"
)

P_IPS2=(
  "141.61.94.99"
  "141.61.94.151"
)
P_IFS2=(
  "enp35s0f2"
  "enp35s0f2"
)

# Decode 节点 (2 nodes)
D_IPS=(

  "141.61.94.103"
  "141.61.94.107"
  "141.61.94.111"
  "141.61.94.139"
)
D_IFS=(
  "enp35s0f2"
  "enp35s0f2"
  "enp35s0f2"
  "enp35s0f2"
)

NUM_NPUS_PER_NODE=8

P_NNODES=${#P_IPS[@]}
P_NNODES2=${#P_IPS2[@]}
D_NNODES=${#D_IPS[@]}

# 并行度 (enable-dp-attention 模式下总NPU = TP_SIZE):
#   P: TP = P_NNODES * 8 = 16, DP4  => DP4TP16, EP16
#   D: TP = D_NNODES * 8 = 16, DP4  => DP4TP16, EP16
P_TP_SIZE=$(( P_NNODES * NUM_NPUS_PER_NODE ))
P_TP_SIZE2=$(( P_NNODES2 * NUM_NPUS_PER_NODE ))
D_TP_SIZE=$(( D_NNODES * NUM_NPUS_PER_NODE ))
P_DP_SIZE=16
P_DP_SIZE2=16
D_DP_SIZE=32
# ================================================================

P_MASTER="${P_IPS[0]}"
P_MASTER2="${P_IPS2[0]}"
D_MASTER="${D_IPS[0]}"
P_DIST_INIT="${P_MASTER}:5567"
P_DIST_INIT2="${P_MASTER2}:5569"
D_DIST_INIT="${D_MASTER}:5568"

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

for i in "${!P_IPS2[@]}"; do
  if [[ "$LOCAL_HOST1" == "${P_IPS2[$i]}" || "$LOCAL_HOST2" == "${P_IPS2[$i]}" ]]; then
    export HCCL_SOCKET_IFNAME="${P_IFS2[$i]}"

    echo "========================================"
    echo "Launching GLM5.2 Prefill node ${i}"
    echo "node-rank       : ${i}"
    echo "local IPs       : ${LOCAL_HOST1} ${LOCAL_HOST2}"
    echo "dist-init-addr  : ${P_DIST_INIT2}"
    echo "nnodes          : ${P_NNODES2}"
    echo "tp-size         : ${P_TP_SIZE2}"
    echo "dp-size         : ${P_DP_SIZE2}"
    echo "HCCL interface  : ${HCCL_SOCKET_IFNAME}"
    echo "GLOO interface  : ${GLOO_SOCKET_IFNAME}"
    echo "========================================"
    export DEEPEP_HCCL_BUFFSIZE=2048
    python3 -m sglang.launch_server \
      --model-path ${MODEL_PATH} \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host 0.0.0.0 \
      --port 30000 \
      --nnodes ${P_NNODES2} \
      --node-rank ${i} \
      --dist-init-addr ${P_DIST_INIT2} \
      --tp-size ${P_TP_SIZE2} \
      --dp ${P_DP_SIZE2} \
      --enable-dp-attention \
      --enable-dp-lm-head \
      --load-balance-method round_robin \
      --disaggregation-mode prefill \
      --disaggregation-transfer-backend ascend \
      --disaggregation-bootstrap-port 8999 \
      --trust-remote-code \
      --attention-backend ascend \
      --device npu \
      --watchdog-timeout 9000 \
      --max-running-requests 256 \
      --mem-fraction-static 0.86 \
      --tokenizer-worker-num 12 \
      --enable-dynamic-batch-tokenizer \
      --quantization modelslim \
      --enable-dynamic-batch-tokenizer \
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


# ===== Launch Decode nodes =====================================
for i in "${!D_IPS[@]}"; do
  if [[ "$LOCAL_HOST1" == "${D_IPS[$i]}" || "$LOCAL_HOST2" == "${D_IPS[$i]}" ]]; then
    export HCCL_SOCKET_IFNAME="${D_IFS[$i]}"

    echo "========================================"
    echo "Launching GLM5.2 Decode node ${i}"
    echo "node-rank       : ${i}"
    echo "local IPs       : ${LOCAL_HOST1} ${LOCAL_HOST2}"
    echo "dist-init-addr  : ${D_DIST_INIT}"
    echo "nnodes          : ${D_NNODES}"
    echo "tp-size         : ${D_TP_SIZE}"
    echo "dp-size         : ${D_DP_SIZE}"
    echo "HCCL interface  : ${HCCL_SOCKET_IFNAME}"
    echo "GLOO interface  : ${GLOO_SOCKET_IFNAME}"
    echo "========================================"
    export DEEPEP_HCCL_BUFFSIZE=250
    export HCCL_BUFFSIZE=70
    # LM-head TP
    export SGLANG_LM_HEAD_TP=4
#    export SGLANG_ATTN_O_TP_SIZE=4
#    export SGLANG_NPU_USE_MULTI_STREAM=1
#    export SGLANG_NPU_PROFILING=1
     export SGLANG_NPU_FINE_GRAINED_MOE_DUAL_STREAM=1
     export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/home/fuyong/hot_map

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
      --watchdog-timeout 9000 \
      --max-running-requests 192 \
      --mem-fraction-static 0.945 \
      --quantization modelslim \
      --kv-cache-dtype "fp8_e4m3" \
      --moe-a2a-backend deepep \
      --deepep-mode auto \
      --cuda-graph-bs 1 4 6 \
      --context-len 133120 \
      --tokenizer-worker-num 8 \
      --speculative-algorithm NEXTN \
      --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
      --speculative-draft-kv-cache-dtype bf16  --expert-distribution-recorder-buffer-size -1  --expert-distribution-recorder-mode stat --ep-dispatch-algorithm static \
      --enable-expert-distribution-metrics \
      --enable-metrics

    exit 1
  fi
done

echo "ERROR: local IPs [${LOCAL_HOST1} ${LOCAL_HOST2}] not found in P_IPS=[${P_IPS[*]}] or D_IPS=[${D_IPS[*]}]"
exit 1

# ===== Router (在独立节点或任一 P/D 节点上手动执行) ============
# python -m sglang_router.launch_router \
#     --pd-disaggregation --policy cache_aware \
#     --prefill http://141.61.33.11:30000 \
#     --decode http://141.61.33.13:30000 \
#     --host 0.0.0.0 --port 6677