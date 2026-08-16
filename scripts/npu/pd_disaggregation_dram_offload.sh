#!/bin/bash
# ============================================================================
# PD 分离 + Decode KV cache DRAM offload 启动脚本（A5 / ASCEND_950，memfabric URMA）
#
# 拓扑: Prefill 单机 8 卡 (141.61.50.31) + Decode 单机 8 卡 (141.61.49.195)
#
# 用法（无需参数, 按 hostname -I 自动识别本机角色）:
#   Prefill 节点 (141.61.50.31): bash pd_disaggregation_dram_offload.sh
#   Decode  节点 (141.61.49.195): bash pd_disaggregation_dram_offload.sh
#
# 新特性 (commit d83bba4): --disaggregation-decode-dram-pool-size <GB>
#   HBM 不足时 Prefill 直接远写 Decode DRAM 池, commit 时 AIV sparse_copy 提升回 HBM
#   要求 ASCEND_MF_TRANSFER_PROTOCOL=device_urma (跨机 URMA, 仅 A5)
#   互斥: --disaggregation-decode-enable-offload-kvcache /
#          --disaggregation-decode-enable-radix-cache / --enable-hisparse
# ============================================================================

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1

MODEL_PATH=${MODEL_PATH:-/home/weights/GLM-5.2-W8A8C8-mxfp8}
# dummy 模式: 目录已含 config.json/tokenizer/量化配置(quantization_config)等除权重外的
# 全部文件, 仅缺 .safetensors; 随机权重启动, 量化方式由 config.json 自动识别
# 权重就绪后 LOAD_FORMAT=auto 切回真实加载
LOAD_FORMAT=${LOAD_FORMAT:-dummy}
DRAM_POOL_GB=${DRAM_POOL_GB:-64}          # Decode DRAM 接收池大小 (GB)
MEM_FRACTION=${MEM_FRACTION:-0.78}       # 压小可提前触发 KV 落 DRAM

if [[ "$LOAD_FORMAT" == "dummy" ]]; then
    # dummy 不支持 --model-loader-extra-config（loader.py 直接 raise）; 量化方式由 config.json 自动识别
    LOADER_ARGS=(--load-format dummy)
    QUANT_ARGS=()
else
    LOADER_ARGS=(--model-loader-extra-config '{"enable_multithread_load": true}')
    QUANT_ARGS=(--quantization modelslim)
fi

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_SOCKET_IFNAME=enp196s0f0
export GLOO_SOCKET_IFNAME=enp196s0f0
export STREAMS_PER_DEVICE=32

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export HCCL_BUFFSIZE=2000
export DEEPEP_NORMAL_LONG_SEQ_ROUND=64
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512

export HCCL_OP_EXPANSION_MODE=AIV

export PYTHONPATH=`pwd`/python:$PYTHONPATH

# --------------------- PD 拓扑: P 单机 8 卡 + D 单机 8 卡 ---------------------
P_IP=('141.61.50.31')
D_IP=('141.61.49.195')

# 跨机 URMA 传输 + DRAM 池远端直写（P/D 两侧均需, 仅 A5 支持）
export ASCEND_MF_TRANSFER_PROTOCOL=device_urma
# session store, P/D 两侧均可达（挂在 Prefill 节点）
export ASCEND_MF_STORE_URL="tcp://141.61.50.31:24669"
# offload 组件依赖库（如已装到默认路径可不设）
# export MEMFABRIC_HYBRID_EXTEND_LIB_PATH=/path/to/libmf_hybm_accoffload.so

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "Prefill -> ${P_IP[$i]}"

        sglang serve \
            ${LOADER_ARGS[@]} \
            --disaggregation-mode prefill --disaggregation-transfer-backend ascend \
            --disaggregation-bootstrap-port $((8998+$i)) \
            --model-path $MODEL_PATH \
            --tokenizer-path $MODEL_PATH \
            --trust-remote-code \
            --attention-backend ascend \
            --device npu \
            ${QUANT_ARGS[@]} \
            --dtype bfloat16 \
            --tp-size 8 \
            --mem-fraction-static $MEM_FRACTION \
            --chunked-prefill-size 8192 \
            --max-running-requests 64 \
            --host 0.0.0.0 \
            --port 30000 \
            --moe-a2a-backend deepep \
            --deepep-mode normal \
            --disable-cuda-graph

        exit 1
    fi
done

for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "Decode -> ${D_IP[$i]}"

        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
        export HCCL_BUFFSIZE=2000

        sglang serve \
            ${LOADER_ARGS[@]} \
            --disaggregation-mode decode --disaggregation-transfer-backend ascend \
            --model-path $MODEL_PATH \
            --tokenizer-path $MODEL_PATH \
            --trust-remote-code \
            --attention-backend ascend \
            --device npu \
            ${QUANT_ARGS[@]} \
            --dtype bfloat16 \
            --tp-size 8 \
            --mem-fraction-static $MEM_FRACTION \
            --chunked-prefill-size 8192 \
            --cuda-graph-bs 16 \
            --max-running-requests 64 \
            --host 0.0.0.0 \
            --port 30000 \
            --moe-a2a-backend deepep \
            --deepep-mode low_latency \
            --disaggregation-decode-dram-pool-size $DRAM_POOL_GB \
            --num-reserved-decode-tokens 2048 \
            --disaggregation-decode-polling-interval 2

        exit 1
    fi
done

# --------------- router + 压测（本机非 P/D 节点时才会走到这里）----------------
python -m sglang_router.launch_router \
    --pd-disaggregation --policy cache_aware \
    --prefill http://141.61.50.31:30000 8998 \
    --decode http://141.61.49.195:30000 \
    --host 0.0.0.0 --port 6688

curl --location 'http://141.61.50.31:30000/flush_cache' --header 'Content-Type: application/json'
python -m sglang.bench_serving \
    --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --dataset-name random \
    --backend sglang \
    --host 141.61.49.195 \
    --port 6688 \
    --max-concurrency 1 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --num-prompts 1 \
    --disable-ignore-eos \
    --random-range-ratio 1 \
    --warmup-request 0

python3 -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --backend sglang --host 141.61.49.195 \
    --port 6688 \
    --max-concurrency 1 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 127620 \
    --gsp-question-len 1280 \
    --gsp-output-len 1000 \
    --warmup-requests 4
