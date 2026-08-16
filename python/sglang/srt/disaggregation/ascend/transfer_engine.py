import logging
import os
from typing import List

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.utils.network import NetworkAddress

try:
    from memfabric_hybrid import TransferEngine

    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)


class AscendTransferEngine(MooncakeTransferEngine):

    def __init__(
        self,
        hostname: str,
        npu_id: int,
        disaggregation_mode: DisaggregationMode,
    ):
        if import_error is not None:
            logger.warning(
                "Please install memfabric_hybrid, for details, see docs/docs/advanced_features/pd_disaggregation.mdx"
            )
            raise import_error

        self.engine = TransferEngine()
        self.hostname = hostname
        self.npu_id = npu_id

        # Centralized storage address of the AscendTransferEngine
        self.store_url = os.getenv("ASCEND_MF_STORE_URL")
        if disaggregation_mode == DisaggregationMode.PREFILL:
            self.role = "Prefill"
        elif disaggregation_mode == DisaggregationMode.DECODE:
            self.role = "Decode"
        else:
            logger.error(f"Unsupported DisaggregationMode: {disaggregation_mode}")
            raise ValueError(f"Unsupported DisaggregationMode: {disaggregation_mode}")
        rpc_port = self.engine.get_rpc_port()
        self.session_id = NetworkAddress(self.hostname, rpc_port).to_host_port_str()
        self.initialize()
        if rpc_port == 0:
            rpc_port = self.engine.get_rpc_port()
            self.session_id = NetworkAddress(self.hostname, rpc_port).to_host_port_str()

    def initialize(self) -> None:
        from sglang.srt.distributed.parallel_state import (
            get_world_group,
            get_world_size,
        )

        transfer_protocol = self._get_transfer_protocol()
        if transfer_protocol is None or transfer_protocol == "sdma":
            trans_op_type = TransferEngine.TransDataOpType.SDMA
        elif transfer_protocol == "device_urma":
            # Cross-node PD transfer: DEVICE_URMA supports both HBM and DRAM
            # (registered host memory) remote writes; RDMA cannot target host
            # memory registered by the offload URMA_POOL.
            from sglang.srt.utils.common import is_npu_atlas_a5

            # DEVICE_URMA is only supported on A5 (ASCEND_950); earlier SoCs
            # (e.g. A3/910C) fail inside OpenDevice with a cryptic error, so
            # fail fast with an explicit message instead.
            if not is_npu_atlas_a5(self.npu_id):
                logger.error(
                    "ASCEND_MF_TRANSFER_PROTOCOL=device_urma is only supported "
                    "on Atlas A5 (ASCEND_950) hardware, npu_id=%d",
                    self.npu_id,
                )
                raise RuntimeError(
                    "device_urma transfer requires Atlas A5 (ASCEND_950); "
                    "this device is not A5. Use sdma or device_rdma instead."
                )
            logger.info(
                f"[DRAM] protocol=device_urma selected (A5 verified, npu_id={self.npu_id})"
            )
            trans_op_type = TransferEngine.TransDataOpType.DEVICE_URMA
        else:
            trans_op_type = TransferEngine.TransDataOpType.DEVICE_RDMA
            """with device RDMA for PD transfer"""
            tmp_tensor = torch.zeros(1, device="npu")
            output_tensor_list = [
                torch.empty_like(tmp_tensor) for _ in range(get_world_size())
            ]
            # Initialize hccl in advance through all_gather to avoid conflicts with rdma initialization.
            torch.distributed.all_gather(
                output_tensor_list, tmp_tensor, group=get_world_group().device_group
            )
        """Initialize the ascend transfer instance."""
        if transfer_protocol is None:
            from sglang.srt.utils.common import is_npu_atlas_a5

            if is_npu_atlas_a5(self.npu_id):
                logger.info(
                    "No ASCEND_MF_TRANSFER_PROTOCOL set; tip: for cross-node "
                    "PD transfer on A5 use ASCEND_MF_TRANSFER_PROTOCOL=device_urma"
                )
        ret_value = self.engine.initialize(
            self.store_url, self.session_id, self.role, self.npu_id, trans_op_type
        )
        if ret_value != 0:
            logger.error("Ascend Transfer Engine initialization failed.")
            raise RuntimeError("Ascend Transfer Engine initialization failed.")

    def batch_register(self, ptrs: List[int], lengths: List[int]):
        try:
            ret_value = self.engine.batch_register_memory(ptrs, lengths)
        except Exception:
            # Mark register as failed
            ret_value = -1
        if ret_value != 0:
            logger.debug(f"Ascend memory registration for ptr {ptrs} failed.")

    @staticmethod
    def _get_transfer_protocol():
        protocol = os.getenv("ASCEND_MF_TRANSFER_PROTOCOL")
        # device_urma: cross-node transfer over UBC (EID addressing), required
        # for the decode DRAM offload path.
        allowed_protocols = {"device_rdma", "sdma", "device_urma"}
        if protocol and protocol.lower() in allowed_protocols:
            return protocol.lower()
        else:
            logger.warning(
                "Invalid or no transfer protocol specified, using default protocol."
            )
            return None
