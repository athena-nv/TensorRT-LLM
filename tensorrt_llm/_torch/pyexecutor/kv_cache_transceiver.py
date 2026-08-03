from abc import ABC, abstractmethod
from os import getenv
from typing import Any, Dict, List, Optional

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm.bindings import WorldConfig
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

from .llm_request import LlmRequest
from .mamba_cache_manager import (BaseMambaCacheManager,
                                  CppMambaHybridCacheManager,
                                  MambaHybridCacheManager,
                                  MixedMambaHybridCacheManager)
from .resource_manager import KVCacheManager

CacheTransceiverCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransceiver
AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
CacheTransBufferManagerCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransBufferManager
BackendTypeCpp = tensorrt_llm.bindings.executor.CacheTransceiverBackendType


def resolve_cache_transceiver_config(
        cache_transceiver_config: Optional[CacheTransceiverConfig]) -> None:
    """Resolve defaults and validate runtime-independent configuration."""
    if cache_transceiver_config is None or cache_transceiver_config.backend is None:
        return

    if cache_transceiver_config.backend == "DEFAULT":
        # NIXL is the default backend. Legacy environment variables override it
        # in priority order.
        cache_transceiver_config.backend = "NIXL"
        backend_env_vars = [
            ("TRTLLM_USE_NIXL_KVCACHE", "NIXL"),
            ("TRTLLM_USE_UCX_KVCACHE", "UCX"),
            ("TRTLLM_USE_MOONCAKE_KVCACHE", "MOONCAKE"),
            ("TRTLLM_USE_MPI_KVCACHE", "MPI"),
        ]
        for env_var, backend in backend_env_vars:
            if getenv(env_var) == "1":
                logger.warning(
                    f"{env_var}=1 is set, but it's recommended to set cache_transceiver_config.backend in yaml config"
                )
                cache_transceiver_config.backend = backend
                break

    runtime = cache_transceiver_config.transceiver_runtime
    enable_pipelined_transfer = cache_transceiver_config.enable_pipelined_transfer
    if runtime is None and enable_pipelined_transfer:
        if cache_transceiver_config.backend != "NIXL":
            raise ValueError(
                f"enable_pipelined_transfer is set but backend "
                f"'{cache_transceiver_config.backend}' requires the C++ "
                f"transceiver, which does not support pipelined transfer. Use NIXL backend to "
                f"enable pipelined transfer.")
        logger.warning(
            "enable_pipelined_transfer is set; auto-selecting the Python "
            "transceiver instead of the C++ transceiver to enable "
            "pipelined KV cache transfer. "
            "Set transceiver_runtime='CPP' to disable this auto-selection.")
        cache_transceiver_config.transceiver_runtime = "PYTHON"
    elif runtime == "CPP" and enable_pipelined_transfer:
        raise ValueError(
            "enable_pipelined_transfer is set but transceiver_runtime='CPP' "
            "explicitly disables Python auto-selection. Use transceiver_runtime='PYTHON' to enable pipelined transfer."
        )

    if (cache_transceiver_config.transceiver_runtime == "PYTHON"
            and cache_transceiver_config.backend != "NIXL"):
        raise ValueError(
            f"Python transceiver currently only supports NIXL backend, "
            f"got {cache_transceiver_config.backend}. "
            f"Please use transceiver_runtime='CPP' for MPI, UCX, or MOONCAKE backends."
        )
    if (enable_pipelined_transfer
            and cache_transceiver_config.kv_cache_bounce_size_mb > 0):
        raise ValueError(
            "kv_cache_bounce_size_mb must be 0 when enable_pipelined_transfer is set."
        )


def mapping_to_world_config(mapping: Mapping) -> WorldConfig:

    return WorldConfig(tensor_parallelism=mapping.tp_size,
                       pipeline_parallelism=mapping.pp_size,
                       context_parallelism=mapping.cp_size,
                       rank=mapping.rank,
                       gpus_per_node=mapping.gpus_per_node,
                       device_ids=None,
                       enable_attention_dp=mapping.enable_attention_dp)


def create_kv_cache_transceiver(
        mapping: Mapping,
        dist: Distributed,
        kv_cache_manager: KVCacheManager,
        attention_type: AttentionTypeCpp,
        cache_transceiver_config: CacheTransceiverConfig,
        mamba_cache_manager: Optional[BaseMambaCacheManager] = None,
        enable_chunked_prefill: bool = False):
    resolve_cache_transceiver_config(cache_transceiver_config)
    if cache_transceiver_config is None or cache_transceiver_config.backend is None:
        logger.info("cache_transceiver is disabled")
        return None

    if cache_transceiver_config.backend == "MPI":
        logger.warning(
            "MPI CacheTransceiver is deprecated, UCX or NIXL is recommended")
    elif cache_transceiver_config.backend == "UCX":
        logger.info(
            "Using UCX kv-cache transceiver. If your devices are not in the same domain, please consider setting "
            "UCX_CUDA_IPC_ENABLE_MNNVL=n, UCX_RNDV_SCHEME=put_zcopy and/or unset UCX_NET_DEVICES upon server "
            "hangs or lower-than-expected performance.")

    if (cache_transceiver_config.enable_pipelined_transfer
            and not enable_chunked_prefill):
        raise ValueError(
            "enable_chunked_prefill is required when enable_pipelined_transfer is set."
        )
    is_kv_cache_sender = getenv("TRTLLM_DISAGG_ROLE") != "generation"
    if (cache_transceiver_config.enable_pipelined_transfer
            and is_kv_cache_sender and mapping.pp_size != 1):
        raise ValueError(
            "pipeline_parallel_size=1 is required when enable_pipelined_transfer is set."
        )

    # Select transceiver implementation based on transceiver_runtime
    # transceiver_runtime == None or "CPP" -> use C++ transceiver (default)
    # transceiver_runtime == "PYTHON" -> use Python transceiver
    if cache_transceiver_config.transceiver_runtime == "PYTHON":
        if (isinstance(kv_cache_manager, MambaHybridCacheManager)
                and not isinstance(kv_cache_manager,
                                   MixedMambaHybridCacheManager)):
            raise ValueError(
                "Python transceiver requires MixedMambaHybridCacheManager "
                f"for hybrid models, got {type(kv_cache_manager).__name__}.")
        from tensorrt_llm._torch.disaggregation.transceiver import \
            KvCacheTransceiverV2
        logger.info("Using KvCacheTransceiverV2")
        # MixedMambaHybridCacheManager contains both the KV and Mamba pools.
        return KvCacheTransceiverV2(mapping, dist, kv_cache_manager,
                                    cache_transceiver_config)

    # Default: use C++ transceiver (transceiver_runtime is None or "CPP")
    return BindKvCacheTransceiver(mapping, dist, kv_cache_manager,
                                  attention_type, cache_transceiver_config,
                                  mamba_cache_manager)


class KvCacheTransceiver(ABC):

    @property
    def pipeline_transfer_enabled(self) -> bool:
        """Whether pipelined prefill-transfer is enabled."""
        return False

    def has_inflight_transfer(self, req: LlmRequest) -> bool:
        """Whether this transceiver still owns transfer resources for req.

        Independent of ``LlmRequestState``: with pipelined transfer a chunk can
        be in flight while the request is still in its context-compute phase.
        True means the request's KV pages may be read by the fabric and must
        not be released.
        """
        return False

    def has_any_inflight_transfer(self) -> bool:
        """Whether any request has transfer resources in flight."""
        return False

    @abstractmethod
    def respond_and_send_async(self, req: LlmRequest):
        raise NotImplementedError

    @abstractmethod
    def request_and_receive_sync(self, req: LlmRequest):
        raise NotImplementedError

    @abstractmethod
    def request_and_receive_async(self, req: LlmRequest):
        raise NotImplementedError

    @abstractmethod
    def check_context_transfer_status(self, at_least_request_num: int):
        raise NotImplementedError

    @abstractmethod
    def check_gen_transfer_status(self, at_least_request_num: int):
        raise NotImplementedError

    @abstractmethod
    def check_gen_transfer_complete(self):
        raise NotImplementedError

    @abstractmethod
    def cancel_request(self, req: LlmRequest):
        raise NotImplementedError

    @abstractmethod
    def prepare_context_requests(self, requests: List[LlmRequest]):
        """
        Prepare the context request for the cache transceiver in generation-first mode.
        This method should set the context request state to DISAGG_CONTEXT_WAIT_SCHEDULER
        so that it won't be scheduled if the responding generation kvcache request is not
        yet received otherwise set it to CONTEXT_INIT.
        """
        ...

    @abstractmethod
    def get_disaggregated_params(self) -> Dict[str, Any]:
        """
        Return a dictionary form of DisaggregatedParams to be set in the generation request.
        The generation server will use it to get kvcache in generation-first mode.
        """
        ...

    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        """Commit received KV blocks to the radix tree for prefix reuse. No-op by default."""

    def shutdown(self):
        """Shut down the transceiver and release registered resources."""


class BindKvCacheTransceiver(KvCacheTransceiver):

    def __init__(self,
                 mapping: Mapping,
                 dist: Distributed,
                 kv_cache_manager: KVCacheManager,
                 attention_type: AttentionTypeCpp,
                 cache_transceiver_config: CacheTransceiverConfig,
                 mamba_cache_manager: Optional[BaseMambaCacheManager] = None):
        world_config = mapping_to_world_config(mapping)
        # Filter out mamba/recurrent state layers (kv_heads == 0) so that
        # CacheState::ModelConfig::mNbKvHeadsPerLayer only contains attention
        # layers — matching the factory path (modelConfig.getNumKvHeadsPerLayer()).
        # This is critical: splitKVCacheDispatch uses mNbKvHeadsPerLayer.size()
        # as the layer count for the CUDA kernel grid dimension.
        total_num_kv_heads_per_layer = [
            h for h in kv_cache_manager.total_num_kv_heads_per_layer if h > 0
        ]
        head_dim = kv_cache_manager.head_dim
        tokens_per_block = kv_cache_manager.tokens_per_block
        dtype = kv_cache_manager.dtype
        # Get the *attention* layer count per PP rank (C++ uses this as
        # mAttentionLayerNumPerPP).  For CppMambaHybridCacheManager the local
        # pp_layers list includes mamba layers (kv_heads == 0); those must be
        # excluded so the C++ buffer-size calculations stay correct.
        pp_layer_num = sum(1 for h in kv_cache_manager.num_kv_heads_per_layer
                           if h > 0)
        pp_layer_num_per_pp_rank = dist.pp_allgather(pp_layer_num)

        self.kv_transfer_timeout_ms = cache_transceiver_config.kv_transfer_timeout_ms
        self.kv_transfer_sender_future_timeout_ms = cache_transceiver_config.kv_transfer_sender_future_timeout_ms
        self.kv_transfer_poll_interval_ms = cache_transceiver_config.kv_transfer_poll_interval_ms

        # Get RNN layer distribution if mamba_cache_manager is provided.
        rnn_layer_num_per_pp_rank = []
        if mamba_cache_manager is not None:
            if isinstance(mamba_cache_manager, CppMambaHybridCacheManager):
                # Unified pool path: RNN model config is in LinearAttentionMetadata,
                # C++ reads it from BlockManager during CacheTransceiver construction.
                rnn_layer_num_per_pp_rank = dist.pp_allgather(
                    mamba_cache_manager.local_num_mamba_layers)
            else:
                # MixedMambaHybridCacheManager with PythonMambaCacheManager.
                rnn_layer_num_per_pp_rank = dist.pp_allgather(
                    len(mamba_cache_manager._impl.mamba_layer_offsets))
                logger.info(
                    f"RNN state transfer enabled: rnn_layer_num_per_pp={rnn_layer_num_per_pp_rank}"
                )

        self.impl = CacheTransceiverCpp(kv_cache_manager.impl,
                                        total_num_kv_heads_per_layer, head_dim,
                                        tokens_per_block, world_config,
                                        pp_layer_num_per_pp_rank, dtype,
                                        attention_type,
                                        cache_transceiver_config._to_pybind(),
                                        rnn_layer_num_per_pp_rank)

    def respond_and_send_async(self, req: LlmRequest):
        return self.impl.respond_and_send_async(req)

    def request_and_receive_sync(self, req: LlmRequest):
        return self.impl.request_and_receive_sync(req)

    def request_and_receive_async(self, req: LlmRequest):
        return self.impl.request_and_receive_async(req)

    def check_context_transfer_status(self, at_least_request_num: int):
        return self.impl.check_context_transfer_status(at_least_request_num)

    def check_gen_transfer_status(self, at_least_request_num: int):
        return self.impl.check_gen_transfer_status(at_least_request_num)

    def check_gen_transfer_complete(self):
        return self.impl.check_gen_transfer_complete()

    def cancel_request(self, req: LlmRequest):
        return self.impl.cancel_request(req)

    def prepare_context_requests(self, requests: List[LlmRequest]):
        # not implemented, an empty placeholder to allow being invoked unconditionally
        ...

    def get_disaggregated_params(self):
        # Cpp kv cache transceiver will set the disaggregated params to context response
        # Only new py cache transceiver will support gen-first disagg
        return {}


class CacheTransBufferManager:

    def __init__(self, kv_cache_manager: KVCacheManager, max_num_tokens: int):
        self.impl = CacheTransBufferManagerCpp(kv_cache_manager.impl,
                                               max_num_tokens)

    @staticmethod
    def pre_alloc_buffer_size(
            kv_cache_size_bytes_per_token_per_window: dict[int, int],
            tokens_per_block: int,
            cache_transceiver_config: CacheTransceiverConfig):
        return CacheTransBufferManagerCpp.pre_alloc_buffer_size(
            kv_cache_size_bytes_per_token_per_window, tokens_per_block,
            cache_transceiver_config)
