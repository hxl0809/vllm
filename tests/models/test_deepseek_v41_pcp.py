# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PR #59857 regression coverage adapted to the v0.31.0 model interface."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.kernel import MEGA_MOE_BACKENDS
from vllm.models.deepseek_v41.nvidia import model as model_module
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@pytest.mark.parametrize(
    "pcp_size,use_ep", [(1, False), (2, True), (2, False), (4, False)]
)
@pytest.mark.parametrize("use_sp", [False, True])
@pytest.mark.parametrize("use_engram", [False, True])
def test_local_embeddings_and_engram_precede_gathered_routing_ids(
    pcp_size, use_ep, use_sp, use_engram, monkeypatch
):
    """Keep local token history while aligning MoE IDs with rank-major rows."""
    gather = Mock(
        side_effect=lambda ids, dim: torch.cat(
            [ids + rank * 100 for rank in range(pcp_size)], dim=dim
        )
    )
    monkeypatch.setattr(
        model_module, "get_pcp_group", lambda: SimpleNamespace(all_gather=gather)
    )
    monkeypatch.setattr(
        model_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        model_module, "is_forward_context_available", lambda: use_engram
    )
    monkeypatch.setattr(model_module.envs, "VLLM_MOE_SKIP_PADDING", False)
    monkeypatch.setattr(model_module, "sp_shard", lambda tensor: tensor[::2])
    monkeypatch.setattr(
        model_module, "sp_all_gather", lambda tensor: tensor.repeat_interleave(2, dim=0)
    )
    monkeypatch.setattr(model_module, "mhc_post_tilelang", lambda hidden, *args: hidden)
    monkeypatch.setattr(model_module, "hc_collapse_triton", lambda hidden, mix: hidden)
    monkeypatch.setattr(
        model_module, "image_sentinel_mask", lambda ids: torch.zeros_like(ids).bool()
    )
    monkeypatch.setattr(
        model_module, "gather_engram_hashes", lambda hashes, **kwargs: hashes
    )
    swa_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        slot_mapping=torch.arange(4),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
    )
    monkeypatch.setattr(
        model_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"swa": swa_metadata}),
    )
    embed = Mock(side_effect=lambda ids: ids[:, None, None].float())
    engram_hash = Mock(side_effect=lambda ids, *args: ids[:, None, None])
    engram_hash.ensure_cache.return_value = True
    layers = []
    for _ in range(2):
        layer = Mock(
            side_effect=lambda hidden, *args, **kwargs: (
                hidden,
                None,
                None,
                None,
                hidden.new_ones((hidden.shape[0], 1)),
                None,
            )
        )
        layer.engram = (
            SimpleNamespace(layer_hash_index=0, prepare_embeddings=Mock())
            if use_engram
            else None
        )
        layers.append(layer)
    model = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size, enable_expert_parallel=use_ep
        ),
        use_mega_moe=False,
        use_sequence_parallel=use_sp,
        engram_hash=engram_hash if use_engram else None,
        engram_swa_prefix="swa",
        engram_dp_shared_memory=False,
        embed_input_ids=embed,
        start_layer=0,
        end_layer=len(layers),
        layers=layers,
        aux_hidden_state_layers=(),
        fuse_mhc_all_reduce=False,
        _mtp_hidden_buffer=None,
        norm=lambda hidden: hidden,
    )
    for offset in (0, 10):
        local_ids = torch.tensor([3, 7, 11, 15]) + offset
        positions = torch.arange(4)
        lookback = torch.tensor([[1, 2]])
        model_module.DeepseekV4Model.forward(
            model, local_ids, positions, None, lookback_token_ids=lookback
        )
        assert embed.call_args.args[0] is local_ids
        routing_ids = local_ids[::2] if use_sp else local_ids
        needs_gather = pcp_size > 1 and not use_ep
        if needs_gather:
            torch.testing.assert_close(gather.call_args.args[0], routing_ids)
            routing_ids = torch.cat(
                [routing_ids + rank * 100 for rank in range(pcp_size)]
            )
        for layer in layers:
            torch.testing.assert_close(layer.call_args.args[2], routing_ids)
        if use_engram:
            assert engram_hash.call_args.args[0] is local_ids
            assert engram_hash.call_args.args[1] is positions
            assert engram_hash.call_args.args[4] is lookback
            for layer in layers:
                torch.testing.assert_close(
                    layer.engram.prepare_embeddings.call_args.args[0],
                    local_ids[:, None],
                )
    assert gather.call_count == 2 * int(pcp_size > 1 and not use_ep)


def _pcp_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        quant_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        kernel_config=SimpleNamespace(moe_backend="triton"),
    )


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "use_ubatching", True),
        (None, "speculative_config", object()),
        (None, "kv_transfer_config", object()),
        ("cache_config", "enable_prefix_caching", True),
        ("kernel_config", "moe_backend", next(iter(MEGA_MOE_BACKENDS))),
    ],
)
def test_pcp_rejects_unsupported_execution_modes(section, field, value):
    config = _pcp_config()
    setattr(config if section is None else getattr(config, section), field, value)
    with pytest.raises(NotImplementedError, match="PP=DP=DCP=1"):
        model_module.DeepseekV4Model(vllm_config=config)


@pytest.mark.parametrize(
    "backend",
    [
        None,
        AttentionBackendEnum.FLASHMLA_SPARSE,
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV4,
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV41,
    ],
)
def test_pcp_selects_flashmla_before_automatic_device_dispatch(backend):
    config = _pcp_config()
    config.attention_config = SimpleNamespace(backend=backend)
    assert (
        model_module._select_dsv4_attn_cls(config)
        is model_module.DeepseekV4FlashMLAAttention
    )


def test_pcp_rejects_incompatible_attention_backend():
    config = _pcp_config()
    config.attention_config = SimpleNamespace(
        backend=AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV41
    )
    with pytest.raises(NotImplementedError, match="requires FlashMLA"):
        model_module._select_dsv4_attn_cls(config)
