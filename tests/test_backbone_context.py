import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import GptOssConfig, GptOssForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from masked_ttt import Backbone, TTTConfig
import masked_ttt.model as loader


@pytest.mark.parametrize("requested,native,rope_scaling", [
    (128, 64, {"rope_type": "linear", "factor": 4.0}),
    (32, 64, {"rope_type": "linear", "factor": 4.0}),
    (32, 64, None),
])
def test_qwen_loader_preserves_native_rope_and_limits_working_context(
    monkeypatch, requested, native, rope_scaling
):
    torch.manual_seed(19)
    model_config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=native, rope_scaling=rope_scaling, pad_token_id=0, eos_token_id=2)
    expected_rope = Qwen3RotaryEmbedding(model_config)
    expected_frequencies = expected_rope.inv_freq.clone()
    expected_type = expected_rope.rope_type
    seen = {}
    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained",
                        lambda *_args, **_kwargs: SimpleNamespace(eos_token_id=2))
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: model_config)

    def model_factory(_path, *, config, **_kwargs):
        seen["native_limit"] = config.max_position_embeddings
        seen["rope_scaling"] = config.rope_scaling
        config._attn_implementation = "eager"
        return Qwen3ForCausalLM(config)

    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", model_factory)
    runtime = TTTConfig(max_context=requested, context_reserve_tokens=16, value_rank=2)
    backbone = Backbone.load("synthetic", runtime, device="cpu", local_files_only=True)
    rotary = backbone.model.model.rotary_emb
    assert seen == {"native_limit": native, "rope_scaling": rope_scaling}
    assert backbone.config.max_context == min(requested, native)
    assert backbone.model.config.max_position_embeddings == native
    assert rotary.rope_type == expected_type
    torch.testing.assert_close(rotary.inv_freq, expected_frequencies, rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in backbone.model.parameters())


def test_gpt_oss_loader_keeps_yarn_and_native_context(monkeypatch):
    torch.manual_seed(23)
    scaling = dict(rope_type="yarn", factor=4., original_max_position_embeddings=32,
                   beta_fast=32., beta_slow=1., truncate=False)
    model_config = GptOssConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        num_local_experts=2, num_experts_per_tok=1, max_position_embeddings=128,
        layer_types=["sliding_attention", "full_attention"], sliding_window=16,
        rope_scaling=scaling, pad_token_id=0, eos_token_id=2)
    model_config.quantization_config = {"quant_method": "mxfp4"}
    expected = GptOssRotaryEmbedding(model_config).inv_freq.clone()
    seen = {}
    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained",
                        lambda *_args, **_kwargs: SimpleNamespace(eos_token_id=2))
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: model_config)

    def model_factory(_path, *, config, **kwargs):
        seen["dequantize"] = kwargs["quantization_config"].dequantize
        seen["native_limit"] = config.max_position_embeddings
        seen["rope_scaling"] = config.rope_scaling
        config._attn_implementation = "eager"
        return GptOssForCausalLM(config)

    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", model_factory)
    runtime = TTTConfig(max_context=256, context_reserve_tokens=16, value_rank=2)
    backbone = Backbone.load("synthetic", runtime, device="cpu", local_files_only=True)
    assert seen == {"dequantize": True, "native_limit": 128, "rope_scaling": scaling}
    assert backbone.config.max_context == 128
    torch.testing.assert_close(backbone.model.model.rotary_emb.inv_freq, expected, rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in backbone.model.parameters())


def test_current_qwen_default_rope_matches_legacy_context_override():
    """The existing 2507 checkpoint has no scaling and keeps its old predictions."""
    torch.manual_seed(29)
    native = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=262144, rope_scaling=None, rope_theta=5000000,
        pad_token_id=0, eos_token_id=2)
    native._attn_implementation = "eager"
    prior = copy.deepcopy(native)
    prior.max_position_embeddings = 131072
    old_model = Qwen3ForCausalLM(prior).eval()
    native_model = Qwen3ForCausalLM(native).eval()
    native_model.load_state_dict(old_model.state_dict())
    ids = torch.tensor([[3, 4, 5, 6]])
    positions = torch.tensor([[0, 8192, 65535, 131071]])
    with torch.no_grad():
        old = old_model(input_ids=ids, position_ids=positions, use_cache=False).logits
        new = native_model(input_ids=ids, position_ids=positions, use_cache=False).logits
    torch.testing.assert_close(old, new, rtol=0, atol=0)
