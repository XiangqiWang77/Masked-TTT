"""Exercise the actual offline loader with small local checkpoints."""
import json

import pytest
import torch
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import (GptOssConfig, GptOssForCausalLM, PreTrainedTokenizerFast,
                          Qwen3Config, Qwen3ForCausalLM)

from masked_ttt import Backbone, FrozenJSONSelector, MaskedTTT, Request, TTTConfig


def cached_source(directory):
    chunks = ["Other."] * 40
    chunks[1], chunks[2], chunks[6] = ("Daniel grabbed the milk.",
        "Daniel moved to the garden.", "Daniel dropped the milk.")
    document, question = "\n".join(chunks), "Where is the milk?"
    spans, cursor = [], 0
    for chunk in chunks:
        start = document.index(chunk, cursor)
        spans.append([start, start + len(chunk)])
        cursor = start + len(chunk)
    records = [dict(chunk=i, utility_output_valid=True,
        decision_probabilities={"yes": {1:.9, 2:.05, 3:.8, 6:.85}.get(i,.01)})
        for i in range(len(chunks))]
    (directory / "scores").mkdir()
    (directory / "case_to_score_file.json").write_text(json.dumps({"case":"case.json"}))
    (directory / "scores/case.json").write_text(json.dumps(dict(
        schema="masked-ttt-source-scores-v1", complete=True, case_key="case",
        question=question, reference_answers_used=False, source_spans=spans,
        canonical_tokenizer="synthetic", scoring_seconds=0., estimated_cost_usd=0., records=records)))
    return document, question


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
def test_load_unquantized_checkpoint_from_disk_without_network(tmp_path, architecture):
    torch.manual_seed(41)
    words = ["<unk>", "<pad>", "<eos>", "\n\n", "\n", "Other.",
             "Daniel grabbed the milk.", "Daniel moved to the garden.",
             "Daniel dropped the milk."] + [f"token{i}" for i in range(55)]
    tokenizer = Tokenizer(models.WordLevel({token: i for i, token in enumerate(words)},
                                            unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"\n\n|\n|[^\n]+"), behavior="isolated")
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>",
                                   pad_token="<pad>", eos_token="<eos>")
    fast.chat_template = ("{% for message in messages %}<|start|>{{ message['role'] }}"
        "<|message|>{{ message['content'] }}<|end|>{% endfor %}"
        "{% if add_generation_prompt %}<|start|>assistant{% endif %}")
    assert fast.encode("\n\n", add_special_tokens=False)
    fast.save_pretrained(tmp_path)
    common = dict(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=512, pad_token_id=1, eos_token_id=2)
    if architecture == "qwen3":
        config = Qwen3Config(**common, rope_scaling={"rope_type": "linear", "factor": 4.})
        model = Qwen3ForCausalLM(config)
    else:
        config = GptOssConfig(**common, num_local_experts=2, num_experts_per_tok=1,
            layer_types=["sliding_attention", "full_attention"], sliding_window=16,
            rope_scaling=None)
        model = GptOssForCausalLM(config)
        for layer in model.model.layers:
            with torch.no_grad():
                layer.self_attn.sinks.fill_(.2)
    model.eval()
    model.save_pretrained(tmp_path)
    runtime = TTTConfig(max_context=384, context_reserve_tokens=32, chunk_tokens=32,
        prefill_chunk_tokens=64, value_rank=2, steps=2, max_answer_steps=2,
        student_rollout_tokens=3, teacher_max_new_tokens=3, self_question_count=0,
        guard_enabled=False, release_full_before_cartridge=True,
        release_selected_source_before_decode=True)
    backbone = Backbone.load(tmp_path, runtime, device="cpu", local_files_only=True)
    backbone.eos_ids = {-1}
    assert backbone.model.config.model_type == architecture
    assert backbone.config.max_context == 384
    assert backbone.model.config.max_position_embeddings == 512
    assert backbone.model.config.rope_scaling == config.rope_scaling
    assert all(not parameter.requires_grad for parameter in backbone.model.parameters())
    assert model.state_dict().keys() == backbone.model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, backbone.model.state_dict()[name], rtol=0, atol=0)
    scores = tmp_path / "cached_scores"
    scores.mkdir()
    document, question = cached_source(scores)
    selector = FrozenJSONSelector(scores)
    selector.bind_case("case")
    result = MaskedTTT(backbone, selector).answer(
        Request(document=document, question=question, max_new_tokens=3), compact_standalone=True)
    assert result["training"]["objective"] == "on-policy KL(teacher || student)"
    assert result["training"]["teacher_conditioned_on_student_prefix"]
    assert result["training"]["student_rollouts_refreshed_each_step"]
    assert result["teacher"]["fresh_selected_context"]
    assert any(step["gain_gradient_norm"] > 0 for step in result["training"]["steps"])
    assert any(step["value_gradient_norm"] > 0 for step in result["training"]["steps"])
    assert result["backbone_unchanged"] and result["original_document_KV_unchanged"]
    assert result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
    memory = result["standalone_cartridge"]["memory"]
    assert memory["kv_only_materialized"] and memory["cartridge_adapter_bytes"] == 0
    assert memory["selected_kv_representation"] == "independently_prefilled_selected_text"
    assert memory["additional_source_kv_bytes_before_decode"] == 0
    assert memory["additional_training_adapter_bytes_before_decode"] == 0
