"""Training and generation must use one immutable selected cache."""
from dataclasses import replace

import pytest
import torch
from transformers import GptOssConfig, GptOssForCausalLM

from masked_ttt import Backbone, MaskedTTT, Request, TTTConfig
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_batched_sd import ByteTokenizer
from test_standalone import make_engine


class Selector:
    def prepare(self, backbone, request):
        layout = prepare_layout(backbone, request)
        return PreparedRequest(request, layout, [len(layout.chunks) // 2],
                               dict(backend='unit_test'))


def request():
    return Request(document=('Old report says red. New report says blue. ' * 6),
                   question='What is the new report?', max_new_tokens=3)


def test_matched_selected_cache_uses_both_gradient_paths_and_remains_frozen():
    engine = make_engine()
    engine.config = replace(engine.config, matched_cache_probe=True)
    result = MaskedTTT(engine, Selector()).answer(request())
    probe = result['matched_cache_study']
    assert probe['accepted_steps'] == 2
    assert probe['same_document_cache_for_training_and_generation']
    assert probe['same_suffix_and_budget_for_zero_and_trained']
    assert probe['source_selected_cache_unchanged']
    assert probe['teacher_target_on_student_prefix']
    assert probe['zero_update']['generated_token_ids']
    assert probe['trained']['generated_token_ids']
    assert probe['zero_format']['generated_token_ids'] == result['teacher']['tokens']
    assert probe['trained_format']['generated_token_ids']
    assert probe['same_document_cache_and_budget_for_format_comparison']
    assert not probe['trained_format']['teacher_answer_in_final_prompt']
    assert not probe['trained']['teacher_answer_in_final_prompt']
    assert not probe['reference_answers_used']
    assert all(step['gain_gradient_norm'] > 0 and step['value_gradient_norm'] > 0
               for step in probe['steps'])
    assert result['original_document_KV_unchanged'] and result['backbone_unchanged']
    assert all(p.grad is None and not p.requires_grad for p in engine.model.parameters())


def test_zero_updates_match_unadapted_selected_teacher_with_identical_prompt():
    engine = make_engine()
    engine.config = replace(engine.config, matched_cache_probe=True, steps=0,
                            teacher_response_instruction='')
    result = MaskedTTT(engine, Selector()).answer(request())
    probe = result['matched_cache_study']
    assert probe['accepted_steps'] == 0
    assert probe['zero_update']['generated_token_ids'] == probe['trained']['generated_token_ids']
    assert probe['zero_format']['generated_token_ids'] == probe['trained_format']['generated_token_ids']
    assert probe['zero_update']['prediction'] == result['teacher']['prediction']


def test_hybrid_gpt_oss_matched_cache_runs_on_cpu():
    torch.manual_seed(71)
    torch.set_num_threads(1)
    config = GptOssConfig(vocab_size=512, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=1024, layer_types=['sliding_attention', 'full_attention'],
        sliding_window=16, pad_token_id=0, eos_token_id=2, head_dim=8,
        num_local_experts=2, num_experts_per_tok=1, rope_scaling=None)
    config._attn_implementation = 'eager'
    model = GptOssForCausalLM(config).eval()
    for block in model.model.layers:
        with torch.no_grad():
            block.self_attn.sinks.fill_(0.2)
    class HarmonyTokenizer(ByteTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return '<|start|>user<|message|>' + messages[0]['content'] + '<|end|><|start|>assistant'
    options = TTTConfig(chunk_tokens=32, max_context=1024, context_reserve_tokens=128,
        value_rank=2, steps=2, teacher_max_new_tokens=3, self_question_count=0,
        prefill_chunk_tokens=64, guard_enabled=False, standalone_cartridge_enabled=True,
        matched_cache_probe=True)
    engine = Backbone(model, HarmonyTokenizer(), options)
    result = MaskedTTT(engine, Selector()).answer(request())
    probe = result['matched_cache_study']
    assert probe['accepted_steps'] == 2
    assert probe['trained']['generated_token_ids']
    assert probe['source_selected_cache_unchanged']
    assert result['training']['adapted_layers'] == [1]
    assert result['backbone_unchanged']


def test_probe_requires_original_standalone_control():
    with pytest.raises(ValueError, match='needs the original standalone'):
        TTTConfig(matched_cache_probe=True, standalone_cartridge_enabled=False)
