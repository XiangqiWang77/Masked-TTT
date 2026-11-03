from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from masked_ttt import Backbone, TTTConfig


class ByteTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return "USER:\n" + messages[0]["content"] + "\nASSISTANT:\n"

    def encode(self, text, **kwargs):
        return list(text.encode("ascii"))

    def decode(self, tokens, **kwargs):
        return "".join(chr(i) if 32 <= i < 127 else f"<{i}>" for i in tokens)

    def __call__(self, text, **kwargs):
        return dict(
            input_ids=self.encode(text), offset_mapping=[(i, i + 1) for i in range(len(text))]
        )


@pytest.fixture
def tiny_backbone():
    torch.set_num_threads(1)
    torch.manual_seed(14)
    config = Qwen3Config(
        vocab_size=512,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=1024,
        pad_token_id=0,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    options = TTTConfig(
        chunk_tokens=16,
        max_context=1024,
        context_reserve_tokens=128,
        value_rank=2,
        steps=2,
        teacher_max_new_tokens=3,
        prefill_chunk_tokens=64,
    )
    return Backbone(model, ByteTokenizer(), options)


@pytest.fixture
def memory_fixture(tiny_backbone):
    from masked_ttt.cache import AppendOnlyDocumentCache
    from masked_ttt.memory import KVReadMemory
    from masked_ttt.reader import MemoryReader
    from masked_ttt.runtime import private_forward

    engine = tiny_backbone
    ids = torch.tensor([3, 4, 5, 6, 7, 8])
    cache = AppendOnlyDocumentCache(engine.model.config, 6)
    with torch.no_grad():
        private_forward(engine.model, ids, 0, cache)
    memory = KVReadMemory(
        engine.model.config, [[0, 1, 2], [3, 4, 5]], [1], 6, rank=2, sink_tokens=2
    )
    memory.prepare_value_basis(cache, [0, 1])
    memory.value_enabled = True
    return SimpleNamespace(
        engine=engine, cache=cache, memory=memory, reader=MemoryReader(engine, memory)
    )
