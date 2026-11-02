"""Native OLMo3/GPT-OSS attention, including local windows and learned sinks."""

from types import MethodType, SimpleNamespace
import torch
from torch.nn import functional as F

from .attention import native_attention, address_attention


def model_head_dim(config):
    return getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)


def global_layers(config):
    types = getattr(config, "layer_types", ["full_attention"] * config.num_hidden_layers)
    return [i for i, kind in enumerate(types) if kind == "full_attention"]


def project_qkv(module, hidden, embeddings):
    kind = module.config.model_type
    shape = (*hidden.shape[:-1], -1, module.head_dim)
    if kind == "olmo3":
        from transformers.models.olmo3.modeling_olmo3 import apply_rotary_pos_emb

        q = module.q_norm(module.q_proj(hidden)).view(shape).transpose(1, 2)
        k = module.k_norm(module.k_proj(hidden)).view(shape).transpose(1, 2)
    elif kind == "gpt_oss":
        from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb

        q = module.q_proj(hidden).view(shape).transpose(1, 2)
        k = module.k_proj(hidden).view(shape).transpose(1, 2)
    else:
        raise ValueError("Unsupported hybrid architecture")
    v = module.v_proj(hidden).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *embeddings)
    return q, k, v


def hybrid_attention(
    module,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    *,
    memory=None,
    **kwargs,
):
    q, k, v = project_qkv(module, hidden_states, position_embeddings)
    active = memory is not None and module.layer_idx in memory.adapted_layers
    augmented = active and getattr(past_key_values, "_masked_ttt_augmented", False)
    if augmented:
        padded = ((module.head_dim + 1 + 31) // 32) * 32
        k, v = F.pad(k, (0, padded - module.head_dim)), F.pad(v, (0, padded - module.head_dim))
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, module.layer_idx, dict(cache_position=cache_position))
    if active and memory.capture_attention:
        memory.observe(module.layer_idx, q, k)
    options = dict(
        scale=module.scaling,
        sliding_window=module.sliding_window,
        sinks=getattr(module, "sinks", None),
    )
    if not active or memory.native_mode:
        output = native_attention(q, k, v, **options)
    elif augmented:
        qp = F.pad(
            torch.cat((q, torch.ones_like(q[..., :1])), -1), (0, k.shape[-1] - module.head_dim - 1)
        )
        output = native_attention(qp, k, v, **options)[..., : module.head_dim]
    else:
        output = address_attention(
            q,
            k,
            memory.values(module.layer_idx, v),
            memory.biases(module.layer_idx, k.shape[-2]).to(k.device),
            sinks=getattr(module, "sinks", None),
            sliding_window=module.sliding_window,
        )
    output = output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).contiguous()
    return module.o_proj(output), None


def install_hybrid_attention(model, memory=None):
    for block in model.layers:

        def forward(module, *args, **kwargs):
            return hybrid_attention(module, *args, memory=memory, **kwargs)

        block.self_attn.forward = MethodType(forward, block.self_attn)


def hybrid_forward(model, tokens, start, cache):
    """Explicit device movement preserves autograd across a sharded frozen model."""
    embed_device = model.embed_tokens.weight.device
    positions = torch.arange(start, start + tokens.numel(), device=embed_device)
    hidden = model.embed_tokens(tokens.to(embed_device)[None])
    embeddings = rotary_embeddings(model, hidden, positions)
    for block in model.layers:
        device = block.self_attn.q_proj.weight.device
        hidden = block(
            hidden.to(device),
            attention_mask=None,
            position_ids=positions.to(device)[None],
            past_key_values=cache,
            use_cache=True,
            cache_position=positions.to(device),
            position_embeddings=tuple(x.to(device) for x in layer_rotary(embeddings, block)),
        )
    hidden = model.norm(hidden.to(model.norm.weight.device))
    return SimpleNamespace(last_hidden_state=hidden, past_key_values=cache)


def rotary_embeddings(model, hidden, positions):
    if model.config.model_type == "olmo3":
        return {kind: module(hidden, positions[None]) for kind, module in model.rotary_embs.items()}
    return model.rotary_emb(hidden, positions[None])


def layer_rotary(embeddings, block):
    return (
        embeddings[block.self_attn.attention_type] if isinstance(embeddings, dict) else embeddings
    )


class FinalChannelTokenizer:
    """Keep GPT-OSS Harmony boundaries; request an answer in the final channel."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, key):
        return getattr(self.tokenizer, key)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, messages, **kwargs):
        tokenize = kwargs.pop("tokenize", False)
        kwargs.pop("enable_thinking", None)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, reasoning_effort="low", **kwargs
        )
        if kwargs.get("add_generation_prompt"):
            if not text.endswith("<|start|>assistant"):
                raise ValueError("Unexpected GPT-OSS Harmony generation boundary")
            text += "<|channel|>final<|message|>"
        return self.tokenizer.encode(text, add_special_tokens=False) if tokenize else text


def install_sparse_experts(model):
    """Use Transformers' differentiable sparse expert branch with frozen weights."""
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

    class SparseView:
        training = True

        def __init__(self, module):
            self.module = module

        def __getattr__(self, name):
            return getattr(self.module, name)

    for block in model.layers:
        experts = block.mlp.experts
        if not isinstance(experts, GptOssExperts):
            raise TypeError("GPT-OSS adaptation needs the differentiable BF16 expert path")

        def forward(module, hidden_states, router_indices=None, routing_weights=None):
            return GptOssExperts.forward(
                SparseView(module), hidden_states, router_indices, routing_weights
            )

        experts.forward = MethodType(forward, experts)
