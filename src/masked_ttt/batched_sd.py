"""Causally batch the fixed teacher prefix with one checkpoint per layer.

The document K/V are immutable. Only the selected document-read state receives
gradients. Each layer's temporary branch is independently rebuilt on backward.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .architectures import layer_rotary,rotary_embeddings
from .attention import address_dimensions, address_key
from .cache import fork_prefix


def batched_teacher_logits(engine,reader,cache,suffix,targets,*,checkpoint_layers=True,
                           target_positions=None):
    if suffix.ndim!=1 or targets.ndim!=1 or not len(suffix) or not len(targets):
        raise ValueError('Need nonempty one-dimensional question and teacher tokens')
    memory=reader.memory;model=reader.reader.model
    if memory.native_mode or memory.capture_attention:
        raise ValueError('Require the selected differentiable read path')
    if model.config.model_type not in ('qwen3','olmo3','gpt_oss'):
        raise ValueError('Unsupported backbone')
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('Backbone must be frozen')
    length=memory.document_length
    input_ids=torch.cat((suffix,targets[:-1]))
    embed_device=model.embed_tokens.weight.device
    positions=torch.arange(length,length+len(input_ids),device=embed_device)
    hidden=model.embed_tokens(input_ids.to(embed_device)[None])
    with torch.no_grad():
        embeddings=rotary_embeddings(model,hidden,positions)
    d=memory.dim
    _,padded=address_dimensions(d,memory.query_heads,memory.heads,memory.gain_heads)
    for layer_index,block in enumerate(model.layers[:model.config.num_hidden_layers]):
        device=block.self_attn.q_proj.weight.device
        hidden=hidden.to(device);local_positions=positions.to(device)
        local_embeddings=tuple(x.to(device) for x in layer_rotary(embeddings,block))

        def apply_layer(value,block=block,layer_index=layer_index,
                        positions=local_positions,embeddings=local_embeddings):
            engine.budget.check()
            branch=fork_prefix(cache,length)
            if hasattr(branch,'_document_storage_owner'):
                del branch._document_storage_owner
            item=branch.layers[layer_index]
            if layer_index in memory.adapted_layers:
                k,v=item.keys,item.values
                bias=memory.biases(layer_index,k.shape[-2]).to(k.device)
                item.keys=address_key(k,bias,memory.query_heads)
                item.values=F.pad(memory.values(layer_index,v),(0,padded-d))
            branch._masked_ttt_augmented=True
            if getattr(item,'is_sliding',False):
                item.keys=item.keys[...,-item.sliding_window+1:,:]
                item.values=item.values[...,-item.sliding_window+1:,:]
                item.cumulative_length=int(positions[0])
            return block(value,attention_mask=None,position_ids=positions[None],
                past_key_values=branch,use_cache=True,cache_position=positions,
                position_embeddings=embeddings)

        hidden=(checkpoint(apply_layer,hidden,use_reentrant=False,preserve_rng_state=False)
                if checkpoint_layers and torch.is_grad_enabled() else apply_layer(hidden))
    hidden=model.norm(hidden.to(model.norm.weight.device))[0,len(suffix)-1:]
    if target_positions is not None:hidden=hidden.index_select(0,target_positions)
    return engine.model.lm_head(hidden).float()
