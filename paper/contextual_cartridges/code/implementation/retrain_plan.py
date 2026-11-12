"""Pure planning rules for the fixed-data, larger-budget paired experiment."""
import hashlib
import math

PERCENTAGES = (5, 10, 15, 20)
METHODS = ('vanilla_cartridges', 'contextual_cartridges')


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prefix_for_percent(whole_kv_tokens, percent):
    if type(whole_kv_tokens) is not int or whole_kv_tokens < 40:
        raise ValueError('Need the actual positive Whole-KV prefix length')
    if percent not in PERCENTAGES:
        raise ValueError('Only 5%, 10%, 15%, 20% are authorized')
    # Floor, not ceil: even after rounding the native KV never exceeds 20%.
    return whole_kv_tokens * percent // 100


def matched_updates(packed_counts, batch, epochs, minimum):
    if not packed_counts or min(packed_counts) < 1 or min(batch, epochs, minimum) < 1:
        raise ValueError('Positive packed data and training budgets required')
    return max(minimum, math.ceil(max(packed_counts) * epochs / batch))


def predicted_peak_bytes(prefix, model_config):
    head = model_config.get('head_dim', model_config['hidden_size']//model_config['num_attention_heads'])
    kv = 2 * model_config['num_hidden_layers'] * model_config['num_key_value_heads'] * head * 2 * prefix
    # Native BF16 K,V + gradients + Adam's two moments + temporary KV; plus
    # a 4B backbone and 8 GiB headroom. An estimate, not a measured guarantee.
    return {'native_kv_bytes': kv, 'training_peak_estimate_bytes': 5*kv + 16*1024**3,
            'estimate_only': True}
