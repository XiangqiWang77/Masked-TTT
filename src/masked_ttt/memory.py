"""The only trainable state: selected chunk read biases and low-rank value maps."""

import hashlib
import json
import math
from bisect import bisect_left
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import repeat_kv


class KVReadMemory(nn.Module):
    def __init__(
        self,
        config,
        chunks,
        selected,
        document_length,
        *,
        rank=4,
        gain_bound=6.0,
        value_radius=0.15,
        sink_tokens=4,
        seed=42,
        value_parameterization="per_chunk",
        gain_scope="selected_only",
    ):
        super().__init__()
        self.layers = config.num_hidden_layers
        self.heads = config.num_key_value_heads
        self.dim = config.head_dim
        if not 1 <= rank <= self.dim:
            raise ValueError("Native value rank must be between1 and head_dim")
        if value_parameterization != "per_chunk":
            raise ValueError("This method requires per-chunk value coefficients")
        self.value_parameterization = value_parameterization
        if gain_scope != "selected_only":
            raise ValueError("Declare the chunk gain scope")
        self.gain_scope = gain_scope
        self.selected_chunks = tuple(sorted(set(selected)))
        self.bank_by_chunk = {chunk: bank for (bank, chunk) in enumerate(self.selected_chunks)}
        self.document_length = document_length
        self.gain_bound = float(gain_bound)
        self.value_radius = float(value_radius)
        index = torch.full((document_length,), -1, dtype=torch.long)
        for i, chunk in enumerate(chunks):
            index[torch.tensor(chunk)] = i
        index[:sink_tokens] = -1
        keep = torch.zeros(len(chunks), dtype=torch.bool)
        keep[torch.tensor(selected, dtype=torch.long)] = True
        self.register_buffer("selected_chunk_mask", keep)
        self.register_buffer("chunk_index", index)
        self.register_buffer("selected_positions", (index >= 0) & keep[index.clamp_min(0)])
        positions = [
            (index == chunk).nonzero().flatten().tolist() for chunk in self.selected_chunks
        ]
        width = max((len(p) for p in positions), default=0)
        bank_positions = torch.full((len(positions), width), -1, dtype=torch.long)
        (source_positions, source_slots, source_banks) = ([], [], [])
        self.bank_starts = []
        for bank, ps in enumerate(positions):
            if ps:
                self.bank_starts.append(ps[0])
                bank_positions[bank, : len(ps)] = torch.tensor(ps, dtype=torch.long)
                source_positions.extend(ps)
                source_slots.extend((bank * width + j for j in range(len(ps))))
                source_banks.extend([bank] * len(ps))
            else:
                self.bank_starts.append(document_length)
        if source_positions != sorted(set(source_positions)):
            raise ValueError("Chunk value banks require chronological disjoint source positions")
        self.source_positions = tuple(source_positions)
        self.source_banks = tuple(source_banks)
        self.register_buffer("bank_positions", bank_positions)
        self.register_buffer(
            "bank_source_indices", torch.tensor(source_positions, dtype=torch.long)
        )
        self.register_buffer("bank_scatter_slots", torch.tensor(source_slots, dtype=torch.long))
        self.register_buffer("value_layers", torch.zeros(self.layers, dtype=torch.bool))
        self.active_value_layers = set()
        self.gain = nn.Parameter(torch.zeros(self.layers, self.heads, len(chunks)))
        generator = torch.Generator().manual_seed(seed)
        basis = torch.randn(self.layers, self.heads, self.dim, rank, generator=generator)
        basis = torch.linalg.qr(basis, mode="reduced").Q
        self.register_buffer("value_basis", basis)
        shape = (self.layers, self.heads, len(self.selected_chunks), rank, self.dim)
        self.value_coefficients = nn.Parameter(torch.zeros(shape))
        self.value_enabled = False
        self.native_mode = False
        self.capture_attention = False
        self.attention_records = {}

    def biases(self, layer, length):
        index = self.chunk_index[: min(length, self.document_length)]
        mask = self.selected_positions[: len(index)]
        values = self.gain[layer][:, index.clamp_min(0)] * mask[None]
        return F.pad(values, (0, max(0, length - self.document_length)))

    def values(self, layer, native):
        if not self.value_enabled or layer not in self.active_value_layers:
            return native
        return self._chunk_values(layer, native)

    def _chunk_values(self, layer, native):
        """Batched chunk writes: changing bank i never rewrites another bank.

        Padding is only a compute layout. Each real selected source position
        is scattered exactly once; sinks and the Question suffix are intact.
        """
        n = min(native.shape[-2], self.document_length)
        count = bisect_left(self.source_positions, n)
        if not count:
            return native + self.value_coefficients[layer].sum() * 0.0
        banks = self.source_banks[count - 1] + 1
        positions = self.bank_positions[:banks]
        visible = (positions >= 0) & (positions < n)
        gathered = native[..., positions.clamp(min=0, max=n - 1), :]
        basis = self.value_basis[layer][None, :, None]
        coefficients = self.value_coefficients[layer, :, :banks][None]
        delta = gathered.float() @ basis.float() @ coefficients.float()
        radius = self.value_radius * gathered.float().norm(dim=-1, keepdim=True)
        ratio = (radius / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
        delta = torch.where(visible[None, None, :, :, None], delta * ratio, torch.zeros_like(delta))
        flat = delta.flatten(-3, -2).to(native.dtype)
        selected_delta = flat.index_select(-2, self.bank_scatter_slots[:count])
        source = self.bank_source_indices[:count]
        updated = native[..., :n, :].index_copy(-2, source, native[..., source, :] + selected_delta)
        return torch.cat((updated, native[..., n:, :]), dim=-2)

    @torch.no_grad()
    def prepare_value_basis(self, cache, layers):
        self.active_value_layers = set(layers)
        self.value_layers.zero_()
        self.value_layers[list(layers)] = True
        for layer in layers:
            x = cache.layers[layer].values[0, :, self.selected_positions].float()
            covariance = x.transpose(-1, -2) @ x / max(1, x.shape[-2])
            (_, vectors) = torch.linalg.eigh(covariance)
            basis = vectors[..., -self.value_basis.shape[-1] :]
            pivots = basis.abs().argmax(dim=-2, keepdim=True)
            signs = basis.gather(-2, pivots).sign()
            self.value_basis[layer].copy_(basis * signs)

    @torch.no_grad()
    def initialize_gain(self, selected, layers, bias):
        self.gain.zero_()
        indices = torch.tensor(selected, device=self.gain.device)
        for layer in layers:
            self.gain[layer].index_fill_(1, indices, float(bias))

    @torch.no_grad()
    def constrain(self):
        self.gain.clamp_(-self.gain_bound, self.gain_bound)
        if self.gain_scope == "selected_only":
            self.gain.masked_fill_(~self.selected_chunk_mask[None, None, :], 0.0)

    def snapshot(self):
        return dict(
            gain=self.gain.detach().clone(),
            value_coefficients=self.value_coefficients.detach().clone(),
            native_mode=self.native_mode,
            value_enabled=self.value_enabled,
        )

    @torch.no_grad()
    def restore(self, snapshot):
        self.gain.copy_(snapshot["gain"])
        self.value_coefficients.copy_(snapshot["value_coefficients"])
        self.native_mode = snapshot["native_mode"]
        self.value_enabled = snapshot["value_enabled"]

    @torch.no_grad()
    def select_actual_read_path(self):
        """Evaluate exactly the native/augmented branch used for generation."""
        has_values = self.value_enabled and bool(self.value_coefficients.any())
        self.native_mode = not bool(self.gain.any()) and (not has_values)

    def digest(self):
        digest = hashlib.sha256()
        for tensor in (
            self.gain,
            self.value_coefficients,
            self.value_layers,
            self.value_basis,
            self.chunk_index,
            self.selected_positions,
            self.bank_positions,
            self.bank_source_indices,
            self.bank_scatter_slots,
        ):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(
            json.dumps(
                dict(
                    native_mode=self.native_mode,
                    value_enabled=self.value_enabled,
                    gain_bound=self.gain_bound,
                    value_radius=self.value_radius,
                    document_length=self.document_length,
                    value_parameterization=self.value_parameterization,
                    selected_chunks=self.selected_chunks,
                    gain_scope=self.gain_scope,
                ),
                sort_keys=True,
            ).encode()
        )
        return digest.hexdigest()

    @torch.no_grad()
    def observe(self, layer, q, k):
        q = q[..., -min(4, q.shape[-2]) :, :].float()
        kk = repeat_kv(k.float(), q.shape[1] // k.shape[1])
        scores = q @ kk.transpose(-1, -2) / math.sqrt(self.dim)
        qpos = torch.arange(q.shape[-2], device=q.device) + k.shape[-2] - q.shape[-2]
        visible = torch.arange(k.shape[-2], device=q.device)[None] <= qpos[:, None]
        probabilities = scores.masked_fill(~visible[None, None], float("-inf")).softmax(-1)
        weights = probabilities.mean((0, 1, 2))[: self.document_length]
        self.attention_records[layer] = dict(
            weights=weights, selected_mass=float(weights[self.selected_positions].sum())
        )
