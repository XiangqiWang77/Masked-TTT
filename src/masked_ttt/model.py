"""Load a frozen Qwen3 backbone; make all request-specific state disposable."""

from contextlib import nullcontext
from dataclasses import dataclass, replace
import os
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import TTTConfig


class Budget:
    def __init__(self, seconds=None):
        self.deadline = time.monotonic() + seconds if seconds is not None else float("inf")

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Request time budget reached")


@dataclass
class Backbone:
    model: object
    tokenizer: object
    config: TTTConfig

    def __post_init__(self):
        kind = self.model.config.model_type
        if kind not in ("qwen3", "olmo3", "gpt_oss"):
            raise ValueError("Supported backbones: Qwen3, OLMo3, GPT-OSS")
        if kind == "qwen3" and self.model.model.has_sliding_layers:
            raise ValueError("The Qwen3 port uses full-attention checkpoints")
        self.model.eval().requires_grad_(False)
        if kind in ("olmo3", "gpt_oss"):
            from .architectures import install_hybrid_attention

            install_hybrid_attention(self.model.model)
        if kind == "gpt_oss":
            from .architectures import FinalChannelTokenizer, install_sparse_experts

            if not isinstance(self.tokenizer, FinalChannelTokenizer):
                self.tokenizer = FinalChannelTokenizer(self.tokenizer)
            install_sparse_experts(self.model.model)
        self.device = next(self.model.parameters()).device
        self.cfg = dict(
            max_context=self.config.max_context,
            chunk_tokens=self.config.chunk_tokens,
            chunk_boundary_mode="legacy",
            prefill_chunk_tokens=self.config.prefill_chunk_tokens,
        )
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.eos_ids.add(self.tokenizer.eos_token_id)
        self.eos_ids.discard(None)
        self.budget = Budget()

    @classmethod
    def load(
        cls, model_path, config=None, *, device="cuda:0", local_files_only=False, shard_gpus=1
    ):
        config = config or TTTConfig()
        target = torch.device(device)
        if target.type == "cuda":
            if (
                torch.cuda.is_initialized()
                and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            ):
                raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
            torch.use_deterministic_algorithms(True)
            torch.utils.deterministic.fill_uninitialized_memory = False
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
        model_config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
        # Preserve each checkpoint's native RoPE/YARN and context contract.
        if model_config.model_type == "qwen3":
            model_config.max_position_embeddings = config.max_context
            model_config.rope_scaling = None
        else:
            config = replace(
                config, max_context=min(config.max_context, model_config.max_position_embeddings)
            )
        device_map = {"": str(target)}
        if shard_gpus > 1:
            if target.type != "cuda" or shard_gpus > torch.cuda.device_count():
                raise ValueError("Requested model shards are not allocated")
            if model_config.model_type == "qwen3":
                raise ValueError("Qwen3 uses its validated single-device runtime")
            rotary = (
                "model.rotary_embs" if model_config.model_type == "olmo3" else "model.rotary_emb"
            )
            device_map = {"model.embed_tokens": 0, rotary: 0, "model.norm": 0, "lm_head": 0}
            for i in range(model_config.num_hidden_layers):
                device_map[f"model.layers.{i}"] = i * shard_gpus // model_config.num_hidden_layers
        options = {}
        if model_config.model_type == "gpt_oss":
            from transformers import Mxfp4Config

            # Frozen expert weights are dequantized once; input gradients remain
            # available on A40/L40S as well as newer GPUs.
            options["quantization_config"] = Mxfp4Config(dequantize=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=model_config,
            torch_dtype=torch.bfloat16 if target.type == "cuda" else torch.float32,
            device_map=device_map,
            attn_implementation=(
                "flash_attention_2"
                if target.type == "cuda" and model_config.model_type != "gpt_oss"
                else "eager"
            ),
            local_files_only=local_files_only,
            **options,
        )
        if model_config.model_type in ("olmo3", "gpt_oss"):
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(model, recurse=True)
        return cls(model, tokenizer, config)

    def synchronize(self):
        if self.device.type == "cuda":
            for device in {p.device for p in self.model.parameters()}:
                torch.cuda.synchronize(device)

    def device_context(self):
        return torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext()

    def weight_stamp(self):
        return [(p, p._version, p.data_ptr()) for p in self.model.parameters()]

    @staticmethod
    def assert_frozen(stamp):
        if any(
            p._version != version
            or p.data_ptr() != pointer
            or p.requires_grad
            or p.grad is not None
            for p, version, pointer in stamp
        ):
            raise AssertionError("A request changed a backbone weight")
