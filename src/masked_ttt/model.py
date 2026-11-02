"""Load a frozen Qwen3 backbone; make all request-specific state disposable."""

from contextlib import nullcontext
from dataclasses import dataclass
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
        if self.model.config.model_type != "qwen3" or self.model.model.has_sliding_layers:
            raise ValueError("This implementation supports Qwen3 with full attention layers")
        self.model.eval().requires_grad_(False)
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
    def load(cls, model_path, config=None, *, device="cuda:0", local_files_only=False):
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
        model_config.max_position_embeddings = config.max_context
        model_config.rope_scaling = None
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=model_config,
            torch_dtype=torch.bfloat16 if target.type == "cuda" else torch.float32,
            device_map={"": str(target)},
            attn_implementation="flash_attention_2" if target.type == "cuda" else "eager",
            local_files_only=local_files_only,
        )
        return cls(model, tokenizer, config)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

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
