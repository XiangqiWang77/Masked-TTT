"""An independent, request-local selected-KV cartridge for the same question."""
from copy import copy
import gc
import math
import threading
import time

import torch

from .cache import fork_prefix
from .memory_accounting import cache_tensors, module_tensors, storage_bytes
from .model import Budget
from .runtime import private_forward
from .standalone import StandaloneReady


class QueryCartridge:
    """Reusable original-question decoding from already-materialized selected KV.

    The object retains the frozen backbone, a private reader, selected document
    KV, and the original question suffix. It does not retain the full document
    KV, Teacher cache, optimizer, or trainable adapter. Each generation forks
    a fresh suffix cache and cannot append to the retained document prefix.
    It is request-specific: it is not a general-purpose document index.
    """

    def __init__(self, engine, reader, branch, suffix, length, limit):
        self._engine, self._reader, self._branch = engine, reader, branch
        self._suffix, self._length, self._limit = suffix, length, limit
        self._lock = threading.RLock()
        self._weight_stamp = engine.weight_stamp()
        self._cache_stamp = [(x, x._version, x.data_ptr()) for x in cache_tensors(branch)]
        self._layer_lengths = tuple(layer.get_seq_length() for layer in branch.layers)
        self._kv_bytes = storage_bytes(cache_tensors(branch))

    @classmethod
    def from_ready(cls, engine, ready):
        """Capture a compact prefix only after the request has released its sources."""
        if (not isinstance(ready, StandaloneReady) or not ready.release_source or not ready.release_training or
            not ready.accounting.get("kv_only_materialized") or module_tensors(ready.reader.memory)):
            raise ValueError("A released, KV-only standalone cartridge is required")
        prefix = fork_prefix(ready.branch, ready.length)
        # A native-mode fork can otherwise retain the entire source cache via
        # this helper reference even though its selected prefix tensors suffice.
        if hasattr(prefix, "_document_storage_owner"):
            del prefix._document_storage_owner
        gc.collect()
        artifact = cache_tensors(prefix)
        source = [ref() for ref, _, _ in ready.source_stamp if ref() is not None]
        training = [ref() for ref in ready.training_refs if ref() is not None]
        if (storage_bytes(artifact + source) != storage_bytes(artifact) or
            storage_bytes(artifact + source + training) != storage_bytes(artifact)):
            raise RuntimeError("Source KV or training state is still retained outside the cartridge")
        return cls(engine, ready.reader, prefix, ready.suffix.detach(), ready.length, ready.limit)

    @property
    def document_kv_bytes(self):
        """Bytes of materialized document KV; excludes backbone and suffix."""
        with self._lock:
            self._require_open()
            return self._kv_bytes

    def _require_open(self):
        if self._branch is None:
            raise RuntimeError("The query cartridge is closed")

    def _assert_intact(self):
        self._require_open()
        self._engine.assert_frozen(self._weight_stamp)
        if (tuple(layer.get_seq_length() for layer in self._branch.layers) != self._layer_lengths or
            any(t._version != version or t.data_ptr() != pointer for t, version, pointer in self._cache_stamp)):
            raise RuntimeError("The retained cartridge KV was modified")

    @torch.no_grad()
    def answer(self, *, max_new_tokens=None, time_budget_seconds=None):
        """Generate again for the original question using a fresh suffix branch."""
        with self._lock:
            self._assert_intact()
            limit = self._limit if max_new_tokens is None else max_new_tokens
            if type(limit) is not int or limit < 1:
                raise ValueError("max_new_tokens must be a positive integer")
            if self._length + len(self._suffix) + limit > self._engine.config.max_context:
                raise ValueError("The question and generation exceed the model context")
            if time_budget_seconds is not None and (type(time_budget_seconds) not in (int, float) or
                                                    not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0):
                raise ValueError("time_budget_seconds must be positive")
            engine = copy(self._engine)
            engine.budget = Budget(time_budget_seconds)
            branch = fork_prefix(self._branch, self._length)
            started = time.perf_counter()
            engine.budget.check()
            output = private_forward(self._reader.reader, self._suffix, self._length, branch)
            tokens = []
            while len(tokens) < limit:
                token = int(engine.model.lm_head(output.last_hidden_state[0, -1:])[0].argmax())
                tokens.append(token)
                if token in engine.eos_ids or len(tokens) == limit:
                    break
                engine.budget.check()
                output = private_forward(self._reader.reader, torch.tensor([token], device=engine.device),
                    self._length + len(self._suffix) + len(tokens) - 1, branch)
            engine.synchronize()
            self._assert_intact()
            return dict(prediction=engine.tokenizer.decode(tokens, skip_special_tokens=True),
                generated_token_ids=tokens, finish_reason="eos" if tokens[-1] in engine.eos_ids else "length",
                document_kv_bytes=self._kv_bytes, generation_seconds=time.perf_counter() - started,
                original_question_only=True, teacher_answer_in_final_prompt=False)

    def close(self):
        """Discard request-specific KV and suffix references."""
        with self._lock:
            self._engine = self._reader = self._branch = self._suffix = None
            self._weight_stamp = self._cache_stamp = self._layer_lengths = None
            self._kv_bytes = 0

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()
