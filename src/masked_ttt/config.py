"""Method settings, with the submitted Selective SD defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optimizer import UpdateOptions


def default_update_options():
    from .optimizer import UpdateOptions

    return UpdateOptions(
        native_KL_budget=None,
        direction_preconditioner="adam",
        adaptive_probes=True,
        preservation_aware_geometry=True,
        query_guard=True,
        allow_native_zero_boundary=True,
    )


@dataclass(frozen=True)
class TTTConfig:
    chunk_tokens: int = 256
    keep_fraction: float = 0.10
    max_context: int = 262144
    context_reserve_tokens: int = 4096
    prefill_chunk_tokens: int = 1024
    steps: int = 16
    value_rank: int = 8
    value_radius: float = 0.15
    gain_bound: float = 6.0
    sink_tokens: int = 4
    teacher_max_new_tokens: int = 1024
    checkpoint_at_tokens: int = 65536
    seed: int = 42
    update: UpdateOptions = field(default_factory=default_update_options)

    def __post_init__(self):
        for name in (
            "chunk_tokens",
            "max_context",
            "context_reserve_tokens",
            "prefill_chunk_tokens",
            "value_rank",
            "teacher_max_new_tokens",
            "checkpoint_at_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.chunk_tokens < 2 or self.context_reserve_tokens >= self.max_context:
            raise ValueError("Invalid chunk or context budget")
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if (
            type(self.keep_fraction) not in (float, int)
            or not 0 < self.keep_fraction <= 1
            or not math.isfinite(self.keep_fraction)
        ):
            raise ValueError("keep_fraction must be in (0, 1]")
        if not 0 < self.value_radius <= 1 or not 0 < self.gain_bound:
            raise ValueError("Invalid KV trust bounds")
        if type(self.sink_tokens) is not int or self.sink_tokens < 0:
            raise ValueError("sink_tokens must be nonnegative")
        if self.update.native_KL_budget is not None:
            raise ValueError(
                "This method uses a fixed teacher target without a cumulative native anchor"
            )


@dataclass(frozen=True)
class DecisionConfig:
    api_key_file: str | None = None
    cache_dir: str = str(Path.home() / ".cache" / "masked_ttt" / "decisions")
    max_cost_usd: float = 15.0
    batch_size: int = 32
    workers: int = 2
    timeout_seconds: float = 45.0

    def __post_init__(self):
        if not 0 < self.max_cost_usd <= 50:
            raise ValueError("Decision API cost ceiling must be in (0, 50] USD")
        if not 1 <= self.batch_size <= 400 or not 1 <= self.workers <= 8:
            raise ValueError("Invalid Decisions batch size or worker count")
        if not 0 < self.timeout_seconds <= 45:
            raise ValueError("Decision timeout must be in (0, 45] seconds")

    def client_options(self):
        root = Path(self.cache_dir).expanduser().resolve()
        return dict(
            decision_api_key_file=self.api_key_file,
            decision_cache_directory=str(root / "responses"),
            decision_budget_ledger=str(root / "cost.json"),
            decision_max_cost_usd=self.max_cost_usd,
            decision_timeout_seconds=self.timeout_seconds,
        )
