"""Method settings, with the submitted Selective SD defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
@dataclass(frozen=True)
class TTTConfig:
    chunk_tokens: int = 256
    keep_fraction: float = 0.10
    max_context: int = 131072
    context_reserve_tokens: int = 4096
    prefill_chunk_tokens: int = 1024
    steps: int = 2
    value_rank: int = 8
    value_radius: float = 0.15
    gain_bound: float = 6.0
    sink_tokens: int = 4
    teacher_max_new_tokens: int = 32
    student_rollout_tokens: int = 8
    self_question_count: int = 2
    self_question_max_tokens: int = 192
    self_question_verify_tokens: int = 16
    practice_boundary_fallback: bool = False
    guard_enabled: bool = True
    guard_minimum_confidence: float = 0.65
    guard_minimum_label_mass: float = 0.20
    standalone_cartridge_enabled: bool = True
    gain_learning_rate: float = 0.01
    value_learning_rate: float = 0.0001
    gradient_clip: float = 10.0
    # Optional self-distillation from the adapted full reader into the same
    # state evaluated on the independently prefetched selected cache.
    selected_transfer_weight: float = 0.0
    teacher_target_scope: str = 'choice_or_all'
    max_answer_steps: int = 8
    answer_KL_stop: float = 0.5
    teacher_response_instruction: str = 'Return only the final answer in the required format. Do not add an explanation.'
    seed: int = 42

    def __post_init__(self):
        for name in (
            "chunk_tokens",
            "max_context",
            "context_reserve_tokens",
            "prefill_chunk_tokens",
            "value_rank",
            "teacher_max_new_tokens",
            "student_rollout_tokens",
            "self_question_max_tokens",
            "self_question_verify_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.chunk_tokens < 2 or self.context_reserve_tokens >= self.max_context:
            raise ValueError("Invalid chunk or context budget")
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if type(self.self_question_count) is not int or not 0 <= self.self_question_count <= 4:
            raise ValueError('self_question_count must be between 0 and 4')
        if type(self.practice_boundary_fallback) is not bool:
            raise ValueError('practice_boundary_fallback must be boolean')
        if type(self.guard_enabled) is not bool or type(self.standalone_cartridge_enabled) is not bool:
            raise ValueError('guard and standalone settings must be boolean')
        for name in ('guard_minimum_confidence','guard_minimum_label_mass'):
            value=getattr(self,name)
            if type(value) not in (float,int) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError(name+' must be in [0, 1]')
        if (
            type(self.keep_fraction) not in (float, int)
            or not 0 < self.keep_fraction <= 1
            or not math.isfinite(self.keep_fraction)
        ):
            raise ValueError("keep_fraction must be in (0, 1]")
        if not 0 < self.value_radius <= 1 or not 0 < self.gain_bound:
            raise ValueError("Invalid KV trust bounds")
        for name in ("gain_learning_rate", "value_learning_rate", "gradient_clip"):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f"{name} must be positive and finite")
        if (type(self.selected_transfer_weight) not in (float,int) or
            not math.isfinite(self.selected_transfer_weight) or self.selected_transfer_weight<0):
            raise ValueError("selected_transfer_weight must be nonnegative and finite")
        if self.teacher_target_scope not in ('all','choice_or_all'):
            raise ValueError('Unsupported teacher token scope')
        if type(self.max_answer_steps) is not int or self.max_answer_steps<self.steps:
            raise ValueError('Maximum answer steps must include the base budget')
        if not math.isfinite(self.answer_KL_stop) or self.answer_KL_stop<0:
            raise ValueError('Nonnegative answer-KL stopping threshold required')
        if not isinstance(self.teacher_response_instruction,str):
            raise ValueError('Teacher response instruction must be text')
        if type(self.sink_tokens) is not int or self.sink_tokens < 0:
            raise ValueError("sink_tokens must be nonnegative")


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
