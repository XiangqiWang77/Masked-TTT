from pathlib import Path
import sys

import pytest
import torch

from experiments.kv_only_cartridge.reusable_probe import run_if_needed, validate_loaded_reusable
from test_materialization_equivalence import gpt_gqa_engine
from test_standalone import make_engine


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss_gqa"])
def test_optional_probe_compares_first_and_reused_logits_and_restores_runtime(architecture):
    engine = make_engine() if architecture == "qwen3" else gpt_gqa_engine()
    config, rng = engine.config, torch.get_rng_state().clone()
    previous = sys.modules.get("masked_ttt.cartridge")
    result = validate_loaded_reusable(engine, ROOT / "src/masked_ttt/pipeline.py",
                                     ROOT / "src/masked_ttt/cartridge.py")
    assert result["passed"] and result["same_first_and_replay_logits"]
    assert result["full_and_source_kv_released"] and not result["adapter_tensors_retained"]
    assert engine.config is config and torch.equal(torch.get_rng_state(), rng)
    assert sys.modules.get("masked_ttt.cartridge") is previous


def test_optional_probe_restores_rng_and_configuration_on_failure(tmp_path):
    engine = make_engine()
    config, rng = engine.config, torch.get_rng_state().clone()
    previous = sys.modules.get("masked_ttt.cartridge")
    with pytest.raises(FileNotFoundError):
        validate_loaded_reusable(engine, ROOT / "src/masked_ttt/pipeline.py", tmp_path / "missing.py")
    assert engine.config is config and torch.equal(torch.get_rng_state(), rng)
    assert sys.modules.get("masked_ttt.cartridge") is previous
    with pytest.raises(ValueError, match="allocated GPU"):
        run_if_needed(engine, "qwen", {})
