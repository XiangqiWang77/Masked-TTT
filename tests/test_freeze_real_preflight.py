from copy import deepcopy

import pytest

from experiments.kv_only_cartridge.freeze_real_preflight import freeze
from experiments.kv_only_cartridge.in_job_report import REQUIRED


def sample(model, version):
    result = dict(model=model, model_type="qwen3" if model == "qwen" else "gpt_oss",
        probe_version=version, passed=True, identity=dict(gpu="NVIDIA B200"),
        generic_teacher_answer_in_prompt=False, online_api_calls=0, reference_answers_used=False,
        created=1., generated_tokens=[123])
    result.update({key: True for key in REQUIRED if version >= 4 or
                   key not in ("same_checkpoint_rope", "native_checkpoint_config_checked")})
    return result


def test_real_gpu_preflight_snapshot_keeps_scope_and_omits_tokens():
    value = freeze(sample("qwen", 3), sample("gpt", 4), dict(state="passed", gpu="NVIDIA B200"))
    assert value["models"]["gpt"]["native_rope_on_gpu_checked"]
    assert not value["models"]["qwen"]["native_rope_on_gpu_checked"]
    assert value["qwen_v4_gpu_rope_check_pending"] and value["quality_not_measured"]
    assert "generated_tokens" not in value["models"]["qwen"]


def test_missing_gpu_rope_or_teacher_is_rejected():
    qwen, gpt = sample("qwen", 3), sample("gpt", 4)
    gpt["same_checkpoint_rope"] = False
    with pytest.raises(ValueError, match="gpt"):
        freeze(qwen, gpt, dict(state="passed", gpu="NVIDIA B200"))
    gpt = sample("gpt", 4); qwen = deepcopy(qwen); qwen["reference_answers_used"] = True
    with pytest.raises(ValueError, match="qwen"):
        freeze(qwen, gpt, dict(state="passed", gpu="NVIDIA B200"))
