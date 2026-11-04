"""The scheduled candidate keeps the OLMo protocol and a single GPU."""
from pathlib import Path

import pytest

from experiments.gqa_head_gain.prepare import make_driver, report_stub, slurm_script


def test_driver_changes_only_the_rollout_cap_and_requires_the_request_date_fix():
    source = ("manifest.get('student_rollout_tokens')!=32\n"
              "t['student_rollout_limit']!=32\n"
              "student_rollout_tokens=32,\n"
              "Original SD / 32-token rollout / independent cartridge contract failed\n"
              "MASKED_TTT_FIXED_CHAT_DATE\n")
    frozen = make_driver(source)
    assert "student_rollout_tokens=8," in frozen
    assert "t['student_rollout_limit']!=8" in frozen
    assert "MASKED_TTT_FIXED_CHAT_DATE" in frozen
    with pytest.raises(ValueError):
        make_driver(source.replace("MASKED_TTT_FIXED_CHAT_DATE", ""))
    with pytest.raises(ValueError):
        make_driver(source.replace("student_rollout_tokens=32,", "student_rollout_tokens=16,"))


def test_submission_and_report_are_single_gpu_and_use_the_cached_protocol():
    root = Path("/tmp/query_cartridges_gqa_gain_dev_20261104")
    script = slurm_script(root)
    assert "#SBATCH --gpus-per-task=1" in script and "#SBATCH --requeue" in script
    assert "unset OPENAI_API_KEY OPENAI_API_KEY_FILE" in script
    assert "test_cuda_materialization.py" in script and "gqa_head_gain/execute.py" in script
    stub = report_stub(Path("/tmp/workspace"), root, Path("/tmp/baseline"))
    assert "gqa_head_gain.report" in stub and "/tmp/baseline" in stub
