from pathlib import Path

import pytest

from experiments.long_rollout_steps.prepare import make_driver, pilot_slurm, select_pilot


def test_pilot_is_fixed_by_source_and_type_without_answer_labels():
    development = [f"babilong-{length}-qa{task}-{300000 + task * 10 + source}"
        for task in range(1, 11) for source in range(5) for length in (8, 16, 32, 64)]
    cases = select_pilot(development)
    assert len(cases) == 56 and sum("-qa3-" in case for case in cases) == 20
    for task in [x for x in range(1, 11) if x != 3]:
        assert sum(f"-qa{task}-" in case for case in cases) == 4
    with pytest.raises(ValueError):
        select_pilot(development[:-1])


def test_frozen_driver_and_job_preserve_single_gpu_and_32_token_path():
    source = "count not in (200,800)\nMASKED_TTT_FIXED_CHAT_DATE\nstudent_rollout_tokens=32,"
    frozen = make_driver(source)
    assert "count not in (56,200,800)" in frozen and "student_rollout_tokens=32," in frozen
    with pytest.raises(ValueError):
        make_driver(source.replace("MASKED_TTT_FIXED_CHAT_DATE", ""))
    script = pilot_slurm(Path("/tmp/query_cartridges_longsteps_pilot_20261104"))
    assert "#SBATCH --gpus-per-task=1" in script and "#SBATCH --requeue" in script
    assert "long_rollout_steps/execute.py" in script and "unset OPENAI_API_KEY" in script
