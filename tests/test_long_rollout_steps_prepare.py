from pathlib import Path

import pytest

from experiments.long_rollout_steps.prepare import make_driver, pilot_slurm, select_pilot
from experiments.long_rollout_steps.report import collect, validate_manifest


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


def test_200_case_extension_remains_source_disjoint_from_800_holdout(tmp_path):
    development_sources = [f"qa{task}-{400000 + task * 100 + source}"
                           for task in range(1, 11) for source in range(5)]
    holdout_sources = [f"qa{task}-{500000 + task * 100 + source}"
                       for task in range(1, 11) for source in range(20)]
    holdout_sources[0], holdout_sources[20] = "qa1-300023", "qa2-300037"
    cases = lambda sources: [f"babilong-{length}-{source}" for source in sources
                             for length in (8, 16, 32, 64)]
    development, holdout = cases(development_sources), cases(holdout_sources)
    manifest = dict(case_count=200, case_ids=development, development_case_ids=development,
                    holdout_case_ids=holdout, step_budgets=[2, 4, 8], student_rollout_tokens=32,
                    selected_transfer_weight=0., selection_policy="decision_entity_link_v1",
                    keep_fraction=.1, reference_answers_used_in_inference=False,
                    online_api_calls=0, models=["qwen"])
    assert validate_manifest(manifest) == 200
    report = collect(manifest, tmp_path, {}, lambda *args: (0, True))
    assert report["expected_per_model"] == 200 and report["models"]["qwen"]["overall"]["n"] == 0
    manifest["case_ids"] = development[:-1] + holdout[:1]
    with pytest.raises(ValueError, match="holdout"):
        validate_manifest(manifest)
