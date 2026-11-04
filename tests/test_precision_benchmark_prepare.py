from pathlib import Path

from experiments.precision_cartridge.benchmark_prepare import CPU_BATCHES, CPU_TESTS, precision_slurm, report_stub


def test_precision_job_is_single_gpu_and_uses_frozen_report():
    root = Path("/tmp/query_cartridges_precision_pilot_20261104")
    script = precision_slurm(root)
    assert "#SBATCH --gpus-per-task=1" in script and "#SBATCH --requeue" in script
    assert "precision_cartridge/benchmark_execute.py" in script and "unset OPENAI_API_KEY" in script
    report = report_stub(Path("/tmp/workspace"), root)
    assert "precision_cartridge.benchmark_report" in report and "/runtime/scorer" in report
    assert [item for batch in CPU_BATCHES for item in batch] == CPU_TESTS
