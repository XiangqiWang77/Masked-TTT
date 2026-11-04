import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.gqa_head_gain.cache_environment import install


@pytest.mark.parametrize("root_variable", ["ROOT", "TASK_ROOT"])
def test_future_interpreters_use_scratch_caches_without_changing_other_environment(tmp_path, root_variable):
    (tmp_path / "job.sbatch").write_text(
        f'{root_variable}={tmp_path}\nexport PYTHONPATH="$TASK_REPO/src:${root_variable}"\n')
    first, again = install(tmp_path), install(tmp_path)
    assert first["installed"] and not first["already_present"] and again["already_present"]
    env = dict(os.environ, PYTHONPATH=str(tmp_path), TRITON_CACHE_DIR="/wrong/triton",
               TORCH_EXTENSIONS_DIR="/wrong/extensions", QUERY_CARTRIDGE_MARKER="unchanged")
    code = "import json,os;print(json.dumps([os.environ[k] for k in " \
           "('TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR','QUERY_CARTRIDGE_MARKER')]))"
    process = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path,
                             capture_output=True, text=True, check=True)
    actual = json.loads(process.stdout)
    assert actual == [str(tmp_path / "runtime_cache/triton"),
                      str(tmp_path / "runtime_cache/torch_extensions"), "unchanged"]
    assert (tmp_path / "runtime_cache/triton").is_dir()
    assert (tmp_path / "runtime_cache/torch_extensions").is_dir()


def test_does_not_replace_an_existing_user_customization(tmp_path):
    (tmp_path / "job.sbatch").write_text(
        f'TASK_ROOT={tmp_path}\nexport PYTHONPATH="$TASK_REPO/src:$TASK_ROOT"\n')
    target = tmp_path / "sitecustomize.py"
    target.write_text("# existing work\n")
    with pytest.raises(ValueError, match="must not be overwritten"):
        install(tmp_path)
    assert target.read_text() == "# existing work\n"


def test_rejects_job_without_root_on_pythonpath(tmp_path):
    (tmp_path / "job.sbatch").write_text("export PYTHONPATH=/unrelated\n")
    with pytest.raises(ValueError, match="not explicitly"):
        install(tmp_path)
    assert not (tmp_path / "sitecustomize.py").exists()
