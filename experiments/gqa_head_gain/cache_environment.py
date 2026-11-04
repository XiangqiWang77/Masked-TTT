"""Point future Python starts in existing jobs at request-local scratch caches.

The batch scripts already include their job root on PYTHONPATH. A narrowly
scoped sitecustomize file changes only Triton and Torch extension cache paths;
the frozen model, training code, and submitted Slurm script are not edited.
An interpreter already running will retain its current environment.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile


def content(root):
    cache = root / "runtime_cache"
    return ("# Task-local compile/autotune caches; no model or training changes.\n"
            "import os as _os\n"
            f"_os.environ['TRITON_CACHE_DIR'] = {str(cache / 'triton')!r}\n"
            f"_os.environ['TORCH_EXTENSIONS_DIR'] = {str(cache / 'torch_extensions')!r}\n")


def install(root):
    root = Path(root).resolve()
    batch = (root / "job.sbatch").read_text()
    # Both historical and current scripts explicitly place the job root at
    # the end of PYTHONPATH. Reject unrelated jobs instead of guessing.
    rooted = (f"ROOT={root}" in batch and ':$ROOT"' in batch) or (
        f"TASK_ROOT={root}" in batch and ':$TASK_ROOT"' in batch)
    if not rooted or "PYTHONPATH=" not in batch:
        raise ValueError("The job root is not explicitly on its Python path")
    for directory in (root / "runtime/repo/src", root / "runtime/repo", root / "runtime/helpers"):
        if (directory / "sitecustomize.py").exists():
            raise ValueError("An earlier Python path contains another sitecustomize")
    target, text = root / "sitecustomize.py", content(root)
    if target.exists():
        if not target.is_file() or target.read_text() != text:
            raise ValueError("An existing job customization must not be overwritten")
        return dict(installed=True, already_present=True, applies_to_future_python_starts=True)
    for directory in (root / "runtime_cache/triton", root / "runtime_cache/torch_extensions"):
        directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".sitecustomize-", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_file() and target.read_text() == text:
                return dict(installed=True, already_present=True, applies_to_future_python_starts=True)
            raise ValueError("An existing job customization must not be overwritten")
    finally:
        os.unlink(temporary)
    return dict(installed=True, already_present=False, applies_to_future_python_starts=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_roots", nargs="+", type=Path)
    args = parser.parse_args()
    for root in args.job_roots:
        print(json.dumps(dict(root=str(root), **install(root))))


if __name__ == "__main__":
    main()
