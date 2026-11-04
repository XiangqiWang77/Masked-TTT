"""Serve preselected SD step counts on source-disjoint holdout questions."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import weakref

from experiments.babilong_linked_selection.long_rollout_runtime import linked_long_rollout_runner
from experiments.long_rollout_steps.runner import preflight as checkpoint_preflight
from masked_ttt.config import TTTConfig


@contextmanager
def fixed_step_runner(implementation, runtime, paired_controls, *, root, linked_selector,
                      physical_controls, model_path_steps):
    """Use only a model-specific step fixed on development, with no answer labels.

    This context is process-local and is intended for the sequential single-GPU
    driver.  The result is decoded from a standalone selected KV after release.
    """
    if not model_path_steps:
        raise ValueError("At least one preselected model and step is required")
    paths = {str(Path(path).resolve()): value for path, value in model_path_steps.items()}
    if len(paths) != len(model_path_steps) or any(type(x) is not int or x not in (2, 4, 8)
                                                  for x in paths.values()):
        raise ValueError("Model paths must be unique and steps must be 2, 4, or 8")
    klass = runtime.Backbone
    descriptor = klass.__dict__["load"]
    if not isinstance(descriptor, classmethod):
        raise TypeError("Expected the frozen Backbone class loader")
    current = {}
    old_preflight = implementation.preflight

    def load(cls, model_path, config=None, **kwargs):
        name = str(Path(model_path).resolve())
        if name not in paths or not isinstance(config, TTTConfig):
            raise ValueError("The model or configuration is not in the frozen holdout manifest")
        selected = paths[name]
        engine = descriptor.__get__(None, cls)(model_path,
            replace(config, steps=selected, max_answer_steps=selected, gain_head_mode="kv"), **kwargs)
        # Do not retain the previous model when the driver loads the next one.
        current.update(step=selected, engine=weakref.ref(engine))
        return engine

    def preflight(engine):
        if current.get("engine") is None or current["engine"]() is not engine:
            raise ValueError("Preflight is using a different loaded model")
        return checkpoint_preflight(engine, final_base_step=current["step"])

    with linked_long_rollout_runner(implementation, runtime, paired_controls, root=root,
                                    linked_selector=linked_selector, physical_controls=physical_controls):
        base_validate = implementation.validate

        def validate(result, entry):
            base_validate(result, entry)
            training = result["training"]
            expected = current.get("step")
            practices = result["self_study"]["verified"]
            if (expected not in (2, 4, 8) or training["gain_head_mode"] != "kv" or
                len(training["steps"]) != expected + practices or
                result["generated_token_ids"] != result["standalone_cartridge"]["generated_token_ids"]):
                raise ValueError("The frozen holdout step or final cartridge changed")

        klass.load = classmethod(load)
        implementation.preflight = preflight
        implementation.validate = validate
        try:
            yield
        finally:
            klass.load = descriptor
            implementation.preflight = old_preflight
            implementation.validate = base_validate
            current.clear()
