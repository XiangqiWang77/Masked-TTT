"""Optional study-only wording for a request-local, same-GPU comparison."""
from contextlib import contextmanager

from experiments.self_study_prompt.probe import variant_request


@contextmanager
def study_only_prompt():
    """Reframe only exercise generation; leave the main question unchanged.

    The original exercise parser and independent source verification remain in
    force. The hook is process-local and cannot run alongside other requests.
    """
    import masked_ttt.pipeline as pipeline

    original = pipeline.generate_self_study

    def generate(engine, teacher_reader, teacher, request, layout, selected):
        return original(engine, teacher_reader, teacher, variant_request(request), layout, selected)

    pipeline.generate_self_study = generate
    try:
        yield
    finally:
        pipeline.generate_self_study = original
