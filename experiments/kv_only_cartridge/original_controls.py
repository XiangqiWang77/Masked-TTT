"""Same-question SD controls followed by a physically self-contained cartridge."""
from contextlib import contextmanager

from experiments.anchored_short_verifier.paired_controls import original_question_controls
from masked_ttt.kv_only import kv_only_cartridge


@contextmanager
def kv_only_original_question_controls():
    """Measure zero/adapted answers and discard duplicate adapter tensors.

    The paired diagnostic uses the source selected cache to compare the same
    question and generation limit. The subsequent formal answer must come from
    a new materialized cartridge after the full, source, and training state have
    been released; its generated tokens must equal the adapted diagnostic.
    Both hooks are process-local and must not be used by concurrent requests.
    """
    with kv_only_cartridge():
        with original_question_controls():
            yield
