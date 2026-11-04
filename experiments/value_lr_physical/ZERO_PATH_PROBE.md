# Zero-correction native versus materialized readout

When the learned gain and Value correction are zero, the native selected-cache
read and the augmented/materialized read have the same mathematical attention
scores and Values. The actual kernels may operate on different padded head
dimensions, so floating-point results should be checked on the real hardware.

`measure_zero_path_parity` builds a separate local selected-cache reader,
zeros only that reader's parameters, and decodes the same original question
with the native path and the zero-correction augmented path. It records the
first-token distribution KL, logit differences, top-token agreement, and whether
the short generated sequences differ. It checks that the selected source KV is
unchanged and does not use references or the Teacher answer.

The Qwen3 and GPT-OSS miniature CPU tests passed with matching short outputs and
unchanged original state. The probe has not yet been run on the actual models
or GPUs, and no production training, prediction or active job is modified by
these checks. A real-model discrepancy would identify an implementation or
numerical control to investigate; a match would not by itself establish an
accuracy improvement.

[Probe](zero_path_probe.py) · [Tests](../../tests/test_zero_path_probe.py)
