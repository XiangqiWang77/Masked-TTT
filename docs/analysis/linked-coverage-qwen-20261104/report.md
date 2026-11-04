# Linked selection: coverage and paired outcomes

Qwen3-4B development cases; labels are used only after generation.

| Support coverage | n | Score-only Cartridge | Linked zero | Linked Teacher | Linked Cartridge | New right / wrong | Δ pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 200 | 53.00 | 57.50 | 59.50 | 58.50 | 22 / 11 | 5.50 |
| newly_complete | 36 | 19.44 | 61.11 | 63.89 | 58.33 | 16 / 2 | 38.89 |
| lost_complete | 4 | 75.00 | 50.00 | 50.00 | 50.00 | 1 / 2 | -25.00 |
| both_complete | 121 | 69.42 | 64.46 | 66.94 | 66.94 | 2 / 5 | -2.48 |
| both_incomplete | 39 | 30.77 | 33.33 | 33.33 | 33.33 | 3 / 2 | 2.56 |
| ambiguous | 0 | — | — | — | — | 0 / 0 | — |

Unavailable or mismatched cases: 0.
Coverage is label-based and does not establish that the model used the facts. Groups are observational and were analyzed on development data.
