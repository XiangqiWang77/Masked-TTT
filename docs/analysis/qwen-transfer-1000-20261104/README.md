# Qwen 1,000-question whole-to-cartridge audit

| Group | n | Whole | Adapted whole | Selected zero | ICL Teacher | Cartridge |
|---|---:|---:|---:|---:|---:|---:|
| All | 1000 | 54.40% | 56.50% | 56.00% | 56.60% | 56.70% |
| qa1 | 100 | 78.00% | 79.00% | 55.00% | 56.00% | 56.00% |
| qa2 | 100 | 34.00% | 39.00% | 26.00% | 24.00% | 27.00% |
| qa3 | 100 | 19.00% | 19.00% | 26.00% | 27.00% | 26.00% |
| qa4 | 100 | 64.00% | 71.00% | 74.00% | 74.00% | 74.00% |
| qa5 | 100 | 71.00% | 74.00% | 80.00% | 79.00% | 80.00% |
| qa6 | 100 | 76.00% | 78.00% | 86.00% | 84.00% | 84.00% |
| qa7 | 100 | 40.00% | 38.00% | 29.00% | 37.00% | 33.00% |
| qa8 | 100 | 43.00% | 43.00% | 48.00% | 49.00% | 51.00% |
| qa9 | 100 | 75.00% | 76.00% | 87.00% | 86.00% | 86.00% |
| qa10 | 100 | 44.00% | 48.00% | 49.00% | 50.00% | 50.00% |

## Paired changes

* Full Student: 33 new correct, 12 new wrong (+2.10 percentage points).
* Standalone Cartridge against zero-update selected KV: 15 new correct, 8 new wrong (+0.70 points).
* Whole correct / selected zero wrong: 130 cases; the adapted whole is correct on 119, and the Cartridge on 1.
* In those whole-only cases, annotated support was incomplete in 91 and complete in 39; support annotations are used only after generation.
* Selected zero correct / whole wrong: 146 cases; the Cartridge remains correct on 144.
* Of 33 new correct answers on the full path, selected zero was already correct and remained correct on 23; selected zero was wrong on 10, of which the Cartridge corrected 2.

These groups are defined using post-generation answer labels. They describe paired outcomes and do not by themselves establish the causal reason for an answer change. The 1,000 rows represent 250 source questions at four context lengths.

Whole, adapted whole, selected zero, and Cartridge use the original question format; the ICL Teacher additionally has the original generic response instruction. This format difference matters for the Teacher comparison. The score is the same task-aware loose lexical scorer for every arm, rather than an LLM semantic judge.

The report uses the completed original Decision top-10% selection, with no model or online API call. Adaptation uses selected ICL Teacher, full Student on-policy KL, and independent selected-cache materialization; no reverse transfer loss is added.
