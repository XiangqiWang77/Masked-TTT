# GPT-OSS-20B: 1,000-question whole-to-cartridge audit

| Group | n | Whole | Adapted whole | Selected zero | ICL Teacher | Cartridge |
|---|---:|---:|---:|---:|---:|---:|
| All | 1000 | 48.70% | 49.20% | 55.10% | 57.10% | 55.40% |
| qa1 | 100 | 58.00% | 58.00% | 56.00% | 56.00% | 55.00% |
| qa2 | 100 | 28.00% | 28.00% | 18.00% | 20.00% | 18.00% |
| qa3 | 100 | 23.00% | 23.00% | 23.00% | 22.00% | 25.00% |
| qa4 | 100 | 28.00% | 29.00% | 55.00% | 57.00% | 56.00% |
| qa5 | 100 | 71.00% | 73.00% | 78.00% | 81.00% | 77.00% |
| qa6 | 100 | 48.00% | 52.00% | 74.00% | 76.00% | 73.00% |
| qa7 | 100 | 68.00% | 68.00% | 66.00% | 67.00% | 67.00% |
| qa8 | 100 | 56.00% | 54.00% | 57.00% | 54.00% | 57.00% |
| qa9 | 100 | 73.00% | 73.00% | 86.00% | 88.00% | 86.00% |
| qa10 | 100 | 34.00% | 34.00% | 38.00% | 50.00% | 40.00% |

## Paired changes

* Full Student: 15 new correct, 10 new wrong (+0.50 percentage points).
* Standalone Cartridge against zero-update selected KV: 8 new correct, 5 new wrong (+0.30 points).
* Whole correct / selected zero wrong: 94 cases; the adapted whole is correct on 87, and the Cartridge on 1.
* In those whole-only cases, annotated support was incomplete in 65 and complete in 29; support annotations are used only after generation.
* Selected zero correct / whole wrong: 158 cases; the Cartridge remains correct on 156.
* Of 15 new correct answers on the full path, selected zero was already correct and remained correct on 13; selected zero was wrong on 2, of which the Cartridge corrected 0.

* With the Teacher's generic format instruction applied to both selected-cache controls, zero-update scores 57.10% and the trained Cartridge scores 57.50% (training change +0.40 points). For the trained Cartridge alone, changing only the question format changes accuracy by +2.10 points.

These groups are defined using post-generation answer labels. They describe paired outcomes and do not by themselves establish the causal reason for an answer change. The 1,000 rows represent 250 source questions at four context lengths.

Whole, adapted whole, selected zero, and Cartridge use the original question format; the ICL Teacher additionally has the original generic response instruction. This format difference matters for the Teacher comparison. The score is the same task-aware loose lexical scorer for every arm, rather than an LLM semantic judge.

The report uses the completed original Decision top-10% selection, with no model or online API call. Adaptation uses selected ICL Teacher, full Student on-policy KL, and independent selected-cache materialization; no reverse transfer loss is added. This historical paired diagnostic decoded the selected cache while retaining its source cache and training state, so it is a quality comparison, not a measurement of a fully released serving footprint. The full document KV was released first.
