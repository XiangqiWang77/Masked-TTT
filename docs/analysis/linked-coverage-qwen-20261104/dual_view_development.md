# Full and selected views: paired development outcomes

| Model / type | n | Whole | Selected zero | Teacher | Cartridge | Whole-only recovered | Selected-only retained |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen | 200 | 58.50 | 57.50 | 59.50 | 58.50 | 0 / 23 | 21 / 21 |
| qwen QA3 | 20 | 20.00 | 35.00 | 35.00 | 35.00 | 0 / 1 | 4 / 4 |
| gpt | 200 | 53.50 | 61.00 | 62.00 | 61.00 | 0 / 16 | 30 / 31 |
| gpt QA3 | 20 | 35.00 | 30.00 | 30.00 | 30.00 | 0 / 5 | 4 / 4 |

Whole-only and selected-only are determined by post-generation labels on the same case. A recovery means the final Cartridge answer was correct in a whole-only case; retention is defined analogously on selected-only cases. Teacher has the original generic formatting requirement, whereas whole, selected zero, and Cartridge use the same original question format.
Each 200-case model subset contains 50 source questions at four context lengths; these are development results and do not establish why a case changed.
