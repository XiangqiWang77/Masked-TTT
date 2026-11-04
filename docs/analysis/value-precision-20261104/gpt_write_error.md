# Selected Value write precision: fixed GPT-OSS development snapshot

181 paired cases; 50 source questions.

| Value LR | Accuracy | Components changed | Written relative norm | Squared write error | vs zero (new right / wrong) |
|---|---:|---:|---:|---:|---:|
| 0.0001 | 53.59 | 23.85% | 0.088% | 62.87% | 1 / 1 |
| 0.0003 | 54.70 | 54.69% | 0.333% | 22.38% | 2 / 0 |
| 0.0010 | 54.70 | 82.23% | 1.046% | 2.56% | 3 / 1 |

The error is the squared norm of the BF16-written correction minus the intended correction, divided by the squared norm of the intended correction, measured on sampled selected Value elements. Each row is a separate training learning rate, so accuracy differences cannot be attributed to rounding alone. Answers are scored only after generation; no new model or API call was used.
