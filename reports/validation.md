# Honest validation on temporal cut-offs

Metric: **accuracy** -- one prediction and one target per user, so accuracy, precision, recall and F1 coincide.

Training is everything before the cut-off; the target is each user's first borrow at or after it.

Users per cut-off (warm / all): 2023-01-01 = 510/736, 2023-07-01 = 422/621, 2024-01-01 = 497/573, 2024-07-01 = 368/409.
A *cold* user has no borrow before the cut-off, so no model can know anything about them; they are excluded from the warm column used to rank models and included in the all-users column, which estimates the submission score.

| model | 2023-01-01 | 2023-07-01 | 2024-01-01 | 2024-07-01 | mean (warm) | mean (all) | min | max |
|---|---|---|---|---|---|---|---|---|
| profile60 + cf0.25 | 0.0176 | 0.0142 | 0.0362 | 0.0217 | **0.0225** | 0.0182 | 0.0142 | 0.0362 |
| profile60 + cf1.0 | 0.0176 | 0.0118 | 0.0362 | 0.0217 | **0.0219** | 0.0178 | 0.0118 | 0.0362 |
| profile tau=60d | 0.0196 | 0.0166 | 0.0282 | 0.0190 | **0.0208** | 0.0166 | 0.0166 | 0.0282 |
| profile tau=30d | 0.0196 | 0.0190 | 0.0221 | 0.0190 | **0.0199** | 0.0157 | 0.0190 | 0.0221 |
| most_recent | 0.0235 | 0.0166 | 0.0221 | 0.0163 | **0.0196** | 0.0154 | 0.0163 | 0.0235 |
| profile tau=365d | 0.0275 | 0.0118 | 0.0201 | 0.0190 | **0.0196** | 0.0154 | 0.0118 | 0.0275 |
| profile tau=180d | 0.0235 | 0.0118 | 0.0262 | 0.0163 | **0.0195** | 0.0154 | 0.0118 | 0.0262 |
| profile no decay | 0.0196 | 0.0071 | 0.0020 | 0.0082 | **0.0092** | 0.0069 | 0.0020 | 0.0196 |
| most_frequent | 0.0137 | 0.0047 | 0.0060 | 0.0082 | **0.0082** | 0.0063 | 0.0047 | 0.0137 |
| global_popularity | 0.0020 | 0.0000 | 0.0000 | 0.0082 | **0.0025** | 0.0022 | 0.0000 | 0.0082 |

The spread across cut-offs is as large as the differences between models. With 400-740 evaluable users per cut-off and an accuracy near 2%, one standard error is about 0.006: **only the recency signal is real**. Everything else, including collaborative filtering, is inside the noise.

For reference, ``submissions/submission.csv`` -- which predicts each user's most recent borrow and was reported as F1 0.0376 -- corresponds to the ``most_recent`` row here.
