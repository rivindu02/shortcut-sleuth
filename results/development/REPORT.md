# Development validation

All results use development data seed 73. The final study uses seed 2026. No final-study test predictions were used for these choices.

| Attempt | Train objects | Epochs | Batch | CNN correlated | ViT correlated | CNN independent | ViT independent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| initial-30-small | 1080 | 30 | 128 | 95.00% | 95.00% | 98.89% | 33.33% |
| batch32-small | 1080 | 30 | 32 | 95.00% | 95.00% | 100.00% | 33.33% |
| full-size-30 | 10800 | 30 | 128 | not run | not run | not run | 75.28% |
| selected-60 | 10800 | 60 | 128 | 100.00% | 95.11% | 100.00% | 93.28% |

The selected recipe passes all four 80% gates.

[validation.json](validation.json) retains the actual per-epoch curves, final validation values, recipe, source/manifest hashes, parameter counts, device and runtime for every completed attempt.

The separate tiny-overfit check originally failed at learning rate 0.003: CNN 100%, ViT 62.96%. At 0.0003 both models reached 100%. [Initial failure](overfit-initial.json), [selected preflight](overfit-selected.json).

Reducing batch size did not resolve the low-data ViT learning failure. Matching the primary training size improved it to 75.28% at 30 epochs; the shared schedule was then extended to 60 epochs before final-test evaluation. Both architectures keep their original designs. These validation attempts are development evidence and cannot establish a general architecture ranking.
