# Build plan

## Outcome

Deliver a reproducible experiment and an interactive comparison of background shortcut learning in a small CNN and a small Vision Transformer. Report measured results without assuming either architecture will outperform the other.

**Status:** experiment design. Data generation, models, training, evaluation and the application are not implemented yet.

## Milestones

| Milestone | Deliverable | Done when |
| --- | --- | --- |
| 1. Deterministic data | Renderer, split manifests, example grid and colour-only baseline | Class/background frequencies match the protocol; splits are disjoint; background swaps preserve foreground pixels |
| 2. Working models | Small CNN and ViT, training loop and checkpoint metadata | Model shapes and parameter counts are verified; both models fit a tiny balanced sample; a short end-to-end run succeeds |
| 3. Controlled experiment | Two architectures × two training distributions × three seeds | All 12 runs use frozen configurations; evaluation produces auditable per-example predictions and per-seed summaries |
| 4. Interactive comparison | Side-by-side predictions, background controls and comparison charts | The application uses real checkpoint predictions, identifies the training condition and matches offline inference on fixed examples |
| 5. Results and reproduction | Results report, limitations, figures and verified commands | The pipeline can be reproduced; the README states measured findings and links to evidence |

## Proposed implementation

- Python, PyTorch, NumPy and Pillow for the experiment.
- Small CNN and ViT built from standard PyTorch layers and trained from scratch.
- Generated examples rendered on demand from versioned manifests and seeds.
- Shared inference code for offline evaluation and a minimal Python API.
- A lightweight HTML/CSS/JavaScript interface for background interventions and comparison charts.
- Resolved configurations, dependency versions, runtime and peak GPU memory recorded with each run.

## Intended code layout

```text
shortcut_sleuth/
  data.py           # object specifications, rendering and split manifests
  models.py         # small CNN and patch-based ViT
  train.py          # reproducible runs and checkpoint metadata
  evaluate.py       # paired interventions and aggregate metrics
  inference.py      # shared inference path for evaluation and application
  server.py         # inference API
configs/            # frozen experiment settings
tests/              # renderer, split, model and inference checks
web/                # interactive comparison
results/            # measured summaries and figures
docs/               # protocol and results report
```

This layout is planned; the code directories are not implemented yet.

## Essential checks

1. Renderer labels, correlation frequencies and split isolation.
2. Pixel-exact foreground preservation under background interventions.
3. Correct tensor/logit shapes and finite loss and gradients for both models.
4. Tiny-sample overfitting before the full experiment.
5. Colour-only baseline matches the protocol's expected metrics.
6. Application inference matches offline inference for identical inputs and checkpoints.
7. Metrics computed from per-example records match the reported summaries.

## Release limits

Background interventions and independent-background training form the primary experiment. Occlusion follows only after that comparison works. Natural-image datasets, pretrained models, saliency methods and architecture sweeps are possible extensions.

An inconclusive architecture comparison is a valid result. Report all three seeds and retain negative findings. Freeze the [experiment protocol](EXPERIMENT_PROTOCOL.md) before the full matrix; do not tune against the final test set.
