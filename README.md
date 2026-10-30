# Shortcut Sleuth

**Did your model learn the shape—or the background?**

A controlled CNN vs. Vision Transformer experiment with an interactive background-swap lab. Train both models from scratch, keep each foreground pixel fixed, and measure what happens when the background shortcut changes.

**Status:** the smoke matrix, both scratch models, checkpoint inference, all automated checks and real desktop/mobile browser behaviour have been verified. All four development learning gates passed; earlier failures and revisions are preserved. **The final 12-run study is pending protocol freeze and execution.** See the [verification report](docs/RESULTS.md).

<p align="center"><img src="results/browser-smoke/lab-desktop.png" width="1100" alt="Actual background-swap lab with CNN and ViT probabilities from four trained smoke checkpoints"></p>

*Actual browser capture using four smoke checkpoints. Two-epoch smoke results demonstrate the pipeline; they are not the primary architecture comparison. [Mobile capture](results/browser-smoke/lab-mobile.png) · [Paired renderer examples](results/paired-examples.png).*

## Features

- Versioned, deterministic 64 × 64 RGB rendering, exact background allocation and hash-checked split isolation.
- Small CNN and patch-based ViT, random initialisation, one shared training recipe and matched data orders.
- Correlated training (95% canonical backgrounds) and an independent-background control.
- Paired evaluation: weighted accuracy, nine-group accuracy, worst-group accuracy, flip rate and true-class probability range.
- Per-object predictions, per-seed reports, runtime, environment and checkpoint provenance.
- Local background lab with checkpoint selection, probabilities and paired sensitivity.

No architecture is assumed to win. Smoke results are development checks; only the frozen three-seed study supports the primary comparison.

## Install

Use Python 3.10+ and an isolated environment. Run commands from the repository root. Check free disk space before installing and current GPU use before explicitly selecting a CUDA device.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU installation; choose the appropriate PyTorch build for your own hardware.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

NumPy and Pillow suffice for data generation, analytic controls and the checkpoint-free interface. PyTorch is required for training and model inference. No pretrained features or external datasets are used. Node.js 22+ is needed only for interface tests.

## Try the lab

```bash
python -m shortcut_sleuth serve
```

Open **http://127.0.0.1:8000**. Without checkpoints, the page exposes exact background swaps and a clearly labelled colour-only rule. CNN and ViT cards explain that trained weights are missing.

To populate those cards:

```bash
python -m shortcut_sleuth smoke --output artifacts/smoke
python -m shortcut_sleuth serve --runs artifacts/smoke/runs
```

The smoke command prepares separate development objects, trains all four model/condition combinations for two epochs, evaluates final checkpoints and writes a Markdown report and SVG figure to `artifacts/smoke/report/`. The [executed smoke report](results/smoke/REPORT.md) retains all four measured outcomes, including the underfit independent controls. Two-epoch scores must not be presented as the main study.

## Verification

```bash
python scripts/verify.py --require-all --output artifacts/verification.json
python -m shortcut_sleuth overfit --output artifacts/overfit.json
```

Verification requires every Python and JavaScript check to execute. The overfit command requires both models to reach 98% accuracy on 27 balanced renders of nine training objects and parameter counts within 20%. This is a learning sanity check, not a generalisation result.

To check a supported subset when PyTorch or local sockets are unavailable, omit `--require-all`; skips remain visible in the JSON record. The [GitHub workflow](.github/workflows/ci.yml) requires all checks, builds the package, runs tiny overfit, executes the smoke pipeline and checks the real browser. Its definition is not evidence of a passing remote run.

For browser verification and reproducible screenshots:

```bash
python -m pip install -e '.[browser]'
python -m playwright install chromium
python scripts/browser_check.py --output artifacts/browser-control
python scripts/browser_check.py --runs artifacts/smoke/runs --output artifacts/browser-smoke
```

Data and controls can also be checked independently:

```bash
python -m shortcut_sleuth prepare --config configs/full.json --output artifacts/data-audit
python -m shortcut_sleuth audit --data artifacts/data-audit --output artifacts/data-audit.json
python -m shortcut_sleuth control
python -m shortcut_sleuth grid --output artifacts/paired-examples.png
```

Published evidence includes **15,600 unique foregrounds, zero cross-split duplicates and 31,200 pixel-exact foreground-swap comparisons**. The colour-only control yields:

| Training-like accuracy | Independent accuracy | Conflicting accuracy | Background flip rate |
| --- | --- | --- | --- |
| 95% | 33⅓% | 0% | 100% |

These validate the intervention and metrics; they are not trained-model results. See [data-audit.json](results/data-audit.json), [colour-only.json](results/colour-only.json) and [verification.json](results/verification.json).

## Full study

Verify the shared recipe on separate development training/validation objects first. These commands do not evaluate the final study test set:

```bash
python -m shortcut_sleuth prepare --config configs/development.json --output artifacts/development-data
python -m shortcut_sleuth matrix --config configs/development.json \
  --data artifacts/development-data --runs artifacts/development-runs
```

Inspect curves and failures. Both architectures and both controls must reach 80% validation accuracy with the shared recipe before freezing. This gate is specified before final-test evaluation. If it fails, investigate and revise the candidate configuration using development data only.

```bash
python -m shortcut_sleuth freeze --config configs/full.json \
  --preflight artifacts/overfit.json --validation-runs artifacts/development-runs \
  --output configs/frozen-v1.json
# Commit the frozen configuration before the main study.
git add configs/frozen-v1.json
git commit -m "experiment: freeze background study v1"

python -m shortcut_sleuth prepare --config configs/frozen-v1.json --output artifacts/full-data
python -m shortcut_sleuth audit --data artifacts/full-data --output artifacts/full-data-audit.json
python -m shortcut_sleuth matrix --config configs/frozen-v1.json \
  --data artifacts/full-data --runs artifacts/full-runs --evaluate
python -m shortcut_sleuth report --runs artifacts/full-runs --output results/background-v1
```

This runs 2 architectures × 2 conditions × 3 seeds = **12 runs**, with 60 epochs each. The original 30-epoch candidate underfit during development; the [protocol history](docs/EXPERIMENT_PROTOCOL.md#candidate-revision-history) explains the change. CPU is the default. After inspecting GPU availability, pass an explicit device such as `--device cuda:0` where appropriate. The project never automatically chooses a GPU. Full training caches about 600 MiB of deterministic input tensors in system RAM, in addition to model/runtime memory.

The final checkpoint is used without test-based selection. Repeating `matrix` resumes completed runs. Partial run directories require inspection or a new output root; existing datasets and evaluations are never silently overwritten. Implementation changes after freezing require a new version and freeze. Changes informed by final-test results also require fresh held-out objects.

Reports retain per-seed values and sample standard deviations; they reject incompatible experiments, mixed test manifests and duplicate seed results. Individual training and evaluation commands are listed by `python -m shortcut_sleuth --help`.

## Layout

```text
shortcut_sleuth/
  data.py             deterministic geometry, manifests and audit
  models.py           small CNN and ViT, both from scratch
  train.py            fixed-budget training, learning preflight and provenance
  inference.py        shared preprocessing and checkpoint predictions
  evaluate.py         paired evaluation and per-object records
  metrics.py          weighted distributions, groups and seed summaries
  report.py           Markdown/JSON reports and SVG charts
  server.py + web/    local interactive background lab
configs/              smoke, development and candidate full recipes
tests/                data, metrics, models, HTTP and interface checks
results/              small verified evidence; no checkpoints or raw manifests
```

Large artifacts, checkpoints and environments stay out of Git. No public deployment or checkpoint download is included. The server is intended for local use.

## Protocol and limitations

Read the [experiment protocol](docs/EXPERIMENT_PROTOCOL.md), [build plan](docs/PLAN.md) and [results and verification report](docs/RESULTS.md).

This study establishes sensitivity to three specified background colours on simple synthetic shapes. Equal parameter counts do not equalise compute or inductive bias. Three training seeds permit limited statistical claims. Background interventions do not identify the complete internal mechanism. Occlusion, attention visualisations and natural-image experiments are outside this release.

Background reading: Geirhos et al., [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780); Dosovitskiy et al., [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929).

## License

MIT. See [LICENSE](LICENSE).
