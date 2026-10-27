# Experiment protocol — proposed v1

Freeze this protocol and record its Git commit before the full experiment. Changes made after examining final test results require a new experiment version and fresh held-out examples.

## Question and hypothesis

Question: how sensitive are a small CNN and a small ViT to a background-colour shortcut under the same controlled training recipe?

Hypothesis: models trained with strongly correlated backgrounds will show worse performance when that correlation changes. Training with independent backgrounds should reduce this effect. There is no directional hypothesis that ViT must outperform CNN.

## Data

- RGB images, 64 × 64 pixels.
- Three equally represented labels: circle, square and triangle.
- A neutral light foreground with an explicit binary foreground mask; three distinct background hues with matched nominal brightness.
- Randomised size, position and orientation. Match foreground-area distributions across classes, and keep the complete foreground inside the image.
- Sample foreground appearance independently of background choice. Do not introduce class-specific borders, filenames, compression or rendering paths.
- Use the same renderer, geometry distributions and interpolation settings for every split and condition. Background assignment is the deliberate distribution change.
- Keep object specifications, foreground masks, seeds and a renderer-version identifier. Each record has a stable object identifier independent of the assigned background.

Proposed split sizes: 10,800 training objects, 1,800 validation objects and 3,000 final test objects, with equal class counts. The training and validation sizes allow exact allocation of both 95%/2.5%/2.5% and one-third background frequencies within every class. Train, validation and test use separate deterministic random streams and disjoint object specifications. Reject duplicate foreground renderings across splits by hashing foreground pixels and masks. Reuse the same foreground specifications across training conditions to isolate the background assignment.

### Training distributions

Fix a canonical shape-to-background mapping.

1. **Correlated:** the canonical background appears with probability 0.95; each other background appears with probability 0.025.
2. **Independent control:** each background appears with probability 1/3 for every shape.

Use fixed per-class allocation counts, not just an unchecked Bernoulli draw. Training labels remain shape labels. Store the realised class/background contingency table for each manifest. Validation uses the corresponding training distribution and independent objects.

## Models and training

Both models start with random weights; neither uses pretrained features.

| Model | Starting design |
| --- | --- |
| Small CNN | Four convolution blocks, starting widths 20/40/80/160, normalisation, GELU, pooling, global average pooling and a three-class head |
| Small ViT | 8 × 8 patches, embedding dimension 64, four encoder blocks, four attention heads, MLP dimension 128, learned position embeddings and a three-class head |

Measure the actual trainable parameter counts before freezing the configurations. Target counts within 20% of each other; adjust widths once during development if needed. Matching counts does not match FLOPs, inductive bias or tuning effort. Report parameter counts, runtime and peak allocated GPU memory separately.

Starting shared recipe: AdamW, learning rate 3e-4, weight decay 1e-4, batch size 128, 30 epochs, cross-entropy loss and a cosine learning-rate schedule. No background-randomising augmentation in the correlated condition. Keep preprocessing identical and fixed across models.

Validate this recipe during development, using training/validation only. If a model cannot learn the task, investigate and document the change before freezing the main experiment. The final test set is not used for hyperparameter choice.

Full matrix: 2 architectures × 2 training distributions × seeds 0, 1 and 2 = **12 runs**. Use the final checkpoint after the fixed epoch budget. For each seed, both architectures receive the same examples and data order. Save the resolved configuration, code commit, manifest hash, dependency versions, optimiser settings and training curves.

A small shared recipe is a bounded comparison, not an exhaustive architecture benchmark. Do not interpret one model being underfit as evidence that its architecture inherently depends on shortcuts.

## Paired background intervention

Render every held-out test object against all three backgrounds. Preserve foreground pixels, position, geometry and mask exactly. This produces 9,000 test images from 3,000 independent objects. Reuse these exact images for every model and seed.

Compute three distributions from those predictions:

- **Training-like accuracy:** per object, weight the canonical-background prediction by 0.95 and the other predictions by 0.025 each.
- **Independent-background accuracy:** weight all three backgrounds equally.
- **Conflicting-background accuracy:** weight only the two noncanonical backgrounds equally.

The training-like metric refers to the correlated distribution even when evaluating the independently trained control. Use explicit labels in the UI and report so readers do not confuse this with the control's training distribution.

Report:

1. The three accuracies above and their differences in percentage points.
2. Accuracy for each of the nine shape/background groups, plus worst-group accuracy.
3. Background flip rate: fraction of test objects whose predicted label changes across the three backgrounds.
4. Mean range of the true-class probability across the three backgrounds.
5. Mean and sample standard deviation across the three training seeds, with per-seed values visible.

The nine thousand rendered images are not nine thousand independent objects. Any uncertainty resampling must preserve the three renders belonging to each object. With three training seeds, avoid strong statistical claims.

### Controls

- **Colour-only rule:** predict the canonical shape associated with the observed background. Expected training-like accuracy is 95%, independent-background accuracy is 1/3, and conflicting-background accuracy is 0% under this protocol. This validates the evaluation weighting.
- **Independent-background training:** same foregrounds and recipe, changed background assignment. Compare both accuracy and background sensitivity.
- **Tiny balanced overfit:** verify both implementations can fit a tiny balanced training sample before launching the full matrix.

Background interventions establish sensitivity to this specified input change. They do not by themselves prove the complete internal mechanism or explain every failure.

## Optional second experiment: occlusion

Only after the background study works, add a seeded neutral occluder and report measured fractions of foreground pixels hidden. Keep the background intervention and occlusion interventions separate in the first analysis. Do not assume equal rectangular areas hide equal fractions of each shape.

## Evidence to publish

- Frozen configurations and renderer/split identifiers.
- Per-seed aggregate metrics and a concise report, including failures.
- Small figures and selected examples linked to their object identifiers.
- Training time, parameter count, peak memory and actual hardware/software versions.
- Verified commands to reproduce rendering, a smoke run, training, evaluation and the demo.

Large artifacts, environments and checkpoints stay out of Git. If checkpoints are later distributed, use a documented artifact location with hashes. Never fabricate benchmark results or a completed-feature screenshot.
