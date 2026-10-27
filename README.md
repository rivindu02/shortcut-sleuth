# Shortcut Sleuth

**Did your model learn the shape—or the background?**

Shortcut Sleuth is a controlled CNN vs. Vision Transformer experiment with an interactive background-swap demo. It investigates how models behave when a useful training shortcut stops working.

**Status:** experiment design. Models, training results and the interactive application are not implemented yet.

## The experiment

Train a small CNN and a small ViT to classify circles, squares and triangles. In the biased training set, each shape usually appears against one particular background colour. At evaluation time, keep the foreground pixels fixed and change only the background.

The central question is how much each model depends on that background. Neither architecture is assumed to win.

For comparison, train the same models on data where shape and background are independent. This tests whether changing the training distribution reduces the measured shortcut dependence.

## Planned experience

- Swap backgrounds and compare predictions side by side.
- Inspect class probabilities and background-induced prediction changes.
- Compare biased training against independent-background training.
- Explore measured accuracy across shape–background groups and three training seeds.
- Add controlled occlusion after the background intervention works.

## Scope

The first release uses generated 64 × 64 images, small models trained from scratch, and one GPU per run. No external dataset download or pretrained checkpoint is needed. Raw training data and model checkpoints stay outside Git history.

This is a small controlled study. Its results will describe the tested models and settings; they cannot establish that CNNs or Transformers are universally more robust. Attention visualisations, if added, will be illustrative rather than evidence of causal explanation.

## Project documents

- [Build plan and completion criteria](docs/PLAN.md)
- [Dataset, training and evaluation protocol](docs/EXPERIMENT_PROTOCOL.md)

Reproduction commands will accompany the implementation. No benchmark scores are claimed yet.

## Background reading

- Geirhos et al., [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780).
- Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

## License

MIT. See [LICENSE](LICENSE).
