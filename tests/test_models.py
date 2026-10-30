"""Torch-dependent checks skip explicitly when the optional local runtime is absent."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(4)

    def test_counts_forward_and_backward(self):
        import torch
        from shortcut_sleuth.models import build_model, parameter_count
        counts = []
        for architecture in ("cnn", "vit"):
            with self.subTest(architecture=architecture):
                model = build_model(architecture)
                counts.append(parameter_count(model))
                logits = model(torch.rand(3, 3, 64, 64))
                self.assertEqual(logits.shape, (3, 3))
                loss = torch.nn.functional.cross_entropy(logits, torch.arange(3))
                loss.backward()
                self.assertTrue(torch.isfinite(loss).all())
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertLessEqual(max(counts) / min(counts), 1.2)

    def test_shared_inference_and_batching(self):
        from shortcut_sleuth.data import make_objects, render
        from shortcut_sleuth.inference import Predictor, preprocess
        from shortcut_sleuth.models import build_model
        objects = make_objects(3, "demo")
        images = np.stack([render(obj, b)[0] for obj in objects for b in range(3)])
        for architecture in ("cnn", "vit"):
            with self.subTest(architecture=architecture):
                predictor = Predictor(build_model(architecture))
                offline = predictor.objects(objects, batch_size=9)
                online = predictor.images(images, batch_size=1).reshape(3, 3, 3)
                np.testing.assert_allclose(offline, online, atol=1e-6)
                np.testing.assert_allclose(offline.sum(-1), 1, atol=1e-6)
                from shortcut_sleuth.server import Demo
                demo = Demo()
                demo.predictors["test"] = predictor
                obj = make_objects(3, "demo", 73)[0]
                np.testing.assert_allclose(demo.example(0, 73)["predictions"]["test"], predictor.objects([obj])[0], atol=1e-6)
        with self.assertRaises(ValueError):
            preprocess(images.astype(float))

    def test_scratch_initialisation_reproducibility(self):
        import torch
        from shortcut_sleuth.models import build_model
        from shortcut_sleuth.train import seed_everything
        for architecture in ("cnn", "vit"):
            seed_everything(7)
            first = build_model(architecture)
            seed_everything(7)
            second = build_model(architecture)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(first.parameters(), second.parameters())))
        vit = build_model("vit")
        self.assertFalse(torch.equal(vit.encoder.layers[0].linear1.weight, vit.encoder.layers[1].linear1.weight))

    def test_train_checkpoint_evaluate_roundtrip(self):
        from shortcut_sleuth.cli import read_config
        from shortcut_sleuth.data import prepare
        from shortcut_sleuth.evaluate import evaluate_checkpoint
        from shortcut_sleuth.inference import Predictor
        from shortcut_sleuth.train import train_run
        config = read_config("configs/smoke.json")
        config["epochs"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare(root / "data", 360, 360, 90, seed=73, protocol=config["protocol"])
            for architecture in ("cnn", "vit"):
                with self.subTest(architecture=architecture):
                    run = root / architecture
                    metadata = train_run(config, root / "data", run, architecture, "independent", 0)
                    self.assertEqual(metadata["checkpoint_selection"], "final epoch")
                    Predictor.load(run / "final.pt")
                    result = evaluate_checkpoint(run / "final.pt", root / "data")
                    self.assertEqual(result["metrics"]["objects"], 90)
                    records = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
                    from shortcut_sleuth.metrics import paired_metrics
                    recalculated = paired_metrics([r["probabilities_by_background"] for r in records], np.array([r["label"] for r in records]))
                    self.assertEqual(recalculated, result["metrics"])
                    with self.assertRaises(FileExistsError):
                        train_run(config, root / "data", run, architecture, "independent", 0)
                    with self.assertRaises(FileExistsError):
                        evaluate_checkpoint(run / "final.pt", root / "data")

    def test_cached_dataset_preserves_rendered_inputs(self):
        import torch
        from shortcut_sleuth.data import prepare, render
        from shortcut_sleuth.inference import preprocess
        from shortcut_sleuth.train import ShapeDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare(root, 360, 360, 90, seed=73)
            for condition in ("correlated", "independent"):
                dataset = ShapeDataset(root / f"train-{condition}.json")
                for i in (0, 1, 2, 119, 359):
                    image, label = dataset[i]
                    expected = preprocess(render(dataset.objects[i], dataset.backgrounds[i])[0][None])[0]
                    self.assertTrue(torch.equal(image, expected))
                    self.assertEqual(label, dataset.objects[i].label)


if __name__ == "__main__":
    unittest.main()
