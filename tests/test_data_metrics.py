import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from shortcut_sleuth.data import (BACKGROUNDS, FOREGROUND, assignment, foreground_hash,
                                 load_manifest, make_objects, mask_for, prepare, render)
from shortcut_sleuth.metrics import aggregate_seeds, colour_only_control, paired_metrics


class DataTests(unittest.TestCase):
    def test_determinism_and_stream_isolation(self):
        first = make_objects(120, "train")
        self.assertEqual(first, make_objects(120, "train"))
        validation = make_objects(120, "validation")
        self.assertFalse({foreground_hash(o) for o in first} & {foreground_hash(o) for o in validation})
        self.assertEqual(len({foreground_hash(o) for o in first}), 120)

    def test_foregrounds_and_area(self):
        objects = make_objects(600, "development")
        errors = []
        for obj in objects:
            images, masks = zip(*(render(obj, b) for b in range(3)))
            mask = masks[0]
            self.assertTrue(mask.any())
            self.assertFalse(mask[[0, -1]].any() or mask[:, [0, -1]].any())
            for b in range(3):
                np.testing.assert_array_equal(masks[b], mask)
                np.testing.assert_array_equal(images[b][mask], np.broadcast_to(FOREGROUND, (mask.sum(), 3)))
                np.testing.assert_array_equal(images[b][~mask], np.broadcast_to(BACKGROUNDS[b], ((~mask).sum(), 3)))
            errors.append(abs(mask.sum() - obj.area) / obj.area)
        self.assertLess(max(errors), .06)
        means = [np.mean([mask_for(o).sum() for o in objects if o.label == label]) for label in range(3)]
        self.assertLess(max(means) - min(means), 30)

    def test_exact_allocation_and_shared_foregrounds(self):
        objects = make_objects(360, "train")
        _, correlated = assignment(objects, "correlated")
        _, independent = assignment(objects, "independent")
        np.testing.assert_array_equal(correlated, [[114, 3, 3], [3, 114, 3], [3, 3, 114]])
        np.testing.assert_array_equal(independent, np.full((3, 3), 40))
        with self.assertRaises(ValueError):
            assignment(make_objects(9, "tiny"), "correlated")

    def test_manifests_integrity_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = prepare(tmp, 360, 360, 90)
            self.assertEqual(summary["unique_foregrounds"], 810)
            records, fingerprints = {}, set()
            for split in ("train", "validation", "test"):
                name = "test-paired" if split == "test" else f"{split}-independent"
                value, objects = load_manifest(Path(tmp) / f"{name}.json")
                current = {foreground_hash(o) for o in objects}
                self.assertFalse(current & fingerprints)
                fingerprints |= current
                records[split] = objects
            _, correlated = load_manifest(Path(tmp) / "train-correlated.json")
            self.assertEqual(records["train"], correlated)
            path = Path(tmp) / "train-correlated.json"
            value = json.loads(path.read_text())
            value["objects"][0]["background"] = 2
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "hash"):
                load_manifest(path)
            with self.assertRaises(FileExistsError):
                prepare(tmp, 360, 360, 90)
        seen = {foreground_hash(o) for o in make_objects(30, "tiny")}
        original = seen.copy()
        new = make_objects(30, "tiny", seen=seen)
        self.assertFalse(original & {foreground_hash(o) for o in new})


class MetricTests(unittest.TestCase):
    def test_colour_only_exact_control(self):
        m = colour_only_control(20)
        self.assertAlmostEqual(m["training_like_accuracy"], .95)
        self.assertAlmostEqual(m["independent_accuracy"], 1 / 3)
        self.assertEqual(m["conflicting_accuracy"], 0)
        self.assertEqual(m["background_flip_rate"], 1)
        self.assertEqual(m["true_class_probability_range"], 1)
        self.assertEqual(m["worst_group_accuracy"], 0)

    def test_perfect_shape_rule(self):
        labels = np.array([0, 1, 2])
        p = np.repeat(np.eye(3)[:, None, :], 3, axis=1)
        m = paired_metrics(p, labels)
        for name in ("training_like_accuracy", "independent_accuracy", "conflicting_accuracy", "worst_group_accuracy"):
            self.assertAlmostEqual(m[name], 1)
        self.assertEqual(m["background_flip_rate"], 0)
        self.assertEqual(m["true_class_probability_range"], 0)

    def test_object_weighting_and_true_class_range(self):
        p = np.array([[[.8, .1, .1], [.2, .7, .1], [.6, .3, .1]],
                      [[.1, .8, .1], [.1, .8, .1], [.1, .8, .1]],
                      [[.2, .1, .7], [.2, .1, .7], [.2, .1, .7]]])
        m = paired_metrics(p, np.arange(3))
        self.assertAlmostEqual(m["training_like_accuracy"], (0.975 + 2) / 3)
        self.assertAlmostEqual(m["independent_accuracy"], 8 / 9)
        self.assertAlmostEqual(m["conflicting_accuracy"], 5 / 6)
        self.assertAlmostEqual(m["true_class_probability_range"], .2)
        self.assertAlmostEqual(m["background_flip_rate"], 1 / 3)

    def test_invalid_probabilities_rejected(self):
        for bad in (np.zeros((3, 3, 3)), np.full((3, 3, 3), np.nan), np.ones((3, 3))):
            with self.assertRaises(ValueError):
                paired_metrics(bad, np.arange(3))
        with self.assertRaises(ValueError):
            paired_metrics(np.tile(np.eye(3), (3, 1, 1)), np.zeros(3, dtype=int))

    def test_sample_standard_deviation(self):
        first = colour_only_control()
        second = copy.deepcopy(first)
        second["independent_accuracy"] += .2
        result = aggregate_seeds([first, second])
        self.assertAlmostEqual(result["metrics"]["independent_accuracy"]["sample_std"], .2 / np.sqrt(2))
        self.assertIsNone(aggregate_seeds([first])["metrics"]["independent_accuracy"]["sample_std"])


if __name__ == "__main__":
    unittest.main()
