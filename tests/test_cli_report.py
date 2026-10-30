import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from shortcut_sleuth.cli import main, read_config
from shortcut_sleuth.data import save_json
from shortcut_sleuth.metrics import colour_only_control
from shortcut_sleuth.report import build_report


class CliTests(unittest.TestCase):
    def test_config_and_control_output(self):
        config = read_config("configs/smoke.json")
        self.assertNotEqual(config["data_seed"], read_config("configs/full.json")["data_seed"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "control.json"
            with redirect_stdout(io.StringIO()):
                main(["control", "--output", str(path)])
            self.assertEqual(json.loads(path.read_text()), colour_only_control())
            bad = copy.deepcopy(config)
            bad["train_size"] = 100
            save_json(path, bad)
            with self.assertRaises(ValueError):
                read_config(path)

    def test_report_retains_individual_seeds_and_rejects_mixing(self):
        # Analytic fixtures live only in this temporary test directory.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (0, 1, 2):
                row = {"protocol": "fixture", "renderer": "fixture", "experiment_sha256": "a", "implementation_sha256": "fixture",
                       "test_manifest_sha256": "b", "architecture": "cnn", "condition": "correlated",
                       "seed": seed, "metrics": colour_only_control()}
                save_json(root / "runs" / str(seed) / "evaluation.json", row)
            result = build_report(root / "runs", root / "report")
            self.assertEqual(result["groups"]["cnn/correlated"]["seeds"], [0, 1, 2])
            self.assertFalse(result["full_matrix_complete"])
            self.assertTrue((root / "report" / "accuracy.svg").is_file())
            row["seed"] = 2
            save_json(root / "runs" / "duplicate" / "evaluation.json", row)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                build_report(root / "runs", root / "report")
            row["test_manifest_sha256"] = "other-test"
            save_json(root / "runs" / "duplicate" / "evaluation.json", row)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                build_report(root / "runs", root / "report")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "No evaluation"):
                build_report(tmp, Path(tmp) / "report")


if __name__ == "__main__":
    unittest.main()
