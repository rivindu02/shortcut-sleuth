"""Paired evaluation of a saved final checkpoint; no fitting or selection."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import digest, load_manifest, save_json
from .metrics import paired_metrics


def evaluate_checkpoint(checkpoint, data_dir, output=None, device="cpu"):
    from .inference import Predictor

    checkpoint, data_dir = Path(checkpoint), Path(data_dir)
    output = Path(output) if output else checkpoint.parent
    if (output / "evaluation.json").exists():
        raise FileExistsError("Evaluation already exists; use a new output directory")
    predictor = Predictor.load(checkpoint, device)
    manifest, objects = load_manifest(data_dir / "test-paired.json")
    metadata = predictor.metadata
    for split in ("train", "validation"):
        value, _ = load_manifest(data_dir / f"{split}-{metadata['condition']}.json")
        if value["sha256"] != metadata[f"{split}_manifest_sha256"]:
            raise ValueError("Checkpoint was trained with different manifests")
        if {row["foreground_hash"] for row in value["objects"]} & {row["foreground_hash"] for row in manifest["objects"]}:
            raise ValueError("Final test foreground leakage")
    summary = json.loads((data_dir / "summary.json").read_text())
    if summary["protocol"] != metadata["config"]["protocol"] or len(objects) != metadata["config"]["test_size"]:
        raise ValueError("Test protocol or size mismatch")
    probabilities = predictor.objects(objects, metadata["config"]["batch_size"])
    metrics = paired_metrics(probabilities, np.array([o.label for o in objects]))
    output.mkdir(parents=True, exist_ok=True)
    prediction_file = output / "predictions.jsonl"
    with prediction_file.open("w") as stream:
        for obj, p in zip(objects, probabilities):
            stream.write(json.dumps({"object_id": obj.object_id, "label": obj.label,
                                     "probabilities_by_background": p.tolist()}, allow_nan=False) + "\n")
    experiment = {k: v for k, v in metadata["config"].items() if k not in ("architecture", "condition", "seed")}
    result = {"schema_version": 1, "architecture": metadata["architecture"], "condition": metadata["condition"],
              "seed": metadata["seed"], "protocol": metadata["config"]["protocol"],
              "stage": metadata["config"]["stage"], "renderer": metadata["renderer"],
              "experiment_sha256": digest(experiment), "implementation_sha256": metadata["implementation_sha256"],
              "test_manifest_sha256": manifest["sha256"],
              "checkpoint_sha256": metadata["checkpoint_sha256"],
              "predictions_sha256": hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
              "code": metadata["code"], "parameters": metadata["parameters"],
              "runtime_seconds": metadata["runtime_seconds"], "peak_gpu_memory_bytes": metadata["peak_gpu_memory_bytes"],
              "environment": metadata["environment"], "metrics": metrics}
    save_json(output / "evaluation.json", result)
    return result
