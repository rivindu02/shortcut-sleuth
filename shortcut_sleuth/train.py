"""Fixed-budget training with a private sampler and explicit provenance."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import RENDERER_VERSION, digest, load_manifest, make_objects, render, save_json
from .inference import Predictor, preprocess
from .models import build_model, parameter_count


def implementation_hash():
    root = Path(__file__).parent
    return digest({name: (root / name).read_text() for name in ("data.py", "models.py", "train.py", "inference.py", "metrics.py", "evaluate.py")})


def seed_everything(seed, threads=2):
    if threads < 1:
        raise ValueError("Thread count must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


def code_state():
    # GIT_DIR/GIT_WORK_TREE may be set for an isolated checkout.
    def git(*args):
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment(device):
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
            "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "Pillow")},
            "device": str(device), "hardware": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else platform.processor() or "CPU",
            "cuda": torch.version.cuda}


def check_device(device):
    resolved = torch.device(device)
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu or an explicitly selected cuda device")
    if resolved.type == "cuda":
        if resolved.index is None:
            raise ValueError("Select an explicit device, e.g. cuda:0; inspect GPU usage first")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
    return resolved


class ShapeDataset(Dataset):
    def __init__(self, manifest_path):
        self.manifest, self.objects = load_manifest(manifest_path)
        self.backgrounds = [row["background"] for row in self.manifest["objects"]]

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, i):
        obj = self.objects[i]
        image = render(obj, self.backgrounds[i])[0]
        return preprocess(image[None])[0], obj.label


def train_run(config, data_dir, run_dir, architecture, condition, seed, device="cpu"):
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite run: {run_dir}")
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid training settings")
    if config["data_seed"] == 2026:
        if not config.get("frozen") or config.get("freeze", {}).get("implementation_sha256") != implementation_hash():
            raise ValueError("Freeze the current implementation after development checks before final-study training")
    device = check_device(device)
    seed_everything(seed, config["threads"])
    train = ShapeDataset(data_dir / f"train-{condition}.json")
    validation = ShapeDataset(data_dir / f"validation-{condition}.json")
    data_summary = json.loads((data_dir / "summary.json").read_text())
    if data_summary["protocol"] != config["protocol"] or data_summary["seed"] != config["data_seed"]:
        raise ValueError("Data protocol or seed does not match configuration")
    if len(train) != config["train_size"] or len(validation) != config["validation_size"]:
        raise ValueError("Data sizes do not match configuration")
    train_hashes = {row["foreground_hash"] for row in train.manifest["objects"]}
    if train_hashes & {row["foreground_hash"] for row in validation.manifest["objects"]}:
        raise ValueError("Foreground leakage between training and validation")
    sampler = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True, generator=sampler, num_workers=0)
    val_loader = DataLoader(validation, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    model = build_model(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    loss_fn = nn.CrossEntropyLoss()
    resolved = {**config, "architecture": architecture, "condition": condition, "seed": seed}
    metadata = {"schema_version": 1, "renderer": RENDERER_VERSION, "architecture": architecture,
                "condition": condition, "seed": seed, "config": resolved, "config_sha256": digest(resolved),
                "code": code_state(), "implementation_sha256": implementation_hash(),
                "environment": environment(device), "parameters": parameter_count(model),
                "train_manifest_sha256": train.manifest["sha256"], "validation_manifest_sha256": validation.manifest["sha256"],
                "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "preprocessing": "uint8 RGB / 255; NCHW",
                "data_order_seed": seed, "checkpoint_selection": "final epoch", "deterministic_algorithms": True}
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", metadata)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    curves = []
    for epoch in range(config["epochs"]):
        model.train()
        loss_sum, correct, total = 0., 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss is not finite")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += int((logits.argmax(1) == labels).sum().item())
            total += len(labels)
        model.eval()
        val_loss, val_correct = 0., 0
        with torch.inference_mode():
            for images, labels in val_loader:
                logits = model(images.to(device))
                labels = labels.to(device)
                val_loss += loss_fn(logits, labels).item() * len(labels)
                val_correct += int((logits.argmax(1) == labels).sum().item())
        row = {"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"],
               "train_loss": loss_sum / total, "train_accuracy": correct / total,
               "validation_loss": val_loss / len(validation), "validation_accuracy": val_correct / len(validation)}
        curves.append(row)
        print(f"{architecture}/{condition}/seed-{seed} epoch {epoch + 1}: loss={row['train_loss']:.4f} val={row['validation_accuracy']:.3f}", flush=True)
        save_json(run_dir / "curves.json", curves)
        scheduler.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metadata.update({"runtime_seconds": time.perf_counter() - start,
                     "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                     "final_validation_accuracy": curves[-1]["validation_accuracy"]})
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    temporary = run_dir / "final.pt.tmp"
    torch.save({"state_dict": state, "metadata": metadata}, temporary)
    temporary.replace(run_dir / "final.pt")
    save_json(run_dir / "metadata.json", metadata)
    return metadata


def tiny_overfit(output, steps=300, device="cpu", threads=2):
    if steps < 1:
        raise ValueError("Steps must be positive")
    device = check_device(device)
    objects = make_objects(9, "tiny", seed=907)
    images = np.stack([render(obj, b)[0] for obj in objects for b in range(3)])
    labels = torch.tensor([obj.label for obj in objects for _ in range(3)], device=device)
    x = preprocess(images).to(device)
    results = {}
    for architecture in ("cnn", "vit"):
        seed_everything(0, threads)
        model = build_model(architecture).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.0001)
        for _ in range(steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss in tiny overfit")
            loss.backward()
            optimizer.step()
        probabilities = Predictor(model, device).images(images)
        accuracy = float((probabilities.argmax(1) == labels.cpu().numpy()).mean())
        results[architecture] = {"accuracy": accuracy, "passed": accuracy >= .98,
                                 "parameters": parameter_count(model), "steps": steps}
        print(f"Tiny overfit {architecture}: {accuracy:.3f}", flush=True)
    ratio = max(r["parameters"] for r in results.values()) / min(r["parameters"] for r in results.values())
    evidence = {"passed": all(r["passed"] for r in results.values()) and ratio <= 1.2,
                "results": results, "parameter_ratio": ratio, "objects": 9, "renders": 27,
                "code": code_state(), "implementation_sha256": implementation_hash(),
                "environment": environment(device), "renderer": RENDERER_VERSION}
    save_json(output, evidence)
    return evidence
