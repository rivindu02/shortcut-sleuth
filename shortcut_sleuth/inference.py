"""One preprocessing and prediction path for training validation, evaluation and UI."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from .data import RENDERER_VERSION, render
from .models import build_model


def preprocess(images):
    images = np.asarray(images)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (64, 64, 3):
        raise ValueError("Expected uint8 RGB images with shape [batch, 64, 64, 3]")
    return torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).float().div(255)


class Predictor:
    def __init__(self, model, device="cpu", metadata=None):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        metadata = checkpoint["metadata"]
        if metadata["renderer"] != RENDERER_VERSION:
            raise ValueError("Checkpoint renderer version mismatch")
        model = build_model(metadata["architecture"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return cls(model, device, {**metadata, "checkpoint_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()})

    def images(self, images, batch_size=128):
        if batch_size <= 0 or len(images) == 0:
            raise ValueError("Nonempty images and positive batch size required")
        outputs = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                logits = self.model(preprocess(images[start:start + batch_size]).to(self.device))
                outputs.append(logits.softmax(-1).cpu().numpy())
        return np.concatenate(outputs)

    def objects(self, objects, batch_size=128):
        # Bound image allocation independently of the evaluation split size.
        outputs = []
        if not objects or batch_size <= 0:
            raise ValueError("Nonempty objects and positive batch size required")
        chunk = max(1, batch_size // 3)
        for start in range(0, len(objects), chunk):
            subset = objects[start:start + chunk]
            images = np.stack([render(obj, b)[0] for obj in subset for b in range(3)])
            outputs.append(self.images(images, batch_size).reshape(len(subset), 3, 3))
        return np.concatenate(outputs)
