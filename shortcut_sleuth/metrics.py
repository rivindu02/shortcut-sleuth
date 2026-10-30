"""Metrics on [object, background, class] probabilities, never unpaired rows."""

import numpy as np


def paired_metrics(probabilities, labels):
    p = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0 or p.shape != (len(labels), 3, 3):
        raise ValueError("Expected nonempty [objects, 3 backgrounds, 3 classes] probabilities")
    if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels > 2)):
        raise ValueError("Labels must be class indices")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(-1), 1, atol=1e-6):
        raise ValueError("Invalid class probabilities")
    if set(labels.tolist()) != {0, 1, 2}:
        raise ValueError("All three shapes must be represented")
    prediction = p.argmax(-1)
    correct = prediction == labels[:, None]
    canonical = np.arange(3)[None, :] == labels[:, None]
    training_like = float((correct * np.where(canonical, .95, .025)).sum(1).mean())
    independent = float(correct.mean())
    conflicting = float((correct * ~canonical).sum(1).mean() / 2)
    groups = np.stack([correct[labels == label].mean(0) for label in range(3)])
    true_probability = np.take_along_axis(p, labels[:, None, None], axis=2).squeeze(2)
    return {"objects": len(labels), "training_like_accuracy": training_like,
            "independent_accuracy": independent, "conflicting_accuracy": conflicting,
            "training_like_minus_independent_pp": 100 * (training_like - independent),
            "training_like_minus_conflicting_pp": 100 * (training_like - conflicting),
            "group_accuracy": groups.tolist(), "worst_group_accuracy": float(groups.min()),
            "background_flip_rate": float(np.any(prediction != prediction[:, :1], axis=1).mean()),
            "true_class_probability_range": float(np.ptp(true_probability, axis=1).mean())}


def colour_only_control(objects_per_class=1):
    if objects_per_class < 1:
        raise ValueError("At least one object per class is required")
    labels = np.repeat(np.arange(3), objects_per_class)
    return paired_metrics(np.tile(np.eye(3), (len(labels), 1, 1)), labels)


def aggregate_seeds(rows):
    if not rows:
        raise ValueError("No seed results supplied")
    result = {}
    for key, value in rows[0].items():
        if key == "objects":
            continue
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[key] = {"mean": values.mean(axis=0).tolist(),
                       "sample_std": values.std(axis=0, ddof=1).tolist() if len(rows) > 1 else None}
    return {"seed_count": len(rows), "metrics": result, "per_seed": rows}
