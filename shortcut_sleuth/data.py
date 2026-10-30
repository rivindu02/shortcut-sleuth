"""Deterministic binary-mask rendering and auditable split manifests.

All shapes use the same analytic pixel-centre rasterisation. The area target is
drawn independently of class, and the enclosing radius is used to keep every
shape inside the frame. RGB foregrounds are assigned only after rasterisation.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

RENDERER_VERSION = "pixel-centres-v1"
SHAPES = ("circle", "square", "triangle")
BACKGROUND_NAMES = ("coral", "mint", "violet")
# Channel permutations have equal arithmetic brightness (120).
BACKGROUNDS = np.array(((184, 88, 88), (88, 184, 88), (88, 88, 184)), dtype=np.uint8)
FOREGROUND = np.array((240, 240, 230), dtype=np.uint8)
SIZE = 64


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    label: int
    x: float
    y: float
    area: float
    angle: float


def mask_for(spec):
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float64)
    xx, yy = xx + 0.5 - spec.x, yy + 0.5 - spec.y
    c, s = math.cos(spec.angle), math.sin(spec.angle)
    x, y = c * xx + s * yy, -s * xx + c * yy
    if spec.label == 0:
        return x * x + y * y <= spec.area / math.pi
    if spec.label == 1:
        half = math.sqrt(spec.area) / 2
        return (np.abs(x) <= half) & (np.abs(y) <= half)
    if spec.label != 2:
        raise ValueError("Shape label must be 0, 1 or 2")
    radius = math.sqrt(4 * spec.area / (3 * math.sqrt(3)))
    # Equilateral triangle, centred at its centroid.
    return (y >= -radius / 2) & (y <= radius - math.sqrt(3) * np.abs(x))


def render(spec, background):
    if not 0 <= background < 3:
        raise ValueError("Background must be 0, 1 or 2")
    mask = mask_for(spec)
    image = np.broadcast_to(BACKGROUNDS[background], (SIZE, SIZE, 3)).copy()
    image[mask] = FOREGROUND
    return image, mask


def foreground_hash(spec):
    # The foreground colour is constant, so the binary mask determines all
    # foreground pixels. Include renderer version and RGB value in the hash.
    return hashlib.sha256(RENDERER_VERSION.encode() + FOREGROUND.tobytes() + mask_for(spec).tobytes()).hexdigest()


def make_objects(size, split, seed=2026, seen=None):
    if size <= 0 or size % 3:
        raise ValueError("Split size must be positive and divisible by three")
    if split not in ("train", "validation", "test", "development", "tiny", "demo"):
        raise ValueError("Unknown split")
    # Hash-based stream identifiers remain stable across processes and Python versions.
    stream = int(hashlib.sha256(split.encode()).hexdigest()[:8], 16)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream])))
    seen = seen if seen is not None else set()
    objects = []
    for i in range(size):
        label = i % 3
        for _ in range(10000):
            area = float(rng.uniform(260, 480))
            # Shared conservative enclosing radius for all labels.
            radius = math.sqrt(4 * area / (3 * math.sqrt(3))) + 2
            spec = ObjectSpec(f"{split}-{i:06d}", label, float(rng.uniform(radius, SIZE - radius)),
                              float(rng.uniform(radius, SIZE - radius)), area, float(rng.uniform(0, 2 * math.pi)))
            fingerprint = foreground_hash(spec)
            if fingerprint not in seen:
                seen.add(fingerprint)
                objects.append(spec)
                break
        else:
            raise RuntimeError("Could not generate a unique foreground")
    return objects


def assignment(objects, condition, seed=2026):
    if condition not in ("correlated", "independent"):
        raise ValueError("Unknown training condition")
    rng = np.random.Generator(np.random.PCG64(seed))
    backgrounds = np.empty(len(objects), dtype=np.int64)
    table = np.zeros((3, 3), dtype=int)
    for label in range(3):
        indices = np.array([i for i, obj in enumerate(objects) if obj.label == label])
        count = len(indices)
        divisor = 40 if condition == "correlated" else 3
        if count == 0 or count % divisor:
            raise ValueError(f"Each class count must be divisible by {divisor} for {condition}")
        counts = [count // divisor] * 3 if condition == "correlated" else [count // 3] * 3
        if condition == "correlated":
            counts[label] = count * 38 // 40
        values = np.repeat(np.arange(3), counts)
        rng.shuffle(values)
        backgrounds[indices] = values
        table[label] = np.bincount(values, minlength=3)
    return backgrounds, table


def manifest(objects, condition, seed):
    backgrounds, table = assignment(objects, condition, seed)
    result = {"renderer": RENDERER_VERSION, "condition": condition, "assignment_seed": seed,
              "contingency": table.tolist(),
              "objects": [{**asdict(o), "foreground_hash": foreground_hash(o), "background": int(b)}
                          for o, b in zip(objects, backgrounds)]}
    return {**result, "sha256": digest(result)}


def load_manifest(path):
    value = json.loads(Path(path).read_text())
    body = {k: v for k, v in value.items() if k != "sha256"}
    if value["sha256"] != digest(body) or value["renderer"] != RENDERER_VERSION:
        raise ValueError("Manifest hash or renderer mismatch")
    objects = [ObjectSpec(**{k: row[k] for k in ObjectSpec.__dataclass_fields__}) for row in value["objects"]]
    if any(foreground_hash(o) != row["foreground_hash"] for o, row in zip(objects, value["objects"])):
        raise ValueError("Foreground hash mismatch")
    return value, objects


def prepare(directory, train_size=10800, validation_size=1800, test_size=3000, seed=2026,
            protocol="background-v1-development"):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Data directory must be empty; manifests are immutable")
    seen = set()
    summary = {"renderer": RENDERER_VERSION, "protocol": protocol, "seed": seed, "splits": {}}
    for split, size in (("train", train_size), ("validation", validation_size), ("test", test_size)):
        objects = make_objects(size, split, seed, seen)
        conditions = ("independent",) if split == "test" else ("correlated", "independent")
        for condition in conditions:
            value = manifest(objects, condition, seed)
            name = f"{split}-{condition}"
            save_json(directory / f"{name}.json", value)
            summary["splits"][name] = {"count": size, "sha256": value["sha256"], "contingency": value["contingency"]}
    summary["unique_foregrounds"] = len(seen)
    save_json(directory / "summary.json", summary)
    return summary


def example_grid(path, seed=2026):
    objects = make_objects(9, "demo", seed)
    grid = Image.new("RGB", (SIZE * 3, SIZE * 9))
    for row, obj in enumerate(objects):
        for background in range(3):
            grid.paste(Image.fromarray(render(obj, background)[0]), (SIZE * background, SIZE * row))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
