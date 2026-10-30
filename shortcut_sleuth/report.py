"""Small, dependency-free reports from measured per-seed evaluation files."""

import html
import json
from pathlib import Path

from .data import BACKGROUND_NAMES, SHAPES, save_json
from .metrics import aggregate_seeds


def build_report(run_root, output):
    files = sorted(Path(run_root).rglob("evaluation.json"))
    if not files:
        raise ValueError("No evaluation files found; train and evaluate first")
    rows = [json.loads(path.read_text()) for path in files]
    for key in ("protocol", "stage", "renderer", "experiment_sha256", "implementation_sha256", "test_manifest_sha256"):
        if len({row[key] for row in rows}) != 1:
            raise ValueError(f"Cannot pool incompatible {key} values")
    identities = [(r["architecture"], r["condition"], r["seed"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate architecture/condition/seed results")
    groups = {}
    for architecture, condition, _ in identities:
        name = f"{architecture}/{condition}"
        subset = sorted((r for r in rows if r["architecture"] == architecture and r["condition"] == condition), key=lambda r: r["seed"])
        groups[name] = {"seeds": [r["seed"] for r in subset], **aggregate_seeds([r["metrics"] for r in subset])}
    seed_sets = {tuple(g["seeds"]) for g in groups.values()}
    complete = rows[0]["stage"] == "study" and len(groups) == 4 and seed_sets == {(0, 1, 2)}
    output = Path(output)
    result = {"protocol": rows[0]["protocol"], "full_matrix_complete": complete, "groups": groups, "runs": rows}
    save_json(output / "metrics.json", result)
    lines = ["# Measured background intervention results", "", f"Protocol: `{rows[0]['protocol']}`.", "",
             "Full three-seed matrix complete." if complete else "Partial/development evidence; the full three-seed matrix is incomplete.", "",
             "All values below are measured from saved predictions. Accuracy is weighted over paired objects. Standard deviations use training seeds, not individual renders.", "",
             "| Model / training | Seeds | Training-like | Independent | Conflicting | Worst group | Flip rate |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    def formatted(stats, scale=100, unit="%"):
        suffix = f" ± {scale * stats['sample_std']:.2f}" if stats["sample_std"] is not None else " (one seed)"
        return f"{scale * stats['mean']:.2f}{unit}{suffix}"
    for name, group in groups.items():
        cells = [formatted(group["metrics"][key]) for key in ("training_like_accuracy", "independent_accuracy", "conflicting_accuracy", "worst_group_accuracy", "background_flip_rate")]
        lines.append(f"| {name} | {', '.join(map(str, group['seeds']))} | " + " | ".join(cells) + " |")
    lines += ["", "![Independent and conflicting accuracy](accuracy.svg)", "", "Per-seed metrics, all nine groups, true-class probability ranges, percentage-point gaps, hashes, parameter counts, runtime and software versions are retained in [metrics.json](metrics.json).",
              "", "Training-like always refers to the 95% correlated distribution, including for the independent control. Three background renders share one object. These results describe the tested configurations and do not establish a universal architecture ranking.", ""]
    lines += ["## Accuracy gaps and probability sensitivity", "",
              "Values are means ± sample standard deviation across training seeds. Gaps and probability ranges are percentage points (pp).", "",
              "| Model / training | Training-like − independent | Training-like − conflicting | True-class probability range |",
              "| --- | --- | --- | --- |"]
    for name, group in groups.items():
        gaps = [formatted(group["metrics"][key], scale=1, unit=" pp") for key in
                ("training_like_minus_independent_pp", "training_like_minus_conflicting_pp")]
        probability_range = formatted(group["metrics"]["true_class_probability_range"], unit=" pp")
        lines.append(f"| {name} | " + " | ".join([*gaps, probability_range]) + " |")
    lines += ["", "## Per-seed results", "",
              "| Model / training | Seed | Training-like | Independent | Conflicting | Worst group | Flip rate | Probability range |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        cells = [f"{100 * row['metrics'][key]:.2f}%" for key in
                 ("training_like_accuracy", "independent_accuracy", "conflicting_accuracy", "worst_group_accuracy", "background_flip_rate")]
        cells.append(f"{100 * row['metrics']['true_class_probability_range']:.2f} pp")
        lines.append(f"| {row['architecture']}/{row['condition']} | {row['seed']} | " + " | ".join(cells) + " |")
    lines += ["", "## All nine shape/background groups", "",
              "Each cell is a mean ± sample standard deviation across training seeds. Worst-group accuracy is computed within each seed before aggregation.", "",
              "| Shape / background | " + " | ".join(groups) + " |",
              "| --- | " + " | ".join("---" for _ in groups) + " |"]
    for label, shape in enumerate(SHAPES):
        for background, hue in enumerate(BACKGROUND_NAMES):
            cells = []
            for group in groups.values():
                stats = group["metrics"]["group_accuracy"]
                std = stats["sample_std"][label][background] if stats["sample_std"] is not None else None
                cells.append(formatted({"mean": stats["mean"][label][background], "sample_std": std}))
            lines.append(f"| {shape} / {hue} | " + " | ".join(cells) + " |")
    lines += ["", "## Training resources", "",
              "Runtime covers the training/validation epoch loop. Cache construction, checkpoint saving and paired evaluation are outside that timer. GPU memory is peak allocated tensor memory; it excludes other processes and driver context.", "",
              "| Model / training | Seed | Parameters | Runtime (s) | Peak GPU (MiB) | Device |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        parameters = f"{row['parameters']:,}" if "parameters" in row else "not recorded"
        runtime = f"{row['runtime_seconds']:.2f}" if "runtime_seconds" in row else "not recorded"
        peak = row.get("peak_gpu_memory_bytes")
        memory = f"{peak / 1024**2:.2f}" if peak is not None else "not allocated / not recorded"
        device = row.get("environment", {}).get("device", "not recorded")
        lines.append(f"| {row['architecture']}/{row['condition']} | {row['seed']} | {parameters} | {runtime} | {memory} | {device} |")
    lines += ["", "The full per-run hardware/software records, source commit, renderer/experiment hashes and checkpoint hashes remain in [metrics.json](metrics.json).", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    bars = []
    for row_index, (name, group) in enumerate(groups.items()):
        y = 56 + row_index * 72
        bars.append(f'<text x="20" y="{y + 15}" fill="#dae1eb">{html.escape(name)}</text>')
        for j, (metric, colour) in enumerate((("independent_accuracy", "#79d9bf"), ("conflicting_accuracy", "#d8afff"))):
            width = 360 * group["metrics"][metric]["mean"]
            bars.append(f'<rect x="190" y="{y + j * 23}" width="{width:.2f}" height="17" fill="{colour}"/><text x="{200 + width:.2f}" y="{y + j * 23 + 13}" fill="#dae1eb">{100 * width / 360:.1f}%</text>')
    height = 100 + len(groups) * 72
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="{height}" role="img" aria-label="Measured independent and conflicting accuracy"><rect width="100%" height="100%" fill="#151d29"/><g font-family="sans-serif" font-size="13"><text x="20" y="28" fill="#79d9bf">Independent accuracy</text><text x="240" y="28" fill="#d8afff">Conflicting accuracy</text>{"".join(bars)}<text x="20" y="{height - 15}" fill="#aab5c5">Bars show means across the available training seeds.</text></g></svg>'
    (output / "accuracy.svg").write_text(svg + "\n")
    return result
