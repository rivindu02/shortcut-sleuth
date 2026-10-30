"""Small, dependency-free reports from measured per-seed evaluation files."""

import html
import json
from pathlib import Path

from .data import save_json
from .metrics import aggregate_seeds


def build_report(run_root, output):
    files = sorted(Path(run_root).rglob("evaluation.json"))
    if not files:
        raise ValueError("No evaluation files found; train and evaluate first")
    rows = [json.loads(path.read_text()) for path in files]
    for key in ("protocol", "renderer", "experiment_sha256", "implementation_sha256", "test_manifest_sha256"):
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
    complete = rows[0]["protocol"] == "background-v1" and len(groups) == 4 and seed_sets == {(0, 1, 2)}
    output = Path(output)
    result = {"protocol": rows[0]["protocol"], "full_matrix_complete": complete, "groups": groups, "runs": rows}
    save_json(output / "metrics.json", result)
    lines = ["# Measured background intervention results", "", f"Protocol: `{rows[0]['protocol']}`.", "",
             "Full three-seed matrix complete." if complete else "Partial/development evidence; the full three-seed matrix is incomplete.", "",
             "All values below are measured from saved predictions. Accuracy is weighted over paired objects. Standard deviations use training seeds, not individual renders.", "",
             "| Model / training | Seeds | Training-like | Independent | Conflicting | Worst group | Flip rate |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    def formatted(stats):
        suffix = f" ± {100 * stats['sample_std']:.2f}" if stats["sample_std"] is not None else " (one seed)"
        return f"{100 * stats['mean']:.2f}%{suffix}"
    for name, group in groups.items():
        cells = [formatted(group["metrics"][key]) for key in ("training_like_accuracy", "independent_accuracy", "conflicting_accuracy", "worst_group_accuracy", "background_flip_rate")]
        lines.append(f"| {name} | {', '.join(map(str, group['seeds']))} | " + " | ".join(cells) + " |")
    lines += ["", "![Independent and conflicting accuracy](accuracy.svg)", "", "Per-seed metrics, all nine groups, true-class probability ranges, percentage-point gaps, hashes, parameter counts, runtime and software versions are retained in [metrics.json](metrics.json).",
              "", "Training-like always refers to the 95% correlated distribution, including for the independent control. Three background renders share one object. These results describe the tested configurations and do not establish a universal architecture ranking.", ""]
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
