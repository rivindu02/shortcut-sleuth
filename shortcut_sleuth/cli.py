"""Command-line entry points. Data/control/server commands do not require torch."""

import argparse
import json
from pathlib import Path
import sys

from .data import audit, example_grid, prepare, save_json
from .metrics import colour_only_control


def read_config(path):
    config = json.loads(Path(path).read_text())
    required = ("protocol", "stage", "train_size", "validation_size", "test_size", "data_seed", "seeds",
                "architectures", "conditions", "epochs", "batch_size", "learning_rate", "weight_decay", "threads")
    if any(key not in config for key in required):
        raise ValueError("Configuration is missing required fields")
    if config["stage"] not in ("study", "development", "smoke"):
        raise ValueError("Stage must be study, development or smoke")
    if config["architectures"] != ["cnn", "vit"] or config["conditions"] != ["correlated", "independent"]:
        raise ValueError("Both architectures and training controls are required")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Training seeds must be nonempty and unique")
    for key in ("train_size", "validation_size"):
        if config[key] < 360 or config[key] % 360:
            raise ValueError(f"{key} must be a positive multiple of 360 for exact allocation")
    if config["test_size"] <= 0 or config["test_size"] % 3:
        raise ValueError("Test size must be a positive multiple of three")
    return config


def matrix(config, data, runs, device, evaluate=False):
    from .train import train_run

    # Candidate final-study configurations cannot silently become benchmarks.
    if config["stage"] == "study" and not config.get("frozen"):
        raise ValueError("Final-study configuration must be frozen after development checks; use freeze first")
    if config.get("frozen"):
        from .train import implementation_hash
        if config["freeze"]["implementation_sha256"] != implementation_hash():
            raise ValueError("Implementation changed after freeze; version the protocol and freeze again")
    results = []
    for seed in config["seeds"]:
        for condition in config["conditions"]:
            for architecture in config["architectures"]:
                directory = Path(runs) / f"{architecture}-{condition}-seed-{seed}"
                if not (directory / "final.pt").exists():
                    train_run(config, data, directory, architecture, condition, seed, device)
                else:
                    from .data import digest
                    metadata = json.loads((directory / "metadata.json").read_text())
                    resolved = {**config, "architecture": architecture, "condition": condition, "seed": seed}
                    if metadata["config_sha256"] != digest(resolved):
                        raise ValueError("Existing run has a different configuration")
                if evaluate:
                    from .evaluate import evaluate_checkpoint
                    if not (directory / "evaluation.json").exists():
                        results.append(evaluate_checkpoint(directory / "final.pt", data, device=device))
    return results


def freeze(config_path, preflight, validation_runs, output):
    from .train import code_state, implementation_hash

    config = read_config(config_path)
    if config["stage"] != "study":
        raise ValueError("Only study configurations can be frozen")
    evidence = json.loads(Path(preflight).read_text())
    current_hash = implementation_hash()
    if not evidence.get("passed") or evidence.get("implementation_sha256") != current_hash:
        raise ValueError("Tiny overfit must pass on the current implementation before freezing")
    metadata_files = sorted(Path(validation_runs).rglob("metadata.json"))
    results = [json.loads(path.read_text()) for path in metadata_files]
    eligible = []
    for result in results:
        recipe = result["config"]
        shared_recipe = all(recipe[key] == config[key] for key in ("epochs", "batch_size", "learning_rate", "weight_decay"))
        if (result.get("implementation_sha256") == current_hash and recipe["data_seed"] != config["data_seed"]
                and shared_recipe and result["final_validation_accuracy"] >= .8):
            eligible.append(result)
    if {(r["architecture"], r["condition"]) for r in eligible} != {(a, c) for a in config["architectures"] for c in config["conditions"]}:
        raise ValueError("Both architectures and controls must reach 80% validation accuracy on separate development data using the shared recipe")
    if Path(output).exists():
        raise FileExistsError("Frozen configuration already exists")
    config.update({"frozen": True, "status": "frozen", "protocol": config["protocol"].removesuffix("-development"),
                   "freeze": {"implementation_sha256": current_hash, "code": code_state(),
                              "tiny_overfit": evidence, "development_validation": [
                                  {key: r[key] for key in ("architecture", "condition", "seed", "final_validation_accuracy", "config_sha256")}
                                  for r in eligible]}})
    save_json(output, config)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Shortcut Sleuth: deterministic paired background experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Create immutable split manifests and reject foreground leakage")
    p.add_argument("--config", default="configs/smoke.json")
    p.add_argument("--output", required=True)
    p = sub.add_parser("audit", help="Check every manifest, split and paired foreground without fitting models")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("grid", help="Render paired examples")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=73)
    p = sub.add_parser("control", help="Validate the exact colour-only baseline")
    p.add_argument("--output")
    p = sub.add_parser("overfit", help="Require both scratch models to fit 27 balanced renders")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=.0003)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("train", help="Train one fixed-budget run without final-test access")
    p.add_argument("--config", default="configs/smoke.json")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--architecture", choices=("cnn", "vit"), required=True)
    p.add_argument("--condition", choices=("correlated", "independent"), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("matrix", help="Run all configured seeds, models and conditions; resume completed runs")
    p.add_argument("--config", default="configs/smoke.json")
    p.add_argument("--data", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--evaluate", action="store_true")
    p = sub.add_parser("smoke", help="Prepare, train, evaluate and report a separate development matrix")
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/smoke.json")
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("evaluate", help="Evaluate a final checkpoint on paired held-out objects")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output")
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("report", help="Aggregate compatible measured runs, retaining per-seed values")
    p.add_argument("--runs", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("freeze", help="Freeze the main protocol after tiny overfit and development validation")
    p.add_argument("--config", default="configs/full.json")
    p.add_argument("--preflight", required=True)
    p.add_argument("--validation-runs", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("serve", help="Start the local background intervention demo")
    p.add_argument("--runs", help="Root containing final.pt checkpoints (optional)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            config = read_config(args.config)
            print(json.dumps(prepare(args.output, config["train_size"], config["validation_size"], config["test_size"],
                                     config["data_seed"], config["protocol"]), indent=2))
        elif args.command == "audit":
            result = audit(args.data)
            save_json(args.output, result)
            print(f"Audit passed: {result['unique_foregrounds']} unique objects; {result['foreground_swap_comparisons']} exact foreground comparisons")
        elif args.command == "grid":
            example_grid(args.output, args.seed)
        elif args.command == "control":
            result = colour_only_control()
            if args.output:
                save_json(args.output, result)
            print(json.dumps(result, indent=2))
        elif args.command == "overfit":
            from .train import tiny_overfit
            if not tiny_overfit(args.output, args.steps, args.device, learning_rate=args.learning_rate)["passed"]:
                parser.exit(1, "Tiny balanced overfit failed; inspect evidence before full training.\n")
        elif args.command == "train":
            from .train import train_run
            config = read_config(args.config)
            if config["stage"] == "study" and not config.get("frozen"):
                raise ValueError("Freeze the final-study configuration before training")
            train_run(config, args.data, args.output, args.architecture, args.condition, args.seed, args.device)
        elif args.command == "matrix":
            matrix(read_config(args.config), args.data, args.runs, args.device, args.evaluate)
        elif args.command == "smoke":
            from .report import build_report
            config = read_config(args.config)
            if config["stage"] == "study" or config.get("frozen"):
                raise ValueError("Smoke runs must use separate development objects")
            root = Path(args.output)
            if not (root / "data" / "summary.json").exists():
                prepare(root / "data", config["train_size"], config["validation_size"], config["test_size"],
                        config["data_seed"], config["protocol"])
            matrix(config, root / "data", root / "runs", args.device, evaluate=True)
            build_report(root / "runs", root / "report")
        elif args.command == "evaluate":
            from .evaluate import evaluate_checkpoint
            print(json.dumps(evaluate_checkpoint(args.checkpoint, args.data, args.output, args.device), indent=2))
        elif args.command == "report":
            from .report import build_report
            build_report(args.runs, args.output)
        elif args.command == "freeze":
            freeze(args.config, args.preflight, args.validation_runs, args.output)
        elif args.command == "serve":
            from .server import serve
            serve(args.runs, args.host, args.port)
    except ModuleNotFoundError as error:
        parser.exit(2, f"Missing dependency: {error.name}. Install the project in an isolated environment (see README).\n")
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
