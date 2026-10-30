"""Run checks and save an honest, compact evidence record (including skips)."""

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/verification.json")
    parser.add_argument("--require-all", action="store_true", help="Fail on any skipped or unavailable checks")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    js = {"executed": False, "passed": False}
    try:
        import os
        environment = dict(os.environ, PYTHON=sys.executable)
        completed = subprocess.run(["node", "--test-reporter=tap", "tests/web.test.cjs"], cwd=root,
                                   env=environment, text=True, capture_output=True)
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        count = re.search(r"# tests (\d+)", completed.stdout)
        js = {"executed": True, "passed": completed.returncode == 0,
              "tests": int(count.group(1)) if count else None}
    except FileNotFoundError:
        print("JavaScript checks unavailable: Node.js is not installed", file=sys.stderr)
    record = {"python": platform.python_version(),
              "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "Pillow")},
              "torch_available": importlib.util.find_spec("torch") is not None,
              "python_checks": {"run": result.testsRun, "passed": result.testsRun - len(result.skipped) - len(result.failures) - len(result.errors),
                                "failures": len(result.failures), "errors": len(result.errors),
                                "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped]},
              "javascript_checks": js,
              "all_required_checks_executed": not result.skipped and js["executed"]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    success = result.wasSuccessful() and (js["passed"] if js["executed"] else not args.require_all)
    if args.require_all and not record["all_required_checks_executed"]:
        success = False
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
