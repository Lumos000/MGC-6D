"""Lightweight repository smoke check.

Default checks intentionally avoid modules that initialize CUDA, load external
checkpoints, or require a RaySt3R checkout. Use the explicit commands in
docs/experiments.md for anchor/query runtime validation.
"""

from pathlib import Path
import importlib
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import setup_project_paths
setup_project_paths()

LIGHT_MODULES = [
    "project_paths",
    "metrics",
]

CLI_HELP = [
    "query_paper.py",
]


def main() -> None:
    for name in LIGHT_MODULES:
        importlib.import_module(name)
        print(f"ok import {name}")
    for script in CLI_HELP:
        subprocess.run(
            [sys.executable, script, "--help"],
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        print(f"ok help {script}")
    print("skipped anchor_paper.py --help by default; it requires RaySt3R import setup")


if __name__ == "__main__":
    main()
