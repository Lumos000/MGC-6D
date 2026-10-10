"""Lightweight repository smoke check.

Default checks intentionally avoid modules that initialize CUDA, load external
checkpoints, or require a RaySt3R checkout. Use the explicit commands in
docs/experiments.md for anchor/query runtime validation.
"""

from pathlib import Path
import argparse
import importlib
import py_compile
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
    "query_paper_toyl.py",
    "run_real275_query.py",
]

FULL_MODULES = [
    "nvdiffrast.torch",
    "pytorch3d",
    "open3d",
    "trimesh",
    "sam2",
    "foundationpose.Utils",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    for name in LIGHT_MODULES:
        importlib.import_module(name)
        print(f"ok import {name}")
    for script in CLI_HELP:
        py_compile.compile(str(PROJECT_ROOT / script), doraise=True)
        print(f"ok compile {script}")
    if args.full:
        for name in FULL_MODULES:
            importlib.import_module(name)
            print(f"ok import {name}")
        from foundationpose.Utils import mycpp

        if mycpp is None or not hasattr(mycpp, "cluster_poses"):
            raise RuntimeError("FoundationPose mycpp.cluster_poses is unavailable")
        print("ok FoundationPose mycpp.cluster_poses")
    else:
        print("use --full to validate CUDA renderers and FoundationPose extensions")


if __name__ == "__main__":
    main()
