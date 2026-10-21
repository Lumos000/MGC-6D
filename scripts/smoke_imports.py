"""Lightweight repository smoke check.

Default checks intentionally avoid modules that initialize CUDA, load external
checkpoints, or require a RaySt3R checkout. Use the explicit commands in
docs/experiments.md for anchor/query runtime validation.
"""

from pathlib import Path
import argparse
import importlib
import os
import py_compile
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
    "query_paper_toyl.py",
    "run_real275_query.py",
]

FULL_CLI_HELP = CLI_HELP + [
    "anchor_paper.py",
    "anchor_paper_toyl.py",
    "anchor_paper_real275.py",
    "anchor_paper_toyl_oryon.py",
    "run_real275_oryon_query.py",
    "run_toyl_oryon_query.py",
]

FULL_MODULES = [
    "nvdiffrast.torch",
    "pytorch3d",
    "open3d",
    "trimesh",
    "openpyxl",
    "rembg",
    "pytorch_lightning",
    "sam2",
    "sam2._C",
    "foundationpose.Utils",
    "estimater",
    "construction",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.cuda and not args.full:
        parser.error("--cuda requires --full")
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
        rayst3r_root = os.environ.get("RAYST3R_ROOT")
        if rayst3r_root:
            if not Path(rayst3r_root).is_dir():
                raise RuntimeError(f"RAYST3R_ROOT is not a directory: {rayst3r_root}")
            importlib.import_module("eval_wrapper.eval")
            print("ok import RaySt3R eval_wrapper.eval")
        if args.cuda:
            import torch
            import nvdiffrast.torch as dr
            from pytorch3d.ops import knn_points

            if not torch.cuda.is_available():
                raise RuntimeError("--cuda requested, but PyTorch cannot access a CUDA device")
            device = torch.device("cuda")
            points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], device=device)
            distances = knn_points(points, points, K=1).dists
            if not torch.equal(distances, torch.zeros_like(distances)):
                raise RuntimeError("PyTorch3D CUDA KNN smoke check returned invalid distances")
            context = dr.RasterizeCudaContext(device=device)
            positions = torch.tensor(
                [[[-0.8, -0.8, 0.0, 1.0], [0.8, -0.8, 0.0, 1.0], [0.0, 0.8, 0.0, 1.0]]],
                device=device,
            )
            triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
            raster, _ = dr.rasterize(context, positions, triangles, resolution=[16, 16])
            if not bool((raster[..., 3] > 0).any()):
                raise RuntimeError("nvdiffrast CUDA rasterizer produced an empty image")
            print(f"ok CUDA kernels on {torch.cuda.get_device_name(device)}")
        for script in FULL_CLI_HELP:
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / script), "--help"],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            print(f"ok CLI help {script}")
    else:
        print("use --full for dependency, extension, and CLI checks; add --cuda for CUDA kernels")


if __name__ == "__main__":
    main()
