"""Shared, inference-only controls for the three released dataset pipelines.

This module deliberately contains no trainable component.  It centralizes the
public query CLI, validation, and provenance fields used by all three datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


PAPER_CANDIDATES = ("rayst3r", "instantmesh", "fused")
CANDIDATE_COLOR_MODES = (
    "native",
    "prior_transfer",
    "anchor_prior_hybrid",
    "uniform_gray",
)


def icp_direction_passes(
    fitness: float,
    rmse: float,
    *,
    tau_f: float,
    tau_r: float,
) -> bool:
    """Return whether one independently optimized ICP direction is valid.

    This is the reviewer-facing strict criterion: fitness must be at least
    ``tau_f`` and RMSE must be at most ``tau_r``.  Non-finite measurements
    always fail closed.
    """

    values = np.asarray([fitness, rmse, tau_f, tau_r], dtype=np.float64)
    if not np.isfinite(values).all():
        return False
    if tau_f < 0.0 or tau_r < 0.0:
        raise ValueError("ICP Fitness/RMSE thresholds must be non-negative")
    return bool(fitness >= tau_f and rmse <= tau_r)


def strict_both_icp_passes(
    inst_to_ray_fitness: float,
    inst_to_ray_rmse: float,
    ray_to_inst_fitness: float,
    ray_to_inst_rmse: float,
    *,
    tau_f: float,
    tau_r: float,
) -> bool:
    """Require both independent ICP directions to satisfy the thresholds."""

    return icp_direction_passes(
        inst_to_ray_fitness,
        inst_to_ray_rmse,
        tau_f=tau_f,
        tau_r=tau_r,
    ) and icp_direction_passes(
        ray_to_inst_fitness,
        ray_to_inst_rmse,
        tau_f=tau_f,
        tau_r=tau_r,
    )


def add_query_protocol_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the common Eq. (22) and candidate-pool arguments."""

    parser.add_argument(
        "--candidate_labels",
        type=str,
        default=",".join(PAPER_CANDIDATES),
        help="Comma-separated subset of rayst3r,instantmesh,fused.",
    )
    parser.add_argument(
        "--candidate_color_mode",
        choices=CANDIDATE_COLOR_MODES,
        default="anchor_prior_hybrid",
        help="Appearance treatment shared by all candidates.",
    )
    parser.add_argument("--w_depth", type=float, default=0.50)
    parser.add_argument("--w_mask", type=float, default=0.25)
    parser.add_argument("--w_geom", type=float, default=0.25)
    parser.add_argument("--score_alpha", type=float, default=0.40)
    parser.add_argument("--score_beta", type=float, default=0.60)


def parse_candidate_labels(raw: str) -> set[str]:
    labels = {value.strip() for value in str(raw).split(",") if value.strip()}
    unknown = labels.difference(PAPER_CANDIDATES)
    if unknown:
        raise ValueError(f"Unknown candidate label(s): {sorted(unknown)}")
    if not labels:
        raise ValueError("At least one candidate label is required")
    return labels


def validate_query_protocol(args: argparse.Namespace) -> set[str]:
    weights = np.asarray(
        [args.w_depth, args.w_mask, args.w_geom, args.score_alpha, args.score_beta],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("Eq. (22) weights must be finite and non-negative")
    if args.w_depth + args.w_mask + args.w_geom <= 0.0:
        raise ValueError("At least one query-loss weight must be positive")
    if args.score_alpha + args.score_beta <= 0.0:
        raise ValueError("--score_alpha and --score_beta cannot both be zero")
    return parse_candidate_labels(args.candidate_labels)


def _git_value(repo_root: Path, *command: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *command],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def runtime_provenance(repo_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Return compact provenance without mutating the repository."""

    root = Path(repo_root).resolve()
    status = _git_value(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
    )
    diff = _git_value(root, "diff", "--no-ext-diff", "--binary", "--", ".")
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda)
        cuda_available = bool(torch.cuda.is_available())
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ] if cuda_available else []
    except Exception:
        torch_version = None
        cuda_version = None
        cuda_available = False
        gpu_names = []
    return {
        "schema": "mgc6d-runtime-provenance-v1",
        "command": [str(value) for value in sys.argv],
        "working_directory": os.getcwd(),
        "repo_root": str(root),
        "git_head": _git_value(root, "rev-parse", "HEAD"),
        # ``git diff`` does not include untracked files.  Use porcelain status
        # for the clean/dirty decision so a formal run cannot silently claim a
        # clean commit while untracked experiment code is present.
        "git_dirty": bool(status),
        "git_status_sha256": (
            hashlib.sha256(status.encode("utf-8")).hexdigest()
            if status
            else None
        ),
        "git_diff_sha256": (
            hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None
        ),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_version,
        "torch_cuda": cuda_version,
        "cuda_available": cuda_available,
        "visible_gpus": gpu_names,
    }


def write_json_atomic(path: str | os.PathLike[str], payload: Any) -> None:
    """Write JSON without leaving a valid-looking partial file."""

    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
