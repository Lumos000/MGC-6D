"""Project-local import path setup for MGC-6D entry points."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _prepend(path: Path | str | None) -> None:
    if path is None:
        return
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def setup_project_paths() -> Path:
    """Make vendored project modules and optional RaySt3R importable.

    The open-source repo keeps RaySt3R external. Set RAYST3R_ROOT to the
    RaySt3R checkout, or place a sibling checkout next to this repository.
    """
    _prepend(PROJECT_ROOT)
    _prepend(PROJECT_ROOT / 'bop_toolkit')

    rayst3r_root = os.environ.get('RAYST3R_ROOT')
    if rayst3r_root:
        _prepend(rayst3r_root)
    else:
        _prepend(PROJECT_ROOT.parent / 'rayst3r')

    return PROJECT_ROOT


__all__ = ['PROJECT_ROOT', 'setup_project_paths']
