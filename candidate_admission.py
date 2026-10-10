"""Shared, GT-free admission rules for reconstruction candidates."""

from __future__ import annotations

from typing import Any, Mapping


def is_instantmesh_only(reconstruction_meta: Mapping[str, Any]) -> bool:
    if reconstruction_meta.get("align_direction") == "instantmesh_only_fallback":
        return True
    attempts = reconstruction_meta.get("depth_unit_candidates", {})
    states = [
        str(value.get("align", ""))
        for value in attempts.values()
        if isinstance(value, Mapping)
    ] if isinstance(attempts, Mapping) else []
    return bool(states) and all(
        state in {"rayst3r_failed", "rayst3r_empty"} for state in states
    )


def is_genuine_fusion(
    reconstruction_meta: Mapping[str, Any], *, instantmesh_only: bool | None = None
) -> bool:
    if instantmesh_only is None:
        instantmesh_only = is_instantmesh_only(reconstruction_meta)
    diagnostics = reconstruction_meta.get("fusion_diagnostics", {}) or {}
    return bool(
        not instantmesh_only
        and diagnostics.get("output_valid", False)
        and not diagnostics.get("fallback_used", False)
    )


__all__ = ["is_genuine_fusion", "is_instantmesh_only"]
