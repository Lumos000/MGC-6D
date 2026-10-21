"""Shared, paper-aligned utilities for multi-candidate query evaluation.

The helpers in this module intentionally avoid candidate-specific gates. A
candidate is selected only by the disclosed scalar score

    alpha * S_anchor_norm + beta * (-weighted_query_loss).

They also provide an optional appearance transfer from the RGB-conditioned
InstantMesh candidate to the other geometries.  The transfer changes vertex
colors only; geometry and anchor poses remain untouched.  This is useful for
FoundationPose, whose render-and-compare models are sensitive to a candidate
being untextured or uniformly gray.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import trimesh
from scipy.spatial import cKDTree


PAPER_CANDIDATE_LABELS = ("rayst3r", "instantmesh", "fused")


def candidate_mesh_diameter(
    vertices: np.ndarray,
    mode: str,
    *,
    sampled_diameter_fn=None,
) -> float:
    """Compute the disclosed FoundationPose crop diameter deterministically.

    ``bbox_diagonal`` is the validated v7 protocol. ``sampled_surface`` exists
    only for an explicit A/B run and delegates to FoundationPose's legacy
    sampled implementation.
    """

    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("Candidate vertices must be non-empty and finite")
    if mode == "bbox_diagonal":
        return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if mode == "sampled_surface":
        if sampled_diameter_fn is None:
            raise ValueError("sampled_surface requires sampled_diameter_fn")
        return float(sampled_diameter_fn(model_pts=points, n_sample=10000))
    raise ValueError(f"Unsupported mesh diameter mode: {mode}")


def parse_candidate_labels(raw: str) -> set[str] | None:
    """Parse the frozen HO3D candidate filter without changing its API."""

    labels = {part.strip() for part in str(raw).split(",") if part.strip()}
    return labels or None


def paper_registry_candidates(entries: Mapping[str, Mapping]) -> dict[str, dict]:
    """Return paper candidates in stable order and reject experiment suffixes.

    A missing geometry is allowed so one failed reconstruction does not abort an
    entire dataset run. Extra labels are rejected because they would silently
    reintroduce scale, appearance, or pose hypotheses into the formal protocol.
    """

    unexpected = sorted(set(entries) - set(PAPER_CANDIDATE_LABELS))
    if unexpected:
        raise ValueError(f"Unsupported candidate registry labels: {unexpected}")
    return {
        label: dict(entries[label])
        for label in PAPER_CANDIDATE_LABELS
        if label in entries
    }


def select_scored_candidate(candidates: list[dict]) -> dict:
    """Select the Eq. (9) argmax before any GT/evaluation fields are attached."""

    if not candidates:
        raise ValueError("Cannot select from an empty candidate list")
    forbidden = {
        key
        for candidate in candidates
        for key in candidate
        if key.startswith(("gt_", "audit_"))
    }
    if forbidden:
        raise AssertionError(
            f"Evaluation fields reached candidate selection: {sorted(forbidden)}"
        )
    return max(candidates, key=lambda row: float(row["total_score"]))


def _minmax_loss(values: Iterable[float], eps: float = 1e-8) -> list[float]:
    """Normalize a loss list to [0, 1], with lower remaining better.

    If every candidate has the same value, that term contains no selection
    information and contributes zero to every candidate.
    """

    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return []
    finite = np.isfinite(arr)
    if not finite.all():
        finite_values = arr[finite]
        replacement = float(finite_values.max() + 1.0) if finite_values.size else 1.0
        arr = np.where(finite, arr, replacement)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo <= eps:
        return [0.0] * int(arr.size)
    return [float(v) for v in ((arr - lo) / (hi - lo))]


def score_candidates(
    candidates: list[dict],
    *,
    w_depth: float,
    w_mask: float,
    w_geom: float,
    score_alpha: float,
    score_beta: float,
    w_appearance: float = 0.0,
    w_consensus: float = 0.0,
    loss_normalization: str = "minmax",
    depth_scale: float = 0.1,
    mask_scale: float = 1.0,
    geom_scale: float = 0.1,
    appearance_scale: float = 1.0,
) -> list[dict]:
    """Attach fully disclosed normalized weighted scores to candidates.

    ``S_anchor`` is divided by the largest non-negative anchor score.  Query
    losses are normalized either with disclosed fixed scales or by candidate-set
    min-max normalization, then combined using the user-visible weights.  The
    optional appearance loss measures RGB agreement, and the consensus term
    records agreement between independently registered geometry candidates. No
    default candidate, threshold replacement, hysteresis, or label-specific
    penalty is applied.
    """

    if not candidates:
        return candidates
    weights = np.asarray(
        [
            w_depth,
            w_mask,
            w_geom,
            w_appearance,
            w_consensus,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or (weights < 0).any() or float(weights.sum()) <= 0:
        raise ValueError("Query score weights must be finite, non-negative, and not all zero")
    if score_alpha < 0 or score_beta < 0 or score_alpha + score_beta <= 0:
        raise ValueError("score_alpha/score_beta must be non-negative and not both zero")

    if loss_normalization == "minmax":
        depth_norm = _minmax_loss(c["l_depth"] for c in candidates)
        mask_norm = _minmax_loss(c["l_mask"] for c in candidates)
        geom_norm = _minmax_loss(c["l_geom"] for c in candidates)
        appearance_norm = _minmax_loss(
            c.get("l_appearance", appearance_scale) for c in candidates
        )
    elif loss_normalization == "fixed_scale":
        scales = np.asarray(
            [depth_scale, mask_scale, geom_scale, appearance_scale],
            dtype=np.float64,
        )
        if not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("Fixed loss scales must be finite and positive")

        def _fixed(values: Iterable[float], scale: float) -> list[float]:
            arr = np.asarray(list(values), dtype=np.float64)
            arr = np.nan_to_num(arr, nan=scale, posinf=scale, neginf=0.0)
            return [float(v) for v in np.clip(arr / scale, 0.0, 1.0)]

        depth_norm = _fixed((c["l_depth"] for c in candidates), depth_scale)
        mask_norm = _fixed((c["l_mask"] for c in candidates), mask_scale)
        geom_norm = _fixed((c["l_geom"] for c in candidates), geom_scale)
        appearance_norm = _fixed(
            (c.get("l_appearance", appearance_scale) for c in candidates),
            appearance_scale,
        )
    else:
        raise ValueError(f"Unsupported loss normalization: {loss_normalization}")
    max_anchor = max(max(float(c.get("s_anchor", 0.0)), 0.0) for c in candidates)
    anchor_den = max(max_anchor, 1e-8)
    consensus = np.asarray(
        [float(c.get("pose_consensus", 1.0)) for c in candidates],
        dtype=np.float64,
    )
    consensus = np.clip(
        np.nan_to_num(consensus, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
    )
    for candidate, ld, lm, lg, la, sc in zip(
        candidates,
        depth_norm,
        mask_norm,
        geom_norm,
        appearance_norm,
        consensus,
    ):
        query_loss = float(
            w_depth * ld
            + w_mask * lm
            + w_geom * lg
            + w_appearance * la
        )
        s_query = float(-query_loss + w_consensus * sc)
        s_anchor_norm = max(float(candidate.get("s_anchor", 0.0)), 0.0) / anchor_den
        total_score = float(score_alpha * s_anchor_norm + score_beta * s_query)
        candidate.update(
            {
                "l_depth_norm": float(ld),
                "l_mask_norm": float(lm),
                "l_geom_norm": float(lg),
                "l_appearance_norm": float(la),
                "s_anchor_norm": float(s_anchor_norm),
                "s_query": float(s_query),
                "pose_consensus": float(sc),
                "total_score": total_score,
            }
        )
    return candidates


def derive_anchor_reliability_scores(
    candidates: list[dict],
    *,
    w_depth: float,
    w_mask: float,
    w_geom: float,
    depth_scale: float = 0.05,
    mask_scale: float = 0.5,
    geom_scale: float = 0.05,
    calibration_weight: float = 0.15,
) -> list[dict]:
    """Derive a bounded anchor reliability score without ground-truth pose.

    Anchor render/depth/geometry losses measure whether each candidate and its
    estimated anchor pose explain the observed anchor RGB-D.  The registry's
    calibration score is retained as a bounded secondary term, so a fused
    candidate cannot dominate solely because its raw ICP score is about three
    times the neutral base-candidate score.
    """

    if not candidates:
        return candidates
    weights = np.asarray([w_depth, w_mask, w_geom], dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Anchor loss weights must be finite, non-negative, and not all zero")
    weights = weights / weights.sum()
    scales = np.asarray([depth_scale, mask_scale, geom_scale], dtype=np.float64)
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Anchor loss scales must be finite and positive")
    if not np.isfinite(calibration_weight) or not 0.0 <= calibration_weight <= 1.0:
        raise ValueError("calibration_weight must lie in [0, 1]")

    raw_scores = np.asarray(
        [max(float(c.get("s_anchor", 0.0)), 0.0) for c in candidates],
        dtype=np.float64,
    )
    calibration_den = max(float(raw_scores.max()), 1e-8)
    for candidate, raw_score in zip(candidates, raw_scores):
        losses = np.asarray(
            [
                float(candidate.get("l_depth", depth_scale)),
                float(candidate.get("l_mask", mask_scale)),
                float(candidate.get("l_geom", geom_scale)),
            ],
            dtype=np.float64,
        )
        losses = np.nan_to_num(losses, nan=scales, posinf=scales, neginf=0.0)
        scaled_loss = float(np.dot(weights, np.clip(losses / scales, 0.0, 20.0)))
        observation_quality = float(np.exp(-scaled_loss))
        calibration_norm = float(raw_score / calibration_den)
        effective_score = float(
            (1.0 - calibration_weight) * observation_quality
            + calibration_weight * calibration_norm
        )
        candidate.update(
            {
                "s_anchor_raw": float(raw_score),
                "anchor_scaled_loss": scaled_loss,
                "anchor_observation_quality": observation_quality,
                "anchor_calibration_norm": calibration_norm,
                "s_anchor": effective_score,
            }
        )
    return candidates


def compute_anchor_reliability_components(
    candidate: Mapping,
    object_metadata: Mapping | None = None,
) -> dict:
    """Compute the shared six-part anchor reliability without GT.

    Older registries do not always contain every construction diagnostic.  A
    missing component is omitted from the weighted mean instead of being
    fabricated.  The output is therefore usable with existing v7 anchors and
    becomes richer when a newly generated registry contains scale/cycle data.
    """

    obj = object_metadata or {}
    depth_loss = max(float(candidate.get("l_depth", 1.0)), 0.0)
    mask_loss = float(np.clip(candidate.get("l_mask", 1.0), 0.0, 1.0))
    components: dict[str, float | None] = {
        "visible_depth": float(np.exp(-depth_loss / 0.05)),
        "silhouette": float(np.exp(-mask_loss / 0.50)),
        "rgbd_coverage": float(np.clip(1.0 - mask_loss, 0.0, 1.0)),
        "cycle_consistency": None,
        "scale_margin": None,
        "topology_preservation": None,
    }

    fusion = obj.get("fusion_metadata", obj.get("fusion_diagnostics", {})) or {}
    bidirectional = obj.get("bidirectional_metadata", obj.get("bidirectional", {})) or {}
    cycle = fusion.get("cycle_confidence", bidirectional.get("cycle_confidence"))
    if cycle is None:
        rotation = bidirectional.get("cycle_rotation_deg")
        translation = bidirectional.get("cycle_translation_normalized")
        if rotation is not None and translation is not None:
            cycle = np.exp(-max(float(rotation), 0.0) / 90.0 - max(float(translation), 0.0))
    if cycle is not None and np.isfinite(float(cycle)):
        components["cycle_consistency"] = float(np.clip(cycle, 0.0, 1.0))

    scale = candidate.get("scale_recovery", {}) or {}
    margins = []
    stack = [scale]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in {"objective_margin", "scale_objective_margin"}:
                    try:
                        number = float(child)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(number):
                        margins.append(max(number, 0.0))
                elif isinstance(child, Mapping):
                    stack.append(child)
    if margins:
        components["scale_margin"] = float(1.0 - np.exp(-min(margins) / 0.01))

    topology_flag = fusion.get("topology_preserved")
    if topology_flag is not None:
        components["topology_preservation"] = 1.0 if bool(topology_flag) else 0.0
    elif candidate.get("base_geometry") in {"instantmesh", "fused"}:
        # These candidates retain the InstantMesh backbone by construction.
        components["topology_preservation"] = 1.0

    fixed_weights = {
        "visible_depth": 0.20,
        "silhouette": 0.15,
        "rgbd_coverage": 0.20,
        "cycle_consistency": 0.15,
        "scale_margin": 0.15,
        "topology_preservation": 0.15,
    }
    observed = {
        key: float(value)
        for key, value in components.items()
        if value is not None and np.isfinite(float(value))
    }
    weight_sum = sum(fixed_weights[key] for key in observed)
    score = sum(fixed_weights[key] * value for key, value in observed.items()) / max(weight_sum, 1e-8)
    return {
        "version": "shared_rgbd_reliability_v1",
        "score": float(np.clip(score, 0.0, 1.0)),
        "low_reliability": bool(score < 0.35),
        "components": components,
    }


def _vertex_rgba(mesh: trimesh.Trimesh) -> np.ndarray | None:
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None
    if getattr(visual, "kind", None) == "texture":
        try:
            visual = visual.to_color()
        except Exception:
            return None
    if getattr(visual, "kind", None) != "vertex":
        return None
    colors = np.asarray(visual.vertex_colors)
    if colors.ndim != 2 or colors.shape[0] != len(mesh.vertices) or colors.shape[1] < 3:
        return None
    if colors.shape[1] == 3:
        colors = np.concatenate(
            [colors, np.full((len(colors), 1), 255, dtype=colors.dtype)], axis=1
        )
    return np.asarray(colors[:, :4], dtype=np.uint8)


def uniform_gray_mesh(mesh: trimesh.Trimesh, value: int = 102) -> trimesh.Trimesh:
    result = mesh.copy()
    rgba = np.tile(np.array([[value, value, value, 255]], dtype=np.uint8), (len(result.vertices), 1))
    result.visual = trimesh.visual.ColorVisuals(mesh=result, vertex_colors=rgba)
    return result


def transfer_prior_vertex_colors(
    target_mesh: trimesh.Trimesh,
    target_pose: np.ndarray,
    prior_mesh: trimesh.Trimesh,
    prior_pose: np.ndarray,
) -> trimesh.Trimesh:
    """Transfer InstantMesh vertex colors in a shared anchor-camera frame.

    The mapping is nearest-neighbour in the prior candidate's local geometry:
    target vertices are first mapped through their candidate-specific anchor
    pose and then through the inverse prior anchor pose.  This avoids assuming
    that independently reconstructed candidate coordinate systems coincide.
    """

    prior_colors = _vertex_rgba(prior_mesh)
    if prior_colors is None or len(prior_mesh.vertices) == 0 or len(target_mesh.vertices) == 0:
        return target_mesh.copy()

    target_pose = np.asarray(target_pose, dtype=np.float64).reshape(4, 4)
    prior_pose = np.asarray(prior_pose, dtype=np.float64).reshape(4, 4)
    prior_from_target = np.linalg.inv(prior_pose) @ target_pose
    target_h = np.concatenate(
        [np.asarray(target_mesh.vertices, dtype=np.float64), np.ones((len(target_mesh.vertices), 1))],
        axis=1,
    )
    target_in_prior = (prior_from_target @ target_h.T).T[:, :3]
    tree = cKDTree(np.asarray(prior_mesh.vertices, dtype=np.float64))
    _, indices = tree.query(target_in_prior, k=1, workers=-1)

    result = target_mesh.copy()
    result.visual = trimesh.visual.ColorVisuals(
        mesh=result, vertex_colors=prior_colors[np.asarray(indices, dtype=np.int64)]
    )
    return result


@dataclass(frozen=True)
class AppearanceSummary:
    mode: str
    source_label: str | None
    transferred_labels: tuple[str, ...]
    projected_labels: tuple[str, ...] = ()
    projected_vertex_counts: tuple[tuple[str, int], ...] = ()


def project_anchor_vertex_colors(
    mesh: trimesh.Trimesh,
    anchor_pose: np.ndarray,
    anchor_rgb: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_k: np.ndarray,
    anchor_depth: np.ndarray | None = None,
    depth_tolerance_m: float = 0.03,
) -> tuple[trimesh.Trimesh, int]:
    """Paint anchor-visible vertices from the observed RGB image.

    Visibility is checked against the anchor mask and, when available, the
    measured depth (a caller-disclosed metric tolerance). Existing colors are
    retained on unseen vertices, so this operation does not invent query-view
    information.
    """

    result = mesh.copy()
    if len(result.vertices) == 0:
        return result, 0
    rgb = np.asarray(anchor_rgb)
    mask = np.asarray(anchor_mask).astype(bool)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    if rgb.ndim != 3 or rgb.shape[:2] != mask.shape or rgb.shape[2] < 3:
        return result, 0

    pose = np.asarray(anchor_pose, dtype=np.float64).reshape(4, 4)
    k = np.asarray(anchor_k, dtype=np.float64).reshape(3, 3)
    vertices_h = np.concatenate(
        [np.asarray(result.vertices, dtype=np.float64), np.ones((len(result.vertices), 1))],
        axis=1,
    )
    vertices_cam = (pose @ vertices_h.T).T[:, :3]
    z = vertices_cam[:, 2]
    projected = (k @ vertices_cam.T).T
    safe_z = np.where(z > 1e-8, z, 1.0)
    u = np.rint(projected[:, 0] / safe_z).astype(np.int64)
    v = np.rint(projected[:, 1] / safe_z).astype(np.int64)
    h, w = mask.shape
    valid = (z > 1e-8) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size:
        valid[valid_idx] &= mask[v[valid_idx], u[valid_idx]]

    if anchor_depth is not None:
        depth = np.asarray(anchor_depth, dtype=np.float64)
        if depth.shape == mask.shape:
            valid_idx = np.flatnonzero(valid)
            if valid_idx.size:
                observed_z = depth[v[valid_idx], u[valid_idx]]
                depth_ok = (observed_z > 1e-6) & (
                    np.abs(z[valid_idx] - observed_z)
                    <= max(float(depth_tolerance_m), 0.0)
                )
                valid[valid_idx] &= depth_ok

    colors = _vertex_rgba(result)
    if colors is None:
        colors = np.tile(
            np.array([[102, 102, 102, 255]], dtype=np.uint8),
            (len(result.vertices), 1),
        )
    else:
        colors = colors.copy()
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size:
        colors[valid_idx, :3] = rgb[v[valid_idx], u[valid_idx], :3].astype(np.uint8)
    result.visual = trimesh.visual.ColorVisuals(mesh=result, vertex_colors=colors)
    return result, int(valid_idx.size)


def apply_candidate_appearance(
    meshes: Mapping[str, trimesh.Trimesh],
    anchor_poses: Mapping[str, np.ndarray],
    mode: str,
    *,
    anchor_rgb: np.ndarray | None = None,
    anchor_mask: np.ndarray | None = None,
    anchor_k: np.ndarray | None = None,
    anchor_depth: np.ndarray | None = None,
    anchor_depth_tolerance_m: float = 0.03,
) -> tuple[dict[str, trimesh.Trimesh], AppearanceSummary]:
    """Apply the selected, explicitly reportable candidate appearance mode."""

    mode = str(mode)
    if mode not in {
        "native",
        "prior_transfer",
        "anchor_prior_hybrid",
        "uniform_gray",
    }:
        raise ValueError(f"Unsupported candidate appearance mode: {mode}")
    output = {label: mesh.copy() for label, mesh in meshes.items()}
    if mode == "uniform_gray":
        output = {label: uniform_gray_mesh(mesh) for label, mesh in output.items()}
        return output, AppearanceSummary(mode, None, tuple(sorted(output)), (), ())
    if mode == "native":
        return output, AppearanceSummary(mode, None, (), (), ())

    source_label = "instantmesh" if "instantmesh" in output else None
    transferred = []
    if source_label is not None and source_label in anchor_poses:
        source_mesh = output[source_label]
        if _vertex_rgba(source_mesh) is not None:
            for label in sorted(output):
                if label == source_label or label not in anchor_poses:
                    continue
                output[label] = transfer_prior_vertex_colors(
                    output[label],
                    anchor_poses[label],
                    source_mesh,
                    anchor_poses[source_label],
                )
                transferred.append(label)
    projected = []
    projected_counts = []
    if mode == "anchor_prior_hybrid":
        if anchor_rgb is None or anchor_mask is None or anchor_k is None:
            raise ValueError(
                f"{mode} appearance requires anchor_rgb, anchor_mask, and anchor_k"
            )
        for label in sorted(output):
            if label not in anchor_poses:
                continue
            output[label], count = project_anchor_vertex_colors(
                output[label],
                anchor_poses[label],
                anchor_rgb,
                anchor_mask,
                anchor_k,
                anchor_depth,
                anchor_depth_tolerance_m,
            )
            if count > 0:
                projected.append(label)
            projected_counts.append((label, int(count)))
    return output, AppearanceSummary(
        mode,
        source_label,
        tuple(transferred),
        tuple(projected),
        tuple(projected_counts),
    )
