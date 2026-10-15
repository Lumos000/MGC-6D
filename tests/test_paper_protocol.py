from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from candidate_selection import (
    PAPER_CANDIDATE_LABELS,
    apply_candidate_appearance,
    parse_candidate_labels,
    paper_registry_candidates,
    score_candidates,
    select_scored_candidate,
)
from scripts import verify_paper_result


ROOT = Path(__file__).resolve().parents[1]


def _load_standalone_function(filename: str, function_name: str):
    """Load a pure helper without importing a CUDA-heavy anchor module."""

    source = (ROOT / filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == function_name
    )
    namespace = {"np": np}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"),
        namespace,
    )
    return namespace[function_name]


def _row(label: str, anchor: float, depth: float, mask: float, geom: float) -> dict:
    return {
        "label": label,
        "s_anchor": anchor,
        "l_depth": depth,
        "l_mask": mask,
        "l_geom": geom,
    }


def test_registry_is_exactly_the_three_paper_families() -> None:
    entries = {label: {"mesh_path": f"{label}.obj"} for label in reversed(PAPER_CANDIDATE_LABELS)}
    filtered = paper_registry_candidates(entries)
    assert tuple(filtered) == PAPER_CANDIDATE_LABELS

    with pytest.raises(ValueError, match="Unsupported candidate registry labels"):
        paper_registry_candidates({**entries, "rayst3r@gray": {}})


def test_ho3d_candidate_parser_contract_is_preserved() -> None:
    assert parse_candidate_labels("rayst3r,instantmesh,fused") == {
        "rayst3r",
        "instantmesh",
        "fused",
    }
    assert parse_candidate_labels("") is None


def test_removed_projection_only_alias_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported candidate appearance mode"):
        apply_candidate_appearance({}, {}, "anchor_" + "projected")


def test_missing_candidate_is_allowed_without_fabricating_a_fallback() -> None:
    filtered = paper_registry_candidates(
        {"rayst3r": {"mesh_path": "ray.obj"}, "fused": {"mesh_path": "fused.obj"}}
    )
    assert tuple(filtered) == ("rayst3r", "fused")
    assert "instantmesh" not in filtered


@pytest.mark.parametrize(
    "filename",
    ("anchor_paper.py", "anchor_paper_toyl.py", "anchor_paper_real275.py"),
)
def test_anchor_rmse_threshold_is_a_soft_calibration_constraint(
    filename: str,
) -> None:
    score = _load_standalone_function(filename, "_compute_anchor_calibration_score")
    kwargs = {
        "label": "fused",
        "c_base": 0.8,
        "w_inlier": 1.0,
        "w_rmse": 1.0,
        "tau_inlier": 0.05,
        "tau_rmse": 0.02,
    }
    below = score(
        {"align_fitness": 0.8, "align_rmse": 0.0199},
        **kwargs,
    )
    above = score(
        {"align_fitness": 0.8, "align_rmse": 0.0201},
        **kwargs,
    )

    assert 0.0 < above < 0.8
    assert abs(above - below) < 0.01


def test_ho3d_fusion_admission_has_no_rmse_gate() -> None:
    source = (ROOT / "anchor_paper.py").read_text(encoding="utf-8")
    start = source.index("if align_valid and fused_path and os.path.exists(fused_path):")
    end = source.index("calibration_scores = {}", start)
    assert "align_rmse" not in source[start:end]


def test_icp_direction_selection_keeps_rmse_penalty() -> None:
    source = (ROOT / "construction.py").read_text(encoding="utf-8")
    assert "align_score_rmse_weight: float = 40.0" in source
    assert "align_high_rmse_penalty: float = 0.15" in source
    assert "score = fitness - cfg.align_score_rmse_weight * rmse" in source
    assert "if rmse > cfg.align_rmse_penalty_threshold:" in source
    assert "score -= cfg.align_high_rmse_penalty" in source


def test_eq22_fixed_scale_replay_and_argmax() -> None:
    rows = [
        _row("rayst3r", 0.8, 0.02, 0.20, 0.03),
        _row("instantmesh", 0.6, 0.04, 0.10, 0.05),
        _row("fused", 1.0, 0.01, 0.15, 0.02),
    ]
    score_candidates(
        rows,
        w_depth=0.50,
        w_mask=0.25,
        w_geom=0.25,
        score_alpha=0.40,
        score_beta=0.60,
        loss_normalization="fixed_scale",
        depth_scale=0.10,
        mask_scale=1.00,
        geom_scale=0.10,
    )
    for row in rows:
        replay = 0.40 * row["s_anchor_norm"] + 0.60 * (
            -0.50 * row["l_depth_norm"]
            -0.25 * row["l_mask_norm"]
            -0.25 * row["l_geom_norm"]
        )
        assert row["total_score"] == pytest.approx(replay, abs=1e-12)
    assert select_scored_candidate(rows)["label"] == "fused"


def test_selection_rejects_gt_or_audit_fields() -> None:
    rows = [_row("rayst3r", 1.0, 0.0, 0.0, 0.0)]
    score_candidates(
        rows,
        w_depth=0.5,
        w_mask=0.25,
        w_geom=0.25,
        score_alpha=0.4,
        score_beta=0.6,
        loss_normalization="fixed_scale",
    )
    rows[0]["gt_pose"] = object()
    with pytest.raises(AssertionError, match="Evaluation fields reached"):
        select_scored_candidate(rows)


def test_failed_frame_counter_counts_only_when_enabled() -> None:
    """Execute the small helper without importing the CUDA-heavy query module."""

    source = (ROOT / "query_paper_toyl.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_record_failed_frame"
    )
    module = ast.Module(body=[node], type_ignores=[])
    calls = []
    namespace = {"_append_zero_frame": lambda *args, **kwargs: calls.append((args, kwargs))}
    exec(compile(module, "<failed-frame-helper>", "exec"), namespace)
    record = namespace["_record_failed_frame"]

    assert record({}, [], "obj_01", 1, 7, 10, 3, False) == (10, 3)
    assert calls == []
    assert record({}, [], "obj_01", 1, 7, 10, 3, True) == (11, 4)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("filename", "selection", "evaluation"),
    [
        (
            "query_paper_toyl.py",
            'winner = select_scored_candidate(candidate_results)',
            "load_object_evaluation_resources()",
        ),
        (
            "run_real275_query.py",
            'winner = select_scored_candidate(candidate_results)',
            "evaluation_resources = _load_real275_evaluation_resources(",
        ),
    ],
)
def test_gt_resources_are_loaded_only_after_selection(
    filename: str, selection: str, evaluation: str
) -> None:
    source = (ROOT / filename).read_text(encoding="utf-8")
    selection_pos = source.index(selection)
    evaluation_pos = source.index(evaluation, selection_pos)
    assert selection_pos < evaluation_pos


@pytest.mark.parametrize(
    ("filename", "expected_calls"),
    [
        ("query_paper_toyl.py", 1),
        ("run_real275_query.py", 1),
    ],
)
def test_query_has_one_registration_call_site(filename: str, expected_calls: int) -> None:
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "register"
    ]
    assert len(calls) == expected_calls


def test_real275_anchor_has_no_scale_trial_layer() -> None:
    source = (ROOT / "anchor_paper_real275.py").read_text(encoding="utf-8")
    for forbidden in (
        "recon_results_by_scale",
        "variant_records",
        "scale_trials",
        "export_trials",
    ):
        assert forbidden not in source


@pytest.mark.parametrize("filename", ["query_paper_toyl.py", "run_real275_query.py"])
def test_dataset_query_protocol_options_are_not_cli_switches(filename: str) -> None:
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    cli_options = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
        ):
            continue
        try:
            option = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            continue
        if isinstance(option, str):
            cli_options.add(option)
    assert not cli_options.intersection(
        {
            "--candidate_color_mode",
            "--query_loss_normalization",
            "--w_depth",
            "--w_mask",
            "--w_geom",
            "--score_alpha",
            "--score_beta",
            "--oom_retry_iteration",
        }
    )


@pytest.mark.parametrize("filename", ["query_paper_toyl.py", "run_real275_query.py"])
def test_dataset_query_requires_explicit_anchor_registry(filename: str) -> None:
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    anchor_args = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--anchor_path"
    ]
    assert len(anchor_args) == 1
    required = next(
        (keyword.value for keyword in anchor_args[0].keywords if keyword.arg == "required"),
        None,
    )
    assert isinstance(required, ast.Constant) and required.value is True


def _cli_default(filename: str, option: str):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == option
        ):
            continue
        default = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "default"),
            None,
        )
        return ast.literal_eval(default)
    raise AssertionError(f"Missing CLI option {option} in {filename}")


def test_restored_protocol_defaults_are_explicit() -> None:
    assert _cli_default(
        "query_paper_toyl.py", "--anchor_color_depth_tolerance"
    ) == pytest.approx(0.03)
    assert _cli_default(
        "run_real275_query.py", "--mesh_diameter_mode"
    ) == "bbox_diagonal"
    for filename in ("query_paper_toyl.py", "run_real275_query.py", "query_paper.py"):
        source = (ROOT / filename).read_text(encoding="utf-8")
        assert "pose_trace_diagnostics" not in source
        assert "record_pose_trace" not in source


def test_formal_launchers_pin_restored_protocol() -> None:
    toyl = (ROOT / "scripts" / "run_toyl_full.sh").read_text(encoding="utf-8")
    real = (ROOT / "scripts" / "run_real275_full.sh").read_text(encoding="utf-8")
    assert "--anchor_color_depth_tolerance 0.03" in toyl
    assert "--mesh_diameter_mode bbox_diagonal" in real


@pytest.mark.parametrize(
    "filename",
    ["run_toyl_full.sh", "run_real275_full.sh", "run_ho3d_full.sh", "run_ho3d_query.sh"],
)
def test_formal_launchers_are_executable(filename: str) -> None:
    assert (ROOT / "scripts" / filename).stat().st_mode & 0o111


def test_result_verifier_replays_eq22_and_reports_metrics(tmp_path, monkeypatch) -> None:
    protocol = dict(verify_paper_result.PROTOCOLS["toyl"])
    protocol["frames"] = 1
    monkeypatch.setitem(verify_paper_result.PROTOCOLS, "toyl", protocol)

    manifest = {
        "dataset": "TOYL",
        "completed": True,
        "selection": {
            "score_alpha": 0.4,
            "score_beta": 0.6,
            "w_depth": 0.5,
            "w_mask": 0.25,
            "w_geom": 0.25,
        },
        "runtime": {"median_seconds": 8.0},
    }
    (tmp_path / "query_run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (tmp_path / "evaluation_coverage.json").write_text(
        json.dumps({"evaluated_frame_instances": 1}), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "Class_ID": "MEAN",
                "ADD-S": 100.0,
                "ADD": 100.0,
                "ADD(-S)": 100.0,
                "AR": 100.0,
                "MSSD": 100.0,
                "MSPD": 100.0,
                "VSD": 100.0,
            }
        ]
    ).to_excel(tmp_path / "0_mean_all_metrics_classes_results.xlsx", index=False)
    pd.DataFrame(
        [
            {
                "Frame_ID": 0,
                "Scene": "000001",
                "Image_ID": 0,
                "Anno_ID": 0,
                "Class": "obj_01",
                "Chosen_Candidate": "rayst3r",
            }
        ]
    ).to_excel(tmp_path / "0_all_frames_metrics_results.xlsx", index=False)
    rows = []
    for label, anchor, depth in (
        ("rayst3r", 1.0, 0.1),
        ("instantmesh", 0.8, 0.2),
        ("fused", 0.7, 0.3),
    ):
        score = 0.4 * anchor + 0.6 * (-0.5 * depth)
        rows.append(
            {
                "Scene": "000001",
                "Image_ID": 0,
                "Anno_ID": 0,
                "Class": "obj_01",
                "Candidate": label,
                "Selected": label == "rayst3r",
                "S_anchor_norm": anchor,
                "L_depth_norm": depth,
                "L_mask_norm": 0.0,
                "L_geom_norm": 0.0,
                "Total_Score": score,
            }
        )
    pd.DataFrame(rows).to_excel(tmp_path / "0_all_candidate_scores.xlsx", index=False)

    summary = verify_paper_result.verify("toyl", tmp_path)
    assert summary["winner_mismatches"] == 0
    assert summary["max_eq22_residual"] < 1e-12
