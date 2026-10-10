# Released experiments

The maintained release consists of one anchor-to-query launcher for each
dataset:

| Dataset | Launcher | Default stride | Anchor mask | Query mask |
|---|---|---:|---|---|
| HO3D | `scripts/run_ho3d_full.sh` | 10 | dataset GT | XMem prediction |
| TOYL | `scripts/run_toyl_full.sh` | 1 | dataset GT instance | dataset GT instance |
| REAL275 | `scripts/run_real275_full.sh` | 10 | dataset GT instance | dataset GT instance |

The complete environment-variable commands are in the repository README. All
three launchers use the same candidate labels, candidate appearance mode, and
Eq. (22) weights. Dataset-specific geometry preprocessing remains explicit in
the launcher so that a saved command can be compared directly with the code.

## Anchor stage

Each object folder is processed with a deterministic object-specific seed.
RaySt3R runs on the first GPU exposed by `ANCHOR_GPUS`; InstantMesh runs on the
second. The stage writes `anchor_run_meta.json`, `candidate_registry.json`, the
candidate meshes and poses, alignment metadata, and construction timing.

The native reconstruction stack may return status 139 while Python is tearing
down after every output has already been flushed. A launcher continues after
139 only when `scripts/verify_anchor_registry.py` independently confirms the
complete source-object set, registry, meshes, poses, calibration files, and
summary files. Any other nonzero status, incomplete output, or stale output
directory fails closed.

## Query stage

Every available candidate is registered independently. The selector uses only
the recorded anchor score and query depth, mask, and geometry losses. Ground
truth is read after selection for metric calculation and cannot affect the
winner. Failed frame instances remain represented by the dataset evaluator's
coverage accounting.

HO3D additionally provides `scripts/run_ho3d_query.sh` for reusing an existing
validated anchor registry. For TOYL or REAL275, set `RUN_ANCHOR=0` and point
`ANCHOR_OUT` to the existing registry before invoking the full launcher.

## Saved evidence

A completed query directory contains:

- `query_run_manifest.json`: command, seed, repository state, parameters, mask
  protocol, candidate fingerprints, runtime, and completion state;
- `evaluation_coverage.json`: attempted, evaluated, failed, and missing-GT
  accounting;
- `0_all_candidate_scores.xlsx`: Eq. (22) components and the selected row;
- aggregate and per-frame metric workbooks;
- candidate-selection counts.

The public validators replay the score and winner without pose estimation:

```bash
python scripts/verify_ho3d_result.py RESULT_DIR
python scripts/verify_paper_result.py --dataset toyl RESULT_DIR
python scripts/verify_paper_result.py --dataset real275 RESULT_DIR
```

The HO3D full protocol samples 2,046 stride-10 frames. Its metadata lacks an
object GT pose for 38 sampled frames, so those frames are reported as
missing-GT and are not inserted into the candidate score table. The remaining
2,008 evaluated queries each have exactly one selected candidate.
