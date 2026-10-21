# Released experiments

## Main paper accuracy protocols

The main quantitative comparisons use one fixed-reference protocol for HO3D
and the official Oryon ordered-pair protocols for TOYL and REAL275.

| Dataset | Launcher | Protocol | Evaluation size |
|---|---|---|---:|
| HO3D | `scripts/run_ho3d_full.sh` | Fixed anchor per object, stride-10 queries | 2,008 valid queries |
| TOYL | `scripts/run_toyl_oryon_2k.sh` | Official Oryon list | 2,000 pairs / 890 references |
| REAL275 | `scripts/run_real275_oryon_2k.sh` | Official Oryon list | 2,000 pairs / 1,447 references |

Every Oryon reference constructs its own RaySt3R, InstantMesh, and fused
candidate pool. Both pair evaluators retain all 2,000 requested pairs in the
denominator and assign zero metrics to failures. The TOYL implementation splits
anchor construction into 21 resumable object batches; it still produces 890
independent reference reconstructions.

The complete environment-variable commands are in the root README. The
launchers use the same candidate labels, candidate appearance mode, Eq. (9)
query weights, and final argmax rule. Dataset-specific scale normalization and
mesh preprocessing remain explicit in the saved anchor command and metadata.

## Fixed-reference diagnostic analysis

The following launchers keep one reference observation per object:

| Dataset | Launcher | Default stride | Role in the paper |
|---|---|---:|---|
| TOYL | `scripts/run_toyl_full.sh` | 1 | Candidate-frequency and runtime analysis |
| REAL275 | `scripts/run_real275_full.sh` | 10 | Candidate-frequency and runtime analysis |

These fixed-reference TOYL and REAL275 runs cover 1,669 and 1,627 queries,
respectively. Their accuracy values must not be compared directly with methods
evaluated on Oryon's variable-reference 2K lists. HO3D uses the same fixed
anchor protocol for its main accuracy, candidate-frequency, and runtime
analyses.

## Anchor stage

Each prepared reference is processed with a deterministic reference-specific
seed. RaySt3R runs on the first GPU exposed by `ANCHOR_GPUS`; InstantMesh runs
on the second. The stage writes `anchor_run_meta.json`,
`candidate_registry.json`, candidate meshes and poses, alignment diagnostics,
and construction timing.

The native reconstruction stack can return status 139 during interpreter
teardown after outputs have been flushed. A launcher continues only when the
independent registry validator confirms that all required registry, mesh,
pose, calibration, and summary artifacts are complete. Other nonzero status or
incomplete output fails closed.

## Query stage

Every available candidate is registered independently. Selection uses the
stored anchor score and query depth, mask, and geometry losses. Ground truth is
opened only after selection for metric calculation and cannot affect the
winner. Run manifests retain the command, seed, protocol, candidate labels,
normalization scales, repository state, and runtime provenance.

## Saved evidence and validation

A completed query directory contains:

- `query_run_manifest.json`: command, seed, repository state, protocol,
  parameters, mask provenance, runtime, and completion state;
- `evaluation_coverage.json` or `summary.json`: attempted, successful, failed,
  and missing-GT accounting;
- candidate-score tables containing every Eq. (9) component and selected row;
- aggregate and per-instance metric tables;
- candidate-selection counts.

Validation is read-only unless `--output` is explicitly supplied:

```bash
python scripts/verify_ho3d_result.py HO3D_RESULT_DIR
python scripts/verify_oryon_toyl_result.py TOYL_2K_RESULT_DIR
python scripts/verify_oryon_real275_result.py REAL275_2K_RESULT_DIR

python scripts/verify_ho3d_result.py HO3D_RESULT_DIR \
  --output audit/ho3d_verification.json
```

The HO3D protocol samples 2,046 stride-10 frames. GT object poses are absent
for 38 sampled frames; these are recorded as missing-GT rather than inserted
into the score table. The remaining 2,008 queries each contain exactly one
selected candidate.
