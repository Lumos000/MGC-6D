# Official Oryon 2K main evaluation

The main TOYL and REAL275 paper results use the exact ordered reference-query
lists released by Oryon. These are the paper's primary accuracy protocols for
the two datasets, not auxiliary fixed-reference sweeps.

## Protocol contract

| Dataset | Pair-list SHA-256 | Pairs | Unique reference reconstructions |
|---|---|---:|---:|
| TOYL | `99eb3a67edb126bb6b3787482b3092acb684debac828c7ee476c7e24e56a5dea` | 2,000 | 890 |
| REAL275 | `1553e4ecfe1a3a3be1d55e6c5da157e7aa199dd80246d7b0501da4fefdbf7424` | 2,000 | 1,447 |

Both sides use dataset instance masks. Every unique reference constructs and
stores its own candidate pool. Ground-truth poses and CAD models are opened
only after the online Eq. (9) winner is selected and are used only for metrics.
Pairs that fail before or after selection remain in the 2,000-pair denominator
with zero metrics.

TOYL's released pair rows use the same object scene on both sides even though
the corresponding archive directory is named `cross_scene_test`.

## Full TOYL run

```bash
cd /path/to/MGC-6D
export PYTHON_BIN=/path/to/mgc6d/python
export RAYST3R_ROOT=/path/to/rayst3r
export RAYST3R_CKPT=/path/to/rayst3r.pth
export MGC6D_DINOV2_REPO=/path/to/dinov2
export TOYL_ROOT=/path/to/TOYL
export TOYL_MODELS="$TOYL_ROOT/models"
export ANCHOR_GPUS=0,1
export QUERY_GPU=1
export SEED=0
export RUN_TAG=toyl_oryon_2k_seed0
bash scripts/run_toyl_oryon_2k.sh
```

The anchor stage is split into 21 resumable object batches. Completed batches
are identity-checked before merging. This produces 890 independent reference
candidate pools, not one candidate pool per object class.

## Full REAL275 run

```bash
cd /path/to/MGC-6D
export PYTHON_BIN=/path/to/mgc6d/python
export RAYST3R_ROOT=/path/to/rayst3r
export RAYST3R_CKPT=/path/to/rayst3r.pth
export MGC6D_DINOV2_REPO=/path/to/dinov2
export REAL_ROOT=/path/to/REAL275
export MODELS_INFO="$PWD/real275_models_info.json"
export ANCHOR_GPUS=0,1
export QUERY_GPU=1
export SEED=0
export RUN_TAG=real275_oryon_2k_seed0
bash scripts/run_real275_oryon_2k.sh
```

REAL275 writes atomic per-reference checkpoints. Status-139 native teardown
failures are retried up to `ANCHOR_MAX_RESTARTS`; configuration-mismatched
checkpoints are rejected.

## Reuse completed stages

Set any of the following to zero:

```bash
export RUN_FETCH=0      # reuse a verified pair-list file
export RUN_PREPARE=0    # reuse prepared reference inputs and manifest
export RUN_ANCHOR=0     # reuse a completed candidate registry
```

TOYL query-only example:

```bash
export RUN_FETCH=0 RUN_PREPARE=0 RUN_ANCHOR=0 RUN_QUERY=1
export ANCHOR_INPUT_ROOT=/path/to/prepared/toyl_oryon_inputs
export PAIR_MANIFEST="$ANCHOR_INPUT_ROOT/oryon_toyl_manifest.json"
export ANCHOR_OUT=/path/to/completed/toyl_oryon_anchor
export QUERY_OUT=/path/to/new/toyl_oryon_query
bash scripts/run_toyl_oryon_2k.sh
```

REAL275 query-only example:

```bash
export RUN_FETCH=0 RUN_PREPARE=0 RUN_ANCHOR=0 RUN_QUERY=1
export ANCHOR_INPUT_ROOT=/path/to/prepared/real275_oryon_inputs
export PAIR_MANIFEST="$ANCHOR_INPUT_ROOT/oryon_real275_manifest.json"
export ANCHOR_OUT=/path/to/completed/real275_oryon_anchor
export QUERY_OUT=/path/to/new/real275_oryon_query
bash scripts/run_real275_oryon_2k.sh
```

Dataset, model, Python, and GPU variables from the full commands remain
required. Every `QUERY_OUT` must be new; launchers refuse to overwrite existing
query output.

## Independent validation

The validators are read-only by default:

```bash
python scripts/verify_oryon_toyl_anchor_registry.py \
  --anchor-output /path/to/toyl_anchor \
  --pair-manifest /path/to/oryon_toyl_manifest.json
python scripts/verify_oryon_toyl_result.py /path/to/toyl_query

python scripts/verify_oryon_anchor_registry.py \
  --anchor-output /path/to/real275_anchor \
  --pair-manifest /path/to/oryon_real275_manifest.json
python scripts/verify_oryon_real275_result.py /path/to/real275_query
```

To save an audit report outside the immutable experiment directory:

```bash
python scripts/verify_oryon_toyl_result.py /path/to/toyl_query \
  --output /path/to/audit/toyl_verification.json
```

Result validation independently replays each saved Eq. (9) score and winner,
checks pair order and the 2,000-pair denominator, and recomputes aggregate
metrics from the per-pair table.

## Fixed-reference diagnostics are separate

`scripts/run_toyl_full.sh` and `scripts/run_real275_full.sh` evaluate one fixed
reference per object. They support the paper's candidate-frequency and runtime
analysis only. Their accuracy output is not interchangeable with the Oryon 2K
main results.
