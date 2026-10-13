# Official Oryon 2K evaluation

These pipelines reproduce MGC-6D on the exact ordered pair lists released by
Oryon. They are separate from the paper's fixed-anchor dataset sweeps.

## Protocol contract

| Dataset | Pair-list SHA-256 | Pairs | Unique reference reconstructions |
|---|---|---:|---:|
| REAL275 | `1553e4ecfe1a3a3be1d55e6c5da157e7aa199dd80246d7b0501da4fefdbf7424` | 2,000 | 1,447 |
| TOYL | `99eb3a67edb126bb6b3787482b3092acb684debac828c7ee476c7e24e56a5dea` | 2,000 | 890 |

Both sides use the dataset instance mask. Each unique reference constructs its
own three-candidate anchor pool. Ground-truth poses and CAD models are opened
only after the online Eq. (22) winner has been selected, and are used only for
metrics. Failed pairs stay in the 2,000-pair denominator with zero metrics.

## Full REAL275 run

```bash
cd /path/to/MGC-6D
export PYTHON_BIN=/path/to/mgc6d/python
export RAYST3R_ROOT=/path/to/rayst3r
export RAYST3R_CKPT=/path/to/rayst3r/checkpoint/rayst3r.pth
export REAL_ROOT=/path/to/REAL275
export MODELS_INFO="$PWD/real275_models_info.json"
export ANCHOR_GPUS=0,1
export QUERY_GPU=1
export SEED=0
export RUN_TAG=real275_oryon_2k_seed0
export SAVE_CANDIDATE_METRICS=1
bash scripts/run_real275_oryon_2k.sh
```

The REAL275 anchor stage writes an atomic checkpoint after every completed
reference. The launcher automatically restarts status 139 crashes up to
`ANCHOR_MAX_RESTARTS` times and rejects configuration-mismatched checkpoints.

## Full TOYL run

```bash
cd /path/to/MGC-6D
export PYTHON_BIN=/path/to/mgc6d/python
export RAYST3R_ROOT=/path/to/rayst3r
export RAYST3R_CKPT=/path/to/rayst3r/checkpoint/rayst3r.pth
export TOYL_ROOT=/path/to/TOYL
export TOYL_MODELS="$TOYL_ROOT/models"
export ANCHOR_GPUS=0,1
export QUERY_GPU=1
export SEED=0
export RUN_TAG=toyl_oryon_2k_seed0
export SAVE_CANDIDATE_METRICS=1
bash scripts/run_toyl_oryon_2k.sh
```

The TOYL anchor stage is split into 21 object batches. Completed batches are
reused, then merged only after their object/reference identities have been
checked. This is still 890 independent reference reconstructions, not a
21-geometry approximation.

## Query-only reuse

REAL275:

```bash
export RUN_FETCH=0 RUN_PREPARE=0 RUN_ANCHOR=0 RUN_QUERY=1
export ANCHOR_INPUT_ROOT=/path/to/prepared/real275_oryon_inputs
export PAIR_MANIFEST="$ANCHOR_INPUT_ROOT/oryon_real275_manifest.json"
export ANCHOR_OUT=/path/to/completed/real275_oryon_anchor
export QUERY_OUT=/path/to/new/real275_oryon_query
bash scripts/run_real275_oryon_2k.sh
```

TOYL:

```bash
export RUN_FETCH=0 RUN_PREPARE=0 RUN_ANCHOR=0 RUN_QUERY=1
export ANCHOR_INPUT_ROOT=/path/to/prepared/toyl_oryon_inputs
export PAIR_MANIFEST="$ANCHOR_INPUT_ROOT/oryon_toyl_manifest.json"
export ANCHOR_OUT=/path/to/completed/toyl_oryon_anchor
export QUERY_OUT=/path/to/new/toyl_oryon_query
bash scripts/run_toyl_oryon_2k.sh
```

The dataset/model/Python/GPU variables from the full command remain required.
Every `QUERY_OUT` must be new; the launchers never overwrite results.

## Independent validation

```bash
python scripts/verify_oryon_anchor_registry.py \
  --anchor-output /path/to/real275_anchor \
  --pair-manifest /path/to/oryon_real275_manifest.json
python scripts/verify_oryon_real275_result.py /path/to/real275_query

python scripts/verify_oryon_toyl_anchor_registry.py \
  --anchor-output /path/to/toyl_anchor \
  --pair-manifest /path/to/oryon_toyl_manifest.json
python scripts/verify_oryon_toyl_result.py /path/to/toyl_query
```

The result validators independently replay every saved Eq. (22) score and
winner, verify the pair order and 2,000-pair denominator, and recompute all
aggregate metrics from the per-pair table.
