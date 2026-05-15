# Datasets And Assets

## HO3D / YCB Models

The main reproducible path in this release is HO3D with YCB Video object
models. Keep datasets outside the repository and pass their paths at runtime:

```bash
export HO3D_ROOT=/path/to/ho3d
export YCB_MODEL_PATH=/path/to/ho3d/YCB_Video_Models
```

Expected layout:

```text
$HO3D_ROOT/
  evaluation/
    MPM10/
    ...
  YCB_Video_Models/
    models/
      003_cracker_box/textured_simple.obj
      ...
```

`models_info.json` is included for the HO3D/YCB object metadata used by the
query scripts.

## Checkpoints

Do not commit checkpoints. Place them in these locations or set the matching
environment variables:

```text
foundationpose/weights/
  2023-10-28-18-33-37/
  2024-01-11-20-02-45/
sam2/checkpoints/sam2.1_hiera_large.pt
instantmesh/ckpts/
```

SAM2 can also be configured with:

```bash
export SAM2_CKPT=/path/to/sam2.1_hiera_large.pt
export SAM2_CFG=configs/sam2.1/sam2.1_hiera_l.yaml
```

RaySt3R weights are downloaded from Hugging Face by default through
`hf_hub_download("bartduis/rayst3r", "rayst3r.pth")`; alternatively pass
`--rayst3r_checkpoint /path/to/rayst3r.pth`.

## REAL275 And Toyota-Light

The paper includes REAL275 and Toyota-Light experiments. The cleaned first
release focuses on HO3D because that path has complete commands, registry
generation, and query evaluation wiring. REAL275/Toyota-Light scripts are
preserved as advanced/experimental entry points and should be checked before
claiming one-command reproduction.
