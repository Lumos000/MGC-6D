# HO3D data and assets

Datasets, generated anchors, model weights, and compiled extensions are kept
outside version control.

## Query dataset

The released evaluator expects the Any6D-style HO3D layout:

```text
$HO3D_ROOT/
  evaluation/
    MPM10/
    MPM11/
    MPM12/
    MPM13/
    MPM14/
    AP10/
    AP11/
    AP12/
    AP13/
    AP14/
    SB11/
    SB13/
    SM1/
  YCB_Video_Models/
    models/
      006_mustard_bottle/textured_simple.obj
      010_potted_meat_can/textured_simple.obj
      019_pitcher_base/textured_simple.obj
      021_bleach_cleanser/textured_simple.obj
```

The query reader consumes the precomputed XMem masks in this layout. Those are
predicted masks, not GT masks. `models_info.json` in this repository supplies
YCB object diameters and symmetry metadata for evaluation metrics. Symmetry
metadata is not used to alter predictions in the main inference protocol.

## Anchor observations

Prepare one object folder per YCB object:

```text
$ANCHOR_SOURCE/
  006_mustard_bottle/
    color.png
    depth.png
    mask.png
    K.txt
    006_mustard_bottle_gt_pose.txt
    source_frame.txt                 # optional provenance
  010_potted_meat_can/
    ...
  019_pitcher_base/
    ...
  021_bleach_cleanser/
    ...
```

`mask.png` is the original dataset GT anchor mask used by the main protocol.
Depth can be millimetres or the prepared RaySt3R uint16 encoding; the released
command tests both conversions and records the selected unit. `K.txt` is a 3x3
camera intrinsic matrix.

The anchor GT pose is copied to the result folder and is loaded only when the
final query prediction is evaluated. It is not used by reconstruction,
geometric alignment, anchor-score computation, query consistency scoring, or
candidate selection. The prepared anchor source must therefore remain fixed
across all methods being compared.

## Checkpoints

The full launcher requires a local RaySt3R checkpoint:

```bash
export RAYST3R_CKPT=/path/to/rayst3r.pth
```

FoundationPose and InstantMesh use their project checkpoint locations:

```text
foundationpose/weights/
  2023-10-28-18-33-37/
  2024-01-11-20-02-45/
instantmesh/ckpts/
```

SAM2 weights are needed only for the explicitly enabled mask-refinement
ablation; the released GT-anchor-mask command does not enable `--refine_mask`.
