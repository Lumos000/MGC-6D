# Dataset layouts and assets

Datasets, prepared anchor observations, model weights, generated meshes, and
results are intentionally kept outside version control.

## HO3D

The evaluator uses the Any6D-style HO3D release layout. `HO3D_ROOT` must point
to the directory containing `evaluation/` and `YCB_Video_Models/`:

```text
$HO3D_ROOT/
  evaluation/
    MPM10/ MPM11/ MPM12/ MPM13/ MPM14/
    AP10/  AP11/  AP12/  AP13/  AP14/
    SB11/  SB13/   SM1/
  YCB_Video_Models/
    models/
      003_cracker_box/textured_simple.obj
      004_sugar_box/textured_simple.obj
      005_tomato_soup_can/textured_simple.obj
      006_mustard_bottle/textured_simple.obj
      010_potted_meat_can/textured_simple.obj
      019_pitcher_base/textured_simple.obj
      021_bleach_cleanser/textured_simple.obj
```

The 13 query sequences use the precomputed XMem masks distributed in the
Any6D-style evaluation data. These are predicted query masks, not ground-truth
masks. `models_info.json` supplies YCB diameters and symmetry metadata for
metric evaluation; symmetry metadata does not alter the released predictions.

### HO3D prepared anchors

The fixed-reference HO3D protocol reconstructs one prepared RGB-D anchor for
each of the seven evaluated YCB objects:

```text
$ANCHOR_SOURCE/
  003_cracker_box/
  004_sugar_box/
  005_tomato_soup_can/
  006_mustard_bottle/
  010_potted_meat_can/
  019_pitcher_base/
  021_bleach_cleanser/
```

Every object directory must contain:

```text
color.png
depth.png
mask.png
K.txt
<object>_gt_pose.txt
source_frame.txt              # strongly recommended provenance record
```

The anchors come from the corresponding HO3D/Any6D reference observations.
`color.png`, `depth.png`, `K.txt`, and the annotated object mask must all refer
to the same source frame. Because the dataset images are not redistributed by
this repository, record the original sequence/frame identifier in
`source_frame.txt` and preserve the prepared anchor directory unchanged across
all compared methods.

`depth.png` may store millimetres or the prepared RaySt3R uint16 encoding. The
launcher tests the supported conversions and records the selected unit.
`K.txt` is the 3x3 camera intrinsic matrix. The GT anchor pose is copied to the
result directory and opened only for final metric evaluation; reconstruction,
alignment, candidate scoring, and candidate selection do not use it.

## TOYL and REAL275

The main TOYL and REAL275 results use Oryon's official ordered-pair lists. The
preparation scripts read reference RGB, depth, intrinsics, instance masks, and
evaluation-only GT poses from the local datasets:

```bash
python scripts/prepare_oryon_toyl.py \
  --toyl-root /path/to/TOYL \
  --pairs-file /path/to/toyl_pair_list.txt \
  --output-root /path/to/toyl_anchor_inputs \
  --resume

python scripts/prepare_oryon_real275.py \
  --real275-data-root /path/to/REAL275 \
  --pairs-file /path/to/real275_pair_list.txt \
  --output-root /path/to/real275_anchor_inputs \
  --resume
```

Prepared RGB and depth files are symlinks into the local dataset. Masks,
intrinsics, source metadata, and evaluation-only GT poses are written below the
chosen output root. TOYL and REAL275 use dataset instance masks on both sides
of every pair.

The fixed-reference TOYL and REAL275 input directories use the same five core
files as HO3D, with dataset-specific object names. Those inputs are used only
for the separate candidate-frequency and runtime analyses, not for the main
Oryon 2K accuracy comparisons.

## Checkpoint layout

The complete checkpoint setup is documented in
[installation.md](installation.md). At minimum, the repository expects:

```text
foundationpose/weights/
  2023-10-28-18-33-37/       # refiner
  2024-01-11-20-02-45/       # scorer

instantmesh/ckpts/
  diffusion_pytorch_model.bin
  instant_mesh_large.ckpt

$TORCH_HOME/hub/checkpoints/
  dinov2_vitl14_reg4_pretrain.pth
```

RaySt3R is external:

```bash
export RAYST3R_ROOT=/path/to/rayst3r
export RAYST3R_CKPT=/path/to/rayst3r.pth
export MGC6D_DINOV2_REPO=/path/to/dinov2
```

Zero123++ (`sudo-ai/zero123plus-v1.2`) is downloaded into the normal Hugging
Face cache on first use. SAM2 is installed because it remains an upstream code
dependency, but the released commands use the benchmark mask sources above and
do not run SAM2 mask refinement.

## Official Oryon pair lists

The launchers extract only the required member from Oryon's v1.0.0 archive and
reject any file whose SHA-256 differs from the released value:

| Dataset | SHA-256 | Pairs | Unique references |
|---|---|---:|---:|
| TOYL | `99eb3a67edb126bb6b3787482b3092acb684debac828c7ee476c7e24e56a5dea` | 2,000 | 890 |
| REAL275 | `1553e4ecfe1a3a3be1d55e6c5da157e7aa199dd80246d7b0501da4fefdbf7424` | 2,000 | 1,447 |

The generated manifests record the pair-file digest, ordered pair IDs,
reference/query instance IDs, mask provenance, and unique-reference count.
