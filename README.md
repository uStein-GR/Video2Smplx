# MoE — Full-Body 3D Animation Pipeline

An end-to-end pipeline for reconstructing 3D full-body animations from monocular video or images. The system integrates three specialist models — **SMPLest-X** (body), **WiLoR** (hands), and **EMOCA** (face) — to produce high-quality SMPL-X parameter files and a rendered 3D animation video.

---

## Overview

```
Input Video
    |
    v
[Stage 0] Frame Extraction (ffmpeg)
    |
    +----> [Stage 1] SMPLest-X  -- Full-body pose + shape (SMPL-X params)
    |
    +----> [Stage 2] WiLoR      -- Hand pose refinement (MANO params)
    |
    +----> [Stage 3] EMOCA      -- Face expression + jaw pose (FLAME params)
    |
    v
[Stage 4] Parameter Fusion  (smplestx_wilor_emoca_fuse.py)
           Replace hand pose  <-- WiLoR
           Replace expression + jaw pose  <-- EMOCA
    |
    v
[Stage 5] Zero Translation + Savitzky-Golay Smooth + Render  (zero_filter_render.py)
    |
    v
Output: smplest_wilor_emoca.mp4  +  fused .pkl per frame
```

---

## Directory Structure

```
MoE/
├── pipeline.py                    <- Master orchestrator (run this)
├── smplestx_wilor_emoca_fuse.py   <- Standalone 3-way fusion script
├── zero_filter_render.py          <- Zero transl + SG smooth + pyrender video
├── memo.txt                       <- Developer notes / quick commands
│
├── demo/
│   ├── input/                     <- Extracted frames (shared, PERSISTENT)
│   ├── result_params_unified/     <- WiLoR output params
│   │   └── params/                <- Per-frame .pkl (right/left hand pose)
│   └── output/                    <- Final rendered output
│
├── SMPLest-X-Inference/           <- Full-body estimation sub-project
│   ├── main/inference.py          <- SMPLest-X entry point
│   ├── models/                    <- ViT-H + TransformerDecoderHead
│   ├── human_models/              <- SMPL-X / SMPL body model files
│   ├── pretrained_models/
│   │   ├── yolov8x.pt             <- YOLOv8x person detector
│   │   └── smplest_x_h/
│   │       ├── smplest_x_h.pth.tar  <- SMPLest-X-H weights (7.7 GB)
│   │       └── config_base.py
│   └── demo/
│       ├── *.mp4                  <- Input videos
│       └── output_params/<name>/ <- Per-frame .pkl output
│
├── WiLoR-Inference/               <- Hand estimation sub-project
│   ├── demo_params_unified.py     <- WiLoR entry point
│   ├── pretrained_models/
│   │   ├── wilor_final.ckpt       <- WiLoR weights (2.4 GB)
│   │   ├── detector.pt            <- YOLO hand detector (51 MB)
│   │   ├── model_config.yaml
│   │   └── dataset_config.yaml
│   ├── mano_data/
│   │   ├── MANO_RIGHT.pkl         <- MANO hand model (~3.7 MB)
│   │   └── mano_mean_params.npz
│   └── wilor/                     <- WiLoR source package
│
└── EMOCA-Inference/               <- Face estimation sub-project
    ├── gdl_apps/EMOCA/demos/
    │   └── visualize3.py          <- EMOCA entry point used by pipeline
    ├── assets/
    │   ├── EMOCA/models/
    │   │   └── EMOCA_v2_lr_mse_20/  <- EMOCA checkpoint (~395 MB)
    │   ├── FLAME/geometry/
    │   │   └── generic_model.pkl    <- FLAME face model (51 MB)
    │   ├── DECA/data/
    │   │   └── deca_model.tar       <- DECA pretrained weights (415 MB)
    │   └── FaceRecognition/
    │       └── resnet50_ft_weight.pkl <- Face recognition weights (158 MB)
    ├── setup_env.sh               <- One-command environment setup (EMOCA)
    └── requirements38.txt
```

---

## Pretrained Models Required

> **All models below must be downloaded manually before running the pipeline. They are too large to include in the repository.**

### SMPLest-X-Inference

| File | Size | Location | Source |
|------|------|----------|--------|
| `smplest_x_h.pth.tar` | **7.7 GB** | `SMPLest-X-Inference/pretrained_models/smplest_x_h/` | [SMPLest-X GitHub](https://github.com/SMPLest-X/SMPLest-X) — model zoo |
| `yolov8x.pt` | **131 MB** | `SMPLest-X-Inference/pretrained_models/` | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| `SMPLX_NEUTRAL.npz` | **104 MB** | `SMPLest-X-Inference/human_models/human_model_files/smplx/` | [SMPL-X Project](https://smpl-x.is.tue.mpg.de/) — requires registration |
| `SMPLX_NEUTRAL_2020.npz` | **160 MB** | same as above | SMPL-X Project — requires registration |
| `SMPLX_MALE.npz` | **104 MB** | same as above | SMPL-X Project — requires registration |
| `SMPLX_FEMALE.npz` | **104 MB** | same as above | SMPL-X Project — requires registration |

### WiLoR-Inference

| File | Size | Location | Source |
|------|------|----------|--------|
| `wilor_final.ckpt` | **2.4 GB** | `WiLoR-Inference/pretrained_models/` | [WiLoR GitHub](https://github.com/rolpotamias/WiLoR) — model release |
| `detector.pt` | **51 MB** | `WiLoR-Inference/pretrained_models/` | WiLoR GitHub — model release |
| `MANO_RIGHT.pkl` | **3.7 MB** | `WiLoR-Inference/mano_data/` | [MANO Project](https://mano.is.tue.mpg.de/) — requires registration |

### EMOCA-Inference

| File | Size | Location | Source |
|------|------|----------|--------|
| EMOCA checkpoint (`*.ckpt`) | **395 MB** | `EMOCA-Inference/assets/EMOCA/models/EMOCA_v2_lr_mse_20/detail/checkpoints/` | [EMOCA Project](https://emoca.is.tue.mpg.de/) — requires registration |
| `deca_model.tar` | **415 MB** | `EMOCA-Inference/assets/DECA/data/` | [DECA Project](https://deca.is.tue.mpg.de/) — requires registration |
| `generic_model.pkl` (FLAME) | **51 MB** | `EMOCA-Inference/assets/FLAME/geometry/` | [FLAME Project](https://flame.is.tue.mpg.de/) — requires registration |
| `resnet50_ft_weight.pkl` | **158 MB** | `EMOCA-Inference/assets/FaceRecognition/` | [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) |
| `mask_inpainting.npz` | **76 MB** | `EMOCA-Inference/assets/DECA/data/` | Included with DECA data |

**Total estimated storage required: ~12 GB**

> Note: SMPL-X, MANO, FLAME, DECA, and EMOCA models require free registration on the Max-Planck Institute website. They are released for **non-commercial research use only**.

---

## Environment Setup

This pipeline runs each sub-system in its own isolated conda environment to avoid dependency conflicts. Merging all three into one environment is **not possible** due to incompatible versions of `timm`, `omegaconf`, and `pytorch-lightning`.

| Environment | Used by | Python |
|-------------|---------|--------|
| `ubuntu` | SMPLest-X, WiLoR, Rendering | 3.10 |
| `work38d` | EMOCA | 3.8 |

### SMPLest-X + WiLoR Environment (`ubuntu`)

```bash
# Key packages
pip install torch torchvision  # CUDA version matching your driver
pip install smplx ultralytics timm einops pyrender trimesh tqdm scipy opencv-python
pip install pytorch-lightning yacs hydra-core

# WiLoR requires chumpy from GitHub
pip install --no-build-isolation git+https://github.com/mattloper/chumpy
```

Or install from the respective requirements files:
```bash
pip install -r SMPLest-X-Inference/requirements.txt
pip install -r WiLoR-Inference/requirements.txt
```

### EMOCA Environment (`work38d`)

```bash
cd EMOCA-Inference
chmod +x setup_env.sh
./setup_env.sh
```

The setup script creates conda environment `work38d` with:
- Python 3.8, PyTorch 1.12.1, CUDA 11.3
- PyTorch3D (pre-built wheel)
- All pip dependencies + GDL package installation
- Compatibility patches (MediaPipe, Chumpy, NumPy)
- WSL2 CUDA path fix (if on WSL2)

---

## Quick Start

### Run the Full Pipeline

```bash
python pipeline.py \
    --video  demo/P.mp4 \
    --output demo/output \
    --smplestx_env ubuntu \
    --wilor_env    ubuntu \
    --emoca_env    work38d
```

### Full Example with All Options

```bash
python pipeline.py \
    --video  demo/P.mp4 \
    --output demo/output \
    --name   my_run \
    --smplestx_env ubuntu  --wilor_env ubuntu  --emoca_env work38d \
    --smplestx_ckpt smplest_x_h \
    --emoca_model   EMOCA_v2_lr_mse_20 \
    --fps 30  --viewport 800 \
    --smooth_window 15  --smooth_poly 3
```

---

## Pipeline Arguments

### I/O

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--video` | Yes | — | Path to input video file |
| `--output` | Yes | — | Root output directory for fused params and final video |
| `--name` | No | video stem | Run name used for SMPLest-X output folder |

### Conda Environments

| Argument | Default | Description |
|----------|---------|-------------|
| `--smplestx_env` | `ubuntu` | Conda env for SMPLest-X |
| `--wilor_env` | `ubuntu` | Conda env for WiLoR |
| `--emoca_env` | `work38d` | Conda env for EMOCA |
| `--render_env` | same as `smplestx_env` | Conda env for rendering (needs smplx, pyrender, scipy, torch) |

### Model Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--smplestx_ckpt` | `smplest_x_h` | SMPLest-X checkpoint folder inside `pretrained_models/` |
| `--emoca_model` | `EMOCA_v2_lr_mse_20` | EMOCA model folder inside `assets/EMOCA/models/` |
| `--smplx_model` | `SMPLX_NEUTRAL.npz` path | Path to SMPLX_NEUTRAL.npz used for rendering |

### Video / Render Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--fps` | `30` | FPS for frame extraction and output video |
| `--viewport` | `800` | Render viewport size in pixels (square) |
| `--smooth_window` | `15` | Savitzky-Golay window length (must be odd) |
| `--smooth_poly` | `3` | Savitzky-Golay polynomial order (must be < window) |

### Skip Flags (Resume a Partial Run)

| Argument | Description |
|----------|-------------|
| `--skip_extract` | Skip frame extraction — reuse existing frames in `demo/input/` |
| `--skip_smplestx` | Skip SMPLest-X inference |
| `--skip_wilor` | Skip WiLoR inference |
| `--skip_emoca` | Skip EMOCA inference |
| `--skip_fuse` | Skip parameter fusion |
| `--skip_render` | Skip rendering |

---

## Pipeline Stages in Detail

### Stage 0 — Frame Extraction
- Uses `ffmpeg` to extract frames from the input video at the specified FPS.
- Frames are saved as `000001.jpg`, `000002.jpg`, ... in `MoE/demo/input/`.
- This folder is **never deleted** by the pipeline — delete manually when no longer needed.

### Stage 1 — SMPLest-X (Full-body)
- Estimates full-body pose, hand pose, face expression, and shape using the SMPL-X parametric model.
- Architecture: ViT-Huge encoder + TransformerDecoderHead (80 task tokens).
- Person detection via YOLOv8x.
- Output: one `.pkl` per frame at `SMPLest-X-Inference/demo/output_params/<name>/`.
- Each `.pkl` is a list of dicts (one per detected person) containing a 182-dim parameter vector.

**SMPL-X parameter vector (182 dims):**
```
global_orient (3) + body_pose (63) + left_hand_pose (45) +
right_hand_pose (45) + jaw_pose (3) + betas (10) + expression (10) + transl (3)
```

### Stage 2 — WiLoR (Hand refinement)
- Specialised hand pose estimation using the MANO hand model.
- Processes frames from `MoE/demo/input/` directly.
- Architecture: ViT backbone + RefineNet head with iterative delta prediction.
- Left-hand reflection correction is applied `(x, -y, -z)` to undo WiLoR's internal mirroring.
- Output: one `.pkl` per frame at `MoE/demo/result_params_unified/params/`.
- Each `.pkl` contains `right_hand_pose` (45-dim) and `left_hand_pose` (45-dim), or `None` if hand not detected.

### Stage 3 — EMOCA (Face refinement)
- Extracts expression and jaw pose from face images using the FLAME face model.
- Architecture: DECA-based coarse + detail two-stage reconstruction.
- Face detection via FAN (face-alignment library).
- Output: one `.pkl` per frame at `EMOCA-Inference/demo/output/`.
- Each `.pkl` contains `exp` (50-dim expression) and `jaw_pose` (3-dim axis-angle).

### Stage 4 — Parameter Fusion
- Merges outputs from all three models into unified SMPL-X `.pkl` files.
- **SMPLest-X** is the base; WiLoR and EMOCA results are used to override specific parameters:
  - `left_hand_pose` and `right_hand_pose` → replaced with WiLoR values (if hand detected)
  - `expression` and `jaw_pose` → replaced with EMOCA values (if face detected)
- Falls back to original SMPLest-X values when a specialist has no detection.
- Output: fused `.pkl` files in `<output>/fused_params/`.

### Stage 5 — Zero / Smooth / Render
Calls `zero_filter_render.py` which performs three in-memory operations:

1. **Zero translation** — sets `transl` to zero to center the avatar.
2. **Savitzky-Golay smoothing** — removes temporal jitter from pose parameters.
3. **Rendering** — feeds each frame through SMPL-X model, renders 3D mesh with `pyrender`, and encodes to MP4 via OpenCV.

- Output video: `<output>/rendered/smplest_wilor_emoca.mp4`
- Output smoothed params: `<output>/rendered/params/*.pkl`

---

## Output Files

After a complete run, the output directory contains:

```
<output>/
├── fused_params/          <- Per-frame fused .pkl (SMPLest-X + WiLoR + EMOCA)
│   ├── 000001_params.pkl
│   ├── 000002_params.pkl
│   └── ...
└── rendered/
    ├── smplest_wilor_emoca.mp4   <- Final 3D animation video
    └── params/                   <- Smoothed + zeroed .pkl (final output params)
        ├── 000001_params.pkl
        └── ...
```

Each final `.pkl` file is structured as a list of dicts (one per person):

```python
import pickle

with open('000001_params.pkl', 'rb') as f:
    frame_data = pickle.load(f)

person = frame_data[0]   # first (and typically only) person
# Keys:
#   global_orient     (3,)   root orientation
#   body_pose         (63,)  21 body joints x3 axis-angle
#   left_hand_pose    (45,)  15 joints x3 axis-angle  (from WiLoR)
#   right_hand_pose   (45,)  15 joints x3 axis-angle  (from WiLoR)
#   jaw_pose          (3,)   jaw rotation              (from EMOCA)
#   betas             (10,)  body shape
#   expression        (10,)  facial expression (10-dim subset from EMOCA's 50-dim)
#   transl            (3,)   root translation (zeroed)
#   smplx_param_vector (182,) all params concatenated
```

---

## Standalone Scripts

### `smplestx_wilor_emoca_fuse.py`
A standalone version of Stage 4 with hardcoded paths — useful for running fusion independently without the pipeline. Edit the path variables at the top of `merge_all()` before running.

```bash
python smplestx_wilor_emoca_fuse.py
```

### `zero_filter_render.py`
A standalone version of Stage 5. Can be run directly with CLI arguments or with hardcoded paths.

```bash
python zero_filter_render.py \
    --input_pkl_folder  <fused_params_dir> \
    --smplx_model_path  <path/to/SMPLX_NEUTRAL.npz> \
    --output_dir        <output_dir> \
    --video_fps 30 \
    --viewport_size 800 \
    --smooth_window_length 15 \
    --smooth_polyorder 3
```

---

## WSL2 Notes

On WSL2, EMOCA requires the CUDA library path to be set. The `setup_env.sh` script handles this automatically. If you see:

```
Could not load library libcudnn_cnn_infer.so.8
```

Run:
```bash
source ~/.bashrc
# or manually:
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/home/$USER/miniconda3/envs/work38d/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
```

---

## Requirements Summary

| Requirement | Notes |
|-------------|-------|
| NVIDIA GPU (CUDA) | Required for all three models |
| Anaconda / Miniconda | For isolated conda environments |
| `ffmpeg` | For frame extraction (Stage 0) |
| ~12 GB disk (models) | See pretrained models table above |
| ~50+ GB disk (data) | For frames, intermediate params, and outputs |

---

## License & Citation

This project integrates three research models, each with their own license:

- **SMPLest-X** — research use; cite: [arXiv:2501.09782](https://arxiv.org/abs/2501.09782)
- **WiLoR** — research use; cite [WiLoR paper](https://github.com/rolpotamias/WiLoR)
- **EMOCA** — non-commercial research use only; cite:

```bibtex
@inproceedings{EMOCA:CVPR:2021,
  title  = {{EMOCA}: {E}motion Driven Monocular Face Capture and Animation},
  author = {Danecek, Radek and Black, Michael J. and Bolkart, Timo},
  booktitle = {CVPR},
  year   = {2022}
}
```

SMPL-X, MANO, and FLAME body models require registration at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) and are for **non-commercial research use only**.
