# WiLoR Inference Mode SuperLight

Minimal inference-only setup of **WiLoR** (Wild-to-Posed Hand Reconstruction) for the research pipeline.
Runs hand pose estimation on images and exports unified hand parameters (axis-angle) compatible with SMPLest-X output format.

---

## Directory Structure

```
WiLoR-Inference-Mode-SuperLight/
├── demo_params_unified.py      ← main run script
├── requirements.txt
│
├── pretrained_models/
│   ├── wilor_final.ckpt        ← WiLoR main model weights (2.4 GB)
│   ├── detector.pt             ← YOLO hand detector weights (52 MB)
│   ├── model_config.yaml       ← model architecture config
│   └── dataset_config.yaml     ← dataset config
│
├── mano_data/
│   ├── MANO_RIGHT.pkl          ← MANO right hand model
│   └── mano_mean_params.npz    ← MANO mean pose parameters
│
└── wilor/                      ← WiLoR source package (inference only)
    ├── configs/
    ├── datasets/
    ├── models/
    │   ├── backbones/          ← ViT-H backbone
    │   └── heads/              ← RefineNet head
    └── utils/                  ← renderer, geometry, logging
```

---

## Requirements

- Python 3.8+
- PyTorch (CUDA recommended)
- Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** `smplx==0.1.28` and `ultralytics==8.1.34` are version-pinned.
> `chumpy` is installed directly from GitHub (required by smplx).

---

## How to Run

```bash
python demo_params_unified.py \
    --img_folder <path/to/input/images> \
    --out_folder <path/to/output> \
    --save_params
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--img_folder` | `images` | Folder containing input images |
| `--out_folder` | `out_demo` | Folder to save output results |
| `--save_params` | `False` | Save hand parameters as `.pkl` per frame |
| `--save_mesh` | `False` | Save hand meshes as `.obj` files |
| `--rescale_factor` | `2.0` | Padding factor around detected bounding box |
| `--file_type` | `*.jpg *.png *.jpeg` | Image file extensions to process |

### Example (GR pipeline)

```bash
python demo_params_unified.py \
    --img_folder ../demo/input \
    --out_folder ../demo/result_params_unified \
    --save_params
```

---

## Output

For each input image, two output types are produced:

### 1. Rendered overlay image
- Saved to `<out_folder>/<image_name>.jpg`
- Input image with 3D hand mesh overlaid

### 2. Unified hand parameters (with `--save_params`)
- Saved to `<out_folder>/params/<image_name>_params.pkl`
- One `.pkl` file per frame containing:

```python
{
    'right_hand_pose':          np.ndarray (45,) or None,   # finger joints axis-angle
    'left_hand_pose':           np.ndarray (45,) or None,
    'right_hand_betas':         np.ndarray (10,) or None,   # hand shape
    'left_hand_betas':          np.ndarray (10,) or None,
    'right_hand_global_orient': np.ndarray (3,)  or None,   # wrist rotation axis-angle
    'left_hand_global_orient':  np.ndarray (3,)  or None,
}
```

> Parameters are in **axis-angle** format, matching the SMPLest-X output convention.
> Left-hand parameters have reflection correction applied `(x, -y, -z)` to undo WiLoR's internal mirroring.

---

## Model Architecture

| Component | Details |
|---|---|
| Backbone | ViT-H (patch16, image size 256x192) |
| Head | RefineNet (delta correction on top of ViT features) |
| Hand model | MANO (right hand, mirrored for left) |
| Detector | YOLO (ultralytics, confidence threshold 0.3) |

---

## Notes

- Requires **GPU** for reasonable speed (falls back to CPU if unavailable).
- Input images can contain both left and right hands simultaneously.
- Frames with no detected hands are skipped silently.
- This folder contains **inference-only** code. Training scripts, dataset loaders, and unused utilities have been removed.
