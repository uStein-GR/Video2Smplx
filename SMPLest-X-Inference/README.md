# SMPLest-X Inference

Inference-only pipeline for **SMPLest-X** — a whole-body expressive human pose and shape estimation model.
Accepts a video or image as input and outputs per-frame **SMPL-X parameters** (body pose, hand pose, facial expression, shape) as `.pkl` files.

Based on: [SMPLest-X: Ultimate Scaling for Expressive Human Pose and Shape Estimation](https://arxiv.org/abs/2501.09782)

---

## Directory Structure

```
SMPLest-X-Inference/
│
├── main/
│   ├── inference.py        ← main run script
│   ├── base.py             ← Tester class (model loading & forward pass)
│   ├── config.py           ← Config loader (dot-notation dict)
│   ├── constants.py        ← shared constants
│   └── visualizer.py       ← SMPL-X hypothesis visualizer
│
├── models/
│   ├── SMPLest_X.py        ← model architecture entry point
│   ├── module.py           ← ViT-H encoder + TransformerDecoderHead
│   └── loss.py             ← training losses (unused at inference)
│
├── datasets/
│   ├── dataset.py          ← base dataset class
│   ├── humandata.py        ← HumanData format handler
│   └── SynHand.py          ← SynHand dataset loader
│
├── utils/
│   ├── data_utils.py       ← image loading, bbox processing, patch generation
│   ├── inference_utils.py  ← NMS for multi-person detection
│   ├── visualization_utils.py  ← mesh overlay rendering
│   ├── transforms.py       ← image/joint transforms
│   ├── distribute_utils.py ← distributed training utilities
│   ├── logger.py           ← logging setup
│   └── timer.py            ← timing utilities
│
├── human_models/
│   ├── human_models.py     ← SMPLX wrapper class
│   └── human_model_files/
│       ├── smplx/          ← SMPL-X body model files (.npz, .pkl, .npy)
│       └── smpl/           ← SMPL body model files (.pkl)
│
├── pretrained_models/
│   ├── yolov8x.pt          ← YOLOv8x person detector
│   └── smplest_x_h/
│       ├── smplest_x_h.pth.tar   ← SMPLest-X-H weights
│       └── config_base.py        ← model configuration
│
├── scripts/
│   └── inference.sh        ← shell script wrapper (ffmpeg-based frame extraction)
│
├── demo/
│   ├── *.mp4               ← place input videos here
│   ├── output_params/      ← inference output .pkl files saved here
│   └── output_frames/      ← rendered mesh overlay frames saved here
│
├── outputs/                ← experiment logs
└── requirements.txt
```

---

## Requirements

- Python 3.8+
- PyTorch with CUDA (GPU required)
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Key dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 1.23.1 | Array operations |
| `opencv-python` | 4.11.0.86 | Image/video I/O |
| `smplx` | 0.1.28 | SMPL-X body model |
| `ultralytics` | 8.3.75 | YOLOv8 person detection |
| `pyrender` | 0.1.45 | Mesh rendering |
| `trimesh` | 4.6.2 | 3D mesh handling |
| `timm` | 1.0.14 | ViT backbone |
| `einops` | 0.8.1 | Tensor operations |

---

## How to Run

Run from the **project root directory** (`SMPLest-X-Inference/`).

### Option 1 — Python script (recommended)

```bash
python main/inference.py --num_gpus 1 --video <video_file.mp4> --ckpt_name smplest_x_h --save_params
```

**Example:**

```bash
python main/inference.py --num_gpus 1 --video P.mp4 --ckpt_name smplest_x_h --save_params
```

### Option 2 — Shell script (requires `ffmpeg`)

```bash
bash scripts/inference.sh smplest_x_h P.mp4 30
```

Arguments: `<ckpt_name> <file_name> [fps=30]`

The shell script handles frame extraction via `ffmpeg` and reassembles the output into a result video.

---

## Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num_gpus` | int | — | Number of GPUs (use `1`) |
| `--video` | str | `None` | Video filename inside `demo/` (e.g. `P.mp4`). Extracts frames automatically and cleans up after. |
| `--file_name` | str | `test` | Pre-extracted frame folder name inside `demo/input_frames/` (alternative to `--video`) |
| `--ckpt_name` | str | `model_dump` | Checkpoint folder name inside `pretrained_models/` |
| `--start` | int | `1` | Start frame index |
| `--end` | int | `1` | End frame index (set automatically when using `--video`) |
| `--save_params` | flag | `False` | Save SMPL-X parameters as `.pkl` per frame |
| `--multi_person` | flag | `False` | Process all detected persons (default: largest person only) |
| `--params_only` | flag | `False` | Skip mesh rendering, only save parameters (faster; requires `--save_params`) |

---

## Model Configuration

Loaded from `pretrained_models/smplest_x_h/config_base.py`:

| Setting | Value |
|---|---|
| Model type | `vit_huge` |
| Input image shape | `(512, 384)` |
| Encoder input shape | `(256, 192)` |
| Encoder | ViT-H (embed_dim=1280, depth=32, heads=16, patch=16) |
| Task tokens | 80 |
| Detector | YOLOv8x, conf=0.5, IoU thr=0.5 |
| Bbox padding ratio | 1.2× |

---

## Output

### Rendered frames
Saved to `demo/output_frames/<video_name>/` as `.jpg` images with the SMPL-X mesh overlaid on the original frame.

### Parameter files (with `--save_params`)
Saved to `demo/output_params/<video_name>/` — one `.pkl` per frame:

```
demo/output_params/P/
├── 000001_params.pkl
├── 000002_params.pkl
└── ...
```

Each `.pkl` is a **list of dicts**, one entry per detected person:

```python
import pickle

with open('000001_params.pkl', 'rb') as f:
    frame_data = pickle.load(f)

# Empty list [] if no person detected in this frame
person = frame_data[0]  # first person

print(person['smplx_param_vector'].shape)  # (182,)
```

### SMPL-X Parameter Dictionary

| Key | Shape | Description |
|---|---|---|
| `smplx_param_vector` | `(182,)` | All parameters concatenated |
| `global_orient` | `(3,)` | Root orientation (axis-angle) |
| `body_pose` | `(63,)` | 21 body joints × 3 (axis-angle) |
| `left_hand_pose` | `(45,)` | 15 left hand joints × 3 |
| `right_hand_pose` | `(45,)` | 15 right hand joints × 3 |
| `jaw_pose` | `(3,)` | Jaw rotation (axis-angle) |
| `betas` | `(10,)` | Body shape coefficients |
| `expression` | `(10,)` | Facial expression coefficients |
| `transl` | `(3,)` | Camera translation |

### Concatenation order of `smplx_param_vector` (182 dims total)

```
global_orient (3) + body_pose (63) + left_hand_pose (45) +
right_hand_pose (45) + jaw_pose (3) + betas (10) + expression (10) + transl (3)
= 182
```

---

## Notes

- Must be run from the **project root** (`SMPLest-X-Inference/`), not from inside `main/`.
- GPU is required; CPU inference is not supported.
- When `--video` is used, frames are extracted automatically to `demo/input_frames/` and **cleaned up** after inference.
- By default, only the **largest bounding box** (largest detected person) per frame is processed. Use `--multi_person` to process all persons.
- Frames with no person detected still produce a `.pkl` containing an **empty list `[]`**.
- Use `--params_only` to skip rendering and run significantly faster when you only need the `.pkl` output.
- Experiment logs are saved to `outputs/inference_<file_name>_<ckpt_name>_<timestamp>/`.
