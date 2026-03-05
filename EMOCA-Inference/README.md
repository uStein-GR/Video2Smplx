# EMOCA Inference

Inference-only build of [EMOCA v2](https://emoca.is.tue.mpg.de/) — 3D face reconstruction with emotion-aware expression capture.

Extracts FLAME parameters (expression, pose, shape) from face images for use in downstream tasks such as SMPL-X animation.

> Based on: *EMOCA: Detail-Rich Expression Capture for Human Avatars* (CVPR 2022)

---

## Requirements

- Linux or WSL2 (Windows Subsystem for Linux 2)
- NVIDIA GPU (CUDA 11.x driver)
- Anaconda or Miniconda

---

## Setup (First Time)

```bash
git clone <this-repo>
cd EMOCA_Inference
chmod +x setup_env.sh
./setup_env.sh
```

The script will:
1. Create conda environment `work38d` (Python 3.8, PyTorch 1.12.1, CUDA 11.3)
2. Install PyTorch3D (pre-built wheel)
3. Install all pip dependencies
4. Install GDL package (this project)
5. Apply all compatibility patches (MediaPipe, Chumpy, NumPy)
6. Apply WSL2 CUDA path fix to `~/.bashrc` (if on WSL2)

---

## Usage

### Activate environment

```bash
conda activate work38d
```

---

### Extract parameters from images (PKL only — fastest)

Outputs one `.pkl` file per image containing: `exp`, `pose`, `global_orient`, `jaw_pose`, `shape`, `cam`, `light`, `tex`.

```bash
python gdl_apps/EMOCA/demos/visualize3.py \
    --input_folder path/to/images \
    --output_folder path/to/output \
    --model_name EMOCA_v2_lr_mse_20 \
    --device cuda
```

---

### Full reconstruction (images + codes + optional mesh)

```bash
python gdl_apps/EMOCA/demos/test_emoca_on_images.py \
    --input_folder path/to/images \
    --output_folder path/to/output \
    --model_name EMOCA_v2_lr_mse_20 \
    --save_images True \
    --save_codes True \
    --save_mesh False \
    --mode detail
```

---

### Video reconstruction

```bash
python gdl_apps/EMOCA/demos/test_emoca_on_video.py \
    --input_video path/to/video.mp4 \
    --output_folder path/to/output \
    --model_name EMOCA_v2_lr_mse_20 \
    --mode detail
```

---

## Output Parameters

Each `.pkl` file contains a dictionary:

| Key | Shape | Description |
|-----|-------|-------------|
| `exp` | (1, 50) | Expression code — primary emotion signal |
| `pose` | (1, 6) | Full pose vector |
| `global_orient` | (1, 3) | Global head rotation (axis-angle) |
| `jaw_pose` | (1, 3) | Jaw rotation (axis-angle) |
| `shape` | (1, 100) | Face identity/shape code |
| `cam` | (1, 3) | Camera: scale, tx, ty |
| `light` | (1, 27) | Spherical harmonics lighting |
| `tex` | (1, 50) | Texture appearance code |

---

## Available Models

| Model | Description |
|-------|-------------|
| `EMOCA_v2_lr_mse_20` | **Recommended** — MediaPipe landmarks + MSE lip reading loss |
| `EMOCA_v2_lr_cos_1.5` | MediaPipe + cosine lip reading (may have artifacts) |
| `EMOCA_v2_mp` | MediaPipe only, no lip reading loss |

---

## Project Structure

```
EMOCA_Inference/
├── gdl/                          # Core deep learning library
│   ├── models/                   # DECA, FLAME, encoders, decoders
│   ├── layers/losses/            # Loss functions
│   ├── datasets/                 # ImageTestDataset and helpers
│   └── utils/                    # FaceDetector, rendering, mesh utils
├── gdl_apps/EMOCA/
│   ├── demos/                    # Inference scripts (run these)
│   │   ├── visualize3.py         # PKL-only parameter extraction
│   │   ├── test_emoca_on_images.py        # Full image output
│   │   ├── test_emoca_on_images_params_fix_1.py
│   │   ├── test_emoca_on_images_params_fix_2.py
│   │   └── test_emoca_on_video.py         # Video pipeline
│   └── utils/
│       ├── load.py               # load_model() entry point
│       └── io.py                 # test(), save_images(), save_codes()
├── assets/
│   ├── EMOCA/models/             # Pre-trained EMOCA checkpoints
│   ├── FLAME/geometry/           # FLAME face model geometry files
│   ├── DECA/data/                # DECA pretrained weights + textures
│   └── FaceRecognition/          # ResNet face recognition weights
├── setup.py
├── requirements38.txt
├── conda-environment_py38_cu11_ubuntu.yml
└── setup_env.sh                  # One-command environment setup
```

---

## WSL2 Notes

On WSL2 you must have the CUDA library path set before running. The setup script does this automatically. If you ever get:

```
Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file
```

Run:
```bash
source ~/.bashrc
```

Or manually:
```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/home/$USER/miniconda3/envs/work38d/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
```

---

## License

Non-commercial research use only. See [EMOCA license](https://emoca.is.tue.mpg.de/license.html).

---

## Citation

```bibtex
@inproceedings{EMOCA:CVPR:2021,
  title = {{EMOCA}: {E}motion Driven Monocular Face Capture and Animation},
  author = {Danecek, Radek and Black, Michael J. and Bolkart, Timo},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022}
}
```
