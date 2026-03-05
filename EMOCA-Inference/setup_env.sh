set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="work38d"
PYTHON="/home/$(whoami)/miniconda3/envs/${ENV_NAME}/bin/python"
PIP="/home/$(whoami)/miniconda3/envs/${ENV_NAME}/bin/pip"

echo "=============================================="
echo " EMOCA Inference Setup"
echo "=============================================="
echo "Project directory: $SCRIPT_DIR"
echo "Conda environment: $ENV_NAME"
echo ""

echo "[1/6] Creating conda environment '$ENV_NAME'..."

if conda env list | grep -q "^${ENV_NAME} \|^${ENV_NAME}$"; then
    echo "  '$ENV_NAME' already exists, skipping."
    echo "  To recreate: conda env remove -n $ENV_NAME"
else
    conda create -n "$ENV_NAME" python=3.8 -y
    echo "  Done."
fi

echo ""
echo "[2/6] Installing PyTorch 1.12.1+cu113..."

"$PIP" install \
    torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    torchaudio==0.12.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

echo "  Done."

echo ""
echo "[3/6] Installing PyTorch3D..."

"$PIP" install fvcore iopath

"$PIP" install \
    --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html

echo "  Done."

echo ""
echo "[4/6] Installing dependencies..."

"$PIP" install mediapipe
"$PIP" install protobuf==3.20.3
"$PIP" install pytorch-lightning==1.4.9
"$PIP" install torchmetrics==0.5.0
"$PIP" install hydra-core==1.1.0
"$PIP" install omegaconf==2.1.1
"$PIP" install numpy==1.23.1
"$PIP" install scikit-image matplotlib
"$PIP" install opencv-python==4.5.5.64
"$PIP" install face-alignment==1.3.5
"$PIP" install kornia==0.6.8
"$PIP" install gdown

"$PIP" install adabound
"$PIP" install pandas scikit-learn
"$PIP" install compress_pickle
"$PIP" install hickle
"$PIP" install imgaug albumentations
"$PIP" install scikit-video
"$PIP" install facenet-pytorch==2.5.2 --no-deps
"$PIP" install wandb
"$PIP" install munch
"$PIP" install torchfile
"$PIP" install chumpy
"$PIP" install scipy
"$PIP" install trimesh==3.14.1
"$PIP" install h5py==3.7.0

"$PIP" install ffmpeg-python

echo "  Done."

echo ""
echo "[5/6] Installing EMOCA (GDL) package..."

"$PIP" install -e "$SCRIPT_DIR"

echo "  Done."

echo ""
echo "[6/6] Applying patches..."

echo "  [A] Patching FaceDetector.py (LandmarksType)..."
FACE_DETECTOR="$SCRIPT_DIR/gdl/utils/FaceDetector.py"
sed -i 's/LandmarksType\._2D/LandmarksType.TWO_D/g' "$FACE_DETECTOR"
sed -i 's/LandmarksType\._TWO_D/LandmarksType.TWO_D/g' "$FACE_DETECTOR"
echo "     Done."

echo "  [B] Patching MediaPipe for Python 3.8..."
"$PYTHON" - <<'EOF'
import os
import re
import subprocess
import sys

# Find mediapipe path WITHOUT importing it (it crashes before being patched)
result = subprocess.run(
    [sys.executable, "-m", "pip", "show", "mediapipe"],
    capture_output=True, text=True
)
location = ""
for line in result.stdout.splitlines():
    if line.startswith("Location:"):
        location = line.split(":", 1)[1].strip()
        break

target_dir = os.path.join(location, "mediapipe")
print(f"    MediaPipe path: {target_dir}")

count = 0
for root, dirs, files in os.walk(target_dir):
    for file in files:
        if not file.endswith(".py"):
            continue
        filepath = os.path.join(root, file)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            original = content

            # Fix 1: Remove unsupported Queue generics (causes unmatched ']' error)
            content = re.sub(r'queue\.Queue\[.*?\]', 'queue.Queue', content)
            # Fix 2: Lowercase collection generics not valid in Python 3.8
            content = content.replace("list[", "List[")
            content = content.replace("tuple[", "Tuple[")
            content = content.replace("type[", "Type[")

            if content != original:
                header = "from typing import List, Tuple, Type, Any, Optional\nimport queue\n"
                if "from typing import List" not in content:
                    content = header + content
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                count += 1
        except Exception as e:
            pass  # skip unreadable files

print(f"    Patched {count} MediaPipe files.")
EOF
echo "     Done."

echo "  [C] Fixing MediaPipe import paths in EMOCA..."
LANDMARK_FILE="$SCRIPT_DIR/gdl/utils/MediaPipeLandmarkLists.py"

sed -i 's/mediapipe\.python\.solutions/mediapipe.solutions/g' "$LANDMARK_FILE"

sed -i 's/from mediapipe.*solutions\.face_mesh_connections/from gdl.utils.face_mesh_connections/g' "$LANDMARK_FILE"
echo "     Done."

echo "  [D] Patching Chumpy..."
"$PYTHON" - <<'EOF'
import os, subprocess, sys

# Find chumpy path WITHOUT importing it (numpy types may be broken at this point)
result = subprocess.run(
    [sys.executable, "-m", "pip", "show", "chumpy"],
    capture_output=True, text=True
)
location = ""
for line in result.stdout.splitlines():
    if line.startswith("Location:"):
        location = line.split(":", 1)[1].strip()
        break

path = os.path.join(location, "chumpy", "__init__.py")
print(f"    Chumpy path: {path}")

with open(path, "r") as f:
    content = f.read()

broken = "from numpy import bool, int, float, complex, object, unicode, str, nan, inf"
fixed  = "from numpy import nan, inf\nbool=bool; int=int; float=float; complex=complex; object=object; unicode=str; str=str"

if broken in content:
    content = content.replace(broken, fixed)
    with open(path, "w") as f:
        f.write(content)
    print("    Chumpy patched.")
else:
    # Fallback: line by line
    with open(path, "r") as f:
        lines = f.readlines()
    with open(path, "w") as f:
        for line in lines:
            if "from numpy import" in line and "bool" in line:
                f.write(fixed + "\n")
            else:
                f.write(line)
    print("    Chumpy force-patched.")
EOF
echo "     Done."

echo "  [E] Patching deprecated numpy types in EMOCA source..."
"$PYTHON" - <<EOF
import os, re

root_dir = "$SCRIPT_DIR/gdl"
replacements = [
    (r"np\.bool\b",    "bool"),
    (r"np\.int\b",     "int"),
    (r"np\.float\b",   "float"),
    (r"np\.object\b",  "object"),
]

count = 0
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if not file.endswith(".py"):
            continue
        filepath = os.path.join(root, file)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        new_content = content
        for pattern, replacement in replacements:
            new_content = re.sub(pattern, replacement, new_content)
        if new_content != content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            count += 1

print(f"    Patched {count} files.")
EOF
echo "     Done."

echo ""
echo "[Optional] Checking for WSL2..."
if grep -qi "microsoft" /proc/version 2>/dev/null; then
    echo "  WSL2 detected. Adding CUDA library fix to ~/.bashrc..."
    TORCH_LIB="$("$PYTHON" -c "import torch, os; print(os.path.dirname(torch.__file__))")/lib"
    FIX_LINE="export LD_LIBRARY_PATH=\"/usr/lib/wsl/lib:${TORCH_LIB}:\$LD_LIBRARY_PATH\""
    if ! grep -q "wsl/lib" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# WSL2 CUDA library path fix" >> ~/.bashrc
        echo "if [[ \":\$LD_LIBRARY_PATH:\" != *\":/usr/lib/wsl/lib:\"* ]]; then" >> ~/.bashrc
        echo "    $FIX_LINE" >> ~/.bashrc
        echo "fi" >> ~/.bashrc
        echo "  Added. Run: source ~/.bashrc"
    else
        echo "  Already present in ~/.bashrc."
    fi
fi

echo ""
echo "=============================================="
echo " Setup complete!"
echo "=============================================="
echo ""
echo "To run inference:"
echo "  conda activate $ENV_NAME"
echo "  cd $SCRIPT_DIR"
echo "  python gdl_apps/EMOCA/demos/visualize3.py \\"
echo "      --input_folder <path/to/images> \\"
echo "      --output_folder <path/to/output> \\"
echo "      --model_name EMOCA_v2_lr_mse_20 \\"
echo "      --device cuda"
echo ""
