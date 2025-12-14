#!/bin/bash
set -euo pipefail

ROOT_DIR="$(pwd)"
DATASET_DIR="${ROOT_DIR}/Dataset"
DINO_PARENT="${ROOT_DIR}/CountingObject/datasets"
DINO_DIR="${DINO_PARENT}/GroundingDINO"

echo "[1/6] Create Dataset directory..."
mkdir -p "${DATASET_DIR}"

echo "[2/6] Install Python deps..."
python -m pip install -U pip setuptools wheel ninja

python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

python -m pip install \
  gdown opencv-python scipy imgaug "numpy<2" \
  git+https://github.com/openai/CLIP.git \
  einops "supervision>=0.22.0" transformers addict yapf pycocotools timm roboflow

echo "[3/6] Export CUDA + runtime libs..."
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Add torch/lib so extension can find libc10.so at runtime
TORCH_LIB="$(python - <<'EOF'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
EOF
)"
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"

# Make GroundingDINO importable without pip -e (simple + stable)
export PYTHONPATH="${DINO_DIR}:${PYTHONPATH:-}"

echo "[4/6] Clone GroundingDINO if missing..."
mkdir -p "${DINO_PARENT}"
if [ ! -d "${DINO_DIR}/.git" ]; then
  git clone https://github.com/IDEA-Research/GroundingDINO.git "${DINO_DIR}"
else
  echo "GroundingDINO already exists: ${DINO_DIR}"
fi

echo "[5/6] Build GroundingDINO CUDA extension (_C)..."
cd "${DINO_DIR}"
python setup.py build_ext --inplace

# Quick sanity check
python - <<'EOF'
import groundingdino
from groundingdino import _C
print("groundingdino:", groundingdino.__file__)
print("_C:", _C.__file__)
EOF

cd "${ROOT_DIR}"

echo "[6/6] Download datasets..."
gdown --id 1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S -O "${DATASET_DIR}/myfile.zip"

if command -v unzip &> /dev/null; then
  unzip -o "${DATASET_DIR}/myfile.zip" -d "${DATASET_DIR}"
  rm -f "${DATASET_DIR}/myfile.zip"
else
  echo "unzip not found, skipping extraction."
fi

curl -L -o "${DATASET_DIR}/FSC-147-S.json" \
  https://raw.githubusercontent.com/cha15yq/T2ICount/main/FSC-147-S.json

curl -L -o "${DATASET_DIR}/ImageClasses_FSC147.txt" \
  https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/master/data/ImageClasses_FSC147.txt

curl -L -o "${DATASET_DIR}/annotation_FSC147_384.json" \
  https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/master/data/annotation_FSC147_384.json

curl -L -o "${DATASET_DIR}/Train_Test_Val_FSC_147.json" \
  https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/master/data/Train_Test_Val_FSC_147.json

echo "Done."
