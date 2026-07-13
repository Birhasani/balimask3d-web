#!/usr/bin/env bash
# Reproduce the dependency setup used by the working InstantMesh notebooks.

set -Eeuo pipefail
trap 'echo "bootstrap_colab.sh failed at line ${LINENO}" >&2' ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
  echo "This workflow targets the Python 3.12 Google Colab runtime." >&2
  exit 1
fi

if ! python -c 'import torch, torchvision' >/dev/null 2>&1; then
  echo "PyTorch and torchvision must come from the selected Colab GPU runtime." >&2
  echo "This bootstrap deliberately does not install or replace them." >&2
  exit 1
fi

SYSTEM_PACKAGES=(build-essential ffmpeg git libgl1 libglib2.0-0)
MISSING_SYSTEM_PACKAGES=()
for package in "${SYSTEM_PACKAGES[@]}"; do
  if ! dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | grep -q 'install ok installed'; then
    MISSING_SYSTEM_PACKAGES+=("${package}")
  fi
done
if ((${#MISSING_SYSTEM_PACKAGES[@]})); then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    "${MISSING_SYSTEM_PACKAGES[@]}"
else
  echo "Required system packages are already installed."
fi

CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "${CONSTRAINTS_FILE}"' EXIT
cat >"${CONSTRAINTS_FILE}" <<'EOF'
numpy==1.26.4
huggingface_hub==0.25.2
accelerate==0.27.2
diffusers==0.26.3
transformers==4.38.2
xatlas==0.0.11
gradio==3.41.2
EOF

# These are the packages installed by the working inference/training notebooks.
# pip skips already-satisfied requirements; no --upgrade or blanket uninstall is used.
PYTHON_REQUIREMENTS=(
  'numpy==1.26.4'
  scipy
  pandas
  openpyxl
  omegaconf
  einops
  pillow
  opencv-python
  imageio
  imageio-ffmpeg
  matplotlib
  plotly
  trimesh
  'xatlas==0.0.11'
  plyfile
  PyMCubes
  torchmetrics
  webdataset
  tensorboard
  safetensors
  sentencepiece
  'huggingface_hub==0.25.2'
  'accelerate==0.27.2'
  'diffusers==0.26.3'
  'transformers==4.38.2'
  'gradio==3.41.2'
  'rembg[cpu]'
  pymatting
  ripser
  persim
  ninja
  setuptools
  wheel
  packaging
  jedi
)
python -m pip install --constraint "${CONSTRAINTS_FILE}" "${PYTHON_REQUIREMENTS[@]}"

read -r DETECTED_ARCH DETECTED_CUDA_HOME < <(
  python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; select a Colab GPU runtime")
if CUDA_HOME is None:
    raise SystemExit("CUDA_HOME is unavailable; nvdiffrast cannot be compiled")
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}", CUDA_HOME)
PY
)
export TORCH_CUDA_ARCH_LIST="${DETECTED_ARCH}"

if [[ ! -x "${DETECTED_CUDA_HOME}/bin/nvcc" ]] && ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is unavailable through CUDA_HOME and PATH; nvdiffrast cannot be compiled." >&2
  exit 1
fi

REQUESTED_MAX_JOBS="${MAX_JOBS:-4}"
if ! [[ "${REQUESTED_MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer." >&2
  exit 1
fi
if ((REQUESTED_MAX_JOBS > 4)); then
  REQUESTED_MAX_JOBS=4
fi
export MAX_JOBS="${REQUESTED_MAX_JOBS}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "MAX_JOBS=${MAX_JOBS}"

if python -c 'import nvdiffrast.torch' >/dev/null 2>&1; then
  echo "nvdiffrast is already importable; leaving the installation unchanged."
else
  python -m pip install \
    'git+https://github.com/NVlabs/nvdiffrast.git' \
    --no-build-isolation
fi

# Creating the CUDA context compiles the extension for the detected architecture
# on first use and is a no-op against the cached build on later runs.
python - <<'PY'
import torch
import nvdiffrast.torch as dr

context = dr.RasterizeCudaContext(device=torch.device("cuda:0"))
del context
torch.cuda.empty_cache()
print("nvdiffrast RasterizeCudaContext: OK")
PY

# The two notebook patches are already incorporated in the reusable code paths:
# texture CHW->HWC conversion and model_state_dict checkpoint unwrapping.
grep -Fq 'texture_map.permute(1, 2, 0)' webapp/instantmesh_service.py
grep -Fq '"model_state_dict"' webapp/checkpoint_loader.py

python scripts/verify_environment.py
echo "Colab bootstrap completed successfully."
