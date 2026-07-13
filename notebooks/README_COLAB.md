# Google Colab Workflow

Use a Python 3.12 GPU runtime. Run the cells in order. The bootstrap preserves
Colab's PyTorch, torchvision, CUDA, and optional xformers installations while
installing the versions used by the working InstantMesh notebooks.

## 1. Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 2. Clone the repository

```python
from pathlib import Path

REPO_DIR = Path("/content/balimask3d-web")
if not REPO_DIR.exists():
    !git clone https://github.com/Birhasani/balimask3d-web.git {REPO_DIR}
%cd {REPO_DIR}
```

## 3. Install dependencies and native extensions

This is idempotent: installed system packages, compatible Python packages, and
an importable nvdiffrast build are retained.

```bash
!bash scripts/bootstrap_colab.sh
```

## 4. Set checkpoint and output paths

Change `CHECKPOINT_DIR` if the five checkpoint files are stored elsewhere.

```python
import os
from pathlib import Path

CHECKPOINT_DIR = Path("/content/drive/MyDrive/Tugas_Akhir/checkpoints")
OUTPUT_DIR = Path("/content/drive/MyDrive/Tugas_Akhir/instantmesh_web_outputs")
SMOKE_IMAGE = Path("/content/balimask3d-web/examples/cartoon_panda.png")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["CHECKPOINT_DIR"] = str(CHECKPOINT_DIR)
os.environ["OUTPUT_DIR"] = str(OUTPUT_DIR)
os.environ["SMOKE_IMAGE"] = str(SMOKE_IMAGE)

print("Checkpoint directory:", CHECKPOINT_DIR)
print("Output directory:", OUTPUT_DIR)
```

Expected fine-tuned checkpoint filenames are defined in
`configs/model_registry.yaml`. Official pretrained weights are downloaded from
`TencentARC/InstantMesh` when they are not already cached.

## 5. Verify the environment

```bash
!python scripts/verify_environment.py
```

## 6. Run the pretrained smoke test

This performs one real inference and requires the MP4, GLB, and mesh ZIP.

```bash
!python tools/smoke_inference.py \
  --image "$SMOKE_IMAGE" \
  --model instantmesh_pretrained \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --output-dir "$OUTPUT_DIR/pretrained" \
  --seed 42 \
  --steps 75
```

## 7. Run a fine-tuned smoke test

The example selects I1. Replace `instantmesh_i1` with another enabled registry
name after placing its matching checkpoint in `CHECKPOINT_DIR`.

```bash
!python tools/smoke_inference.py \
  --image "$SMOKE_IMAGE" \
  --model instantmesh_i1 \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --output-dir "$OUTPUT_DIR/finetuned_i1" \
  --seed 42 \
  --steps 75
```

## 8. Launch Gradio

This cell remains active while the server runs. The `--share` flag creates the
temporary public Colab link; no authentication is enabled.

```bash
!python app.py \
  --share \
  --host 0.0.0.0 \
  --port 7860 \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --output-dir "$OUTPUT_DIR/gradio" \
  --config configs/model_registry.yaml
```
