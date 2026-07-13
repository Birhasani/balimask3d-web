#!/usr/bin/env python3
"""Verify the notebook-compatible InstantMesh runtime without loading models."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import platform
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "huggingface-hub": "0.25.2",
    "accelerate": "0.27.2",
    "diffusers": "0.26.3",
    "transformers": "4.38.2",
    "xatlas": "0.0.11",
    "gradio": "3.41.2",
}

REQUIRED_IMPORTS = (
    "zero123plus.pipeline",
    "src.models.lrm_mesh",
    "src.utils.mesh_util",
    "webapp.instantmesh_service",
    "nvdiffrast.torch",
    "xatlas",
    "mcubes",
    "plyfile",
    "onnxruntime",
    "rembg",
)


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def import_status(module_name: str) -> tuple[bool, str, Any | None]:
    """Import a module while converting rembg-style SystemExit into a failure."""

    try:
        module = importlib.import_module(module_name)
    except (Exception, SystemExit) as error:
        return False, f"FAILED: {error}", None
    module_path = getattr(module, "__file__", None)
    return True, f"OK ({module_path or 'built-in'})", module


def main() -> int:
    errors: list[str] = []
    print(f"Python version: {platform.python_version()}")
    if sys.version_info[:2] != (3, 12):
        errors.append("Python 3.12 is required by the working Colab notebooks")

    try:
        import torch
    except Exception as error:
        print(f"PyTorch version: import failed: {error}")
        print("torchvision version: not checked")
        print("CUDA runtime: not checked")
        print("CUDA available: False")
        print("GPU name: unavailable")
        return 1

    print(f"PyTorch version: {torch.__version__}")
    try:
        import torchvision

        print(f"torchvision version: {torchvision.__version__}")
    except Exception as error:
        print(f"torchvision version: import failed: {error}")
        errors.append(f"torchvision import failed: {error}")

    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    print(f"GPU name: {gpu_name}")
    if not torch.cuda.is_available():
        errors.append("CUDA is unavailable; select a Colab GPU runtime")

    xformers_available = importlib.util.find_spec("xformers") is not None
    if xformers_available:
        ok, detail, xformers = import_status("xformers")
        version = getattr(xformers, "__version__", "unknown") if xformers else "unknown"
        print(f"xformers availability: {ok} (version={version}; {detail})")
    else:
        print("xformers availability: False (optional; PyTorch attention fallback is supported)")

    print(f"diffusers version: {distribution_version('diffusers')}")
    print(f"transformers version: {distribution_version('transformers')}")

    for distribution, expected in EXPECTED_VERSIONS.items():
        installed = distribution_version(distribution)
        if installed != expected:
            errors.append(
                f"{distribution} must be {expected} for the notebook-compatible runtime; "
                f"detected {installed or 'not installed'}"
            )

    print("Required custom module imports:")
    imported: dict[str, Any] = {}
    for module_name in REQUIRED_IMPORTS:
        ok, detail, module = import_status(module_name)
        print(f"  {module_name}: {detail}")
        if ok:
            imported[module_name] = module
        else:
            errors.append(f"required import failed: {module_name}: {detail}")

    onnxruntime = imported.get("onnxruntime")
    if onnxruntime is not None:
        providers = list(onnxruntime.get_available_providers())
        print(f"ONNX Runtime version: {onnxruntime.__version__}")
        print(f"ONNX Runtime providers: {providers}")
        if not providers:
            errors.append("ONNX Runtime has no available execution provider")

    if errors:
        print("Environment verification failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Environment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
