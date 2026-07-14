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
    "tokenizers": "0.15.2",
    "diffusers": "0.26.3",
    "transformers": "4.38.2",
    "xatlas": "0.0.11",
    "gradio": "3.41.2",
    "fastapi": "0.103.0",
    "starlette": "0.27.0",
    "pydantic": "1.10.23",
}

REQUIRED_IMPORTS = (
    "zero123plus.pipeline",
    "src.models.lrm_mesh",
    "src.utils.mesh_util",
    "webapp.instantmesh_service",
    "nvdiffrast.torch",
    "xatlas",
    "trimesh",
    "mcubes",
    "plyfile",
    "onnxruntime",
    "rembg",
)

FORBIDDEN_DISTRIBUTIONS = (
    "peft",
    "cupy",
    "cupy-cuda11x",
    "cupy-cuda12x",
    "cupy-cuda13x",
    "onnxruntime-gpu",
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


def verify_web_stack(errors: list[str]) -> None:
    """Construct a minimal legacy Gradio app without loading InstantMesh."""

    try:
        import gradio as gr

        with gr.Blocks() as demo:
            gr.Markdown("InstantMesh web-stack verification")
        del demo
        print("Minimal gr.Blocks construction: OK")
    except (Exception, SystemExit) as error:
        print(f"Minimal gr.Blocks construction: FAILED ({error})")
        errors.append(f"minimal gr.Blocks construction failed: {error}")


def verify_huggingface_compatibility(errors: list[str]) -> None:
    """Check symbols required by the pinned InstantMesh diffusion runtime."""

    try:
        from huggingface_hub import cached_download

        print(f"huggingface_hub.cached_download import: OK ({cached_download.__module__})")
    except (Exception, SystemExit) as error:
        print(f"huggingface_hub.cached_download import: FAILED ({error})")
        errors.append(f"huggingface_hub.cached_download import failed: {error}")

    try:
        from diffusers import DiffusionPipeline

        print(f"diffusers.DiffusionPipeline import: OK ({DiffusionPipeline.__module__})")
    except (Exception, SystemExit) as error:
        print(f"diffusers.DiffusionPipeline import: FAILED ({error})")
        errors.append(f"diffusers.DiffusionPipeline import failed: {error}")


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
    print(f"huggingface_hub version: {distribution_version('huggingface-hub')}")
    print(f"gradio version: {distribution_version('gradio')}")
    print(f"FastAPI version: {distribution_version('fastapi')}")
    print(f"Starlette version: {distribution_version('starlette')}")
    print(f"Pydantic version: {distribution_version('pydantic')}")

    verify_web_stack(errors)
    verify_huggingface_compatibility(errors)

    print("Forbidden optional packages:")
    for distribution in FORBIDDEN_DISTRIBUTIONS:
        installed = distribution_version(distribution)
        status = "not installed" if installed is None else f"INSTALLED ({installed})"
        print(f"  {distribution}: {status}")
        if installed is not None:
            errors.append(f"{distribution} must not be installed in the InstantMesh runtime")
    peft_module = importlib.util.find_spec("peft")
    cupy_module = importlib.util.find_spec("cupy")
    print(f"PEFT module availability: {peft_module is not None}")
    print(f"CuPy module availability: {cupy_module is not None}")
    if peft_module is not None:
        errors.append("PEFT remains importable after bootstrap cleanup")
    if cupy_module is not None:
        errors.append("CuPy remains importable after bootstrap cleanup")

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
    providers: list[str] = []
    if onnxruntime is not None:
        providers = list(onnxruntime.get_available_providers())
        print(f"ONNX Runtime version: {onnxruntime.__version__}")
        print(f"ONNX Runtime providers: {providers}")
        if "CPUExecutionProvider" not in providers:
            errors.append(
                "ONNX Runtime CPUExecutionProvider is unavailable; install rembg[cpu]"
            )

    rembg = imported.get("rembg")
    if rembg is None:
        print("rembg u2net session: not run because rembg import failed")
    elif "CPUExecutionProvider" not in providers:
        print("rembg u2net session: not run because CPUExecutionProvider is unavailable")
    else:
        try:
            session = rembg.new_session("u2net")
            session_providers = list(session.inner_session.get_providers())
            print(f"rembg u2net session: OK (providers={session_providers})")
            if "CPUExecutionProvider" not in session_providers:
                errors.append("rembg u2net session does not expose CPUExecutionProvider")
            del session
        except (Exception, SystemExit) as error:
            print(f"rembg u2net session: FAILED ({error})")
            errors.append(f'rembg.new_session("u2net") failed: {error}')

    dr = imported.get("nvdiffrast.torch")
    if dr is None or not torch.cuda.is_available():
        print("nvdiffrast RasterizeCudaContext: not run (CUDA or import unavailable)")
    else:
        try:
            context = dr.RasterizeCudaContext(device=torch.device("cuda:0"))
            print("nvdiffrast RasterizeCudaContext: OK")
            del context
            torch.cuda.empty_cache()
        except Exception as error:
            print(f"nvdiffrast RasterizeCudaContext: FAILED ({error})")
            errors.append(f"nvdiffrast RasterizeCudaContext creation failed: {error}")

    if errors:
        print("Environment verification failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Environment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
