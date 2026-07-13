#!/usr/bin/env python3
"""Verify the native dependencies required by InstantMesh mesh export.

This script performs no model loading, training, or inference. Creating a
``RasterizeCudaContext`` may compile nvdiffrast's CUDA extension on first use.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


BLACKWELL_MIN_PYTORCH = (2, 7)
BLACKWELL_MIN_CUDA = (12, 8)
REQUIRED_TRANSFORMERS = "4.57.1"
REQUIRED_XATLAS = "0.0.11"


def parse_version(value: str | None) -> tuple[int, int] | None:
    """Extract a comparable major/minor pair from a version string."""

    if not value:
        return None
    match = re.search(r"(\d+)\.(\d+)", value)
    return (int(match.group(1)), int(match.group(2))) if match else None


def is_blackwell(capability: tuple[int, int]) -> bool:
    """Return whether a CUDA compute capability belongs to Blackwell or newer."""

    return capability[0] >= 10


def find_nvcc(cuda_home: str | None) -> Path | None:
    """Locate nvcc through CUDA_HOME first, then PATH."""

    executable = "nvcc.exe" if sys.platform == "win32" else "nvcc"
    if cuda_home:
        candidate = Path(cuda_home) / "bin" / executable
        if candidate.is_file():
            return candidate
    located = shutil.which("nvcc")
    return Path(located) if located else None


def read_nvcc_version(nvcc_path: Path | None) -> tuple[str, tuple[int, int] | None]:
    """Return nvcc's version output and parsed release number."""

    if nvcc_path is None:
        return "unavailable", None
    try:
        process = subprocess.run(
            [str(nvcc_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return f"failed: {error}", None
    output = (process.stdout or process.stderr).strip()
    release = re.search(r"release\s+(\d+)\.(\d+)", output, flags=re.IGNORECASE)
    parsed = (int(release.group(1)), int(release.group(2))) if release else parse_version(output)
    return output, parsed


def package_version(name: str) -> str | None:
    """Return installed distribution version without importing the package."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_environment(
    *,
    check_nvdiffrast: bool = True,
    check_python_dependencies: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """Collect environment facts and return actionable validation errors."""

    report: dict[str, Any] = {
        "python": platform.python_version(),
        "transformers": package_version("transformers"),
        "tokenizers": package_version("tokenizers"),
        "diffusers": package_version("diffusers"),
        "transformers_pruning_helpers": "not checked",
        "xatlas": package_version("xatlas"),
        "xatlas_module_path": "not checked",
        "xatlas_parametrize": "not checked",
        "mcubes_import": "not checked",
        "plyfile_import": "not checked",
        "rembg_import": "not checked",
        "rembg_u2net_session": "not checked",
        "onnxruntime_version": "not checked",
        "onnxruntime_providers": "not checked",
        "zero123plus_pipeline": "not checked",
    }
    errors: list[str] = []
    if check_python_dependencies:
        if sys.version_info[:2] != (3, 12):
            errors.append(
                "Python 3.12 is required by the Colab bootstrap; "
                f"found Python {platform.python_version()}."
            )

        transformers_version = report["transformers"]
        if transformers_version is None:
            errors.append(
                f"transformers is not installed; required version is {REQUIRED_TRANSFORMERS}"
            )
        else:
            parsed_transformers = parse_version(transformers_version)
            if parsed_transformers and parsed_transformers[0] >= 5:
                errors.append(
                    f"Unsupported Transformers {transformers_version}: major version 5 or newer "
                    f"removed APIs required by InstantMesh. Install transformers=={REQUIRED_TRANSFORMERS}."
                )
            if transformers_version.split("+")[0] != REQUIRED_TRANSFORMERS:
                errors.append(
                    f"Transformers must be pinned to {REQUIRED_TRANSFORMERS}; detected "
                    f"{transformers_version}"
                )
            try:
                from transformers.pytorch_utils import (
                    find_pruneable_heads_and_indices,
                    prune_linear_layer,
                )

                # Keep explicit references so both imports are verified by this process.
                _ = find_pruneable_heads_and_indices, prune_linear_layer
                report["transformers_pruning_helpers"] = "ok"
            except Exception as error:
                report["transformers_pruning_helpers"] = f"failed: {error}"
                errors.append(
                    "Transformers pruning helper import failed: "
                    "find_pruneable_heads_and_indices and prune_linear_layer must both be "
                    f"available ({error})"
                )
        if report["tokenizers"] is None:
            errors.append("tokenizers is not installed")
        if report["diffusers"] is None:
            errors.append("diffusers is not installed")

        xatlas_version = report["xatlas"]
        if xatlas_version is None:
            errors.append(f"xatlas is not installed; required version is {REQUIRED_XATLAS}")
        elif xatlas_version.split("+")[0] != REQUIRED_XATLAS:
            errors.append(
                f"xatlas must be pinned to {REQUIRED_XATLAS}; detected {xatlas_version}"
            )

        try:
            xatlas = importlib.import_module("xatlas")
            report["xatlas_module_path"] = str(Path(xatlas.__file__).resolve())
        except Exception as error:
            report["xatlas_module_path"] = f"import failed: {error}"
            report["xatlas_parametrize"] = "not run: import failed"
            errors.append(f"xatlas import failed: {error}")
        else:
            try:
                import numpy as np

                vertices = np.asarray(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                )
                faces = np.asarray([[0, 1, 2]], dtype=np.uint32)
                vertex_mapping, uv_faces, uvs = xatlas.parametrize(vertices, faces)
                if len(vertex_mapping) == 0:
                    raise ValueError("returned an empty vertex mapping")
                if uv_faces.shape != (1, 3):
                    raise ValueError(f"returned face shape {uv_faces.shape}, expected (1, 3)")
                if uvs.ndim != 2 or uvs.shape[1] != 2 or uvs.shape[0] == 0:
                    raise ValueError(f"returned invalid UV shape {uvs.shape}")
                if not np.isfinite(uvs).all():
                    raise ValueError("returned non-finite UV coordinates")
                report["xatlas_parametrize"] = (
                    f"ok: {uvs.shape[0]} UV vertices, {uv_faces.shape[0]} face"
                )
            except Exception as error:
                report["xatlas_parametrize"] = f"failed: {error}"
                errors.append(
                    "xatlas.parametrize failed to generate UV coordinates for one synthetic "
                    f"triangle: {error}"
                )

        for module_name, report_key, package_name in (
            ("mcubes", "mcubes_import", "PyMCubes"),
            ("plyfile", "plyfile_import", "plyfile"),
        ):
            try:
                module = importlib.import_module(module_name)
                module_path = getattr(module, "__file__", None)
                report[report_key] = f"ok: {module_path or 'module path unavailable'}"
            except Exception as error:
                report[report_key] = f"failed: {error}"
                errors.append(
                    f"{package_name} dependency import failed via '{module_name}': {error}"
                )

        providers: list[str] = []
        try:
            onnxruntime = importlib.import_module("onnxruntime")
            report["onnxruntime_version"] = onnxruntime.__version__
            providers = list(onnxruntime.get_available_providers())
            report["onnxruntime_providers"] = providers
            if not providers:
                errors.append(
                    "ONNX Runtime imported but no execution providers are available; "
                    "install rembg[cpu] to provide CPUExecutionProvider"
                )
        except Exception as error:
            report["onnxruntime_version"] = f"import failed: {error}"
            report["onnxruntime_providers"] = []
            errors.append(
                "ONNX Runtime import failed; install rembg[cpu] (not rembg[gpu] or "
                f"onnxruntime-gpu): {error}"
            )

        try:
            rembg = importlib.import_module("rembg")
            module_path = getattr(rembg, "__file__", None)
            report["rembg_import"] = f"ok: {module_path or 'module path unavailable'}"
        except (Exception, SystemExit) as error:
            report["rembg_import"] = f"failed: {error}"
            report["rembg_u2net_session"] = "not run: import failed"
            errors.append(f"rembg dependency import failed: {error}")
        else:
            if not providers:
                report["rembg_u2net_session"] = "not run: no ONNX Runtime provider"
            else:
                try:
                    session = rembg.new_session("u2net")
                    session_providers = list(session.inner_session.get_providers())
                    if not session_providers:
                        raise RuntimeError("created session has no execution providers")
                    report["rembg_u2net_session"] = (
                        "ok: providers=" + ", ".join(session_providers)
                    )
                    del session
                except (Exception, SystemExit) as error:
                    report["rembg_u2net_session"] = f"failed: {error}"
                    errors.append(f"rembg u2net session creation failed: {error}")

        pipeline_path = Path(__file__).resolve().parents[1] / "zero123plus" / "pipeline.py"
        if pipeline_path.is_file():
            report["zero123plus_pipeline"] = f"ok: {pipeline_path}"
        else:
            report["zero123plus_pipeline"] = f"missing: {pipeline_path}"
            errors.append(
                "Trusted Zero123++ custom pipeline is missing: expected "
                f"'{pipeline_path}'"
            )

    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
    except ImportError as error:
        report.update(
            {
                "pytorch": "unavailable",
                "torch_cuda": None,
                "cuda_available": False,
                "cuda_home": None,
                "torch_arch_list": [],
                "gpu_name": None,
                "compute_capability": None,
                "nvcc_path": None,
                "nvcc_version": "unavailable",
                "nvdiffrast_import": "not checked",
                "rasterize_cuda_context": "not checked",
            }
        )
        errors.append(f"PyTorch import failed: {error}")
        return report, errors

    report["pytorch"] = torch.__version__
    report["torch_cuda"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_home"] = str(CUDA_HOME) if CUDA_HOME else None
    report["torch_arch_list"] = torch.cuda.get_arch_list() if torch.cuda.is_available() else []

    if not torch.cuda.is_available():
        errors.append("CUDA is unavailable in PyTorch; a CUDA-enabled runtime is required")
        report["gpu_name"] = None
        report["compute_capability"] = None
    else:
        capability = torch.cuda.get_device_capability(0)
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["compute_capability"] = f"{capability[0]}.{capability[1]}"
        expected_arch = f"sm_{capability[0]}{capability[1]}"
        if expected_arch not in report["torch_arch_list"]:
            errors.append(
                f"PyTorch does not advertise the detected GPU architecture {expected_arch}; "
                f"torch.cuda.get_arch_list()={report['torch_arch_list']}"
            )

        if is_blackwell(capability):
            pytorch_version = parse_version(torch.__version__)
            torch_cuda_version = parse_version(torch.version.cuda)
            blackwell_problems = []
            if pytorch_version is None or pytorch_version < BLACKWELL_MIN_PYTORCH:
                blackwell_problems.append(
                    f"PyTorch {torch.__version__} is older than {BLACKWELL_MIN_PYTORCH[0]}."
                    f"{BLACKWELL_MIN_PYTORCH[1]}"
                )
            if torch_cuda_version is None or torch_cuda_version < BLACKWELL_MIN_CUDA:
                blackwell_problems.append(
                    f"PyTorch CUDA {torch.version.cuda!r} is older than CUDA "
                    f"{BLACKWELL_MIN_CUDA[0]}.{BLACKWELL_MIN_CUDA[1]}"
                )
            if expected_arch not in report["torch_arch_list"]:
                blackwell_problems.append(f"the PyTorch build lacks {expected_arch}")
            if blackwell_problems:
                errors.append(
                    "Unsupported Blackwell PyTorch/CUDA toolchain: "
                    + "; ".join(blackwell_problems)
                    + ". Install a Blackwell-capable PyTorch build manually; this bootstrap "
                    "will not change PyTorch versions."
                )

    if CUDA_HOME is None:
        errors.append(
            "CUDA_HOME is unavailable; install or expose the CUDA toolkit before building nvdiffrast"
        )

    nvcc_path = find_nvcc(str(CUDA_HOME) if CUDA_HOME else None)
    nvcc_output, nvcc_version = read_nvcc_version(nvcc_path)
    report["nvcc_path"] = str(nvcc_path) if nvcc_path else None
    report["nvcc_version"] = nvcc_output
    if nvcc_path is None:
        errors.append("nvcc is unavailable through CUDA_HOME/bin and PATH")

    capability_value = report.get("compute_capability")
    if capability_value and is_blackwell(tuple(int(item) for item in capability_value.split("."))):
        if nvcc_version is None or nvcc_version < BLACKWELL_MIN_CUDA:
            errors.append(
                "Unsupported Blackwell CUDA toolkit: nvcc must be CUDA "
                f"{BLACKWELL_MIN_CUDA[0]}.{BLACKWELL_MIN_CUDA[1]} or newer; "
                f"detected {nvcc_version or 'unknown'}"
            )

    report["nvdiffrast_import"] = "not checked"
    report["rasterize_cuda_context"] = "not checked"
    if check_nvdiffrast:
        try:
            dr = importlib.import_module("nvdiffrast.torch")
            report["nvdiffrast_import"] = "ok"
        except Exception as error:
            report["nvdiffrast_import"] = f"failed: {error}"
            errors.append(f"nvdiffrast import failed: {error}")
        else:
            if torch.cuda.is_available():
                try:
                    context = dr.RasterizeCudaContext(device=torch.device("cuda:0"))
                    report["rasterize_cuda_context"] = "ok"
                    del context
                    torch.cuda.empty_cache()
                except Exception as error:
                    report["rasterize_cuda_context"] = f"failed: {error}"
                    errors.append(f"RasterizeCudaContext creation failed: {error}")
            else:
                report["rasterize_cuda_context"] = "skipped: CUDA unavailable"

    return report, errors


def print_report(report: dict[str, Any], errors: list[str]) -> None:
    """Print every required environment fact in a stable human-readable form."""

    print(f"Python version: {report.get('python')}")
    print(f"Transformers version: {report.get('transformers')}")
    print(f"Tokenizers version: {report.get('tokenizers')}")
    print(f"Diffusers version: {report.get('diffusers')}")
    print(
        "Transformers pruning helper imports: "
        f"{report.get('transformers_pruning_helpers')}"
    )
    print(f"xatlas version: {report.get('xatlas')}")
    print(f"xatlas module path: {report.get('xatlas_module_path')}")
    print(f"xatlas.parametrize synthetic triangle: {report.get('xatlas_parametrize')}")
    print(f"mcubes import status: {report.get('mcubes_import')}")
    print(f"plyfile import status: {report.get('plyfile_import')}")
    print(f"rembg import status: {report.get('rembg_import')}")
    print(f"ONNX Runtime version: {report.get('onnxruntime_version')}")
    print(f"ONNX Runtime execution providers: {report.get('onnxruntime_providers')}")
    print(f"rembg u2net session status: {report.get('rembg_u2net_session')}")
    print(f"Zero123++ pipeline path: {report.get('zero123plus_pipeline')}")
    print(f"PyTorch version: {report.get('pytorch')}")
    print(f"torch.version.cuda: {report.get('torch_cuda')}")
    print(f"CUDA available: {report.get('cuda_available')}")
    print(f"GPU name: {report.get('gpu_name')}")
    print(f"GPU compute capability: {report.get('compute_capability')}")
    print(f"torch.cuda.get_arch_list(): {report.get('torch_arch_list')}")
    print(f"CUDA_HOME: {report.get('cuda_home')}")
    print(f"nvcc path: {report.get('nvcc_path')}")
    print("nvcc version:")
    print(report.get("nvcc_version"))
    print(f"nvdiffrast import status: {report.get('nvdiffrast_import')}")
    print(f"RasterizeCudaContext creation status: {report.get('rasterize_cuda_context')}")
    if errors:
        print("Environment verification failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
    else:
        print("Environment verification passed.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify xatlas, CUDA, nvcc, nvdiffrast, and RasterizeCudaContext without inference."
        )
    )
    parser.parse_args(argv)
    report, errors = inspect_environment(check_nvdiffrast=True)
    print_report(report, errors)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
