#!/usr/bin/env python3
"""Install verified native and Python dependencies in a CUDA Colab runtime.

The script requires the pinned Transformers version and installs the missing
InstantMesh runtime packages, but deliberately does not install or change
PyTorch, torchvision, CUDA, nvdiffrast when already present, Transformers, or
Diffusers. Run it with the same Python interpreter used for inference.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import subprocess
import sys

from verify_cuda_environment import inspect_environment, print_report


NVDIFFRAST_REPOSITORY = "git+https://github.com/NVlabs/nvdiffrast.git"
MAX_BUILD_JOBS = 4
REQUIRED_TRANSFORMERS_VERSION = "4.57.1"
REQUIRED_XATLAS_VERSION = "0.0.11"
INSTANTMESH_RUNTIME_REQUIREMENTS = ("PyMCubes", "plyfile", "rembg[cpu]")


class BootstrapError(RuntimeError):
    """Raised when the requested runtime dependencies cannot be verified."""


def run_pip(*arguments: str) -> None:
    """Run pip through the active interpreter and fail on installation errors."""

    command = [sys.executable, "-m", "pip", "install", *arguments]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def require_existing_distribution(name: str, required_version: str) -> None:
    """Require an exact installed version without modifying the environment."""

    try:
        installed_version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise BootstrapError(
            f"{name}=={required_version} must already be installed; this bootstrap will not "
            f"install or modify {name}"
        ) from error
    if installed_version.split("+")[0] != required_version:
        raise BootstrapError(
            f"{name}=={required_version} must already be installed; detected "
            f"{installed_version}. This bootstrap will not modify {name}."
        )
    print(f"{name}=={installed_version} is already installed; leaving it unchanged.")


def configure_build_environment() -> tuple[int, int]:
    """Detect the active GPU and configure bounded CUDA extension compilation."""

    try:
        import torch
    except ImportError as error:
        raise BootstrapError(
            "PyTorch must already be installed; this bootstrap will not install or change it"
        ) from error
    if not torch.cuda.is_available():
        raise BootstrapError("CUDA is unavailable; select a GPU Colab runtime before bootstrapping")

    capability = torch.cuda.get_device_capability(0)
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{capability[0]}.{capability[1]}"
    requested_jobs = os.environ.get("MAX_JOBS", str(MAX_BUILD_JOBS))
    try:
        requested = int(requested_jobs)
    except ValueError:
        requested = MAX_BUILD_JOBS
    os.environ["MAX_JOBS"] = str(min(MAX_BUILD_JOBS, max(1, requested)))
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {capability[0]}.{capability[1]}")
    print(f"TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")
    print(f"MAX_JOBS={os.environ['MAX_JOBS']}")
    return capability


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Install InstantMesh runtime dependencies, build NVlabs/nvdiffrast when absent, "
            "and verify the Colab runtime."
        )
    )
    parser.parse_args(argv)
    try:
        if sys.version_info[:2] != (3, 12):
            raise BootstrapError(
                "this bootstrap targets the Python 3.12 Colab runtime; "
                f"detected Python {sys.version.split()[0]}"
            )

        require_existing_distribution("transformers", REQUIRED_TRANSFORMERS_VERSION)
        require_existing_distribution("xatlas", REQUIRED_XATLAS_VERSION)

        # These are build prerequisites only; no ML framework versions are changed.
        run_pip("setuptools", "wheel", "ninja")
        configure_build_environment()

        preflight_report, preflight_errors = inspect_environment(
            check_nvdiffrast=False,
            check_python_dependencies=False,
        )
        print_report(preflight_report, preflight_errors)
        if preflight_errors:
            raise BootstrapError(
                "CUDA preflight failed; nvdiffrast was not built. Resolve the errors above first."
            )

        if importlib.util.find_spec("nvdiffrast") is None:
            run_pip(NVDIFFRAST_REPOSITORY, "--no-build-isolation")
        else:
            print("nvdiffrast is already installed; leaving it unchanged.")

        # rembg[cpu] installs the CPU ONNX Runtime backend. None of these
        # requirements depend on or replace the protected ML/CUDA stack.
        run_pip(*INSTANTMESH_RUNTIME_REQUIREMENTS)

        verification_report, verification_errors = inspect_environment(
            check_nvdiffrast=True,
            check_python_dependencies=True,
        )
        print_report(verification_report, verification_errors)
        if verification_errors:
            raise BootstrapError("runtime dependency verification failed")
    except (BootstrapError, subprocess.CalledProcessError) as error:
        print(f"Colab dependency bootstrap failed: {error}", file=sys.stderr)
        return 1
    print("Colab dependency bootstrap completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
