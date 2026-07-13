#!/usr/bin/env python3
"""Run one real, end-to-end InstantMesh GPU smoke inference."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from webapp.instantmesh_service import InstantMeshService  # noqa: E402
from webapp.settings import InstantMeshSettings  # noqa: E402
from webapp.types import InferenceRequest, InferenceResult  # noqa: E402


@dataclass(frozen=True)
class SmokeModel:
    """CLI-facing model metadata and its optional fine-tuned checkpoint."""

    label: str
    checkpoint_filename: str | None


MODELS = {
    "instantmesh_pretrained": SmokeModel("InstantMesh Pretrained", None),
    "instantmesh_i1": SmokeModel(
        "InstantMesh Fine-tuned I1", "best_meshval_combined_I1_depthnormal.pt"
    ),
    "instantmesh_i2": SmokeModel(
        "InstantMesh Fine-tuned I2",
        "best_meshval_combined_I2_supervision_same_trainable.pt",
    ),
    "instantmesh_i3": SmokeModel(
        "InstantMesh Fine-tuned I3",
        "best_meshval_combined_I3_trainable_plus_supervision.pt",
    ),
    "instantmesh_i4": SmokeModel(
        "InstantMesh Fine-tuned I4", "best_meshval_combined_I4_variation.pt"
    ),
}

EXPECTED_ZIP_MESH_ASSETS = {
    "mesh/mesh.obj",
    "mesh/mesh.mtl",
    "mesh/mesh.png",
    "mesh/mesh.glb",
}


class SmokeValidationError(RuntimeError):
    """Raised when real inference completes without its required artifacts."""


def _require_nonempty(path: Path | None, label: str) -> Path:
    if path is None or not path.is_file() or path.stat().st_size <= 0:
        raise SmokeValidationError(f"missing or empty {label}: {path}")
    return path


def validate_result(result: InferenceResult, *, skip_video: bool) -> dict[str, object]:
    """Validate the smoke-test artifact contract, including archive members."""

    if len(result.multiview_image_paths) != 6:
        raise SmokeValidationError(
            f"expected exactly six multiview images, received {len(result.multiview_image_paths)}"
        )
    for index, path in enumerate(result.multiview_image_paths):
        _require_nonempty(path, f"multiview image {index}")

    if not skip_video:
        _require_nonempty(result.video_path, "MP4 video")
    elif result.video_path is not None:
        raise SmokeValidationError("--skip-video was used but a video path was returned")
    _require_nonempty(result.glb_path, "GLB mesh")
    _require_nonempty(result.obj_path, "OBJ mesh")
    _require_nonempty(result.mtl_path, "MTL material")
    for path in result.texture_paths:
        _require_nonempty(path, "texture")
    if not result.texture_paths:
        raise SmokeValidationError("no texture paths were returned")

    archive_path = _require_nonempty(result.zip_path, "ZIP archive")
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            names = set(archive.namelist())
            bad_member = archive.testzip()
    except (OSError, zipfile.BadZipFile) as error:
        raise SmokeValidationError(f"invalid ZIP archive '{archive_path}': {error}") from error
    if bad_member:
        raise SmokeValidationError(f"corrupt ZIP member: {bad_member}")
    missing_assets = EXPECTED_ZIP_MESH_ASSETS - names
    if missing_assets:
        raise SmokeValidationError(
            "ZIP archive is missing expected mesh assets: " + ", ".join(sorted(missing_assets))
        )

    metadata_path = _require_nonempty(result.metadata_path, "metadata JSON")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SmokeValidationError(f"invalid metadata JSON '{metadata_path}': {error}") from error
    if metadata.get("request_id") != result.request_id:
        raise SmokeValidationError("metadata request_id does not match InferenceResult")

    return {
        "request_id": result.request_id,
        "multiview_count": len(result.multiview_image_paths),
        "video_skipped": skip_video,
        "zip_mesh_assets": sorted(EXPECTED_ZIP_MESH_ASSETS),
        "metadata_path": str(metadata_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one real InstantMesh GPU inference and validate every output artifact."
    )
    parser.add_argument("--image", required=True, type=Path, help="Input image path")
    parser.add_argument("--model", required=True, choices=sorted(MODELS), help="Model variant")
    parser.add_argument(
        "--checkpoint-dir", required=True, type=Path, help="Directory containing fine-tuned checkpoints"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Root for the isolated UUID request output"
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    parser.add_argument("--steps", type=int, default=75, help="Diffusion steps (default: 75)")
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Debug only: skip orbit rendering and MP4 validation",
    )
    return parser


def run(args: argparse.Namespace) -> InferenceResult:
    """Initialize the production service and execute one unmocked request."""

    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; real InstantMesh smoke inference requires a GPU")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    image_path = args.image.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"input image does not exist: {image_path}")
    if args.seed < 0:
        raise ValueError("--seed must be zero or greater")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    selected = MODELS[args.model]
    settings = InstantMeshSettings(outputs_root=output_dir)
    service = InstantMeshService(settings=settings)
    service_variant = service.registry.pretrained_name
    if selected.checkpoint_filename is not None:
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_dir}")
        checkpoint_path = checkpoint_dir / selected.checkpoint_filename
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"checkpoint for {args.model} does not exist: {checkpoint_path}"
            )
        service_variant = args.model
        service.register_checkpoint(
            service_variant,
            checkpoint_path,
            description=selected.label,
        )

    service.initialize()
    overlay_report = service.select_variant(service_variant)
    compatibility = overlay_report or service.base_checkpoint_report
    print("Checkpoint compatibility:")
    if compatibility is None:
        print(json.dumps({"variant": service_variant, "status": "pretrained restore succeeded"}, indent=2))
    else:
        print(json.dumps(compatibility.to_dict(), indent=2))

    result = service.infer(
        InferenceRequest(
            input_image_path=image_path,
            model_variant=service_variant,
            remove_background=False,
            seed=args.seed,
            diffusion_steps=args.steps,
            skip_video=args.skip_video,
        )
    )
    verification = validate_result(result, skip_video=args.skip_video)
    print("Smoke verification:")
    print(json.dumps(verification, indent=2))
    print(f"Request output: {result.metadata_path.parent}")
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run(args)
    except Exception as error:
        logging.getLogger(__name__).exception("GPU smoke inference failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
