"""Shared value objects for the InstantMesh inference service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelVariant:
    """A named reconstruction checkpoint layered over the pretrained model."""

    name: str
    checkpoint_path: Path | None = None
    description: str = ""


@dataclass(frozen=True)
class InferenceRequest:
    """Input accepted by :class:`InstantMeshService`."""

    input_image_path: Path
    model_variant: str = "pretrained"
    remove_background: bool = False
    seed: int | None = None
    diffusion_steps: int | None = None
    skip_video: bool = False


@dataclass(frozen=True)
class InferenceResult:
    """Validated files produced for one inference request."""

    request_id: str
    processed_image_path: Path
    multiview_image_paths: tuple[Path, ...]
    video_path: Path | None
    glb_path: Path
    obj_path: Path
    mtl_path: Path | None
    texture_paths: tuple[Path, ...]
    zip_path: Path
    metadata_path: Path

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the result."""

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            return value

        return {key: convert(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Compatibility details from applying a reconstruction checkpoint."""

    checkpoint_path: Path
    state_dict_path: str
    checkpoint_tensor_count: int
    compatible_tensor_count: int
    compatibility_ratio: float
    stripped_prefix_counts: dict[str, int] = field(default_factory=dict)
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()
    shape_mismatches: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report."""

        data = asdict(self)
        data["checkpoint_path"] = str(self.checkpoint_path)
        return data


@dataclass(frozen=True)
class OutputPaths:
    """Canonical paths reserved for a single request."""

    request_id: str
    root: Path
    processed_image: Path
    multiview_grid: Path
    multiview_images: tuple[Path, ...]
    video: Path
    glb: Path
    obj: Path
    mtl: Path
    texture: Path
    archive: Path
    metadata: Path
