"""Per-request output allocation, validation, metadata, and archiving."""

from __future__ import annotations

import json
import uuid
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .types import InferenceResult, OutputPaths


class OutputValidationError(RuntimeError):
    """Raised when inference did not produce its required artifact contract."""


class OutputManager:
    """Own the isolated ``outputs/<uuid>/`` lifecycle for inference requests."""

    def __init__(self, outputs_root: Path) -> None:
        self.outputs_root = Path(outputs_root).expanduser().resolve()

    def create_request_paths(self, request_id: str | None = None) -> OutputPaths:
        """Create and return a collision-resistant request directory."""

        identifier = request_id or str(uuid.uuid4())
        try:
            parsed = uuid.UUID(identifier)
        except ValueError as error:
            raise OutputValidationError(f"request_id must be a valid UUID: {identifier}") from error
        identifier = str(parsed)
        root = (self.outputs_root / identifier).resolve()
        if root.parent != self.outputs_root:
            raise OutputValidationError("request output path escaped the configured outputs root")
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise OutputValidationError(f"request output directory already exists: {root}") from error

        multiview_dir = root / "multiview"
        mesh_dir = root / "mesh"
        multiview_dir.mkdir()
        mesh_dir.mkdir()
        return OutputPaths(
            request_id=identifier,
            root=root,
            processed_image=root / "processed_input.png",
            multiview_grid=root / "multiview_grid.png",
            multiview_images=tuple(multiview_dir / f"view_{index:02d}.png" for index in range(6)),
            video=root / "orbit.mp4",
            glb=mesh_dir / "mesh.glb",
            obj=mesh_dir / "mesh.obj",
            mtl=mesh_dir / "mesh.mtl",
            texture=mesh_dir / "mesh.png",
            archive=root / "artifacts.zip",
            metadata=root / "metadata.json",
        )

    @staticmethod
    def _require_nonempty(paths: Iterable[Path], label: str) -> None:
        failures = [str(path) for path in paths if not path.is_file() or path.stat().st_size <= 0]
        if failures:
            raise OutputValidationError(f"missing or empty {label}: {', '.join(failures)}")

    def write_metadata(self, paths: OutputPaths, metadata: dict[str, Any]) -> None:
        """Write deterministic, human-readable request metadata."""

        paths.metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        self._require_nonempty((paths.metadata,), "metadata file")

    def create_archive(self, paths: OutputPaths) -> None:
        """Zip every produced artifact except the archive itself."""

        with zipfile.ZipFile(paths.archive, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in sorted(paths.root.rglob("*")):
                if file_path.is_file() and file_path != paths.archive:
                    archive.write(file_path, file_path.relative_to(paths.root))
        self._require_nonempty((paths.archive,), "zip archive")

    def validate_and_build_result(
        self,
        paths: OutputPaths,
        *,
        require_video: bool = True,
        require_mtl: bool,
        texture_paths: tuple[Path, ...],
    ) -> InferenceResult:
        """Validate the complete output contract and construct ``InferenceResult``."""

        self._require_nonempty(
            (
                paths.processed_image,
                paths.multiview_grid,
                *paths.multiview_images,
                paths.glb,
                paths.obj,
                paths.metadata,
                paths.archive,
            ),
            "required inference output",
        )
        if require_video:
            self._require_nonempty((paths.video,), "MP4 video")
        if require_mtl:
            self._require_nonempty((paths.mtl,), "MTL file")
        if texture_paths:
            self._require_nonempty(texture_paths, "texture file")
        elif require_mtl:
            raise OutputValidationError("textured OBJ output requires at least one texture file")

        return InferenceResult(
            request_id=paths.request_id,
            processed_image_path=paths.processed_image,
            multiview_image_paths=paths.multiview_images,
            video_path=paths.video if require_video else None,
            glb_path=paths.glb,
            obj_path=paths.obj,
            mtl_path=paths.mtl if require_mtl else None,
            texture_paths=texture_paths,
            zip_path=paths.archive,
            metadata_path=paths.metadata,
        )
