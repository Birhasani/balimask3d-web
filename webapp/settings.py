"""Configuration for the notebook-compatible InstantMesh service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def repository_root() -> Path:
    """Return the repository root containing this package."""

    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class InstantMeshSettings:
    """Stable inference defaults copied from the working fine-tuned notebook."""

    repo_root: Path = field(default_factory=repository_root)
    config_path: Path | None = None
    outputs_root: Path | None = None
    model_cache_dir: Path | None = None
    diffusion_model_id: str = "sudo-ai/zero123plus-v1.2"
    model_repository_id: str = "TencentARC/InstantMesh"
    unet_filename: str = "diffusion_pytorch_model.bin"
    base_checkpoint_filename: str = "instant_mesh_large.ckpt"
    seed: int = 42
    diffusion_steps: int = 75
    view_count: int = 6
    image_size: int = 320
    input_camera_radius: float = 4.0
    input_camera_fov: float = 30.0
    orbit_frames: int = 120
    orbit_radius: float = 4.5
    orbit_elevation: float = 20.0
    render_resolution: int = 512
    render_chunk_size: int = 1
    video_fps: int = 8
    texture_resolution: int = 1024
    foreground_ratio: float = 0.85
    use_fp16_diffusion: bool = True
    export_texture_map: bool = True
    save_video: bool = True
    minimum_checkpoint_compatibility: float = 0.10
    diffusion_device: str = "cuda"
    reconstruction_device: str = "cuda"

    def __post_init__(self) -> None:
        root = Path(self.repo_root).resolve()
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(
            self,
            "config_path",
            Path(self.config_path).resolve()
            if self.config_path
            else root / "configs" / "instant-mesh-large.yaml",
        )
        object.__setattr__(
            self,
            "outputs_root",
            Path(self.outputs_root).resolve() if self.outputs_root else root / "outputs",
        )
        object.__setattr__(
            self,
            "model_cache_dir",
            Path(self.model_cache_dir).resolve() if self.model_cache_dir else root / "ckpts",
        )
        if not 0 < self.minimum_checkpoint_compatibility <= 1:
            raise ValueError("minimum_checkpoint_compatibility must be in (0, 1]")
        audited_values = {
            "seed": 42,
            "diffusion_steps": 75,
            "view_count": 6,
            "image_size": 320,
            "input_camera_radius": 4.0,
            "input_camera_fov": 30.0,
            "orbit_frames": 120,
            "orbit_radius": 4.5,
            "orbit_elevation": 20.0,
            "render_resolution": 512,
            "render_chunk_size": 1,
            "video_fps": 8,
            "texture_resolution": 1024,
            "use_fp16_diffusion": True,
            "export_texture_map": True,
            "save_video": True,
        }
        changed = {
            name: (getattr(self, name), expected)
            for name, expected in audited_values.items()
            if getattr(self, name) != expected
        }
        if changed:
            details = ", ".join(
                f"{name}={actual!r} (required {expected!r})"
                for name, (actual, expected) in changed.items()
            )
            raise ValueError(
                "audited InstantMesh inference parameters cannot be changed: " + details
            )
