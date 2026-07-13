"""Locked, reusable InstantMesh inference orchestration.

This module composes the official InstantMesh model, camera, preprocessing, and
export utilities. It intentionally contains no model architecture definitions.
"""

from __future__ import annotations

import logging
import random
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .checkpoint_loader import (
    CheckpointError,
    apply_checkpoint,
    extract_state_dict,
    load_checkpoint_cpu,
    prepare_compatible_state_dict,
)
from .model_registry import ModelRegistry
from .output_manager import OutputManager
from .preprocessing import preprocess_image, split_multiview_grid
from .settings import InstantMeshSettings
from .types import CheckpointLoadReport, InferenceRequest, InferenceResult, ModelVariant, OutputPaths


class InferenceServiceError(RuntimeError):
    """Raised when model initialization, switching, or inference cannot complete."""


class InstantMeshService:
    """Maintain one pretrained reconstruction model and serialize GPU operations.

    Construction is side-effect free. Models are initialized on the first call
    to :meth:`initialize`, :meth:`select_variant`, or :meth:`infer`.
    """

    def __init__(
        self,
        settings: InstantMeshSettings | None = None,
        registry: ModelRegistry | None = None,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings or InstantMeshSettings()
        self.registry = registry or ModelRegistry()
        self.output_manager = OutputManager(self.settings.outputs_root)
        self.logger = logger or logging.getLogger(__name__)
        self._gpu_lock = threading.RLock()
        self._pipeline: Any | None = None
        self._model: torch.nn.Module | None = None
        self._config: Any | None = None
        self._infer_config: dict[str, Any] = {}
        self._base_checkpoint_path: Path | None = None
        self._base_checkpoint_report: CheckpointLoadReport | None = None
        self._active_variant: str | None = None
        self._active_checkpoint_report: CheckpointLoadReport | None = None

    @property
    def active_variant(self) -> str | None:
        """Name of the reconstruction weights currently active on GPU."""

        return self._active_variant

    @property
    def base_checkpoint_report(self) -> CheckpointLoadReport | None:
        """Compatibility report from the latest official pretrained restore."""

        return self._base_checkpoint_report

    @property
    def active_checkpoint_report(self) -> CheckpointLoadReport | None:
        """Compatibility report for the active fine-tuned overlay, if any."""

        return self._active_checkpoint_report

    @property
    def gpu_lock(self) -> threading.RLock:
        """Expose the process-local lock for integrations that need coordination."""

        return self._gpu_lock

    def register_checkpoint(
        self, name: str, checkpoint_path: Path, *, description: str = "", replace: bool = False
    ) -> None:
        """Register a fine-tuned reconstruction variant without loading it."""

        self.registry.register_checkpoint(
            name, checkpoint_path, description=description, replace=replace
        )

    def initialize(self) -> None:
        """Load official pretrained reconstruction weights, then Zero123++."""

        with self._gpu_lock:
            self._ensure_initialized_locked()

    def _ensure_initialized_locked(self) -> None:
        if self._model is not None and self._pipeline is not None:
            return
        if not torch.cuda.is_available():
            raise InferenceServiceError(
                "InstantMesh inference requires CUDA; CPU is supported only for infrastructure tests"
            )

        try:
            from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
            from omegaconf import OmegaConf

            from src.utils.train_util import instantiate_from_config
        except ImportError as error:
            raise InferenceServiceError(
                "InstantMesh runtime dependencies are unavailable; install the repository requirements"
            ) from error

        config_path = Path(self.settings.config_path)
        if not config_path.is_file():
            raise InferenceServiceError(f"InstantMesh config does not exist: {config_path}")

        try:
            config = OmegaConf.load(config_path)
            model = instantiate_from_config(config.model_config)
            base_path = self._resolve_checkpoint_path(
                configured_path=config.infer_config.model_path,
                filename=self.settings.base_checkpoint_filename,
            )
            self._model = model
            self._base_checkpoint_path = base_path
            self._restore_pretrained_locked()

            reconstruction_device = torch.device(self.settings.reconstruction_device)
            model = model.to(reconstruction_device).float()
            model.init_flexicubes_geometry(reconstruction_device, fovy=30.0)
            model.eval()
            self._model = model
            self._active_variant = self.registry.pretrained_name

            diffusion_dtype = torch.float16 if self.settings.use_fp16_diffusion else torch.float32
            pipeline = DiffusionPipeline.from_pretrained(
                self.settings.diffusion_model_id,
                custom_pipeline="zero123plus",
                torch_dtype=diffusion_dtype,
                cache_dir=str(self.settings.model_cache_dir),
                # This executes custom pipeline code. zero123plus/pipeline.py
                # must come from the trusted InstantMesh project source.
                trust_remote_code=True,
            )
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipeline.scheduler.config,
                timestep_spacing="trailing",
            )
            unet_path = self._resolve_checkpoint_path(
                configured_path=config.infer_config.unet_path,
                filename=self.settings.unet_filename,
            )
            unet_state = load_checkpoint_cpu(unet_path)
            if not isinstance(unet_state, dict):
                raise InferenceServiceError(
                    f"white-background UNet checkpoint is not a state dictionary: {unet_path}"
                )
            pipeline.unet.load_state_dict(unet_state, strict=True)
            self._pipeline = pipeline.to(torch.device(self.settings.diffusion_device))
            self._config = config
            self._infer_config = OmegaConf.to_container(
                config.infer_config, resolve=True
            )
            self.logger.info("Initialized official pretrained InstantMesh model")
        except Exception as error:
            self._pipeline = None
            self._model = None
            self._config = None
            self._active_variant = None
            if isinstance(error, InferenceServiceError):
                raise
            raise InferenceServiceError(f"failed to initialize InstantMesh: {error}") from error

    def _resolve_checkpoint_path(self, *, configured_path: str, filename: str) -> Path:
        configured = Path(str(configured_path))
        if not configured.is_absolute():
            configured = self.settings.repo_root / configured
        if configured.is_file():
            return configured.resolve()
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise InferenceServiceError(
                f"checkpoint '{filename}' is not local and huggingface_hub is unavailable"
            ) from error
        downloaded = hf_hub_download(
            repo_id=self.settings.model_repository_id,
            filename=filename,
            repo_type="model",
            cache_dir=str(self.settings.model_cache_dir),
        )
        return Path(downloaded).resolve()

    def _restore_pretrained_locked(self) -> None:
        """Reload pristine official weights before every variant overlay."""

        if self._model is None or self._base_checkpoint_path is None:
            raise InferenceServiceError("reconstruction model is not ready for pretrained restore")
        payload = load_checkpoint_cpu(self._base_checkpoint_path)
        state_dict, state_path = extract_state_dict(payload)
        compatible, report = prepare_compatible_state_dict(
            self._model,
            state_dict,
            checkpoint_path=self._base_checkpoint_path,
            state_dict_path=state_path,
            minimum_compatibility=0.90,
        )
        incompatible = self._model.load_state_dict(compatible, strict=False)
        if incompatible.missing_keys:
            raise InferenceServiceError(
                "official pretrained checkpoint does not fully cover the configured model; "
                f"missing keys (first 20): {incompatible.missing_keys[:20]}"
            )
        self.logger.info(
            "Restored pretrained reconstruction state (%d/%d compatible tensors, %d ignored)",
            report.compatible_tensor_count,
            report.checkpoint_tensor_count,
            len(report.unexpected_keys),
        )
        if report.unexpected_keys:
            self.logger.warning(
                "Ignored pretrained checkpoint keys (first 20): %s",
                report.unexpected_keys[:20],
            )
        self._base_checkpoint_report = report
        self._active_checkpoint_report = None

    def select_variant(self, name: str) -> CheckpointLoadReport | None:
        """Atomically restore pretrained weights and apply a selected overlay."""

        with self._gpu_lock:
            self._ensure_initialized_locked()
            return self._select_variant_locked(name)

    def _select_variant_locked(self, name: str) -> CheckpointLoadReport | None:
        if name == self._active_variant:
            return self._active_checkpoint_report
        variant = self.registry.get(name)
        self._restore_pretrained_locked()
        self._active_variant = self.registry.pretrained_name
        self._active_checkpoint_report = None

        if variant.checkpoint_path is None:
            self._model.eval()
            return None
        if not variant.checkpoint_path.is_file():
            raise InferenceServiceError(
                f"checkpoint for variant '{variant.name}' does not exist: {variant.checkpoint_path}"
            )
        try:
            report = apply_checkpoint(
                self._model,
                variant.checkpoint_path,
                minimum_compatibility=self.settings.minimum_checkpoint_compatibility,
                logger=self.logger,
            )
        except CheckpointError as error:
            # The base state remains active because compatibility is checked before application.
            raise InferenceServiceError(
                f"could not activate model variant '{variant.name}': {error}"
            ) from error
        self._model.eval()
        self._active_variant = variant.name
        self._active_checkpoint_report = report
        return report

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run one complete request while excluding concurrent GPU operations."""

        with self._gpu_lock:
            self._ensure_initialized_locked()
            self._select_variant_locked(request.model_variant)
            paths = self.output_manager.create_request_paths()
            try:
                with torch.inference_mode():
                    return self._infer_locked(request, paths)
            except Exception as error:
                if isinstance(error, InferenceServiceError):
                    raise
                raise InferenceServiceError(
                    f"inference request {paths.request_id} failed; partial outputs remain at "
                    f"'{paths.root}': {error}"
                ) from error

    def _infer_locked(self, request: InferenceRequest, paths: OutputPaths) -> InferenceResult:
        from einops import rearrange
        from torchvision.transforms import v2

        from src.utils.camera_util import get_zero123plus_input_cameras
        from src.utils.infer_util import save_video
        from src.utils.mesh_util import save_glb, save_obj_with_mtl

        seed = self.settings.seed if request.seed is None else request.seed
        diffusion_steps = (
            self.settings.diffusion_steps
            if request.diffusion_steps is None
            else request.diffusion_steps
        )
        if seed < 0:
            raise InferenceServiceError("seed must be zero or greater")
        if diffusion_steps <= 0:
            raise InferenceServiceError("diffusion_steps must be positive")
        self._seed_all(seed)
        processed = preprocess_image(
            request.input_image_path,
            paths.processed_image,
            remove_background=request.remove_background,
            foreground_ratio=self.settings.foreground_ratio,
        )

        grid = self._pipeline(
            processed,
            num_inference_steps=diffusion_steps,
        ).images[0].convert("RGB")
        grid.save(paths.multiview_grid, format="PNG")
        split_multiview_grid(grid, paths.multiview_images)

        images = np.asarray(grid, dtype=np.float32) / 255.0
        images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
        images = rearrange(images, "c (n h) (m w) -> (n m) c h w", n=3, m=2)
        images = images.unsqueeze(0).to(torch.device(self.settings.reconstruction_device))
        images = v2.functional.resize(
            images,
            self.settings.image_size,
            interpolation=3,
            antialias=True,
        ).clamp(0, 1)
        input_cameras = get_zero123plus_input_cameras(
            batch_size=1,
            radius=self.settings.input_camera_radius,
            fov=self.settings.input_camera_fov,
        ).to(torch.device(self.settings.reconstruction_device))

        planes = self._model.forward_planes(images, input_cameras)
        mesh_kwargs = dict(self._infer_config)
        mesh_kwargs["texture_resolution"] = self.settings.texture_resolution
        textured = self._model.extract_mesh(
            planes,
            use_texture_map=self.settings.export_texture_map,
            **mesh_kwargs,
        )
        if not self.settings.export_texture_map:
            raise InferenceServiceError(
                "the service output contract requires export_texture_map=True"
            )
        vertices, faces, uvs, mesh_tex_idx, texture_map = textured
        save_obj_with_mtl(
            self._to_numpy(vertices),
            self._to_numpy(uvs),
            self._to_numpy(faces),
            self._to_numpy(mesh_tex_idx),
            self._to_numpy(texture_map.permute(1, 2, 0)),
            str(paths.obj),
        )

        # The official app exports GLB from the vertex-color extraction branch.
        glb_vertices, glb_faces, glb_colors = self._model.extract_mesh(
            planes,
            use_texture_map=False,
            **mesh_kwargs,
        )
        glb_vertices = self._to_numpy(glb_vertices)[:, [1, 2, 0]]
        save_glb(
            glb_vertices,
            self._to_numpy(glb_faces),
            self._to_numpy(glb_colors),
            str(paths.glb),
        )

        if not request.skip_video:
            render_cameras = self._get_render_cameras().to(
                torch.device(self.settings.reconstruction_device)
            )
            frames = self._render_frames(planes, render_cameras)
            save_video(frames, str(paths.video), fps=self.settings.video_fps)

        checkpoint_metadata = (
            self._active_checkpoint_report.to_dict()
            if self._active_checkpoint_report
            else None
        )
        if checkpoint_metadata is not None:
            checkpoint_metadata["checkpoint_path"] = Path(
                checkpoint_metadata["checkpoint_path"]
            ).name
        metadata = {
            "request_id": paths.request_id,
            "input_image_path": Path(request.input_image_path).name,
            "model_variant": self._active_variant,
            "checkpoint_report": checkpoint_metadata,
            "parameters": {
                "seed": seed,
                "diffusion_steps": diffusion_steps,
                "view_count": self.settings.view_count,
                "image_size": self.settings.image_size,
                "input_camera_radius": self.settings.input_camera_radius,
                "input_camera_fov": self.settings.input_camera_fov,
                "orbit_frames": self.settings.orbit_frames,
                "orbit_radius": self.settings.orbit_radius,
                "orbit_elevation": self.settings.orbit_elevation,
                "render_resolution": self.settings.render_resolution,
                "video_fps": self.settings.video_fps,
                "video_skipped": request.skip_video,
                "fp16_diffusion": self.settings.use_fp16_diffusion,
                "export_texture_map": self.settings.export_texture_map,
            },
        }
        self.output_manager.write_metadata(paths, metadata)
        self.output_manager.create_archive(paths)
        return self.output_manager.validate_and_build_result(
            paths,
            require_video=not request.skip_video,
            require_mtl=True,
            texture_paths=(paths.texture,),
        )

    def _get_render_cameras(self) -> torch.Tensor:
        """Build the exact 120-view FlexiCubes orbit used by the notebook."""

        from src.utils.camera_util import get_circular_camera_poses

        c2ws = get_circular_camera_poses(
            M=self.settings.orbit_frames,
            radius=self.settings.orbit_radius,
            elevation=self.settings.orbit_elevation,
        )
        return torch.linalg.inv(c2ws).unsqueeze(0)

    def _render_frames(self, planes: torch.Tensor, render_cameras: torch.Tensor) -> torch.Tensor:
        """Render orbit frames through the existing FlexiCubes geometry path."""

        frames = []
        for start in range(0, render_cameras.shape[1], self.settings.render_chunk_size):
            frame = self._model.forward_geometry(
                planes,
                render_cameras[:, start : start + self.settings.render_chunk_size],
                render_size=self.settings.render_resolution,
            )["img"]
            frames.append(frame)
        return torch.cat(frames, dim=1)[0]

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
