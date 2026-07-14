"""Lazy Gradio interface for :class:`webapp.instantmesh_service.InstantMeshService`.

Importing this module or building the component tree never initializes either
InstantMesh model. The first queued Generate request constructs one reusable
service, whose own lock serializes checkpoint switching and GPU inference.
"""

from __future__ import annotations

import ast
import json
import inspect
import logging
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import gradio as gr
import yaml

from .instantmesh_service import InstantMeshService
from .model_registry import ModelRegistry
from .settings import InstantMeshSettings, repository_root
from .types import InferenceRequest, InferenceResult


LOGGER = logging.getLogger(__name__)
GPU_CONCURRENCY_ID = "instantmesh-gpu"
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![\w])([A-Za-z]:[\\/][^\s'\"]+)")
_POSIX_ABSOLUTE_PATH = re.compile(r"(?<![:\w])(/[^\s'\"]+)")


class UIConfigurationError(ValueError):
    """Raised when the UI registry or cleanup policy is invalid."""


@dataclass(frozen=True)
class CleanupPolicy:
    """Age-based policy that retains completed downloads before deletion."""

    retention_seconds: int = 24 * 60 * 60
    interval_seconds: int = 30 * 60

    def __post_init__(self) -> None:
        if self.retention_seconds < 60 * 60:
            raise UIConfigurationError("output retention must be at least one hour")
        if self.interval_seconds <= 0:
            raise UIConfigurationError("cleanup interval must be positive")

    @classmethod
    def from_hours(cls, *, retention_hours: float, interval_minutes: float) -> "CleanupPolicy":
        """Build a policy from user-facing CLI units."""

        return cls(
            retention_seconds=int(retention_hours * 60 * 60),
            interval_seconds=int(interval_minutes * 60),
        )


@dataclass(frozen=True)
class RegistryConfiguration:
    """Validated registry and Gradio dropdown metadata."""

    registry: ModelRegistry
    choices: tuple[str, ...]
    label_to_model_id: Mapping[str, str]
    default_variant: str
    default_label: str


def load_registry_configuration(
    registry_path: Path, checkpoint_dir: Path
) -> RegistryConfiguration:
    """Load enabled model variants without reading checkpoint tensor data."""

    registry_path = Path(registry_path).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not registry_path.is_file():
        raise UIConfigurationError(f"model registry does not exist: {registry_path}")
    try:
        payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise UIConfigurationError(f"could not parse model registry: {error}") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("models"), Mapping):
        raise UIConfigurationError("model registry must contain a 'models' mapping")

    enabled: list[tuple[str, Mapping[str, Any]]] = []
    pretrained_names: list[str] = []
    for name, raw_entry in payload["models"].items():
        if not isinstance(raw_entry, Mapping):
            raise UIConfigurationError(f"model '{name}' configuration must be a mapping")
        if not bool(raw_entry.get("enabled", True)):
            continue
        model_type = str(raw_entry.get("type", "")).lower()
        if model_type not in {"pretrained", "finetuned"}:
            raise UIConfigurationError(
                f"enabled model '{name}' has unsupported type '{model_type}'"
            )
        enabled.append((str(name), raw_entry))
        if model_type == "pretrained":
            pretrained_names.append(str(name))

    if len(pretrained_names) != 1:
        raise UIConfigurationError(
            "model registry must contain exactly one enabled pretrained model"
        )
    registry = ModelRegistry(pretrained_name=pretrained_names[0])
    choices: list[str] = []
    label_to_model_id: dict[str, str] = {}
    for name, entry in enabled:
        label = str(entry.get("label") or name)
        if label in label_to_model_id:
            raise UIConfigurationError(f"duplicate enabled model label: {label}")
        choices.append(label)
        label_to_model_id[label] = name
        if str(entry.get("type")).lower() == "pretrained":
            if entry.get("checkpoint") not in (None, ""):
                raise UIConfigurationError(
                    f"pretrained model '{name}' must not define a fine-tuned checkpoint"
                )
            continue

        checkpoint_name = entry.get("checkpoint")
        if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
            raise UIConfigurationError(f"fine-tuned model '{name}' requires a checkpoint")
        checkpoint_path = (checkpoint_dir / checkpoint_name).resolve()
        try:
            checkpoint_path.relative_to(checkpoint_dir)
        except ValueError as error:
            raise UIConfigurationError(
                f"checkpoint for model '{name}' escapes --checkpoint-dir"
            ) from error
        registry.register_checkpoint(name, checkpoint_path, description=label)

    return RegistryConfiguration(
        registry=registry,
        choices=tuple(choices),
        label_to_model_id=label_to_model_id,
        default_variant=pretrained_names[0],
        default_label=next(
            label for label, model_id in label_to_model_id.items()
            if model_id == pretrained_names[0]
        ),
    )


def normalize_model_selection(
    raw_value: Any, label_to_model_id: Mapping[str, str]
) -> str:
    """Resolve Gradio and legacy dropdown values to an exact registry model ID."""

    valid_model_ids = frozenset(label_to_model_id.values())
    if isinstance(raw_value, (tuple, list)):
        if len(raw_value) != 2:
            raise UIConfigurationError("model selection tuple/list must contain two values")
        for candidate in (raw_value[1], raw_value[0]):
            try:
                return normalize_model_selection(candidate, label_to_model_id)
            except UIConfigurationError:
                continue
        raise UIConfigurationError(f"unknown model selection: {raw_value!r}")

    if isinstance(raw_value, str):
        selected = raw_value.strip()
        if selected in valid_model_ids:
            return selected
        if selected in label_to_model_id:
            return label_to_model_id[selected]
        try:
            legacy_value = ast.literal_eval(selected)
        except (SyntaxError, ValueError):
            legacy_value = None
        if isinstance(legacy_value, (tuple, list)):
            return normalize_model_selection(legacy_value, label_to_model_id)

    raise UIConfigurationError(f"unknown model selection: {raw_value!r}")


def cleanup_stale_outputs(
    outputs_root: Path,
    policy: CleanupPolicy,
    *,
    now: float | None = None,
) -> tuple[Path, ...]:
    """Remove only expired, non-symlink UUID directories below ``outputs_root``."""

    root = Path(outputs_root).expanduser().resolve()
    if not root.is_dir():
        return ()
    cutoff = (time.time() if now is None else now) - policy.retention_seconds
    removed: list[Path] = []
    for candidate in root.iterdir():
        if candidate.is_symlink() or not candidate.is_dir():
            continue
        try:
            uuid.UUID(candidate.name)
        except ValueError:
            continue
        resolved = candidate.resolve()
        if resolved.parent != root or resolved.stat().st_mtime > cutoff:
            continue
        shutil.rmtree(resolved)
        removed.append(resolved)
    return tuple(removed)


def _public_path_name(value: str) -> str:
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return PureWindowsPath(value).name
    return Path(value).name


def sanitize_metadata(value: Any, *, key: str = "") -> Any:
    """Strip absolute server paths from metadata displayed in the browser."""

    if isinstance(value, Mapping):
        return {str(item_key): sanitize_metadata(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_metadata(item, key=key) for item in value]
    if isinstance(value, str):
        is_absolute = Path(value).is_absolute() or bool(re.match(r"^[A-Za-z]:[\\/]", value))
        if is_absolute or key.endswith("_path"):
            return _public_path_name(value)
    return value


def _safe_error_message(error: Exception) -> str:
    """Return an actionable message with absolute filesystem locations redacted."""

    message = str(error).strip() or error.__class__.__name__
    message = _WINDOWS_ABSOLUTE_PATH.sub("<server-path>", message)
    message = _POSIX_ABSOLUTE_PATH.sub("<server-path>", message)
    return message


class InstantMeshUIController:
    """Lazily own one service and serialize complete UI requests."""

    def __init__(
        self,
        service_factory: Callable[[], InstantMeshService],
        *,
        outputs_root: Path,
        cleanup_policy: CleanupPolicy,
        registry: ModelRegistry,
        label_to_model_id: Mapping[str, str],
        logger: logging.Logger | None = None,
    ) -> None:
        self._service_factory = service_factory
        self._outputs_root = Path(outputs_root).expanduser().resolve()
        self._cleanup_policy = cleanup_policy
        self._registry = registry
        self._label_to_model_id = dict(label_to_model_id)
        self._logger = logger or LOGGER
        self._service: InstantMeshService | None = None
        self._service_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._last_cleanup = 0.0

    @property
    def service_created(self) -> bool:
        """Expose lazy state for diagnostics and module-import tests."""

        return self._service is not None

    def _get_service(self) -> InstantMeshService:
        if self._service is None:
            with self._service_lock:
                if self._service is None:
                    self._service = self._service_factory()
        return self._service

    def _maybe_cleanup(self) -> None:
        now = time.time()
        if now - self._last_cleanup < self._cleanup_policy.interval_seconds:
            return
        removed = cleanup_stale_outputs(self._outputs_root, self._cleanup_policy, now=now)
        self._last_cleanup = now
        if removed:
            self._logger.info("Removed %d stale UUID output directories", len(removed))

    @staticmethod
    def _read_public_metadata(result: InferenceResult) -> dict[str, Any]:
        try:
            raw = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"could not read inference metadata: {error}") from error
        public = sanitize_metadata(raw)
        public["artifacts"] = {
            "glb": result.glb_path.name,
            "zip": result.zip_path.name,
            "video": result.video_path.name if result.video_path else None,
            "multiview_count": len(result.multiview_image_paths),
        }
        return public

    def generate(
        self,
        image_path: str | None,
        model_variant: Any,
        seed: int | float,
        diffusion_steps: int | float,
        remove_background: bool,
        save_video: bool,
    ) -> tuple[str, list[str], str, str | None, dict[str, Any], str, str]:
        """Execute one complete request and return browser-safe Gradio values."""

        try:
            if not image_path:
                raise ValueError("Upload an image or capture one with the webcam first.")
            input_path = Path(image_path).expanduser().resolve()
            if not input_path.is_file():
                raise ValueError("The uploaded image is no longer available; upload it again.")
            seed_value = int(seed)
            step_value = int(diffusion_steps)
            if float(seed) != seed_value or seed_value < 0:
                raise ValueError("Seed must be a non-negative integer.")
            if float(diffusion_steps) != step_value or step_value <= 0:
                raise ValueError("Diffusion steps must be a positive integer.")

            normalized_model_id = normalize_model_selection(
                model_variant, self._label_to_model_id
            )
            selected_variant = self._registry.get(normalized_model_id)
            checkpoint_filename = (
                selected_variant.checkpoint_path.name
                if selected_variant.checkpoint_path is not None
                else None
            )
            self._logger.info(
                "Model selection: raw=%r normalized=%s checkpoint=%s",
                model_variant,
                normalized_model_id,
                checkpoint_filename or "<official pretrained>",
            )

            # This lock spans cleanup, lazy initialization, checkpoint switching,
            # and inference, so a model variant cannot change mid-request.
            with self._request_lock:
                self._maybe_cleanup()
                result = self._get_service().infer(
                    InferenceRequest(
                        input_image_path=input_path,
                        model_variant=normalized_model_id,
                        remove_background=bool(remove_background),
                        seed=seed_value,
                        diffusion_steps=step_value,
                        # Production output always includes the rotating MP4.
                        # The checkbox is retained as a visible contract indicator.
                        skip_video=False,
                    )
                )
                metadata = self._read_public_metadata(result)

            return (
                str(result.processed_image_path),
                [str(path) for path in result.multiview_image_paths],
                str(result.glb_path),
                str(result.video_path) if result.video_path else None,
                metadata,
                str(result.glb_path),
                str(result.zip_path),
            )
        except Exception as error:
            self._logger.exception("InstantMesh UI request failed")
            raise gr.Error(f"Generation failed: {_safe_error_message(error)}") from None


def build_demo(
    controller: InstantMeshUIController,
    registry: RegistryConfiguration,
    *,
    cleanup_policy: CleanupPolicy,
) -> gr.Blocks:
    """Build the component tree and single-concurrency queue."""

    blocks_kwargs: dict[str, Any] = {"title": "InstantMesh"}
    if "delete_cache" in inspect.signature(gr.Blocks).parameters:
        blocks_kwargs["delete_cache"] = (
            cleanup_policy.interval_seconds,
            cleanup_policy.retention_seconds,
        )
    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown(
            "# InstantMesh\nGenerate a textured 3D mesh and rotating preview from one image."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_parameters = inspect.signature(gr.Image).parameters

                def image_input(label: str, source: str):
                    kwargs: dict[str, Any] = {
                        "label": label,
                        "type": "filepath",
                        "image_mode": "RGBA",
                    }
                    if "sources" in image_parameters:
                        kwargs["sources"] = [source]
                    else:
                        kwargs["source"] = source
                    return gr.Image(**kwargs)

                with gr.Tabs():
                    with gr.Tab("Upload"):
                        uploaded_image = image_input("Upload image", "upload")
                    with gr.Tab("Webcam"):
                        webcam_image = image_input("Capture image", "webcam")
                model = gr.Dropdown(
                    label="Reconstruction model",
                    choices=list(registry.choices),
                    value=registry.default_label,
                    interactive=True,
                )
                with gr.Row():
                    seed = gr.Number(label="Seed", value=42, precision=0)
                    steps = gr.Slider(
                        label="Diffusion steps", minimum=1, maximum=100, value=75, step=1
                    )
                with gr.Row():
                    remove_background = gr.Checkbox(label="Remove background", value=True)
                    save_video = gr.Checkbox(
                        label="Save rotating video (required)",
                        value=True,
                        interactive=False,
                    )
                generate = gr.Button("Generate", variant="primary")

            with gr.Column(scale=2):
                processed = gr.Image(label="Processed input", interactive=False)
                gallery = gr.Gallery(
                    label="Six generated views",
                    columns=3,
                    rows=2,
                    object_fit="contain",
                    interactive=False,
                )

        with gr.Row():
            model_viewer = gr.Model3D(
                label="Interactive GLB viewer",
                interactive=False,
            )
            video = gr.Video(label="Rotating preview", format="mp4", interactive=False)

        json_kwargs: dict[str, Any] = {"label": "Inference metadata"}
        if "open" in inspect.signature(gr.JSON).parameters:
            json_kwargs["open"] = False
        metadata = gr.JSON(**json_kwargs)
        with gr.Row():
            glb_download = gr.File(label="Download GLB", interactive=False)
            zip_download = gr.File(
                label="Download OBJ, MTL and texture ZIP", interactive=False
            )

        def generate_from_sources(upload_path, webcam_path, *request_options):
            return controller.generate(upload_path or webcam_path, *request_options)

        click_kwargs: dict[str, Any] = {
            "fn": generate_from_sources,
            "inputs": [
                uploaded_image,
                webcam_image,
                model,
                seed,
                steps,
                remove_background,
                save_video,
            ],
            "outputs": [
                processed,
                gallery,
                model_viewer,
                video,
                metadata,
                glb_download,
                zip_download,
            ],
            "api_name": "generate",
        }
        click_parameters = inspect.signature(generate.click).parameters
        if "concurrency_limit" in click_parameters:
            click_kwargs["concurrency_limit"] = 1
            click_kwargs["concurrency_id"] = GPU_CONCURRENCY_ID
        if "trigger_mode" in click_parameters:
            click_kwargs["trigger_mode"] = "once"
        generate.click(**click_kwargs)

    queue_kwargs: dict[str, Any] = {"max_size": 10}
    queue_parameters = inspect.signature(demo.queue).parameters
    if "default_concurrency_limit" in queue_parameters:
        queue_kwargs["default_concurrency_limit"] = 1
    elif "concurrency_count" in queue_parameters:
        queue_kwargs["concurrency_count"] = 1
    demo.queue(**queue_kwargs)
    return demo


def create_app(
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    registry_config: Path,
    cleanup_policy: CleanupPolicy | None = None,
) -> gr.Blocks:
    """Create an import-safe Gradio app whose service remains lazy."""

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    policy = cleanup_policy or CleanupPolicy()
    registry = load_registry_configuration(registry_config, checkpoint_dir)
    settings = InstantMeshSettings(
        repo_root=repository_root(),
        outputs_root=output_dir,
        model_cache_dir=checkpoint_dir,
    )

    def service_factory() -> InstantMeshService:
        return InstantMeshService(settings=settings, registry=registry.registry)

    controller = InstantMeshUIController(
        service_factory,
        outputs_root=output_dir,
        cleanup_policy=policy,
        registry=registry.registry,
        label_to_model_id=registry.label_to_model_id,
    )
    return build_demo(controller, registry, cleanup_policy=policy)
