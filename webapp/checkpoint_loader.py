"""Safe checkpoint parsing and compatibility-aware state application."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .types import CheckpointLoadReport


STATE_DICT_KEYS = (
    "model_state_dict",
    "state_dict",
    "model",
    "module",
    "net",
    "network",
    "weights",
)
INSPECTED_PREFIXES = (
    "module.",
    "model.",
    "lrm_generator.",
    "reconstruction_model.",
    "instant_mesh.",
)


class CheckpointError(RuntimeError):
    """Base class for actionable checkpoint failures."""


class InvalidCheckpointError(CheckpointError):
    """Raised when a file does not contain a usable state dictionary."""


class IncompatibleCheckpointError(CheckpointError):
    """Raised when too few checkpoint tensors match the reconstruction model."""


def load_checkpoint_cpu(path: Path) -> Any:
    """Load a checkpoint on CPU using PyTorch's restricted loader."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise InvalidCheckpointError(f"checkpoint does not exist or is not a file: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise InvalidCheckpointError(f"could not safely load checkpoint '{checkpoint_path}': {error}") from error


def extract_state_dict(payload: Any) -> tuple[Mapping[str, torch.Tensor], str]:
    """Select the notebook-supported state dictionary from a checkpoint payload."""

    if not isinstance(payload, Mapping):
        raise InvalidCheckpointError(
            f"checkpoint root must be a mapping, received {type(payload).__name__}"
        )

    candidates: list[tuple[Mapping[str, torch.Tensor], str, int]] = []

    def add_candidate(value: Any, path: str, rank: int) -> None:
        if not isinstance(value, Mapping):
            return
        tensors = {
            key: tensor
            for key, tensor in value.items()
            if isinstance(key, str) and torch.is_tensor(tensor)
        }
        if tensors:
            candidates.append((tensors, path, rank))

    add_candidate(payload, "<root>", len(STATE_DICT_KEYS))
    for rank, key in enumerate(STATE_DICT_KEYS):
        add_candidate(payload.get(key), key, rank)

    if not candidates:
        raise InvalidCheckpointError("checkpoint contains no supported tensor state dictionary")
    state_dict, path, _ = min(candidates, key=lambda item: (item[2], -len(item[0])))
    return state_dict, path


def _matching_name(
    checkpoint_name: str, model_keys: set[str]
) -> tuple[str | None, tuple[str, ...]]:
    if checkpoint_name in model_keys:
        return checkpoint_name, ()

    candidate = checkpoint_name
    stripped: list[str] = []
    while True:
        prefix = next((item for item in INSPECTED_PREFIXES if candidate.startswith(item)), None)
        if prefix is None:
            return None, tuple(stripped)
        candidate = candidate[len(prefix) :]
        stripped.append(prefix)
        if candidate in model_keys:
            return candidate, tuple(stripped)


def prepare_compatible_state_dict(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    checkpoint_path: Path,
    state_dict_path: str,
    minimum_compatibility: float,
) -> tuple[dict[str, torch.Tensor], CheckpointLoadReport]:
    """Normalize inspected wrappers and reject overwhelmingly incompatible states."""

    if not 0 < minimum_compatibility <= 1:
        raise ValueError("minimum_compatibility must be in (0, 1]")

    model_state = model.state_dict()
    model_keys = set(model_state)
    compatible: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    shape_mismatches: list[str] = []
    stripped_counts = {prefix: 0 for prefix in INSPECTED_PREFIXES}

    for original_name, tensor in state_dict.items():
        matched_name, stripped = _matching_name(original_name, model_keys)
        if matched_name is None:
            unexpected.append(original_name)
            continue
        if tuple(tensor.shape) != tuple(model_state[matched_name].shape):
            shape_mismatches.append(
                f"{original_name}: checkpoint {tuple(tensor.shape)} != model "
                f"{tuple(model_state[matched_name].shape)}"
            )
            continue
        if matched_name in compatible:
            raise InvalidCheckpointError(
                f"multiple checkpoint keys normalize to model parameter '{matched_name}'"
            )
        compatible[matched_name] = tensor
        for prefix in stripped:
            stripped_counts[prefix] += 1

    ratio = len(compatible) / max(len(state_dict), 1)
    if not compatible or ratio < minimum_compatibility:
        raise IncompatibleCheckpointError(
            f"checkpoint '{checkpoint_path}' is incompatible: {len(compatible)}/"
            f"{len(state_dict)} tensors match ({ratio:.1%}); required at least "
            f"{minimum_compatibility:.1%}"
        )

    missing = tuple(key for key in model_state if key not in compatible)
    report = CheckpointLoadReport(
        checkpoint_path=Path(checkpoint_path),
        state_dict_path=state_dict_path,
        checkpoint_tensor_count=len(state_dict),
        compatible_tensor_count=len(compatible),
        compatibility_ratio=ratio,
        stripped_prefix_counts={key: value for key, value in stripped_counts.items() if value},
        missing_keys=missing,
        unexpected_keys=tuple(unexpected),
        shape_mismatches=tuple(shape_mismatches),
    )
    return compatible, report


def apply_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    minimum_compatibility: float = 0.10,
    logger: logging.Logger | None = None,
) -> CheckpointLoadReport:
    """Apply full or partial compatible weights to an already-pretrained model."""

    log = logger or logging.getLogger(__name__)
    payload = load_checkpoint_cpu(checkpoint_path)
    state_dict, state_path = extract_state_dict(payload)
    compatible, report = prepare_compatible_state_dict(
        model,
        state_dict,
        checkpoint_path=Path(checkpoint_path),
        state_dict_path=state_path,
        minimum_compatibility=minimum_compatibility,
    )
    incompatible = model.load_state_dict(compatible, strict=False)
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(dict.fromkeys((*report.unexpected_keys, *incompatible.unexpected_keys)))
    report = CheckpointLoadReport(
        checkpoint_path=report.checkpoint_path,
        state_dict_path=report.state_dict_path,
        checkpoint_tensor_count=report.checkpoint_tensor_count,
        compatible_tensor_count=report.compatible_tensor_count,
        compatibility_ratio=report.compatibility_ratio,
        stripped_prefix_counts=report.stripped_prefix_counts,
        missing_keys=missing,
        unexpected_keys=unexpected,
        shape_mismatches=report.shape_mismatches,
    )
    log.info(
        "Loaded checkpoint %s: compatible=%d/%d (%.1f%%), missing=%d, unexpected=%d, "
        "shape_mismatches=%d",
        report.checkpoint_path,
        report.compatible_tensor_count,
        report.checkpoint_tensor_count,
        report.compatibility_ratio * 100,
        len(report.missing_keys),
        len(report.unexpected_keys),
        len(report.shape_mismatches),
    )
    if report.missing_keys:
        log.warning("Missing checkpoint keys (first 20): %s", report.missing_keys[:20])
    if report.unexpected_keys:
        log.warning("Unexpected checkpoint keys (first 20): %s", report.unexpected_keys[:20])
    if report.shape_mismatches:
        log.warning("Shape mismatches (first 20): %s", report.shape_mismatches[:20])
    return report
