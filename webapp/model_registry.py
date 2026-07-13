"""Thread-safe registry of pretrained and fine-tuned model variants."""

from __future__ import annotations

import re
import threading
from pathlib import Path

from .types import ModelVariant


_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ModelRegistryError(ValueError):
    """Raised for invalid or conflicting model registry operations."""


class UnknownModelVariantError(KeyError):
    """Raised when a requested model variant is not registered."""


class ModelRegistry:
    """Store checkpoint metadata without constructing model instances."""

    def __init__(self, pretrained_name: str = "pretrained") -> None:
        self._lock = threading.RLock()
        self._pretrained_name = pretrained_name
        self._variants: dict[str, ModelVariant] = {
            pretrained_name: ModelVariant(
                name=pretrained_name,
                checkpoint_path=None,
                description="Official pretrained InstantMesh reconstruction model",
            )
        }

    @property
    def pretrained_name(self) -> str:
        return self._pretrained_name

    def register(self, variant: ModelVariant, *, replace: bool = False) -> None:
        """Register a variant; fine-tuned variants must provide a checkpoint path."""

        if not _VALID_NAME.fullmatch(variant.name):
            raise ModelRegistryError(
                "variant name must start with an alphanumeric character and contain only "
                "letters, numbers, '.', '_' or '-'"
            )
        if variant.name == self._pretrained_name and variant.checkpoint_path is not None:
            raise ModelRegistryError("the pretrained registry entry cannot reference a fine-tuned checkpoint")
        if variant.name != self._pretrained_name and variant.checkpoint_path is None:
            raise ModelRegistryError("a fine-tuned variant requires checkpoint_path")

        normalized = variant
        if variant.checkpoint_path is not None:
            normalized = ModelVariant(
                name=variant.name,
                checkpoint_path=Path(variant.checkpoint_path).expanduser().resolve(),
                description=variant.description,
            )
        with self._lock:
            if variant.name in self._variants and not replace:
                raise ModelRegistryError(f"model variant already registered: {variant.name}")
            self._variants[variant.name] = normalized

    def register_checkpoint(
        self, name: str, checkpoint_path: Path, *, description: str = "", replace: bool = False
    ) -> None:
        """Convenience wrapper for registering a fine-tuned checkpoint."""

        self.register(ModelVariant(name, checkpoint_path, description), replace=replace)

    def get(self, name: str) -> ModelVariant:
        """Return a variant or raise an actionable error."""

        with self._lock:
            try:
                return self._variants[name]
            except KeyError as error:
                available = ", ".join(sorted(self._variants))
                raise UnknownModelVariantError(
                    f"unknown model variant '{name}'; available variants: {available}"
                ) from error

    def list(self) -> tuple[ModelVariant, ...]:
        """Return an immutable, name-sorted registry snapshot."""

        with self._lock:
            return tuple(self._variants[name] for name in sorted(self._variants))
