#!/usr/bin/env python3
"""Safely inspect PyTorch checkpoint structure without loading a model."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


WRAPPER_KEYS = (
    "model_state_dict",
    "state_dict",
    "model",
    "module",
    "net",
    "network",
    "weights",
)
COMMON_PREFIXES = (
    "module.",
    "model.",
    "lrm_generator.",
    "reconstruction_model.",
    "instant_mesh.",
)
INSTANTMESH_FAMILIES = ("encoder.", "transformer.", "synthesizer.")
PARAMETER_LIMIT = 50


def _direct_tensors(value: Any) -> list[tuple[str, torch.Tensor]]:
    if not isinstance(value, Mapping):
        return []
    return [
        (key, item)
        for key, item in value.items()
        if isinstance(key, str) and torch.is_tensor(item)
    ]


def find_state_dict_candidates(payload: Any) -> list[dict[str, Any]]:
    """Find mappings containing direct tensor values, without copying tensors."""
    candidates: list[dict[str, Any]] = []
    visited: set[int] = set()

    def visit(value: Any, path: str, depth: int, wrapper_rank: int | None) -> None:
        if not isinstance(value, Mapping) or id(value) in visited or depth > 3:
            return
        visited.add(id(value))

        tensors = _direct_tensors(value)
        if tensors:
            candidates.append(
                {
                    "path": path,
                    "mapping": value,
                    "tensors": tensors,
                    "wrapper_rank": wrapper_rank,
                }
            )

        for key, child in value.items():
            if not isinstance(key, str) or not isinstance(child, Mapping):
                continue
            rank = WRAPPER_KEYS.index(key) if key in WRAPPER_KEYS else None
            child_path = key if path == "<root>" else f"{path}.{key}"
            visit(child, child_path, depth + 1, rank)

    visit(payload, "<root>", 0, None)
    return candidates


def select_primary_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    known = [candidate for candidate in candidates if candidate["wrapper_rank"] is not None]
    if known:
        return min(
            known,
            key=lambda candidate: (candidate["wrapper_rank"], -len(candidate["tensors"])),
        )
    return max(candidates, key=lambda candidate: len(candidate["tensors"]))


def normalize_parameter_name(name: str) -> tuple[str, set[str]]:
    """Strip repeated leading wrappers and return every detected prefix."""
    normalized = name
    detected: set[str] = set()
    while True:
        match = next((prefix for prefix in COMMON_PREFIXES if normalized.startswith(prefix)), None)
        if match is None:
            break
        detected.add(match)
        normalized = normalized[len(match) :]
    return normalized, detected


def analyze_state_dict(tensors: list[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    prefix_counts = {prefix: 0 for prefix in COMMON_PREFIXES}
    normalized_names: list[str] = []

    for name, _ in tensors:
        normalized, detected = normalize_parameter_name(name)
        normalized_names.append(normalized)
        for prefix in detected:
            prefix_counts[prefix] += 1

    family_counts = {
        family: sum(name.startswith(family) for name in normalized_names)
        for family in INSTANTMESH_FAMILIES
    }
    present = [family for family, count in family_counts.items() if count]
    if len(present) == len(INSTANTMESH_FAMILIES):
        classification = "full"
        reason = "all expected InstantMesh families are present"
    elif present:
        classification = "partial"
        reason = "only a subset of expected InstantMesh families is present"
    else:
        classification = "unknown"
        reason = "no expected InstantMesh families were identified"

    return {
        "prefix_counts": prefix_counts,
        "family_counts": family_counts,
        "classification": classification,
        "reason": reason,
        "tensor_count": len(tensors),
        "scalar_count": sum(tensor.numel() for _, tensor in tensors),
    }


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"checkpoint is not a file: {path}")

    # weights_only prevents arbitrary pickle globals from being constructed.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    candidates = find_state_dict_candidates(payload)
    if not candidates:
        raise ValueError("no likely state dictionary containing tensors was found")

    primary = select_primary_candidate(candidates)
    analysis = analyze_state_dict(primary["tensors"])
    return {
        "path": path,
        "payload": payload,
        "candidates": candidates,
        "primary": primary,
        "analysis": analysis,
    }


def print_report(report: dict[str, Any]) -> None:
    payload = report["payload"]
    primary = report["primary"]
    analysis = report["analysis"]

    print(f"Checkpoint: {report['path']}")
    print(f"Loaded object type: {type(payload).__name__}")
    if isinstance(payload, Mapping):
        print(f"Top-level keys ({len(payload)}): {list(payload.keys())}")
    else:
        print("Top-level keys: <not a mapping>")

    print("Likely state dictionaries:")
    for candidate in report["candidates"]:
        print(f"  - {candidate['path']}: {len(candidate['tensors'])} tensors")
    print(f"Selected state dictionary: {primary['path']}")

    shown = min(PARAMETER_LIMIT, len(primary["tensors"]))
    print(f"Parameters (first {shown} of {len(primary['tensors'])}):")
    for index, (name, tensor) in enumerate(primary["tensors"][:PARAMETER_LIMIT], start=1):
        print(f"  [{index:02d}] {name} | shape={tuple(tensor.shape)} | dtype={tensor.dtype}")

    print(f"Tensor count: {analysis['tensor_count']}")
    print(f"Scalar value count: {analysis['scalar_count']}")
    print("Common prefix counts:")
    for prefix, count in analysis["prefix_counts"].items():
        print(f"  {prefix} {count}")
    print("InstantMesh family counts:")
    for family, count in analysis["family_counts"].items():
        print(f"  {family} {count}")
    print(f"Coverage classification: {analysis['classification']}")
    print(f"Classification reason: {analysis['reason']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a PyTorch checkpoint safely on CPU without loading InstantMesh."
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint file path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = inspect_checkpoint(args.checkpoint)
        print_report(report)
    except Exception as error:
        print(f"error: failed to inspect checkpoint: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
