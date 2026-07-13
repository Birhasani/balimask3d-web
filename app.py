#!/usr/bin/env python3
"""Launch the lazy, single-GPU InstantMesh Gradio application."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from webapp.ui import CleanupPolicy, UIConfigurationError, create_app


REPOSITORY_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser without constructing model objects."""

    parser = argparse.ArgumentParser(description="Launch the InstantMesh Gradio web interface.")
    parser.add_argument("--share", action="store_true", help="Create a Gradio public share link")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPOSITORY_ROOT / "ckpts",
        help="Directory containing official and fine-tuned checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs",
        help="Root directory for UUID-scoped inference outputs",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "model_registry.yaml",
        help="Model registry YAML used to populate the dropdown",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose server logging")
    parser.add_argument(
        "--output-retention-hours",
        type=float,
        default=float(os.getenv("INSTANTMESH_OUTPUT_RETENTION_HOURS", "24")),
        help="Minimum age before a completed UUID output may be removed (default: 24)",
    )
    parser.add_argument(
        "--cleanup-interval-minutes",
        type=float,
        default=float(os.getenv("INSTANTMESH_CLEANUP_INTERVAL_MINUTES", "30")),
        help="Minimum interval between stale-output cleanup passes (default: 30)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build and launch Gradio; model weights remain lazy until Generate is clicked."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        cleanup_policy = CleanupPolicy.from_hours(
            retention_hours=args.output_retention_hours,
            interval_minutes=args.cleanup_interval_minutes,
        )
        demo = create_app(
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            registry_config=args.config,
            cleanup_policy=cleanup_policy,
        )
    except (OSError, ValueError, UIConfigurationError) as error:
        logging.getLogger(__name__).error("Could not configure InstantMesh UI: %s", error)
        return 1

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
