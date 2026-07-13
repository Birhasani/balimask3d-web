from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from webapp.checkpoint_loader import (
    IncompatibleCheckpointError,
    apply_checkpoint,
    prepare_compatible_state_dict,
)
from webapp.model_registry import (
    ModelRegistry,
    ModelRegistryError,
    UnknownModelVariantError,
)
from webapp.output_manager import OutputManager, OutputValidationError
from webapp.preprocessing import ImagePreprocessingError, load_input_image, split_multiview_grid
from webapp.settings import InstantMeshSettings
from webapp.types import ModelVariant


class TinyInstantMesh(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2, bias=False)
        self.transformer = torch.nn.Linear(2, 2, bias=False)
        self.synthesizer = torch.nn.Linear(2, 1, bias=False)


class SettingsAndPathTests(unittest.TestCase):
    def test_settings_resolve_repository_paths_and_preserve_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = InstantMeshSettings(repo_root=root)

        self.assertEqual(settings.config_path, root.resolve() / "configs" / "instant-mesh-large.yaml")
        self.assertEqual(settings.outputs_root, root.resolve() / "outputs")
        self.assertEqual(settings.view_count, 6)
        self.assertEqual(settings.diffusion_steps, 75)
        self.assertEqual(settings.video_fps, 8)

    def test_settings_reject_view_order_changes(self):
        with self.assertRaisesRegex(ValueError, "cannot be changed"):
            InstantMeshSettings(view_count=4)

    def test_input_path_and_multiview_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.png"
            Image.new("RGB", (4, 4), (1, 2, 3)).save(input_path)
            loaded = load_input_image(input_path)
            self.assertEqual(loaded.mode, "RGBA")

            grid = np.zeros((6, 4, 3), dtype=np.uint8)
            colors = [(index * 10, index * 10 + 1, index * 10 + 2) for index in range(6)]
            for index, color in enumerate(colors):
                row, column = divmod(index, 2)
                grid[row * 2 : (row + 1) * 2, column * 2 : (column + 1) * 2] = color
            outputs = tuple(root / f"view_{index}.png" for index in range(6))
            tiles = split_multiview_grid(Image.fromarray(grid), outputs)

            self.assertEqual(len(tiles), 6)
            for index, tile in enumerate(tiles):
                self.assertEqual(tile.getpixel((0, 0)), colors[index])
                self.assertGreater(outputs[index].stat().st_size, 0)

    def test_invalid_grid_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = tuple(Path(directory) / f"view_{index}.png" for index in range(6))
            with self.assertRaises(ImagePreprocessingError):
                split_multiview_grid(Image.new("RGB", (5, 5)), outputs)


class RegistryTests(unittest.TestCase):
    def test_registry_registers_and_lists_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fine.pt"
            registry = ModelRegistry()
            registry.register_checkpoint("i4-topology", checkpoint, description="I4")

            variant = registry.get("i4-topology")
            self.assertEqual(variant.checkpoint_path, checkpoint.resolve())
            self.assertEqual([item.name for item in registry.list()], ["i4-topology", "pretrained"])

    def test_registry_rejects_traversal_and_unknown_names(self):
        registry = ModelRegistry()
        with self.assertRaises(ModelRegistryError):
            registry.register(ModelVariant("../escape", Path("checkpoint.pt")))
        with self.assertRaises(UnknownModelVariantError):
            registry.get("missing")


class CheckpointLoaderTests(unittest.TestCase):
    def test_partial_checkpoint_is_applied_over_existing_model(self):
        model = TinyInstantMesh()
        original_encoder = model.encoder.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.pt"
            expected = torch.full_like(model.synthesizer.weight, 7)
            torch.save(
                {"model_state_dict": {"module.lrm_generator.synthesizer.weight": expected}},
                path,
            )
            report = apply_checkpoint(model, path, minimum_compatibility=0.10)

        self.assertTrue(torch.equal(model.synthesizer.weight, expected))
        self.assertTrue(torch.equal(model.encoder.weight, original_encoder))
        self.assertEqual(report.compatibility_ratio, 1.0)
        self.assertEqual(report.stripped_prefix_counts["module."], 1)
        self.assertEqual(report.stripped_prefix_counts["lrm_generator."], 1)
        self.assertIn("encoder.weight", report.missing_keys)

    def test_nearly_all_incompatible_checkpoint_is_rejected_without_loading(self):
        model = TinyInstantMesh()
        before = model.encoder.weight.detach().clone()
        state = {f"unknown.layer.{index}": torch.zeros(1) for index in range(19)}
        state["encoder.weight"] = torch.ones_like(model.encoder.weight)

        with self.assertRaises(IncompatibleCheckpointError):
            prepare_compatible_state_dict(
                model,
                state,
                checkpoint_path=Path("incompatible.pt"),
                state_dict_path="state_dict",
                minimum_compatibility=0.10,
            )
        self.assertTrue(torch.equal(model.encoder.weight, before))

    def test_prefix_is_not_stripped_without_a_model_key_match(self):
        model = TinyInstantMesh()
        state = {"module.not_a_model_key": torch.zeros(1)}
        with self.assertRaises(IncompatibleCheckpointError):
            prepare_compatible_state_dict(
                model,
                state,
                checkpoint_path=Path("bad.pt"),
                state_dict_path="<root>",
                minimum_compatibility=0.10,
            )


class OutputManagerTests(unittest.TestCase):
    @staticmethod
    def write_nonempty(path: Path, content: bytes = b"x") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def test_request_paths_are_uuid_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = OutputManager(Path(directory) / "outputs")
            paths = manager.create_request_paths()

        self.assertEqual(str(uuid.UUID(paths.request_id)), paths.request_id)
        self.assertEqual(paths.root.parent, manager.outputs_root)
        self.assertEqual(len(paths.multiview_images), 6)

    def test_output_contract_validates_and_archives_nonempty_files(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = OutputManager(Path(directory) / "outputs")
            paths = manager.create_request_paths()
            for path in (
                paths.processed_image,
                paths.multiview_grid,
                *paths.multiview_images,
                paths.video,
                paths.glb,
                paths.obj,
                paths.mtl,
                paths.texture,
            ):
                self.write_nonempty(path)
            manager.write_metadata(paths, {"request_id": paths.request_id})
            manager.create_archive(paths)
            result = manager.validate_and_build_result(
                paths, require_mtl=True, texture_paths=(paths.texture,)
            )

            self.assertEqual(result.request_id, paths.request_id)
            self.assertEqual(len(result.multiview_image_paths), 6)
            self.assertGreater(result.zip_path.stat().st_size, 0)
            self.assertEqual(json.loads(result.metadata_path.read_text())["request_id"], paths.request_id)

    def test_missing_or_empty_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = OutputManager(Path(directory) / "outputs")
            paths = manager.create_request_paths()
            paths.processed_image.touch()
            with self.assertRaisesRegex(OutputValidationError, "missing or empty"):
                manager.validate_and_build_result(
                    paths, require_mtl=True, texture_paths=(paths.texture,)
                )


if __name__ == "__main__":
    unittest.main()
