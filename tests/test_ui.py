from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from webapp.types import InferenceResult
from webapp.ui import (
    CleanupPolicy,
    InstantMeshUIController,
    cleanup_stale_outputs,
    create_app,
    load_registry_configuration,
    sanitize_metadata,
)


class UIConfigurationTests(unittest.TestCase):
    def test_repository_registry_populates_all_enabled_models(self):
        with tempfile.TemporaryDirectory() as directory:
            loaded = load_registry_configuration(
                Path("configs/model_registry.yaml"), Path(directory)
            )

        self.assertEqual(loaded.default_variant, "instantmesh_pretrained")
        self.assertEqual(len(loaded.choices), 5)
        self.assertEqual(
            loaded.registry.get("instantmesh_i4").checkpoint_path.name,
            "best_meshval_combined_I4_variation.pt",
        )

    def test_create_app_does_not_construct_service(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("webapp.ui.InstantMeshService") as service_class:
                demo = create_app(
                    checkpoint_dir=root / "ckpts",
                    output_dir=root / "outputs",
                    registry_config=Path("configs/model_registry.yaml"),
                )

        self.assertIsNotNone(demo)
        service_class.assert_not_called()


class CleanupTests(unittest.TestCase):
    def test_cleanup_removes_only_expired_uuid_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / str(uuid.uuid4())
            fresh = root / str(uuid.uuid4())
            unrelated = root / "keep-me"
            for path in (old, fresh, unrelated):
                path.mkdir()
            now = time.time()
            os.utime(old, (now - 7200, now - 7200))
            policy = CleanupPolicy(retention_seconds=3600, interval_seconds=60)

            removed = cleanup_stale_outputs(root, policy, now=now)

            self.assertEqual(removed, (old.resolve(),))
            self.assertFalse(old.exists())
            self.assertTrue(fresh.exists())
            self.assertTrue(unrelated.exists())


class PublicOutputTests(unittest.TestCase):
    def test_metadata_sanitization_removes_absolute_paths(self):
        sanitized = sanitize_metadata(
            {
                "input_image_path": "/srv/gradio/input.png",
                "checkpoint_report": {"checkpoint_path": r"D:\models\fine.pt"},
                "model_variant": "instantmesh_i1",
            }
        )

        self.assertEqual(sanitized["input_image_path"], "input.png")
        self.assertEqual(sanitized["checkpoint_report"]["checkpoint_path"], "fine.pt")
        self.assertEqual(sanitized["model_variant"], "instantmesh_i1")

    def test_controller_reuses_one_lazy_service(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.png"
            input_path.write_bytes(b"image")
            metadata_path = root / "metadata.json"
            metadata_path.write_text(
                json.dumps({"input_image_path": str(input_path.resolve())}), encoding="utf-8"
            )
            result = InferenceResult(
                request_id=str(uuid.uuid4()),
                processed_image_path=root / "processed.png",
                multiview_image_paths=tuple(root / f"view_{index}.png" for index in range(6)),
                video_path=None,
                glb_path=root / "mesh.glb",
                obj_path=root / "mesh.obj",
                mtl_path=root / "mesh.mtl",
                texture_paths=(root / "mesh.png",),
                zip_path=root / "artifacts.zip",
                metadata_path=metadata_path,
            )

            class FakeService:
                def __init__(self) -> None:
                    self.requests = []

                def infer(self, request):
                    self.requests.append(request)
                    return result

            service = FakeService()
            creations = 0

            def factory():
                nonlocal creations
                creations += 1
                return service

            controller = InstantMeshUIController(
                factory,
                outputs_root=root / "outputs",
                cleanup_policy=CleanupPolicy(),
            )
            self.assertFalse(controller.service_created)
            first = controller.generate(input_path, "pretrained", 42, 75, True, False)
            second = controller.generate(input_path, "pretrained", 42, 75, False, False)

        self.assertEqual(creations, 1)
        self.assertEqual(len(service.requests), 2)
        self.assertFalse(service.requests[0].skip_video)
        self.assertFalse(service.requests[1].skip_video)
        self.assertEqual(first[4]["input_image_path"], "input.png")
        self.assertEqual(second[2], str(result.glb_path))


if __name__ == "__main__":
    unittest.main()
