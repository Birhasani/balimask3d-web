from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


MODULE_PATH = Path(__file__).parents[1] / "tools" / "inspect_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("inspect_checkpoint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
inspect_checkpoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspect_checkpoint)


class CheckpointParserTests(unittest.TestCase):
    def save_payload(self, directory: str, payload, name: str = "checkpoint.pt") -> Path:
        path = Path(directory) / name
        torch.save(payload, path)
        return path

    def test_finetuned_wrapper_is_full(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save_payload(
                directory,
                {
                    "model_state_dict": {
                        "encoder.model.weight": torch.zeros(2, 3),
                        "transformer.layers.0.weight": torch.ones(4),
                        "synthesizer.decoder.bias": torch.zeros(1, dtype=torch.float16),
                    },
                    "stage": "I1",
                    "epoch": 1,
                },
            )
            report = inspect_checkpoint.inspect_checkpoint(path)

        self.assertEqual(report["primary"]["path"], "model_state_dict")
        self.assertEqual(report["analysis"]["classification"], "full")
        self.assertEqual(report["analysis"]["tensor_count"], 3)

    def test_official_lrm_generator_checkpoint_is_full(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save_payload(
                directory,
                {
                    "state_dict": {
                        "lrm_generator.encoder.weight": torch.zeros(1),
                        "lrm_generator.transformer.weight": torch.zeros(1),
                        "lrm_generator.synthesizer.weight": torch.zeros(1),
                    }
                },
            )
            report = inspect_checkpoint.inspect_checkpoint(path)

        self.assertEqual(report["analysis"]["classification"], "full")
        self.assertEqual(report["analysis"]["prefix_counts"]["lrm_generator."], 3)

    def test_chained_requested_prefixes_are_detected(self):
        tensors = [
            ("module.model.encoder.weight", torch.zeros(1)),
            ("reconstruction_model.transformer.weight", torch.zeros(1)),
            ("instant_mesh.synthesizer.weight", torch.zeros(1)),
        ]
        result = inspect_checkpoint.analyze_state_dict(tensors)

        self.assertEqual(result["classification"], "full")
        self.assertEqual(result["prefix_counts"]["module."], 1)
        self.assertEqual(result["prefix_counts"]["model."], 1)
        self.assertEqual(result["prefix_counts"]["reconstruction_model."], 1)
        self.assertEqual(result["prefix_counts"]["instant_mesh."], 1)

    def test_partial_and_unknown_classification(self):
        partial = inspect_checkpoint.analyze_state_dict(
            [("synthesizer.decoder.weight", torch.zeros(1))]
        )
        unknown = inspect_checkpoint.analyze_state_dict(
            [("unet.down_blocks.0.weight", torch.zeros(1))]
        )

        self.assertEqual(partial["classification"], "partial")
        self.assertEqual(unknown["classification"], "unknown")

    def test_report_limits_parameters_and_prints_shape_and_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {f"encoder.layer.{i}": torch.zeros(2, dtype=torch.float32) for i in range(55)}
            path = self.save_payload(directory, {"model_state_dict": state})
            report = inspect_checkpoint.inspect_checkpoint(path)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                inspect_checkpoint.print_report(report)

        text = output.getvalue()
        parameter_lines = [line for line in text.splitlines() if line.startswith("  [")]
        self.assertEqual(len(parameter_lines), 50)
        self.assertIn("shape=(2,)", text)
        self.assertIn("dtype=torch.float32", text)
        self.assertNotIn("encoder.layer.54 |", text)

    def test_invalid_and_tensor_free_files_return_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "invalid.pt"
            invalid.write_text("not a checkpoint", encoding="utf-8")
            tensor_free = self.save_payload(directory, {"epoch": 2}, "metadata.pt")

            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(inspect_checkpoint.main(["--checkpoint", str(invalid)]), 1)
                self.assertEqual(inspect_checkpoint.main(["--checkpoint", str(tensor_free)]), 1)

    def test_inspection_does_not_modify_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save_payload(
                directory, {"state_dict": {"encoder.weight": torch.zeros(1)}}
            )
            before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            before_mtime = path.stat().st_mtime_ns
            inspect_checkpoint.inspect_checkpoint(path)
            after_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            after_mtime = path.stat().st_mtime_ns

        self.assertEqual(before_hash, after_hash)
        self.assertEqual(before_mtime, after_mtime)

    def test_loader_is_restricted_and_cpu_only(self):
        fake_payload = {"state_dict": {"encoder.weight": torch.zeros(1)}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.touch()
            with mock.patch.object(inspect_checkpoint.torch, "load", return_value=fake_payload) as load:
                inspect_checkpoint.inspect_checkpoint(path)

        load.assert_called_once_with(path, map_location="cpu", weights_only=True)


if __name__ == "__main__":
    unittest.main()
