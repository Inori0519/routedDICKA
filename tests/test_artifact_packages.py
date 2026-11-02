import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main"
VERIFY = ROOT / "verification"
for path in (MAIN, VERIFY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_artifact_packages import (
    _write_package_manifest,
    export_submission_package,
    export_verification_package,
)
from verify_artifact_packages import verify_package


class ArtifactPackageTests(unittest.TestCase):
    def _verification_package(self, directory):
        package = Path(directory)
        for relative in (
            "project/config/paper_suite.json",
            "outputs/manifest.json",
            "outputs/audits/m4_same_point_certificate.json",
        ):
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        _write_package_manifest(package, "verification")
        return package

    def test_nested_run_manifest_is_hashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = self._verification_package(temporary)
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("outputs/manifest.json", manifest["files"])
            self.assertTrue(verify_package(package)["passed"])

    def test_changed_run_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = self._verification_package(temporary)
            (package / "outputs" / "manifest.json").write_text(
                '{"changed": true}\n', encoding="utf-8"
            )
            audit = verify_package(package)
            self.assertFalse(audit["passed"])
            self.assertIn("hash mismatch: outputs/manifest.json", audit["errors"])

    def test_unlisted_old_figure_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = self._verification_package(temporary)
            stale_figure = package / "outputs" / "figures" / "old_curve.png"
            stale_figure.parent.mkdir(parents=True)
            stale_figure.write_bytes(b"old figure")
            audit = verify_package(package)
            self.assertFalse(audit["passed"])
            self.assertIn(
                "unexpected file: outputs/figures/old_curve.png", audit["errors"]
            )

    def test_verification_package_includes_tests_and_omits_unreferenced_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for relative in (
                "raw/current.json",
                "raw/old.json",
                "audits/m4_same_point_certificate.json",
            ):
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            (output / "manifest.json").write_text(
                json.dumps({"point_files": ["raw/current.json"]}), encoding="utf-8"
            )
            package = export_verification_package(
                ROOT / "config" / "paper_suite.json", output
            )
            self.assertTrue((package / "project" / "tests" / Path(__file__).name).is_file())
            self.assertTrue((package / "outputs" / "raw" / "current.json").is_file())
            self.assertFalse((package / "outputs" / "raw" / "old.json").exists())
            self.assertTrue(verify_package(package)["passed"])

    def test_legacy_submission_export_needs_the_former_split_manuscript(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "figures").mkdir()
            (output / "figure_data").mkdir()
            for suffix in ("pdf", "png", "svg"):
                (output / "figures" / f"example.{suffix}").write_bytes(b"figure")
            (output / "figure_data" / "example.json").write_text(
                '{"series": []}\n', encoding="utf-8"
            )
            figure_manifest = {
                "files": {
                    f"figures/example.{suffix}": "unused" for suffix in ("pdf", "png", "svg")
                },
                "figure_data": {"figure_data/example.json": "unused"},
            }
            (output / "figures" / "figure_manifest.json").write_text(
                json.dumps(figure_manifest), encoding="utf-8"
            )
            package = export_submission_package(
                ROOT / "config" / "paper_suite.json", output
            )

            self.assertTrue((package / "outputs" / "figures" / "example.pdf").is_file())
            self.assertTrue((package / "outputs" / "figures" / "example.png").is_file())
            self.assertFalse((package / "outputs" / "figures" / "example.svg").exists())
            self.assertTrue((package / "outputs" / "figure_data" / "example.json").is_file())
            self.assertTrue((package / "paper" / "main.tex").is_file())
            self.assertTrue((package / "paper" / "PROOF_DATA.md").is_file())
            self.assertTrue(
                (package / "paper" / "figures" / "routed_protocols_visibility.pdf").is_file()
            )
            self.assertTrue((package / "numerical_summary.json").is_file())
            self.assertEqual(
                verify_package(package)["errors"],
                ["missing required package file: paper/supplement.tex"],
            )

            (package / "outputs" / "figures" / "old_curve.png").write_bytes(b"old")
            package = export_submission_package(
                ROOT / "config" / "paper_suite.json", output
            )
            self.assertFalse((package / "outputs" / "figures" / "old_curve.png").exists())
            self.assertEqual(
                verify_package(package)["errors"],
                ["missing required package file: paper/supplement.tex"],
            )


if __name__ == "__main__":
    unittest.main()
