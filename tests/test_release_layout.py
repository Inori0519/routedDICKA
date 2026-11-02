"""Publication checks that need neither numerical dependencies nor a solver."""

from hashlib import sha256
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


class ReleaseLayoutTests(unittest.TestCase):
    def test_scientific_sources_are_exactly_the_plotted_snapshot(self):
        expected = load("outputs/manifest.json")["source_hashes"]
        actual = {}
        for directory in ("src", "main", "verification"):
            for path in (ROOT / directory).rglob("*.py"):
                actual[path.relative_to(ROOT).as_posix()] = digest(path)
        self.assertEqual(actual, expected)

    def test_no_complete_table_scanner_or_duplicate_paper_is_shipped(self):
        for relative in (
            "src/routed_dicka/ternary_models.py",
            "config/ternary_efficiency_suite.json",
            "scripts/run_ternary_efficiency.ps1",
            "main/check_ternary_models.py",
            "main/export_ternary_code_package.py",
            "paper/supplement.tex",
            "paper/main.pdf",
            "outputs/packages",
            "outputs_full_ternary",
        ):
            self.assertFalse((ROOT / relative).exists(), msg=relative)
        for directory in ("src", "main", "verification"):
            for path in (ROOT / directory).rglob("*.py"):
                self.assertNotIn("physical-ternary-v1", path.read_text(encoding="utf-8"))

    def test_configuration_and_all_recorded_artifact_hashes_match(self):
        manifest = load("outputs/manifest.json")
        for group in ("point_hashes", "auxiliary_hashes", "table_hashes"):
            for relative, expected in manifest[group].items():
                self.assertEqual(digest(ROOT / "outputs" / relative), expected, msg=relative)
        for relative in manifest["point_files"]:
            path = ROOT / "outputs" / relative
            self.assertTrue(path.is_file(), msg=relative)
            record = json.loads(path.read_text(encoding="utf-8"))
            for file_key, hash_key in (
                ("target_file", "target_sha256"),
                ("attack_target_file", "attack_target_sha256"),
            ):
                if record.get(file_key):
                    target = ROOT / "outputs" / record[file_key]
                    self.assertTrue(target.is_file(), msg=record[file_key])
                    payload = json.loads(target.read_text(encoding="utf-8"))
                    claimed = payload.pop("sha256")
                    canonical = json.dumps(
                        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode("utf-8")
                    self.assertEqual(sha256(canonical).hexdigest(), claimed, msg=record[file_key])
                    self.assertEqual(claimed, record[hash_key], msg=record[file_key])
            if record.get("entropy_source_file"):
                self.assertTrue((ROOT / "outputs" / record["entropy_source_file"]).is_file())
            for node in record["entropy"].get("nodes", []):
                for file_key in ("certificate_file", "affine_file"):
                    if node.get(file_key):
                        self.assertTrue((ROOT / "outputs" / node[file_key]).is_file(), msg=node[file_key])
        reference = load("outputs/audits/reference_data.json")
        for group in ("inputs", "files"):
            for relative, expected in reference[group].items():
                self.assertEqual(digest(ROOT / "outputs" / relative), expected, msg=relative)
        figures = load("outputs/figures/figure_manifest.json")
        self.assertEqual(digest(ROOT / "config/paper_suite.json"), figures["config_sha256"])
        for group in ("input_tables", "files", "figure_data"):
            for relative, expected in figures[group].items():
                self.assertEqual(digest(ROOT / "outputs" / relative), expected, msg=relative)

    def test_manuscript_figures_are_present_and_match_the_published_exports(self):
        manuscript = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
        figures = re.findall(
            r"\\(?:includegraphics(?:\[[^]]*\])?|resultfigure)\{([^}]+)\}",
            manuscript,
        )
        figures = [name for name in figures if not name.startswith("#")]
        self.assertEqual(len(figures), 6)
        for relative in figures:
            self.assertTrue(relative.endswith(".png"), msg=relative)
            self.assertTrue((ROOT / "paper/figures" / relative).is_file(), msg=relative)
        for relative, expected in load("outputs/figures/figure_manifest.json")["files"].items():
            self.assertEqual(digest(ROOT / "paper" / relative), expected, msg=relative)

    def test_local_credentials_and_oversized_files_are_not_shipped(self):
        ignored = {"reproduction", "__pycache__", ".git", ".pytest_cache"}
        for path in ROOT.rglob("*"):
            relative = path.relative_to(ROOT)
            if set(relative.parts) & ignored or not path.is_file():
                continue
            self.assertNotEqual(path.suffix.lower(), ".lic", msg=str(relative))
            self.assertNotIn(path.name, {".env", "id_rsa", "id_ed25519"}, msg=str(relative))
            self.assertLess(path.stat().st_size, 100 * 1024 * 1024, msg=str(relative))
        self.assertIn("* -text", (ROOT / ".gitattributes").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
