"""Build compact submission and complete verification artifact packages."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routed_dicka.io import file_sha256, utc_now, write_json  # noqa: E402


def _reset_package(path: Path, packages_root: Path) -> None:
    path = path.resolve()
    packages_root = packages_root.resolve()
    if path.parent != packages_root:
        raise ValueError(f"Refusing to reset package outside {packages_root}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_package_manifest(package: Path, package_type: str) -> Path:
    files = sorted(
        path
        for path in package.rglob("*")
        if path.is_file() and path != package / "manifest.json"
    )
    manifest = {
        "schema": "routed-dicka-artifact-package-v1",
        "created_utc": utc_now(),
        "package_type": package_type,
        "files": {
            path.relative_to(package).as_posix(): file_sha256(path) for path in files
        },
    }
    path = package / "manifest.json"
    write_json(path, manifest)
    return path


def export_submission_package(config_path: Path, output: Path) -> Path:
    config_path = Path(config_path).resolve()
    output = Path(output).resolve()
    packages_root = output / "packages"
    packages_root.mkdir(parents=True, exist_ok=True)
    package = packages_root / "submission"
    _reset_package(package, packages_root)

    figure_manifest_path = output / "figures" / "figure_manifest.json"
    if not figure_manifest_path.is_file():
        raise FileNotFoundError(f"Missing figure manifest: {figure_manifest_path}")
    figure_manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    for relative in figure_manifest.get("files", {}):
        source = output / relative
        if source.suffix.lower() in {".pdf", ".png"}:
            _copy(source, package / "outputs" / relative)
    for relative in figure_manifest.get("figure_data", {}):
        source = output / relative
        if source.suffix.lower() == ".json":
            _copy(source, package / "outputs" / relative)
    _copy(
        figure_manifest_path,
        package / "outputs" / "figures" / "figure_manifest.json",
    )
    _copy(config_path, package / "paper_suite.json")
    root = config_path.parents[1]
    for pattern in ("*.tex", "*.bib", "*.pdf", "*.png", "*.md"):
        for source in sorted((root / "paper").rglob(pattern)):
            _copy(source, package / "paper" / source.relative_to(root / "paper"))

    manuscript_numerics_path = output / "tables" / "manuscript_numerics.json"
    manuscript_numerics = (
        json.loads(manuscript_numerics_path.read_text(encoding="utf-8"))
        if manuscript_numerics_path.is_file()
        else {}
    )
    summary = {
        "schema": "routed-dicka-manuscript-numerics-v1",
        "created_utc": utc_now(),
        "curve_definitions": json.loads(config_path.read_text(encoding="utf-8"))[
            "curve_plots"
        ],
        "strict_separation": _csv_rows(output / "tables" / "strict_separation.csv"),
        "mwrf": _csv_rows(output / "tables" / "mwrf.csv"),
        "recording_ablation": _csv_rows(output / "tables" / "recording_ablation.csv"),
        "radau_sensitivity": _csv_rows(output / "tables" / "radau_sensitivity.csv"),
        **{
            key: value
            for key, value in manuscript_numerics.items()
            if key not in {"schema", "created_utc", "inputs"}
        },
    }
    write_json(package / "numerical_summary.json", summary)
    _write_package_manifest(package, "submission")
    return package


def export_verification_package(config_path: Path, output: Path) -> Path:
    config_path = Path(config_path).resolve()
    output = Path(output).resolve()
    root = config_path.parents[1]
    packages_root = output / "packages"
    packages_root.mkdir(parents=True, exist_ok=True)
    package = packages_root / "verification"
    _reset_package(package, packages_root)

    source_patterns = {
        "config": ("*.json",),
        "main": ("*.py",),
        "src": ("*.py",),
        "verification": ("*.py",),
        "tests": ("*.py",),
        "scripts": ("*.ps1", "*.bat"),
        "paper": ("*.tex", "*.bib", "*.pdf", "*.png", "*.md"),
        "source_data": ("*.json",),
    }
    for directory, patterns in source_patterns.items():
        base = root / directory
        for pattern in patterns:
            for source in sorted(base.rglob(pattern)):
                _copy(source, package / "project" / source.relative_to(root))
    for source in (
        root / "README.md",
        root / "requirements.txt",
        root / "routed_DICKA_unified_framework_v3.md",
        *sorted((root / "docs").glob("*.md")),
        *sorted((root / "source_data").glob("*.md")),
    ):
        if source.is_file():
            _copy(source, package / "project" / source.relative_to(root))

    run_manifest_path = output / "manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {run_manifest_path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    _copy(run_manifest_path, package / "outputs" / "manifest.json")
    referenced = set(run_manifest.get("point_files", []))
    referenced.update(run_manifest.get("auxiliary_hashes", {}))
    referenced.update(run_manifest.get("table_hashes", {}))
    referenced.update(
        f"tables/{name}"
        for name in (
            "ideal_rates.csv",
            "strict_separation.csv",
            "protocol_envelopes.csv",
            "manuscript_numerics.json",
        )
    )
    for relative in sorted(referenced):
        source = output / relative
        if source.is_file():
            _copy(source, package / "outputs" / relative)

    audit_names = (
        "model_contract.json",
        "v21_attack.json",
        "v21_routed_certificate.json",
        "m4_same_point_certificate.json",
        "reference_data.json",
        "output_audit.json",
        "witness_envelopes.json",
        "witness_envelopes_independent.json",
        "review_refinements.json",
    )
    for name in audit_names:
        source = output / "audits" / name
        if source.is_file():
            _copy(source, package / "outputs" / "audits" / name)

    _write_package_manifest(package, "verification")
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "paper_suite.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    submission = export_submission_package(args.config, args.output)
    verification = export_verification_package(args.config, args.output)
    print(
        json.dumps(
            {
                "submission": str(submission),
                "verification": str(verification),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
