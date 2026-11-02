"""Verify hashes and required structure of exported artifact packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routed_dicka.io import file_sha256, write_json  # noqa: E402


def verify_package(package: Path) -> dict[str, object]:
    package = Path(package).resolve()
    manifest_path = package / "manifest.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        return {"passed": False, "package": str(package), "errors": ["missing manifest"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "routed-dicka-artifact-package-v1":
        errors.append("unexpected manifest schema")
    package_type = manifest.get("package_type")
    required = {
        "submission": (
            "paper/main.tex",
            "paper/supplement.tex",
            "numerical_summary.json",
            "outputs/figures/figure_manifest.json",
        ),
        "verification": (
            "project/config/paper_suite.json",
            "outputs/manifest.json",
            "outputs/audits/m4_same_point_certificate.json",
        ),
    }.get(str(package_type), ())
    for relative in required:
        if not (package / relative).is_file():
            errors.append(f"missing required package file: {relative}")
    actual_files = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path != manifest_path
    }
    for relative in sorted(actual_files - set(manifest.get("files", {}))):
        errors.append(f"unexpected file: {relative}")
    for relative, expected_hash in manifest.get("files", {}).items():
        path = package / relative
        if not path.is_file():
            errors.append(f"missing file: {relative}")
        elif file_sha256(path) != expected_hash:
            errors.append(f"hash mismatch: {relative}")
    return {
        "passed": not errors,
        "package": str(package),
        "package_type": package_type,
        "file_count": len(manifest.get("files", {})),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    output = args.output.resolve()
    results = [
        verify_package(output / "packages" / package_type)
        for package_type in ("submission", "verification")
    ]
    audit = {
        "schema": "routed-dicka-artifact-package-audit-v1",
        "passed": all(result["passed"] for result in results),
        "packages": results,
    }
    write_json(output / "audits" / "artifact_packages.json", audit)
    print(json.dumps(audit, indent=2))
    raise SystemExit(0 if audit["passed"] else 1)


if __name__ == "__main__":
    main()
