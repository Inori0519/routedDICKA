"""Independently rebuild and compare the dense affine-certificate envelopes."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
from math import isclose
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN = ROOT / "main"
for path in (SRC, MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_witness_envelopes import _load_manifest_witnesses, build_rows  # noqa: E402
from routed_dicka.io import file_sha256, write_json  # noqa: E402


def _close(left: object, right: object) -> bool:
    return isclose(float(left), float(right), rel_tol=2e-11, abs_tol=2e-11)


def route_order_errors(rows: list[dict[str, object]]) -> list[str]:
    """Check matched far/routed bounds independently of their construction."""
    def key(row):
        return (str(row["protocol"]), *(float(row[field]) for field in (
            "nu", "eta_b", "eta_c", "theta"
        )), str(row["statistics_recording"]), str(row["reconciliation_recording"]))

    far = {key(row): row for row in rows if row["data_scope"] == "far_only"}
    errors = []
    for row in rows:
        if row["data_scope"] != "routed" or key(row) not in far:
            continue
        baseline = far[key(row)]
        if not _close(row["EC_cost"], baseline["EC_cost"]):
            errors.append(f"{key(row)}: far/routed reconciliation costs differ")
        if Fraction(str(row["entropy_lower_exact"])) < Fraction(str(baseline["entropy_lower_exact"])):
            errors.append(f"{key(row)}: routed entropy is below its far-only guarantee")
        if float(row["key_rate_lower"]) + 2e-11 < float(baseline["key_rate_lower"]):
            errors.append(f"{key(row)}: routed key rate is below its far-only guarantee")
    return errors


def verify(config_path: Path, output: Path) -> dict[str, object]:
    audit_path = output / "audits" / "witness_envelopes.json"
    table_path = output / "tables" / "protocol_envelopes.csv"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if audit.get("passed") is not True:
        errors.append("Envelope producer audit did not pass")
    if audit.get("config_sha256") != file_sha256(config_path):
        errors.append("Envelope config hash mismatch")
    if audit.get("table_sha256") != file_sha256(table_path):
        errors.append("Envelope table hash mismatch")
    for relative, expected in audit.get("witness_files", {}).items():
        path = output / relative
        if not path.is_file() or file_sha256(path) != expected:
            errors.append(f"Witness artifact hash mismatch: {relative}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    banks, current_hashes = _load_manifest_witnesses(output)
    if current_hashes != audit.get("witness_files"):
        errors.append("Current manifest witness bank differs from the producer audit")
    expected_rows = build_rows(config, banks)
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        actual_rows = list(csv.DictReader(handle))
    ordering_errors = route_order_errors(actual_rows)
    errors.extend(ordering_errors)
    if len(actual_rows) != len(expected_rows):
        errors.append("Envelope row count mismatch")

    def key(row):
        return (
            str(row["slice_id"]),
            str(row["protocol"]),
            f"{float(row['eta_b']):.15g}",
        )

    expected_by_key = {key(row): row for row in expected_rows}
    actual_by_key = {key(row): row for row in actual_rows}
    if set(actual_by_key) != set(expected_by_key):
        errors.append("Envelope plotting grid or protocol set mismatch")
    for row_key in set(actual_by_key) & set(expected_by_key):
        actual = actual_by_key[row_key]
        expected = expected_by_key[row_key]
        for field in (
            "entropy_lower",
            "EC_cost",
            "raw_key_rate_lower",
            "key_rate_lower",
        ):
            if not _close(actual[field], expected[field]):
                errors.append(f"{row_key}: {field} mismatch")
        for field in (
            "entropy_lower_exact",
            "bound_source_protocol",
            "bound_source_scope",
            "selected_nodes",
        ):
            if str(actual[field]) != str(expected[field]):
                errors.append(f"{row_key}: {field} mismatch")

    return {
        "schema": "routed-dicka-witness-envelope-independent-audit-v1",
        "passed": not errors,
        "errors": errors,
        "rows": len(actual_rows),
        "routed_far_ordering_checked": not ordering_errors,
        "far_only_inherited_rows": sum(
            row["data_scope"] == "routed" and row["bound_source_scope"] == "far_only"
            for row in actual_rows
        ),
        "witness_counts": {
            f"{protocol}|{'routed' if include_near else 'long_only'}": len(value)
            for (protocol, include_near), value in banks.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "paper_suite.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    result = verify(args.config.resolve(), args.output.resolve())
    path = args.output.resolve() / "audits" / "witness_envelopes_independent.json"
    write_json(path, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
