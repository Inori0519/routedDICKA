"""Evaluate verified reduced certificates on dense manuscript plotting grids."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VERIFICATION = ROOT / "verification"
for path in (SRC, VERIFICATION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from routed_dicka.io import file_sha256, utc_now, write_csv, write_json  # noqa: E402
from routed_dicka.protocols import TargetPoint  # noqa: E402
from routed_dicka.reduced_models import build_reduced_model  # noqa: E402
from routed_dicka.target import build_target  # noqa: E402
from routed_dicka.witness_bank import (  # noqa: E402
    evaluate_witness_envelope,
    load_verified_witness,
)
from verify_reduced_certificate import verify_node_certificate  # noqa: E402


PROTOCOLS = ("rPCHSH", "rPhase", "rHybrid")


def _grid(minimum: float, maximum: float, points: int) -> list[float]:
    return [minimum + (maximum - minimum) * index / (points - 1) for index in range(points)]


BankKey = tuple[str, bool]


def validate_curve_matrix_rows(
    config: dict[str, object], rows: list[dict[str, object]]
) -> dict[str, int]:
    """Require one dense plotting grid for every configured protocol slice."""
    curve_specification = config["experiments"]["protocol_curve_anchors"]
    expected_families = {
        (str(curve_slice["slice_id"]), str(protocol))
        for curve_slice in curve_specification["slices"]
        for protocol in curve_slice["protocols"]
    }
    points = int(config["analytic_curves"]["certificate_plot_grid"]["plot_points"])
    actual = Counter((str(row["slice_id"]), str(row["protocol"])) for row in rows)
    actual_families = set(actual)
    missing = sorted(expected_families - actual_families)
    unexpected = sorted(actual_families - expected_families)
    wrong_counts = sorted(
        (family, actual[family])
        for family in expected_families & actual_families
        if actual[family] != points
    )
    if missing or unexpected or wrong_counts:
        raise ValueError(
            "Curve matrix row contract failed: "
            f"missing={missing}, unexpected={unexpected}, wrong_counts={wrong_counts}"
        )
    return {
        "families": len(expected_families),
        "points_per_family": points,
        "rows": len(rows),
    }


def _load_manifest_witnesses(
    output: Path,
) -> tuple[dict[BankKey, list[dict[str, object]]], dict[str, str]]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    banks: dict[BankKey, list[dict[str, object]]] = {
        (protocol, include_near): []
        for protocol in PROTOCOLS
        for include_near in (True, False)
    }
    hashes: dict[str, str] = {}
    seen: set[str] = set()
    for relative in manifest["point_files"]:
        record = json.loads((output / relative).read_text(encoding="utf-8"))
        if record.get("evidence_kind") != "verified_generated_certificate":
            continue
        if record.get("metadata", {}).get("statistics_recording") != "ternary":
            continue
        protocol = str(record.get("metadata", {}).get("protocol"))
        include_near = bool(record.get("metadata", {}).get("include_near", True))
        key_for_bank = (protocol, include_near)
        if key_for_bank not in banks:
            continue
        for node in record["entropy"]["nodes"]:
            certificate_path = output / node["certificate_file"]
            affine_path = output / node["affine_file"]
            key = str(certificate_path.resolve())
            if key in seen:
                continue
            verified = verify_node_certificate(certificate_path)
            if verified.get("verified") is not True:
                raise AssertionError(f"Certificate did not verify: {certificate_path}")
            witness = load_verified_witness(certificate_path, affine_path)
            banks[key_for_bank].append(witness)
            hashes[certificate_path.relative_to(output).as_posix()] = file_sha256(certificate_path)
            hashes[affine_path.relative_to(output).as_posix()] = file_sha256(affine_path)
            seen.add(key)
    return banks, hashes


def _point(config: dict[str, object], *, nu: float, eta_s: float, eta: float) -> TargetPoint:
    common = config["common_target"]
    return TargetPoint(
        nu=nu,
        eta_s=eta_s,
        eta_b=eta,
        eta_c=eta,
        eta_a=float(common.get("eta_a", 1.0)),
        far_angle_policy=str(common.get("far_angle_policy", "matched")),
        near_angle_policy=str(common.get("near_angle_policy", "matched")),
    )


def _evaluate_row(
    config: dict[str, object],
    witnesses: list[dict[str, object]],
    *,
    protocol: str,
    slice_id: str,
    nu: float,
    eta_s: float,
    eta: float,
    include_near: bool,
    far_witnesses: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    behavior = build_target(_point(config, nu=nu, eta_s=eta_s, eta=eta))
    model = build_reduced_model(
        behavior,
        protocol,
        statistics_recording="ternary",
        include_near=include_near,
    )
    bound = evaluate_witness_envelope(witnesses, model)
    source_scope = "routed" if include_near else "far_only"
    if include_near and far_witnesses:
        far_model = build_reduced_model(
            behavior, protocol, statistics_recording="ternary", include_near=False
        )
        far_bound = evaluate_witness_envelope(far_witnesses, far_model)
        if Fraction(far_bound["entropy_lower_exact"]) > Fraction(bound["entropy_lower_exact"]):
            bound = far_bound
            source_scope = "far_only"
    h_ab, h_ac = behavior.key_entropies(recording="ternary")
    ec_upper = min(1.0, max(h_ab, h_ac) + 1e-12)
    raw_rate = float(bound["entropy_lower"]) - ec_upper
    return {
        "experiment_kind": "certificate_envelope",
        "slice_id": slice_id,
        "data_scope": "routed" if include_near else "far_only",
        "include_near": include_near,
        "protocol": protocol,
        "nu": nu,
        "eta_s": eta_s,
        "eta_b": eta,
        "eta_c": eta,
        "theta": behavior.point.far_theta,
        "statistics_recording": "ternary",
        "reconciliation_recording": "ternary",
        "evidence_kind": "verified_affine_certificate_envelope",
        "m": bound["m"],
        "entropy_lower": bound["entropy_lower"],
        "entropy_lower_exact": bound["entropy_lower_exact"],
        "EC_cost": ec_upper,
        "raw_key_rate_lower": raw_rate,
        "key_rate_lower": max(0.0, raw_rate),
        "claim_eligible": True,
        "bound_source_protocol": protocol,
        "bound_source_scope": source_scope,
        "selected_nodes": json.dumps(bound["nodes"], separators=(",", ":")),
    }


def build_rows(
    config: dict[str, object],
    banks: dict[BankKey, list[dict[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grid = config["analytic_curves"]["certificate_plot_grid"]
    eta_values = _grid(
        float(grid["eta_min"]),
        float(grid["eta_max"]),
        int(grid["plot_points"]),
    )
    curve_specification = config["experiments"]["protocol_curve_anchors"]
    for curve_slice in curve_specification["slices"]:
        include_near = bool(curve_slice["include_near"])
        for protocol in curve_slice["protocols"]:
            witnesses = banks[(str(protocol), include_near)]
            if not witnesses:
                continue
            for eta in eta_values:
                rows.append(
                _evaluate_row(
                    config,
                    witnesses,
                    protocol=str(protocol),
                    slice_id=str(curve_slice["slice_id"]),
                    nu=float(curve_slice["nu"]),
                    eta_s=float(curve_slice["eta_s"]),
                    eta=eta,
                    include_near=include_near,
                    far_witnesses=banks.get((str(protocol), False)) if include_near else None,
                )
                )

    by_eta: dict[tuple[str, float], dict[str, dict[str, object]]] = {}
    for row in rows:
        by_eta.setdefault(
            (str(row["slice_id"]), float(row["eta_b"])), {}
        )[str(row["protocol"])] = row
    for protocols in by_eta.values():
        hybrid = protocols.get("rHybrid")
        if hybrid is None:
            continue
        source, best = max(
            protocols.items(), key=lambda item: Fraction(item[1]["entropy_lower_exact"])
        )
        if Fraction(best["entropy_lower_exact"]) > Fraction(hybrid["entropy_lower_exact"]):
            for field in (
                "m",
                "entropy_lower",
                "entropy_lower_exact",
                "raw_key_rate_lower",
                "key_rate_lower",
                "selected_nodes",
                "bound_source_scope",
            ):
                hybrid[field] = best[field]
            hybrid["bound_source_protocol"] = source
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "paper_suite.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    output = args.output.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    banks, witness_hashes = _load_manifest_witnesses(output)
    required = {
        (str(protocol), bool(curve_slice["include_near"]))
        for curve_slice in config["experiments"]["protocol_curve_anchors"]["slices"]
        for protocol in curve_slice["protocols"]
    }
    missing = [key for key in sorted(required) if not banks[key]]
    if missing:
        labels = [f"{protocol}/{'routed' if near else 'long-only'}" for protocol, near in missing]
        raise SystemExit(f"Missing ternary witness banks for: {', '.join(labels)}")
    rows = build_rows(config, banks)
    curve_matrix = validate_curve_matrix_rows(config, rows)
    table_path = output / "tables" / "protocol_envelopes.csv"
    write_csv(table_path, rows)
    audit = {
        "schema": "routed-dicka-witness-envelope-audit-v1",
        "created_utc": utc_now(),
        "passed": True,
        "config_sha256": file_sha256(args.config.resolve()),
        "witness_files": witness_hashes,
        "witness_counts": {
            f"{protocol}|{'routed' if include_near else 'long_only'}": len(value)
            for (protocol, include_near), value in banks.items()
        },
        "curve_matrix": curve_matrix,
        "rows": len(rows),
        "far_only_inherited_rows": sum(
            row["include_near"] and row["bound_source_scope"] == "far_only" for row in rows
        ),
        "table": table_path.relative_to(output).as_posix(),
        "table_sha256": file_sha256(table_path),
    }
    audit_path = output / "audits" / "witness_envelopes.json"
    write_json(audit_path, audit)
    print(json.dumps({"passed": True, "rows": len(rows), "witness_counts": audit["witness_counts"]}, indent=2))


if __name__ == "__main__":
    main()
