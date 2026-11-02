"""Independent audit of raw SDP records, tables, and scientific claim gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
from fractions import Fraction
from hashlib import sha256
import json
from math import isclose, log
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
VERIFICATION = ROOT / "verification"
if str(VERIFICATION) not in sys.path:
    sys.path.insert(0, str(VERIFICATION))

from verify_reduced_certificate import (  # noqa: E402
    certified_entropy,
    verify_node_certificate,
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _object_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def resolve_audit_input(output: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (output / path).resolve()


def _close(left: float, right: float, tolerance: float = 2e-10) -> bool:
    return isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_radau_rule(entropy: dict[str, object]) -> dict[str, object]:
    """Validate a right Gauss--Radau rule without imposing a paper-wide order."""

    m = int(entropy["m"])
    _require(m >= 2, f"Radau order m={m} is below the supported minimum")
    rule = list(entropy["radau_rule"])
    _require(len(rule) == m, "Radau rule length does not equal m")
    _require(bool(rule[-1]["endpoint"]), "Final Radau node is not the endpoint")
    _require(_close(rule[-1]["t"], 1.0), "Radau endpoint is not t=1")
    _require(
        _close(rule[-1]["weight"], 1.0 / (m * m)),
        "Radau endpoint weight is not 1/m^2",
    )
    max_moment_error = 0.0
    for degree in range(2 * m - 1):
        approximate = sum(
            float(node["weight"]) * float(node["t"]) ** degree for node in rule
        )
        max_moment_error = max(
            max_moment_error,
            abs(approximate - 1.0 / (degree + 1)),
        )
    _require(
        max_moment_error <= 2e-11,
        f"Radau moment error {max_moment_error:.3e}",
    )
    return {"m": m, "max_moment_error": max_moment_error}


def _validate_rate_identity(record: dict[str, object]) -> dict[str, float]:
    entropy_lower = float(record["entropy"]["entropy_lower"])
    raw_entropy_lower = float(
        record["entropy"].get("raw_entropy_lower", entropy_lower)
    )
    _require(
        _close(entropy_lower, max(0.0, raw_entropy_lower)),
        "Clipped entropy identity failed",
    )
    ec = max(float(record["H_A_given_B"]), float(record["H_A_given_C"]))
    _require(_close(ec, record["EC_cost"]), "Error-correction cost mismatch")
    raw_rate = raw_entropy_lower - ec
    _require(
        _close(raw_rate, record["raw_key_rate_lower"]),
        "Raw key-rate identity failed",
    )
    _require(
        _close(max(0.0, raw_rate), record["key_rate_lower"]),
        "Clipped key-rate identity failed",
    )
    _require(bool(record["claim_eligible"]), "Record is not claim eligible")
    return {
        "entropy_lower": entropy_lower,
        "EC_cost": ec,
        "raw_key_rate_lower": raw_rate,
        "key_rate_lower": max(0.0, raw_rate),
    }


def validate_certified_entropy_identity(
    combined: dict[str, object], recorded: dict[str, object]
) -> dict[str, float]:
    raw_entropy_lower = float(combined["entropy_lower"])
    _require(
        _close(raw_entropy_lower, recorded["raw_entropy_lower"]),
        "Exact certificate raw entropy mismatch",
    )
    entropy_lower = max(0.0, raw_entropy_lower)
    _require(
        _close(entropy_lower, recorded["entropy_lower"]),
        "Exact certificate clipped entropy mismatch",
    )
    return {
        "raw_entropy_lower": raw_entropy_lower,
        "entropy_lower": entropy_lower,
    }


def validate_evidence_identity(record: dict[str, object]) -> dict[str, object]:
    """Recompute the evidence-local numerical identities of one point record."""

    evidence = str(record.get("evidence_kind") or "raw_bff_dual")
    entropy = record["entropy"]
    if evidence == "explicit_compatible_zero_entropy_attack":
        _require(int(entropy["m"]) == 0, "Attack record must not declare Radau nodes")
        _require(not entropy.get("nodes"), "Attack record unexpectedly contains SDP nodes")
        _require(_close(entropy["entropy_lower"], 0.0), "Attack entropy lower value is not zero")
        _require(
            _close(entropy["compatible_attack_entropy_upper"], 0.0),
            "Compatible attack does not establish a zero entropy upper bound",
        )
        _require(
            _close(record["guaranteed_key_rate_upper"], 0.0),
            "Compatible attack does not establish a zero guaranteed-rate upper bound",
        )
        rate = _validate_rate_identity(record)
        return {
            "evidence_kind": evidence,
            "m": 0,
            "sdp_nodes": 0,
            "guaranteed_key_rate_upper": 0.0,
            **rate,
        }

    quadrature = validate_radau_rule(entropy)
    nodes = list(entropy["nodes"])
    _require(len(nodes) == quadrature["m"] - 1, "Wrong number of SDP node records")

    if evidence == "raw_bff_dual":
        recomputed_entropy = 0.0
        for node, quadrature_node in zip(nodes, entropy["radau_rule"][:-1]):
            _require(bool(node["status_optimal"]), f"Nonoptimal node {node['index']}")
            _require(bool(node["gap_accepted"]), f"Rejected gap at node {node['index']}")
            _require(bool(node["claim_eligible"]), f"Ineligible node {node['index']}")
            coefficient = float(quadrature_node["weight"]) / (
                float(quadrature_node["t"]) * log(2.0)
            )
            _require(
                _close(coefficient, node["coefficient"]),
                f"BFF coefficient mismatch at node {node['index']}",
            )
            guarded = float(node["dual"]) - float(node["numerical_guard"])
            _require(
                _close(guarded, node["guarded_dual"]),
                f"Guarded dual mismatch at node {node['index']}",
            )
            recomputed_entropy += coefficient * (1.0 + guarded)
        _require(
            _close(recomputed_entropy, entropy["raw_entropy_lower"]),
            "BFF entropy aggregation mismatch",
        )
        clipped = min(1.0, max(0.0, recomputed_entropy))
        _require(_close(clipped, entropy["entropy_lower"]), "Clipped entropy mismatch")
    elif evidence in {
        "verified_generated_certificate",
        "reused_verified_entropy_certificate",
    }:
        _require(
            bool(entropy.get("all_nodes_claim_eligible")),
            "Certificate record contains an ineligible node",
        )
    else:
        raise AssertionError(f"Unsupported evidence kind: {evidence}")

    rate = _validate_rate_identity(record)
    return {
        "evidence_kind": evidence,
        "m": quadrature["m"],
        "sdp_nodes": len(nodes),
        "max_radau_moment_error": quadrature["max_moment_error"],
        **rate,
    }


class Auditor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output = args.output.resolve()
        self.config = _load(args.config.resolve())
        self.manifest = _load(self.output / "manifest.json")
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.point_summaries: list[dict[str, object]] = []
        self.records: list[dict[str, object]] = []
        self.records_by_source: dict[str, dict[str, object]] = {}
        self.certificate_cache: dict[str, dict[str, object]] = {}

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def verify_sources(self) -> None:
        for relative, expected in self.manifest.get("source_hashes", {}).items():
            path = ROOT / relative
            self.check(path.is_file(), f"Missing source file recorded by manifest: {relative}")
            if path.is_file():
                self.check(_hash(path) == expected, f"Source changed after run: {relative}")

    def verify_artifact_hashes(self) -> None:
        for group in ("point_hashes", "auxiliary_hashes", "table_hashes"):
            hashes = self.manifest.get(group, {})
            self.check(bool(hashes), f"Manifest contains no {group}")
            for relative, expected in hashes.items():
                path = self.output / relative
                self.check(path.is_file(), f"Missing recorded artifact: {relative}")
                if path.is_file():
                    self.check(
                        _hash(path) == expected,
                        f"Artifact hash mismatch: {relative}",
                    )

    def verify_target(self, record: dict[str, object], source: str) -> None:
        target_relative = record.get("target_file")
        self.check(bool(target_relative), f"{source}: missing target_file")
        if not target_relative:
            return
        target_path = self.output / str(target_relative)
        self.check(target_path.is_file(), f"{source}: target file not found: {target_relative}")
        if not target_path.is_file():
            return
        target = _load(target_path)
        claimed = target.get("sha256")
        payload = deepcopy(target)
        payload.pop("sha256", None)
        actual = _object_hash(payload)
        self.check(actual == claimed, f"{source}: target internal hash mismatch")
        self.check(
            claimed == record.get("target_sha256"),
            f"{source}: point/target hash mismatch",
        )
        for route in ("near", "far"):
            for context in target.get(route, []):
                probabilities = context["probabilities"]
                flat = [
                    value
                    for alice in probabilities
                    for bob in alice
                    for value in bob
                ]
                self.check(
                    min(flat) >= -2e-12,
                    f"{source}: negative {route} target probability",
                )
                self.check(
                    abs(sum(flat) - 1.0) <= 2e-12,
                    f"{source}: unnormalized {route} target context",
                )

        attack_relative = record.get("attack_target_file")
        if attack_relative:
            attack_path = self.output / str(attack_relative)
            self.check(
                attack_path.is_file(),
                f"{source}: attack target file not found: {attack_relative}",
            )
            if attack_path.is_file():
                attack_target = _load(attack_path)
                claimed = attack_target.get("sha256")
                attack_payload = deepcopy(attack_target)
                attack_payload.pop("sha256", None)
                self.check(
                    _object_hash(attack_payload) == claimed
                    and claimed == record.get("attack_target_sha256"),
                    f"{source}: attack target hash mismatch",
                )

    def verify_quadrature_and_rate(self, record: dict[str, object], source: str) -> None:
        try:
            summary = validate_evidence_identity(record)
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            self.errors.append(f"{source}: {exc}")
            return

        evidence = str(summary["evidence_kind"])
        if evidence in {
            "verified_generated_certificate",
            "reused_verified_entropy_certificate",
        }:
            self.verify_generated_certificates(record, source)
        self.point_summaries.append({"source_file": source, **summary})

    def verify_generated_certificates(
        self, record: dict[str, object], source: str
    ) -> None:
        entropy = record["entropy"]
        certificate_paths: list[Path] = []
        for node in entropy["nodes"]:
            relative = node.get("certificate_file")
            self.check(bool(relative), f"{source}: node is missing certificate_file")
            if not relative:
                continue
            path = self.output / str(relative)
            self.check(path.is_file(), f"{source}: missing certificate {relative}")
            if not path.is_file():
                continue
            self.check(
                _hash(path) == node.get("certificate_sha256"),
                f"{source}: certificate hash mismatch for {relative}",
            )
            affine_relative = node.get("affine_file")
            self.check(bool(affine_relative), f"{source}: node is missing affine_file")
            if affine_relative:
                affine_path = self.output / str(affine_relative)
                self.check(
                    affine_path.is_file(),
                    f"{source}: missing affine witness {affine_relative}",
                )
                if affine_path.is_file():
                    self.check(
                        _hash(affine_path) == node.get("affine_sha256"),
                        f"{source}: affine hash mismatch for {affine_relative}",
                    )
            cache_key = str(path.resolve())
            try:
                verified = self.certificate_cache.get(cache_key)
                if verified is None:
                    verified = verify_node_certificate(path)
                    self.certificate_cache[cache_key] = verified
            except (AssertionError, KeyError, TypeError, ValueError) as exc:
                self.errors.append(f"{source}: exact certificate check failed: {exc}")
                continue
            recorded = node.get("verified_certificate", {})
            self.check(
                bool(verified.get("verified")) and bool(recorded.get("verified")),
                f"{source}: certificate is not independently verified",
            )
            self.check(
                _close(verified["node_lower"], recorded.get("node_lower")),
                f"{source}: recorded node lower bound differs from exact check",
            )
            self.check(
                str(verified["exact_node_lower"])
                == str(recorded.get("exact_node_lower")),
                f"{source}: exact node lower bound mismatch",
            )
            certificate_paths.append(path)

        if len(certificate_paths) != int(entropy["m"]) - 1:
            return
        try:
            combined = certified_entropy(certificate_paths, m=int(entropy["m"]))
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            self.errors.append(f"{source}: exact entropy aggregation failed: {exc}")
            return
        try:
            validate_certified_entropy_identity(combined, entropy)
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            self.errors.append(f"{source}: {exc}")

    def read_points(self) -> None:
        point_files = self.manifest.get("point_files", [])
        self.check(bool(point_files), "Manifest contains no point files")
        for relative in point_files:
            path = self.output / relative
            self.check(path.is_file(), f"Missing point file: {relative}")
            if not path.is_file():
                continue
            record = _load(path)
            self.records.append(record)
            self.records_by_source[str(relative)] = record
            self.check(
                record.get("run_contract_sha256")
                == self.manifest.get("run_contract_sha256"),
                f"{relative}: run-contract mismatch",
            )
            manifest = record.get("constraint_manifest", {})
            names = [row["name"] for row in manifest.get("coordinates", [])]
            self.check(
                len(names) == len(set(names)) == manifest.get("constraint_count"),
                f"{relative}: duplicate or miscounted constraints",
            )
            constraint_id = sha256("\n".join(names).encode("ascii")).hexdigest()
            self.check(
                constraint_id == manifest.get("constraint_id"),
                f"{relative}: constraint ID mismatch",
            )
            self.verify_target(record, relative)

        for record in self.records:
            relative = str(record["source_file"])
            self.verify_quadrature_and_rate(record, relative)
            if record.get("evidence_kind") == "reused_verified_entropy_certificate":
                source_relative = str(
                    record.get("metadata", {}).get("entropy_source_file", "")
                )
                source_record = self.records_by_source.get(source_relative)
                self.check(
                    source_record is not None,
                    f"{relative}: missing entropy source record {source_relative}",
                )
                if source_record is not None:
                    self.check(
                        source_record.get("evidence_kind")
                        == "verified_generated_certificate",
                        f"{relative}: entropy source is not a generated certificate",
                    )
                    self.check(
                        record["entropy"] == source_record["entropy"],
                        f"{relative}: reused entropy payload differs from its source",
                    )
                    self.check(
                        record.get("target_sha256")
                        == source_record.get("target_sha256"),
                        f"{relative}: reused certificate targets a different behavior",
                    )

    def verify_tables(self) -> None:
        results_path = self.output / "tables" / "results.csv"
        self.check(results_path.is_file(), "Missing aggregate results.csv")
        if not results_path.is_file():
            return
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_source = {row.get("source_file"): row for row in rows}
        self.check(len(by_source) == len(rows), "results.csv has duplicate source_file rows")
        self.check(len(rows) == len(self.records), "results.csv row count does not match point records")
        for record in self.records:
            source = record["source_file"]
            row = by_source.get(source)
            self.check(row is not None, f"results.csv is missing {source}")
            if row is None:
                continue
            metadata = record.get("metadata", {})
            point = record["physical_point"]
            expected_text = {
                "experiment_id": record["experiment_id"],
                "experiment_kind": metadata.get("experiment_kind"),
                "evidence_kind": record.get("evidence_kind", "raw_bff_dual"),
                "backend": record.get("solver", {}).get("backend"),
                "slice_id": metadata.get("slice_id"),
                "point_id": metadata.get("point_id"),
                "protocol": record.get("metadata", {}).get("protocol"),
                "info_model": metadata.get("info_model"),
                "comparison_mode": metadata.get("comparison_mode"),
                "recording_mode": metadata.get("recording_mode"),
                "statistics_recording": metadata.get("statistics_recording"),
                "reconciliation_recording": record.get("reconciliation_recording"),
                "include_near": metadata.get("include_near"),
                "claim_eligible": record.get("claim_eligible"),
                "source_file": source,
                "target_file": record.get("target_file"),
                "target_sha256": record.get("target_sha256"),
                "constraint_id": record["constraint_manifest"]["constraint_id"],
            }
            for field, expected in expected_text.items():
                normalized = "" if expected is None else str(expected)
                self.check(
                    str(row.get(field, "")) == normalized,
                    f"results.csv {source}: {field} mismatch",
                )
            expected_numeric = {
                "nu": point.get("nu"),
                "eta_a": point.get("eta_a"),
                "eta_s": point.get("eta_s"),
                "eta_b": point.get("eta_b"),
                "eta_c": point.get("eta_c"),
                "theta": point.get("far_theta"),
                "npa_level": record.get("solver", {}).get("npa_level"),
                "entropy_lower": record["entropy"]["entropy_lower"],
                "EC_cost": record["EC_cost"],
                "raw_key_rate_lower": record["raw_key_rate_lower"],
                "key_rate_lower": record["key_rate_lower"],
                "m": record["entropy"]["m"],
                "constraint_count": record["constraint_manifest"]["constraint_count"],
                "guaranteed_key_rate_upper": record.get(
                    "guaranteed_key_rate_upper"
                ),
            }
            for field, expected in expected_numeric.items():
                actual = row.get(field, "")
                if expected is not None:
                    self.check(
                        actual != "" and _close(float(actual), float(expected)),
                        f"results.csv {source}: {field} mismatch",
                    )

        constraints_path = self.output / "tables" / "constraint_manifest.csv"
        self.check(constraints_path.is_file(), "Missing aggregate constraint_manifest.csv")
        if constraints_path.is_file():
            with constraints_path.open("r", encoding="utf-8", newline="") as handle:
                constraints = list(csv.DictReader(handle))
            expected_count = sum(
                int(record["constraint_manifest"]["constraint_count"])
                for record in self.records
            )
            self.check(
                len(constraints) == expected_count,
                "constraint_manifest.csv row count does not match point records",
            )
            actual_coordinates: dict[str, dict[str, float]] = defaultdict(dict)
            for row in constraints:
                actual_coordinates[str(row["source_file"])][str(row["name"])] = float(
                    row["value"]
                )
            for record in self.records:
                source = str(record["source_file"])
                expected_coordinates = {
                    str(item["name"]): float(item["value"])
                    for item in record["constraint_manifest"]["coordinates"]
                }
                actual = actual_coordinates.get(source, {})
                self.check(
                    set(actual) == set(expected_coordinates),
                    f"constraint_manifest.csv {source}: coordinate names mismatch",
                )
                for name in set(actual) & set(expected_coordinates):
                    self.check(
                        _close(actual[name], expected_coordinates[name]),
                        f"constraint_manifest.csv {source}: value mismatch for {name}",
                    )

    def verify_mwrf(self) -> None:
        groups: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
        for record in self.records:
            metadata = record.get("metadata", {})
            if metadata.get("experiment_kind") == "mwrf":
                groups[str(metadata["point_id"])][str(metadata["info_model"])] = record
        expected_counts = self.config["scientific_gates"]["mwrf_expected_counts"]
        expected_groups: dict[str, set[str]] = {}
        selected = set(self.manifest.get("selected_experiments", []))
        for experiment_id in selected:
            spec = self.config["experiments"].get(experiment_id, {})
            if spec.get("kind") != "mwrf":
                continue
            points = spec.get("points", [spec])
            for point in points:
                point_id = str(point.get("point_id", spec.get("point_id", "point")))
                expected_groups[point_id] = {
                    *(str(value) for value in spec.get("models", [])),
                    *(str(value) for value in spec.get("attack_models", [])),
                }
        for point_id, models in groups.items():
            expected_models = expected_groups.get(point_id, set(models))
            self.check(
                set(models) == expected_models,
                f"MWRF {point_id}: model set {sorted(models)} != {sorted(expected_models)}",
            )
            sets = {
                model: {
                    row["name"]
                    for row in record["constraint_manifest"]["coordinates"]
                }
                for model, record in models.items()
            }
            for model in models:
                expected = expected_counts[model]
                self.check(
                    len(sets[model]) == int(expected),
                    f"MWRF {point_id}/{model}: {len(sets[model])} constraints, expected {expected}",
                )
            for lower, upper in (("M", "W"), ("M", "R"), ("W", "F"), ("R", "F")):
                if lower not in sets or upper not in sets:
                    continue
                self.check(
                    sets[lower] < sets[upper],
                    f"MWRF {point_id}: coordinate inclusion {lower}<{upper} failed",
                )
            if {"W", "R", "F"} <= set(sets):
                self.check(
                    sets["W"] | sets["R"] == sets["F"],
                    f"MWRF {point_id}: W union R != F",
                )
            for model in set(models) & {"M", "R"}:
                record = models[model]
                self.check(
                    record.get("evidence_kind")
                    == "explicit_compatible_zero_entropy_attack",
                    f"MWRF {point_id}/{model}: missing compatible attack evidence",
                )
                self.check(
                    _close(record.get("guaranteed_key_rate_upper"), 0.0),
                    f"MWRF {point_id}/{model}: zero-rate upper bound failed",
                )
            for model in set(models) & {"W", "F"}:
                self.check(
                    models[model].get("evidence_kind", "raw_bff_dual")
                    in {"raw_bff_dual", "verified_generated_certificate"},
                    f"MWRF {point_id}/{model}: missing numerical lower-bound evidence",
                )

        for experiment_id in selected:
            spec = self.config["experiments"].get(experiment_id, {})
            if spec.get("kind") != "mwrf":
                continue
            dephasing_files = sorted((self.output / "raw" / experiment_id).glob("*_dephasing.json"))
            self.check(bool(dephasing_files), f"{experiment_id}: missing dephasing checks")
            for path in dephasing_files:
                report = _load(path)
                self.check(
                    report.get("only_expected_coordinates_changed") is True,
                    f"{path.name}: dephasing changes unexpected coordinates",
                )
                self.check(
                    report.get("accepted_models") == {"M": True, "W": False, "R": True, "F": False},
                    f"{path.name}: dephasing/MWRF acceptance mismatch",
                )

    def verify_protocol_nesting(self) -> None:
        groups: dict[tuple[object, ...], dict[str, float]] = defaultdict(dict)
        tolerance = float(self.config["solver"]["max_accepted_primal_dual_gap"])
        for record in self.records:
            metadata = record.get("metadata", {})
            if metadata.get("experiment_kind") not in {
                "protocol_curves",
                "protocol_comparison",
                "certificate_anchor",
            }:
                continue
            point = record["physical_point"]
            key = (
                record["experiment_id"],
                metadata.get("slice_id"),
                point["nu"],
                point["eta_s"],
                point["eta_b"],
                point["eta_c"],
                record["entropy"]["m"],
            )
            groups[key][str(metadata["protocol"])] = float(record["entropy"]["entropy_lower"])
        for key, values in groups.items():
            if key[1] == "routed_nu1_s1" and set(values) == {"rPCHSH"}:
                # rPhase and rHybrid use the exact ideal formula on this slice.
                continue
            if set(values) != {"rPCHSH", "rPhase", "rHybrid"}:
                self.warnings.append(f"Protocol nesting group incomplete: {key}")
                continue
            for child in ("rPCHSH", "rPhase"):
                if values[child] > values["rHybrid"] + tolerance:
                    self.warnings.append(
                        f"Independent relaxation lower bound for {child} exceeds the "
                        f"dedicated rHybrid bound at {key}; use the certified envelope."
                    )

    def scientific_gates(self) -> dict[str, object]:
        attack_path = self.output / "audits" / "v21_attack.json"
        attack = _load(attack_path) if attack_path.is_file() else {}
        certificate_path = self.output / "audits" / "m4_same_point_certificate.json"
        certificate = _load(certificate_path) if certificate_path.is_file() else {}
        separation = self.config["reference_points"]["S1"]
        expected_eta = float(separation["eta_l"])
        expected_nu = float(separation["nu"])
        expected_eta_s = float(separation["eta_s"])
        positive_margin = float(
            self.config["scientific_gates"].get("minimum_positive_rate", 0.0)
        )
        reference = certificate.get("reference_point", {})
        certificate_matches = bool(certificate.get("passed")) and all(
            _close(reference.get(key, float("nan")), expected)
            for key, expected in (
                ("nu", expected_nu),
                ("eta_s", expected_eta_s),
                ("eta_b", expected_eta),
                ("eta_c", expected_eta),
            )
        )
        positive = (
            certificate_matches
            and bool(reference.get("positive"))
            and float(reference.get("K_lower", 0.0)) > positive_margin
        )
        exact_attack = bool(attack.get("exactly_verified")) and bool(
            attack.get("exact_degrading_channel_verified")
        ) and int(attack.get("probabilities", 0)) == 288
        attack_matches_point = False
        if exact_attack:
            try:
                attack_matches_point = _close(
                    float(Fraction(str(attack["configured_nu"]))), expected_nu
                ) and _close(
                    float(Fraction(str(attack["configured_eta"]))), expected_eta
                )
            except (KeyError, ValueError, ZeroDivisionError):
                attack_matches_point = False
        exact_attack = exact_attack and attack_matches_point
        separation_path = self.output / "tables" / "strict_separation.csv"
        table_ready = False
        table_rate = None
        table_evidence = None
        if separation_path.is_file():
            with separation_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            by_scope = {row.get("data_scope"): row for row in rows}
            table_ready = (
                set(by_scope) == {"far_only", "routed"}
                and _close(by_scope["far_only"]["guaranteed_key_rate"], 0.0)
                and float(by_scope["routed"]["guaranteed_key_rate"])
                > positive_margin
                and _close(
                    by_scope["routed"]["guaranteed_key_rate"],
                    reference.get("K_lower", float("nan")),
                )
            )
            if "routed" in by_scope:
                table_rate = float(by_scope["routed"]["guaranteed_key_rate"])
                table_evidence = by_scope["routed"].get("evidence_kind")
        return {
            "positive_routed_certificate_at_S1": positive,
            "archived_routed_certificate_positive_at_S1": positive,
            "archived_m4_routed_certificate_positive_at_S1": positive,
            "exact_long_only_attack_at_S1": exact_attack,
            "strict_separation_table_ready": table_ready,
            "strict_same_point_separation_ready": exact_attack and table_ready,
            "configured_point": {
                "nu": expected_nu,
                "eta_s": expected_eta_s,
                "eta_b": expected_eta,
                "eta_c": expected_eta,
                "theta": float(separation["theta"]),
            },
            "minimum_positive_rate": positive_margin,
            "routed_certificate_key_rate_lower": reference.get("K_lower"),
            "strict_table_routed_rate": table_rate,
            "strict_table_routed_evidence": table_evidence,
        }

    def verify_figures(self) -> None:
        manifest_path = self.output / "figures" / "figure_manifest.json"
        self.check(manifest_path.is_file(), "Missing figure manifest")
        if not manifest_path.is_file():
            return
        manifest = _load(manifest_path)
        expected_stems = set(self.config["curve_plots"]["figure_stems"])
        expected_figures = {
            f"figures/{stem}.{extension}"
            for stem in expected_stems
            for extension in ("pdf", "png")
        }
        expected_data = {f"figure_data/{stem}.json" for stem in expected_stems}
        self.check(
            set(manifest.get("files", {})) == expected_figures,
            "Figure manifest does not contain exactly the five PDF/PNG figure pairs",
        )
        self.check(
            set(manifest.get("figure_data", {})) == expected_data,
            "Figure manifest does not contain exactly the five JSON data files",
        )
        for relative, expected in manifest.get("files", {}).items():
            path = self.output / relative
            self.check(path.is_file(), f"Missing figure: {relative}")
            if path.is_file():
                self.check(_hash(path) == expected, f"Figure hash mismatch: {relative}")
        for relative, expected in manifest.get("figure_data", {}).items():
            path = self.output / relative
            self.check(path.is_file(), f"Missing figure data: {relative}")
            if path.is_file():
                self.check(_hash(path) == expected, f"Figure-data hash mismatch: {relative}")
        input_tables = manifest.get("input_tables", {})
        self.check(bool(input_tables), "Figure manifest contains no input tables")
        for relative, expected in input_tables.items():
            path = self.output / relative
            self.check(path.is_file(), f"Missing figure input table: {relative}")
            if path.is_file():
                self.check(
                    _hash(path) == expected,
                    f"Figures were not generated from the current {relative}",
                )
        envelope_table = self.output / "tables" / "protocol_envelopes.csv"
        if envelope_table.is_file():
            for name in (
                "witness_envelopes.json",
                "witness_envelopes_independent.json",
            ):
                path = self.output / "audits" / name
                self.check(path.is_file(), f"Missing envelope audit: {name}")
                if path.is_file():
                    self.check(
                        _load(path).get("passed") is True,
                        f"Envelope audit did not pass: {name}",
                    )

    def verify_reference_data(self) -> None:
        manifest_path = self.output / "audits" / "reference_data.json"
        self.check(manifest_path.is_file(), "Missing reference-data audit")
        if not manifest_path.is_file():
            return
        manifest = _load(manifest_path)
        self.check(manifest.get("passed") is True, "Reference-data audit did not pass")
        for raw_path, expected in manifest.get("inputs", {}).items():
            path = resolve_audit_input(self.output, raw_path)
            self.check(path.is_file(), f"Missing reference-data input: {path}")
            if path.is_file():
                self.check(
                    _hash(path) == expected,
                    f"Reference-data input hash mismatch: {path}",
                )
        for relative, expected in manifest.get("files", {}).items():
            path = self.output / relative
            self.check(path.is_file(), f"Missing reference table: {relative}")
            if path.is_file():
                self.check(
                    _hash(path) == expected,
                    f"Reference table hash mismatch: {relative}",
                )

    def run(self) -> int:
        self.verify_sources()
        self.verify_artifact_hashes()
        self.read_points()
        self.verify_tables()
        self.verify_mwrf()
        self.verify_protocol_nesting()
        self.verify_reference_data()
        gates = self.scientific_gates()
        if self.args.require_figures:
            self.verify_figures()
        if self.args.require_scientific_gates:
            scientific = self.config["scientific_gates"]
            if scientific.get("require_verified_routed_certificate"):
                self.check(
                    bool(gates["positive_routed_certificate_at_S1"]),
                    "Verified routed certificate at S1 is not positive",
                )
            if scientific.get("require_exact_long_only_attack"):
                self.check(
                    bool(gates["exact_long_only_attack_at_S1"]),
                    "Exact long-only attack at S1 failed",
                )
            if scientific.get("require_verified_routed_certificate") and scientific.get(
                "require_exact_long_only_attack"
            ):
                self.check(
                    bool(gates["strict_same_point_separation_ready"]),
                    "Same-point strict separation gate failed",
                )

        audit = {
            "schema": "routed-dicka-output-audit-v2",
            "passed": not self.errors,
            "hard_errors": self.errors,
            "warnings": self.warnings,
            "point_count": len(self.records),
            "point_summaries": self.point_summaries,
            "scientific_gates": gates,
            "figures_checked": self.args.require_figures,
            "scientific_gates_required": self.args.require_scientific_gates,
        }
        audit_path = self.output / "audits" / "output_audit.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"passed": audit["passed"], "point_count": len(self.records), "scientific_gates": gates, "errors": self.errors}, indent=2))
        return 0 if not self.errors else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "paper_suite.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--require-figures", action="store_true")
    parser.add_argument("--require-scientific-gates", action="store_true")
    args = parser.parse_args()
    raise SystemExit(Auditor(args).run())


if __name__ == "__main__":
    main()
