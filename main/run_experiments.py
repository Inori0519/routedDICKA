"""Run the manuscript experiment matrix and write resumable raw point files."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
VERIFY = ROOT / "verification"
if str(VERIFY) not in sys.path:
    sys.path.insert(0, str(VERIFY))

from routed_dicka.io import (  # noqa: E402
    collect_file_hashes,
    environment_record,
    file_sha256,
    load_json,
    object_sha256,
    source_hashes,
    utc_now,
    write_csv,
    write_json,
)
from routed_dicka.keyrate import (  # noqa: E402
    compute_point,
    compute_protocol_point,
    recordings_for_mode,
)
from routed_dicka.protocols import TargetPoint, get_protocol  # noqa: E402
from routed_dicka.quadrature import right_radau_rule  # noqa: E402
from routed_dicka.reduced_certificate import certify_reduced_node  # noqa: E402
from routed_dicka.reduced_models import build_reduced_model  # noqa: E402
from routed_dicka.reduced_solver import ReducedNPA  # noqa: E402
from routed_dicka.sdp import SolverSettings  # noqa: E402
from routed_dicka.statistics import (  # noqa: E402
    constraint_manifest,
    dephasing_difference_report,
    mwrf_coordinate_sets,
    validate_mwrf_lattice,
)
from routed_dicka.target import (  # noqa: E402
    build_route_selective_dephasing_attack,
    build_target,
)
from verify_reduced_certificate import (  # noqa: E402
    certified_entropy,
    verify_node_certificate,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "paper_suite.json")
    parser.add_argument("--profile", default="paper")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--m", type=int)
    parser.add_argument("--level", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--experiments",
        nargs="*",
        help="Optional explicit experiment IDs, overriding the profile.",
    )
    return parser.parse_args()


def _safe_token(value: object) -> str:
    return str(value).replace(".", "p").replace("/", "-").replace("\\", "-")


class SuiteRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config_path = args.config.resolve()
        self.config = load_json(self.config_path)
        if not isinstance(self.config, dict):
            raise TypeError("Configuration root must be a JSON object")
        solver_config = dict(self.config["solver"])
        self.m_override = args.m
        if args.m is not None:
            solver_config["radau_nodes"] = args.m
        if args.level is not None:
            solver_config["level"] = args.level
        self.m = int(solver_config["radau_nodes"])
        if self.m < 2:
            raise ValueError("Right Gauss-Radau quadrature requires m >= 2")
        self.settings = SolverSettings.from_mapping(solver_config)
        self.output = args.output.resolve()
        self.raw_dir = self.output / "raw"
        self.target_dir = self.output / "targets"
        self.table_dir = self.output / "tables"
        self.common = dict(self.config["common_target"])
        self.initial_source_hashes = source_hashes(ROOT)
        computation_source_hashes = {
            path: digest
            for path, digest in self.initial_source_hashes.items()
            if path.startswith("src/") or path == "main/run_experiments.py"
        }
        self.run_contract = {
            "schema": "routed-dicka-run-contract-v2",
            "profile": args.profile,
            "default_m": self.m,
            "m_override": self.m_override,
            "level": self.settings.level,
            "solver": solver_config,
            "config_sha256": file_sha256(self.config_path),
            "computation_source_sha256": object_sha256(computation_source_hashes),
        }
        self.run_contract_sha256 = object_sha256(self.run_contract)
        self.records: list[tuple[Path, dict[str, object]]] = []
        self.targets_written: dict[str, Path] = {}
        self.auxiliary_written: set[Path] = set()

    def experiment_m(self, specification: dict[str, object]) -> int:
        allow_override = bool(specification.get("allow_m_override", True))
        value = (
            self.m_override
            if self.m_override is not None and allow_override
            else specification.get("m", self.m)
        )
        result = int(value)
        if result < 2:
            raise ValueError("Right Gauss-Radau quadrature requires m >= 2")
        return result

    def selected_experiments(self) -> list[str]:
        if self.args.experiments:
            selected = list(self.args.experiments)
        else:
            profiles = self.config["profiles"]
            if self.args.profile not in profiles:
                raise ValueError(
                    f"Unknown profile {self.args.profile!r}; choose from {sorted(profiles)}"
                )
            selected = list(profiles[self.args.profile])
        missing = set(selected) - set(self.config["experiments"])
        if missing:
            raise ValueError(f"Unknown experiment IDs: {sorted(missing)}")
        return selected

    def target(self, values: dict[str, object], construction: str = "honest"):
        point = TargetPoint.from_mapping(values, self.common)
        if construction == "honest":
            behavior = build_target(point)
            suffix = ""
        elif construction == "route-selective-dephasing-attack":
            behavior = build_route_selective_dephasing_attack(point)
            suffix = "_route_dephasing_attack"
        else:
            raise ValueError(f"Unknown target construction: {construction}")
        record = behavior.to_record()
        path = self.target_dir / f"{point.slug}{suffix}.json"
        key = f"{point.slug}{suffix}"
        if key not in self.targets_written:
            if not path.exists() or load_json(path) != record:
                write_json(path, record)
            self.targets_written[key] = path
        return behavior, path, record["sha256"]

    def execute(
        self,
        experiment_id: str,
        file_stem: str,
        specification: dict[str, object],
        compute: Callable[[], dict[str, object]],
    ) -> dict[str, object]:
        signature = object_sha256(
            {
                "run_contract": self.run_contract,
                "experiment_id": experiment_id,
                "specification": specification,
            }
        )
        path = self.raw_dir / experiment_id / f"{file_stem}.json"
        if self.args.resume and path.exists():
            existing = load_json(path)
            if (
                isinstance(existing, dict)
                and existing.get("run_signature") == signature
                and existing.get("claim_eligible") is True
            ):
                print(f"RESUME {path.relative_to(self.output)}", flush=True)
                self.records.append((path, existing))
                return existing
            if isinstance(existing, dict) and existing.get("run_signature") == signature:
                print(f"RETRY ineligible point {path.relative_to(self.output)}", flush=True)

        print(f"RUN {experiment_id} / {file_stem}", flush=True)
        result = compute()
        result["experiment_id"] = experiment_id
        result["run_signature"] = signature
        result["run_contract_sha256"] = self.run_contract_sha256
        result["created_utc"] = utc_now()
        result["source_file"] = path.relative_to(self.output).as_posix()
        write_json(path, result)
        self.records.append((path, result))
        if result.get("claim_eligible") is not True:
            raise RuntimeError(
                f"Point is not eligible for a manuscript claim; diagnostics saved to {path}"
            )
        return result

    def _protocol_point(
        self,
        experiment_id: str,
        values: dict[str, object],
        protocol_id: str,
        statistics_recording: str,
        reconciliation_recording: str,
        include_near: bool,
        metadata: dict[str, object],
        m: int | None = None,
    ) -> dict[str, object]:
        point = TargetPoint.from_mapping(values, self.common)
        m_value = self.m if m is None else int(m)
        stem = "_".join(
            [
                _safe_token(metadata.get("point_id", metadata.get("slice_id", "point"))),
                protocol_id,
                "routed" if include_near else "long_only",
                statistics_recording,
                point.slug,
                f"L{self.settings.level}",
                f"m{m_value}",
            ]
        )
        specification = {
            "values": values,
            "protocol": protocol_id,
            "statistics_recording": statistics_recording,
            "reconciliation_recording": reconciliation_recording,
            "include_near": include_near,
            "metadata": metadata,
            "m": m_value,
        }

        def compute() -> dict[str, object]:
            behavior, target_path, target_hash = self.target(values)
            result = compute_protocol_point(
                behavior,
                get_protocol(protocol_id),
                m=m_value,
                settings=self.settings,
                statistics_recording=statistics_recording,
                reconciliation_recording=reconciliation_recording,
                include_near=include_near,
                metadata=metadata,
            )
            result["target_file"] = target_path.relative_to(self.output).as_posix()
            result["target_sha256"] = target_hash
            return result

        return self.execute(experiment_id, stem, specification, compute)

    @staticmethod
    def _reduced_constraint_manifest(problem: ReducedNPA) -> dict[str, object]:
        rows = [
            {
                "name": problem.algebra.label(word),
                "value": float(value),
            }
            for word, value in sorted(
                problem.fixed.items(), key=lambda item: (len(item[0]), item[0])
            )
        ]
        names = [row["name"] for row in rows]
        return {
            "constraint_id": sha256("\n".join(names).encode("ascii")).hexdigest(),
            "constraint_value_sha256": object_sha256(rows),
            "constraint_count": len(rows),
            "coordinates": rows,
        }

    def _reduced_point(
        self,
        experiment_id: str,
        values: dict[str, object],
        protocol_id: str,
        *,
        statistics_recording: str,
        reconciliation_recording: str,
        metadata: dict[str, object],
        m: int,
        include_near: bool = True,
    ) -> dict[str, object]:
        if m not in (2, 3, 4):
            raise ValueError(
                "The exact reduced-certificate backend currently supports m=2, m=3, or m=4"
            )
        point = TargetPoint.from_mapping(values, self.common)
        stem = "_".join(
            [
                _safe_token(metadata.get("point_id", metadata.get("slice_id", "point"))),
                protocol_id,
                "routed" if include_near else "long_only",
                statistics_recording,
                point.slug,
                f"L{self.settings.level}",
                f"m{m}",
            ]
        )
        specification = {
            "values": values,
            "protocol": protocol_id,
            "statistics_recording": statistics_recording,
            "reconciliation_recording": reconciliation_recording,
            "metadata": metadata,
            "include_near": include_near,
            "m": m,
            "backend": "reduced-exact-certificate",
        }

        def compute() -> dict[str, object]:
            behavior, target_path, target_hash = self.target(values)
            model = build_reduced_model(
                behavior,
                protocol_id,
                statistics_recording=statistics_recording,
                include_near=include_near,
            )
            certificate_dir = self.output / "certificates" / experiment_id / stem
            certificate_paths: list[Path] = []
            node_records = []
            solver_parameters = {
                "relative_gap_tolerance": self.settings.relative_gap_tolerance,
                "primal_feasibility_tolerance": self.settings.primal_feasibility_tolerance,
                "dual_feasibility_tolerance": self.settings.dual_feasibility_tolerance,
                "max_iterations": self.settings.max_iterations,
                "threads": self.settings.threads,
            }
            for node in right_radau_rule(m):
                if node.endpoint:
                    continue
                problem = ReducedNPA(
                    model,
                    t=node.t,
                    level=self.settings.level,
                    enrich=True,
                    localizers=True,
                )
                search = problem.solve(parameters=solver_parameters)
                certificate_path = certificate_dir / f"node_{node.index}.certificate.json"
                affine_path = certificate_dir / f"node_{node.index}.affine.json"
                generated = certify_reduced_node(
                    problem,
                    m=m,
                    node_index=node.index,
                    export=certificate_path,
                )
                verified = verify_node_certificate(
                    certificate_path,
                    affine_export=affine_path,
                )
                certificate_paths.append(certificate_path)
                node_records.append(
                    {
                        "index": node.index,
                        "t": node.t,
                        "weight": node.weight,
                        "search": {
                            key: value
                            for key, value in search.items()
                            if key not in ("x", "dual")
                        },
                        "model": problem.metadata,
                        "generated_certificate": generated,
                        "verified_certificate": verified,
                        "certificate_file": certificate_path.relative_to(self.output).as_posix(),
                        "certificate_sha256": file_sha256(certificate_path),
                        "affine_file": affine_path.relative_to(self.output).as_posix(),
                        "affine_sha256": file_sha256(affine_path),
                        "claim_eligible": bool(verified["verified"]),
                    }
                )
            entropy = certified_entropy(certificate_paths, m=m)
            h_ab, h_ac = behavior.key_entropies(recording=reconciliation_recording)
            ec = max(h_ab, h_ac)
            ec_upper = min(1.0, ec + 1e-12)
            raw_rate = float(entropy["entropy_lower"]) - ec_upper
            representative = ReducedNPA(
                model,
                t=right_radau_rule(m)[0].t,
                level=self.settings.level,
                enrich=True,
                localizers=True,
            )
            result = {
                "schema": "routed-dicka-reduced-certified-point-v1",
                "evidence_kind": "verified_generated_certificate",
                "metadata": {
                    "protocol": protocol_id,
                    "statistics_recording": statistics_recording,
                    "include_near": include_near,
                    **metadata,
                },
                "physical_point": behavior.point.as_dict(),
                "target_construction": behavior.construction,
                "constraint_manifest": self._reduced_constraint_manifest(representative),
                "solver": {
                    "name": "mosek",
                    "backend": "reduced-sparse-real-part",
                    "npa_level": self.settings.level,
                },
                "entropy": {
                    "m": m,
                    "radau_rule": [item.as_dict() for item in right_radau_rule(m)],
                    "nodes": node_records,
                    "raw_entropy_lower": float(entropy["entropy_lower"]),
                    "entropy_lower": min(1.0, max(0.0, float(entropy["entropy_lower"]))),
                    "entropy_lower_exact": entropy["entropy_lower_exact"],
                    "all_nodes_claim_eligible": all(
                        bool(node["claim_eligible"]) for node in node_records
                    ),
                },
                "reconciliation_recording": reconciliation_recording,
                "H_A_given_B": h_ab,
                "H_A_given_C": h_ac,
                "EC_cost": ec_upper,
                "raw_key_rate_lower": raw_rate,
                "key_rate_lower": max(0.0, raw_rate),
                "claim_eligible": all(
                    bool(node["claim_eligible"]) for node in node_records
                ),
                "target_file": target_path.relative_to(self.output).as_posix(),
                "target_sha256": target_hash,
            }
            return result

        return self.execute(experiment_id, stem, specification, compute)

    def run_reduced_point(self, experiment_id: str, spec: dict[str, object]) -> None:
        self._reduced_point(
            experiment_id,
            spec,
            str(spec["protocol"]),
            statistics_recording=str(spec.get("recording", "ternary")),
            reconciliation_recording="ternary",
            metadata={
                "experiment_kind": "reduced_point",
                "point_id": spec["point_id"],
                "scope": spec.get("scope"),
            },
            m=self.experiment_m(spec),
            include_near=bool(spec.get("include_near", True)),
        )

    def run_reduced_protocol_point(
        self, experiment_id: str, spec: dict[str, object]
    ) -> None:
        m = self.experiment_m(spec)
        for protocol_id in spec["protocols"]:
            self._reduced_point(
                experiment_id,
                spec,
                str(protocol_id),
                statistics_recording=str(spec.get("recording", "ternary")),
                reconciliation_recording="ternary",
                metadata={
                    "experiment_kind": "protocol_comparison",
                    "point_id": spec["point_id"],
                },
                m=m,
                include_near=bool(spec.get("include_near", True)),
            )

    def run_reduced_anchor_scan(
        self, experiment_id: str, spec: dict[str, object]
    ) -> None:
        m = self.experiment_m(spec)
        for eta_l in spec["eta_l"]:
            values = {**spec, "eta_l": eta_l}
            for protocol_id in spec["protocols"]:
                self._reduced_point(
                    experiment_id,
                    values,
                    str(protocol_id),
                    statistics_recording=str(spec.get("recording", "ternary")),
                    reconciliation_recording="ternary",
                    metadata={
                        "experiment_kind": "certificate_anchor",
                        "slice_id": spec["slice_id"],
                        "point_id": f"eta{eta_l:g}",
                    },
                    m=m,
                    include_near=bool(spec.get("include_near", True)),
                )

    def run_reduced_curve_matrix(
        self, experiment_id: str, spec: dict[str, object]
    ) -> None:
        m = self.experiment_m(spec)
        for curve_slice in spec["slices"]:
            include_near = bool(curve_slice["include_near"])
            for eta_l in spec["eta_l"]:
                values = {
                    **spec,
                    **curve_slice,
                    "eta_l": eta_l,
                }
                for protocol_id in curve_slice["protocols"]:
                    self._reduced_point(
                        experiment_id,
                        values,
                        str(protocol_id),
                        statistics_recording=str(spec.get("recording", "ternary")),
                        reconciliation_recording="ternary",
                        metadata={
                            "experiment_kind": "certificate_anchor",
                            "slice_id": curve_slice["slice_id"],
                            "point_id": f"eta{eta_l:g}",
                            "data_scope": "routed" if include_near else "far_only",
                        },
                        m=m,
                        include_near=include_near,
                    )

    def run_reduced_recording_ablation(
        self, experiment_id: str, spec: dict[str, object]
    ) -> None:
        m = self.experiment_m(spec)
        expected_modes = {
            "binary_statistics_binary_reconciliation",
            "binary_statistics_ternary_reconciliation",
            "ternary_statistics_ternary_reconciliation",
        }
        if set(spec["report_modes"]) != expected_modes:
            raise ValueError("The recording ablation must declare the three canonical modes")
        for eta_l in spec["eta_l"]:
            values = {**spec, "eta_l": eta_l}
            binary = self._reduced_point(
                experiment_id,
                values,
                str(spec["protocol"]),
                statistics_recording="binary",
                reconciliation_recording="binary",
                metadata={
                    "experiment_kind": "recording_ablation",
                    "recording_mode": "binary_statistics_binary_reconciliation",
                    "point_id": f"eta{eta_l:g}",
                },
                m=m,
            )
            behavior, _, _ = self.target(values)
            h_ab, h_ac = behavior.key_entropies(recording="ternary")
            ec_upper = min(1.0, max(h_ab, h_ac) + 1e-12)
            derived_stem = "_".join(
                [
                    f"eta{_safe_token(eta_l)}",
                    str(spec["protocol"]),
                    "binary_stats_ternary_ec",
                    behavior.point.slug,
                    f"L{self.settings.level}",
                    f"m{m}",
                ]
            )
            derived_specification = {
                "values": values,
                "protocol": spec["protocol"],
                "statistics_recording": "binary",
                "reconciliation_recording": "ternary",
                "derived_from": binary["source_file"],
                "m": m,
            }

            def derive() -> dict[str, object]:
                result = deepcopy(binary)
                result.pop("experiment_id", None)
                result.pop("run_signature", None)
                result.pop("run_contract_sha256", None)
                result.pop("created_utc", None)
                result.pop("source_file", None)
                result["schema"] = "routed-dicka-derived-reconciliation-point-v1"
                result["evidence_kind"] = "reused_verified_entropy_certificate"
                result["metadata"] = {
                    **dict(result["metadata"]),
                    "recording_mode": "binary_statistics_ternary_reconciliation",
                    "entropy_source_file": binary["source_file"],
                }
                result["reconciliation_recording"] = "ternary"
                result["H_A_given_B"] = h_ab
                result["H_A_given_C"] = h_ac
                result["EC_cost"] = ec_upper
                raw_rate = float(result["entropy"]["entropy_lower"]) - ec_upper
                result["raw_key_rate_lower"] = raw_rate
                result["key_rate_lower"] = max(0.0, raw_rate)
                return result

            self.execute(
                experiment_id,
                derived_stem,
                derived_specification,
                derive,
            )
            self._reduced_point(
                experiment_id,
                values,
                str(spec["protocol"]),
                statistics_recording="ternary",
                reconciliation_recording="ternary",
                metadata={
                    "experiment_kind": "recording_ablation",
                    "recording_mode": "ternary_statistics_ternary_reconciliation",
                    "point_id": f"eta{eta_l:g}",
                },
                m=m,
            )

    def run_reduced_radau_sensitivity(
        self, experiment_id: str, spec: dict[str, object]
    ) -> None:
        for m_value in spec["m_values"]:
            self._reduced_point(
                experiment_id,
                spec,
                str(spec["protocol"]),
                statistics_recording=str(spec.get("recording", "ternary")),
                reconciliation_recording="ternary",
                metadata={
                    "experiment_kind": "radau_sensitivity",
                    "point_id": spec["point_id"],
                    "quadrature_order": int(m_value),
                },
                m=int(m_value),
            )

    def run_protocol_curves(self, experiment_id: str, spec: dict[str, object]) -> None:
        for eta_l in spec["eta_l"]:
            values = {**spec, "eta_l": eta_l}
            for protocol in spec["protocols"]:
                self._protocol_point(
                    experiment_id,
                    values,
                    protocol,
                    statistics_recording=str(spec.get("recording", "ternary")),
                    reconciliation_recording="ternary",
                    include_near=bool(spec.get("include_near", True)),
                    metadata={
                        "experiment_kind": "protocol_curves",
                        "slice_id": spec["slice_id"],
                        "point_id": f"eta{eta_l:g}",
                    },
                )

    def run_recording_ablation(self, experiment_id: str, spec: dict[str, object]) -> None:
        for eta_l in spec["eta_l"]:
            values = {**spec, "eta_l": eta_l}
            for mode in spec["modes"]:
                statistics_recording, reconciliation_recording = recordings_for_mode(mode)
                self._protocol_point(
                    experiment_id,
                    values,
                    str(spec["protocol"]),
                    statistics_recording=statistics_recording,
                    reconciliation_recording=reconciliation_recording,
                    include_near=True,
                    metadata={
                        "experiment_kind": "recording_ablation",
                        "recording_mode": mode,
                        "point_id": f"eta{eta_l:g}",
                    },
                )

    def run_routed_long_only(self, experiment_id: str, spec: dict[str, object]) -> None:
        for include_near, mode in ((True, "routed"), (False, "long_only")):
            result = self._protocol_point(
                experiment_id,
                spec,
                str(spec["protocol"]),
                statistics_recording=str(spec.get("recording", "ternary")),
                reconciliation_recording="ternary",
                include_near=include_near,
                metadata={
                    "experiment_kind": "routed_long_only",
                    "comparison_mode": mode,
                    "point_id": mode,
                },
            )
            if include_near and self.config["scientific_gates"].get(
                "require_positive_routed_rate", False
            ):
                minimum = float(
                    self.config["scientific_gates"].get("minimum_positive_rate", 0.0)
                )
                if float(result["raw_key_rate_lower"]) <= minimum:
                    raise RuntimeError(
                        "The configured same-point routed rate did not pass the "
                        f"positive-rate gate ({result['raw_key_rate_lower']} <= {minimum})"
                    )

    def run_mwrf(self, experiment_id: str, spec: dict[str, object]) -> None:
        m = self.experiment_m(spec)
        points = spec.get("points")
        if points is None:
            points = [spec]
        sets = mwrf_coordinate_sets()
        for point_spec in points:
            values = {**spec, **point_spec}
            point_id = str(point_spec.get("point_id", spec.get("point_id", "point")))
            behavior, target_path, target_hash = self.target(values)
            attack, attack_path, attack_hash = self.target(
                values, construction="route-selective-dephasing-attack"
            )
            difference = dephasing_difference_report(behavior, attack)
            difference.update(
                {
                    "schema": "routed-dicka-dephasing-check-v1",
                    "physical_point": behavior.point.as_dict(),
                    "honest_target": target_path.relative_to(self.output).as_posix(),
                    "honest_target_sha256": target_hash,
                    "attack_target": attack_path.relative_to(self.output).as_posix(),
                    "attack_target_sha256": attack_hash,
                }
            )
            difference_path = self.raw_dir / experiment_id / f"{point_id}_dephasing.json"
            write_json(difference_path, difference)
            self.auxiliary_written.add(difference_path)

            for model_name in spec.get("attack_models", []):
                model_name = str(model_name)
                if difference["accepted_models"].get(model_name) is not True:
                    raise AssertionError(
                        f"The explicit dephasing attack is not compatible with {model_name}"
                    )
                stem = "_".join(
                    [point_id, model_name, "attack", behavior.point.slug]
                )
                specification = {
                    "point": values,
                    "point_id": point_id,
                    "model": model_name,
                    "evidence": "route-selective-dephasing-attack",
                }

                def attack_record(model_name=model_name):
                    h_ab, h_ac = behavior.key_entropies(recording="ternary")
                    ec = max(h_ab, h_ac)
                    return {
                        "schema": "routed-dicka-mwrf-attack-point-v1",
                        "evidence_kind": "explicit_compatible_zero_entropy_attack",
                        "metadata": {
                            "experiment_kind": "mwrf",
                            "point_id": point_id,
                            "info_model": model_name,
                            "protocol": "rHybrid",
                            "statistics_recording": "ternary",
                            "include_near": True,
                        },
                        "physical_point": behavior.point.as_dict(),
                        "target_construction": behavior.construction,
                        "constraint_manifest": constraint_manifest(
                            behavior, sets[model_name]
                        ),
                        "solver": {
                            "name": None,
                            "backend": "exact-physical-attack",
                            "npa_level": None,
                        },
                        "entropy": {
                            "m": 0,
                            "nodes": [],
                            "raw_entropy_lower": 0.0,
                            "entropy_lower": 0.0,
                            "compatible_attack_entropy_upper": 0.0,
                            "all_nodes_claim_eligible": True,
                        },
                        "reconciliation_recording": "ternary",
                        "H_A_given_B": h_ab,
                        "H_A_given_C": h_ac,
                        "EC_cost": ec,
                        "raw_key_rate_lower": -ec,
                        "key_rate_lower": 0.0,
                        "guaranteed_key_rate_upper": 0.0,
                        "claim_eligible": True,
                        "attack_target_file": attack_path.relative_to(self.output).as_posix(),
                        "attack_target_sha256": attack_hash,
                        "target_file": target_path.relative_to(self.output).as_posix(),
                        "target_sha256": target_hash,
                    }

                self.execute(experiment_id, stem, specification, attack_record)

            for model_name in spec["models"]:
                stem = "_".join(
                    [
                        point_id,
                        str(model_name),
                        behavior.point.slug,
                        f"L{self.settings.level}",
                        f"m{m}",
                    ]
                )
                specification = {
                    "point": values,
                    "point_id": point_id,
                    "model": model_name,
                    "m": m,
                }

                def compute(model_name=model_name):
                    result = compute_point(
                        behavior,
                        sets[str(model_name)],
                        m=m,
                        settings=self.settings,
                        reconciliation_recording="ternary",
                        metadata={
                            "experiment_kind": "mwrf",
                            "point_id": point_id,
                            "info_model": str(model_name),
                            "protocol": "rHybrid",
                            "statistics_recording": "ternary",
                            "include_near": True,
                        },
                    )
                    result["target_file"] = target_path.relative_to(self.output).as_posix()
                    result["target_sha256"] = target_hash
                    return result

                self.execute(experiment_id, stem, specification, compute)

    def run_radau_sensitivity(self, experiment_id: str, spec: dict[str, object]) -> None:
        for m_value in spec["m_values"]:
            if int(m_value) < 2:
                raise ValueError("Radau sensitivity values must all be at least 2")
            self._protocol_point(
                experiment_id,
                spec,
                str(spec["protocol"]),
                statistics_recording=str(spec.get("recording", "ternary")),
                reconciliation_recording="ternary",
                include_near=bool(spec.get("include_near", True)),
                metadata={
                    "experiment_kind": "radau_sensitivity",
                    "point_id": f"m{m_value}",
                },
                m=int(m_value),
            )

    def aggregate_rows(self) -> list[dict[str, object]]:
        rows = []
        for path, record in self.records:
            metadata = dict(record.get("metadata", {}))
            point = dict(record.get("physical_point", {}))
            entropy = dict(record.get("entropy", {}))
            rows.append(
                {
                    "experiment_id": record["experiment_id"],
                    "experiment_kind": metadata.get("experiment_kind"),
                    "evidence_kind": record.get("evidence_kind", "raw_bff_dual"),
                    "backend": record.get("solver", {}).get("backend"),
                    "slice_id": metadata.get("slice_id"),
                    "point_id": metadata.get("point_id"),
                    "protocol": metadata.get("protocol"),
                    "info_model": metadata.get("info_model"),
                    "comparison_mode": metadata.get("comparison_mode"),
                    "recording_mode": metadata.get("recording_mode"),
                    "statistics_recording": metadata.get("statistics_recording"),
                    "reconciliation_recording": record.get("reconciliation_recording"),
                    "include_near": metadata.get("include_near"),
                    "nu": point.get("nu"),
                    "eta_a": point.get("eta_a"),
                    "eta_s": point.get("eta_s"),
                    "eta_b": point.get("eta_b"),
                    "eta_c": point.get("eta_c"),
                    "theta": point.get("far_theta"),
                    "npa_level": record.get("solver", {}).get("npa_level"),
                    "m": entropy.get("m"),
                    "constraint_count": record.get("constraint_manifest", {}).get("constraint_count"),
                    "constraint_id": record.get("constraint_manifest", {}).get("constraint_id"),
                    "entropy_lower": entropy.get("entropy_lower"),
                    "raw_entropy_lower": entropy.get("raw_entropy_lower"),
                    "H_A_given_B": record.get("H_A_given_B"),
                    "H_A_given_C": record.get("H_A_given_C"),
                    "EC_cost": record.get("EC_cost"),
                    "raw_key_rate_lower": record.get("raw_key_rate_lower"),
                    "key_rate_lower": record.get("key_rate_lower"),
                    "guaranteed_key_rate_upper": record.get("guaranteed_key_rate_upper"),
                    "claim_eligible": record.get("claim_eligible"),
                    "source_file": path.relative_to(self.output).as_posix(),
                    "target_file": record.get("target_file"),
                    "target_sha256": record.get("target_sha256"),
                }
            )
        return rows

    def finalize(self, selected: list[str]) -> None:
        final_source_hashes = source_hashes(ROOT)
        if final_source_hashes != self.initial_source_hashes:
            raise RuntimeError("Python source files changed while the suite was running")
        rows = self.aggregate_rows()
        write_csv(self.table_dir / "results.csv", rows)
        secondary_tables = {}
        for kind, filename in (
            ("mwrf", "mwrf.csv"),
            ("recording_ablation", "recording_ablation.csv"),
            ("radau_sensitivity", "radau_sensitivity.csv"),
        ):
            path = self.table_dir / filename
            write_csv(path, [row for row in rows if row["experiment_kind"] == kind])
            secondary_tables[kind] = path
        manifest_rows = []
        for path, record in self.records:
            for coordinate in record["constraint_manifest"]["coordinates"]:
                manifest_rows.append(
                    {
                        "source_file": path.relative_to(self.output).as_posix(),
                        "constraint_id": record["constraint_manifest"]["constraint_id"],
                        **coordinate,
                    }
                )
        write_csv(self.table_dir / "constraint_manifest.csv", manifest_rows)
        point_paths = [path for path, _ in self.records]
        auxiliary_paths = [*self.targets_written.values(), *self.auxiliary_written]
        for _, record in self.records:
            for node in record.get("entropy", {}).get("nodes", []):
                for key in ("certificate_file", "affine_file"):
                    relative = node.get(key)
                    if relative:
                        auxiliary_paths.append(self.output / str(relative))
        table_paths = [
            self.table_dir / "results.csv",
            self.table_dir / "constraint_manifest.csv",
            *secondary_tables.values(),
        ]
        manifest = {
            "schema": "routed-dicka-run-manifest-v2",
            "created_utc": utc_now(),
            "selected_experiments": selected,
            "run_contract": self.run_contract,
            "run_contract_sha256": self.run_contract_sha256,
            "environment": environment_record(),
            "source_hashes": final_source_hashes,
            "mwrf_lattice": validate_mwrf_lattice(),
            "point_files": [path.relative_to(self.output).as_posix() for path in point_paths],
            "point_hashes": collect_file_hashes(self.output, point_paths),
            "auxiliary_hashes": collect_file_hashes(self.output, auxiliary_paths),
            "table_hashes": collect_file_hashes(self.output, table_paths),
            "result_rows": len(rows),
        }
        write_json(self.output / "manifest.json", manifest)

    def run(self) -> None:
        try:
            import mosek  # noqa: F401
        except ImportError as exc:
            raise SystemExit("MOSEK Python bindings are required for this suite") from exc

        selected = self.selected_experiments()
        handlers = {
            "protocol_curves": self.run_protocol_curves,
            "recording_ablation": self.run_recording_ablation,
            "routed_long_only": self.run_routed_long_only,
            "reduced_point": self.run_reduced_point,
            "reduced_protocol_point": self.run_reduced_protocol_point,
            "reduced_anchor_scan": self.run_reduced_anchor_scan,
            "reduced_curve_matrix": self.run_reduced_curve_matrix,
            "reduced_recording_ablation": self.run_reduced_recording_ablation,
            "reduced_radau_sensitivity": self.run_reduced_radau_sensitivity,
            "mwrf": self.run_mwrf,
            "radau_sensitivity": self.run_radau_sensitivity,
        }
        for experiment_id in selected:
            spec = deepcopy(self.config["experiments"][experiment_id])
            kind = str(spec["kind"])
            if kind not in handlers:
                raise ValueError(f"Unsupported experiment kind: {kind}")
            handlers[kind](experiment_id, spec)
        self.finalize(selected)
        print(f"WROTE {self.output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    SuiteRunner(_parse_args()).run()
