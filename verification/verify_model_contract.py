"""Check the protocol/statistics/BFF configuration without solving an SDP."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routed_dicka.protocols import PROTOCOLS, TargetPoint  # noqa: E402
from routed_dicka.quadrature import right_radau_rule, validate_rule  # noqa: E402
from routed_dicka.statistics import (  # noqa: E402
    mwrf_coordinate_sets,
    protocol_coordinates,
    validate_mwrf_lattice,
)
from routed_dicka.target import build_target  # noqa: E402


def _assert_alpha_squared_localizer(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    value = assignments.get("alpha_squared")
    if not (
        isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.Pow)
        and isinstance(value.right, ast.Constant)
        and value.right.value == 2
    ):
        raise AssertionError("BFF localizer must define alpha_squared = alpha ** 2")
    if "inequalities=inequalities" not in source or "momentinequalities=[]" not in source:
        raise AssertionError(
            "BFF norm polynomials must generate localizing matrices, not scalar moment inequalities"
        )


def _experiment_cost(specification: dict[str, object]) -> dict[str, int]:
    kind = str(specification["kind"])
    attack_records = 0
    report_rows = 0
    if kind == "reduced_point":
        points = 1
        nodes = points * (int(specification["m"]) - 1)
    elif kind == "reduced_protocol_point":
        points = len(specification["protocols"])
        nodes = points * (int(specification["m"]) - 1)
    elif kind == "reduced_anchor_scan":
        points = len(specification["eta_l"]) * len(specification["protocols"])
        nodes = points * (int(specification["m"]) - 1)
    elif kind == "reduced_curve_matrix":
        points = len(specification["eta_l"]) * sum(
            len(item["protocols"]) for item in specification["slices"]
        )
        nodes = points * (int(specification["m"]) - 1)
    elif kind == "reduced_recording_ablation":
        points = len(specification["eta_l"]) * len(
            specification["statistics_recordings"]
        )
        report_rows = len(specification["eta_l"]) * len(specification["report_modes"])
        nodes = points * (int(specification["m"]) - 1)
    elif kind == "mwrf":
        physical_points = len(specification.get("points", [specification]))
        points = physical_points * len(specification["models"])
        attack_records = physical_points * len(specification.get("attack_models", []))
        nodes = points * (int(specification["m"]) - 1)
    elif kind == "reduced_radau_sensitivity":
        orders = [int(value) for value in specification["m_values"]]
        points = len(orders)
        nodes = sum(order - 1 for order in orders)
    else:
        raise AssertionError(f"Unknown experiment kind: {kind}")
    if nodes < 0:
        raise AssertionError(f"Invalid quadrature order in {kind}")
    return {
        "entropy_points": points,
        "sdp_node_solves": nodes,
        "attack_records": attack_records,
        "report_rows": report_rows or points,
    }


def verify(config: dict[str, object]) -> dict[str, object]:
    if str(config["solver"].get("name", "")).lower() != "mosek":
        raise AssertionError("Fresh searches must declare MOSEK as their solver")
    default_m = int(config["solver"]["radau_nodes"])
    if default_m < 2:
        raise AssertionError("The default Radau order must be at least two")

    orders = {default_m}
    for specification in config["experiments"].values():
        if "m" in specification:
            orders.add(int(specification["m"]))
        orders.update(int(value) for value in specification.get("m_values", []))
        if "optional_m" in specification:
            orders.add(int(specification["optional_m"]))
    if min(orders) < 2:
        raise AssertionError("Every declared Radau order must be at least two")
    radau = {str(order): validate_rule(right_radau_rule(order)) for order in sorted(orders)}

    lattice = validate_mwrf_lattice(mwrf_coordinate_sets())
    expected = config["scientific_gates"]["mwrf_expected_counts"]
    if lattice["counts"] != expected:
        raise AssertionError(f"MWRF count mismatch: {lattice['counts']} != {expected}")

    hybrid = protocol_coordinates("rHybrid", recording="ternary", include_near=True)
    nesting = {}
    for protocol in ("rPCHSH", "rPhase"):
        child = protocol_coordinates(protocol, recording="ternary", include_near=True)
        nesting[protocol] = child < hybrid
        if not nesting[protocol]:
            raise AssertionError(f"{protocol} is not a strict subprotocol of rHybrid")

    separation = config["reference_points"]["S1"]
    point = TargetPoint(
        nu=float(separation["nu"]),
        eta_s=float(separation["eta_s"]),
        eta_b=float(separation["eta_l"]),
        eta_c=float(separation["eta_l"]),
        theta=float(separation["theta"]),
    )
    target_validation = build_target(point).validate()
    _assert_alpha_squared_localizer(ROOT / "src" / "routed_dicka" / "sdp.py")

    common = config["common_target"]
    if common.get("far_no_click") != "retain_perp":
        raise AssertionError("The unified paper model must retain remote no-clicks")
    if not math.isclose(
        float(separation["theta"]),
        math.atan(float(separation["eta_l"])),
        rel_tol=0.0,
        abs_tol=2e-15,
    ):
        raise AssertionError("The same-point comparison must use theta=atan(eta_L)")

    curve_matrix = config["experiments"]["protocol_curve_anchors"]
    if int(curve_matrix["m"]) != 4 or curve_matrix.get("allow_m_override") is not False:
        raise AssertionError("The manuscript curve matrix must be locked to m=4")
    eta_values = [float(value) for value in curve_matrix["eta_l"]]
    if eta_values[0] != 0.6 or eta_values[-1] != 1.0:
        raise AssertionError("The manuscript anchor range must cover eta_L in [0.6,1]")
    slices = {str(item["slice_id"]): item for item in curve_matrix["slices"]}
    ideal_slice = slices.get("routed_nu1_s1")
    if ideal_slice is None or set(ideal_slice["protocols"]) != {"rPCHSH"}:
        raise AssertionError(
            "Only rPCHSH should be numerically searched on the exact ideal slice"
        )
    plot_contract = config["curve_plots"]
    routed_nu1_eta_s = {
        float(item["eta_s"])
        for item in slices.values()
        if bool(item["include_near"]) and float(item["nu"]) == 1.0
    }
    if set(map(float, plot_contract["per_protocol_eta_s"])) != routed_nu1_eta_s:
        raise AssertionError("Displayed near-efficiency curves and searched slices disagree")
    long_only = [item for item in slices.values() if not bool(item["include_near"])]
    if len(long_only) != 1 or set(long_only[0]["protocols"]) != set(PROTOCOLS):
        raise AssertionError("Exactly one common long-only slice is required")

    profiles = {}
    experiments = config["experiments"]
    for profile, experiment_ids in config["profiles"].items():
        totals = {
            "entropy_points": 0,
            "sdp_node_solves": 0,
            "attack_records": 0,
            "report_rows": 0,
        }
        for experiment_id in experiment_ids:
            if experiment_id not in experiments:
                raise AssertionError(f"Unknown experiment {experiment_id!r} in {profile!r}")
            cost = _experiment_cost(experiments[experiment_id])
            for key in totals:
                totals[key] += cost[key]
        profiles[profile] = totals

    return {
        "schema": "routed-dicka-model-contract-audit-v2",
        "passed": True,
        "default_m": default_m,
        "radau_orders": radau,
        "protocols": {
            identifier: list(protocol.bob_far_settings)
            for identifier, protocol in PROTOCOLS.items()
        },
        "protocol_nesting": nesting,
        "mwrf": lattice,
        "target_validation": target_validation,
        "bff_localizer_uses_alpha_squared": True,
        "bff_norm_constraints_are_localizing_matrices": True,
        "remote_no_clicks_retained": True,
        "same_point_angle_matches_attack": True,
        "curve_matrix": {
            "m": int(curve_matrix["m"]),
            "anchor_efficiencies": eta_values,
            "numerical_families": sum(
                len(item["protocols"]) for item in curve_matrix["slices"]
            ),
            "long_only_slices": len(long_only),
        },
        "profiles": profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "paper_suite.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "audits" / "model_contract.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = verify(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
