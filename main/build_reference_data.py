"""Build analytic and proof-backed manuscript tables without solving an SDP."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from math import isclose
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routed_dicka.analytic import ideal_symmetric_threshold, symmetric_ideal_rows  # noqa: E402
from routed_dicka.io import file_sha256, utc_now, write_csv, write_json  # noqa: E402


def _same(left: object, right: object, tolerance: float = 2e-12) -> bool:
    return isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def portable_input_key(path: Path, output: Path) -> str:
    return Path(os.path.relpath(path.resolve(), output.resolve())).as_posix()


def build_ideal_rows(config: dict[str, object]) -> list[dict[str, object]]:
    specification = config["analytic_curves"]["ideal"]
    rows = symmetric_ideal_rows(
        points=int(specification["plot_points"]),
        eta_min=float(specification["eta_min"]),
        eta_max=float(specification["eta_max"]),
    )
    allowed = set(specification["protocols"])
    rows = [row for row in rows if row["protocol"] in allowed]
    for row in rows:
        row.update(
            {
                "experiment_kind": "ideal_loss_analytic",
                "nu": 1.0,
                "eta_s": 1.0,
                "eta_l": row["eta"],
                "sdp_optimized": False,
                "certificate_id": "exact_ideal_loss_formula",
            }
        )
    return rows


def build_separation_rows(
    config: dict[str, object],
    certificate_audit: dict[str, object],
    attack_audit: dict[str, object],
) -> list[dict[str, object]]:
    point = config["reference_points"]["S1"]
    nu = float(point["nu"])
    eta_s = float(point["eta_s"])
    eta_l = float(point["eta_l"])
    theta = float(point["theta"])

    if certificate_audit.get("passed") is not True:
        raise AssertionError("The routed certificate audit did not pass")
    evaluated = certificate_audit["reference_point"]
    for key, expected in (
        ("nu", nu),
        ("eta_s", eta_s),
        ("eta_b", eta_l),
        ("eta_c", eta_l),
    ):
        if not _same(evaluated[key], expected):
            raise AssertionError(f"Certificate point mismatch for {key}")
    if evaluated.get("positive") is not True or float(evaluated["K_lower"]) <= 0.0:
        raise AssertionError("The routed certificate does not establish a positive rate")

    if attack_audit.get("exactly_verified") is not True:
        raise AssertionError("The complete remote probability attack was not verified")
    if attack_audit.get("exact_degrading_channel_verified") is not True:
        raise AssertionError("The receiver-simulation channel was not verified")
    if int(attack_audit.get("probabilities", 0)) != 288:
        raise AssertionError("The remote attack does not cover all 288 probabilities")
    attack_nu = float(Fraction(str(attack_audit["configured_nu"])))
    attack_eta = float(Fraction(str(attack_audit["configured_eta"])))
    if not _same(attack_nu, nu) or not _same(attack_eta, eta_l):
        raise AssertionError("Attack and routed certificate are not evaluated at the same point")

    common = {
        "experiment_kind": "strict_separation",
        "point_id": "S1",
        "protocol": str(point["protocol"]),
        "nu": nu,
        "eta_s": eta_s,
        "eta_b": eta_l,
        "eta_c": eta_l,
        "theta": theta,
        "communication_class": "fixed_ZZZ_raw_data_Alice_forward",
        "sdp_optimized": False,
    }
    rows = [
        {
            **common,
            "data_scope": "far_only",
            "evidence_kind": "exact_receiver_simulation_attack",
            "bound_relation": "equals",
            "H_A_given_E_lower": "",
            "EC_cost_upper": "",
            "guaranteed_key_rate": 0.0,
            "probabilities_verified": 288,
            "m": "",
            "certificate_source_file": "",
        },
        {
            **common,
            "data_scope": "routed",
            "evidence_kind": "verified_affine_certificate",
            "bound_relation": "lower_bound",
            "H_A_given_E_lower": float(evaluated["H_lower"]),
            "EC_cost_upper": float(evaluated["EC_upper"]),
            "guaranteed_key_rate": float(evaluated["K_lower"]),
            "probabilities_verified": "",
            "m": evaluated.get("m", ""),
            "certificate_source_file": (
                "source_data/certificates/m4"
                if int(evaluated.get("m", 0) or 0) == 4
                else "archived_v21_near_m2_certificate"
            ),
        },
    ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "paper_suite.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--certificate-audit", type=Path)
    parser.add_argument("--attack-audit", type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    config_path = args.config.resolve()
    certificate_path = (
        args.certificate_audit.resolve()
        if args.certificate_audit
        else output / "audits" / "m4_same_point_certificate.json"
    )
    attack_path = (
        args.attack_audit.resolve()
        if args.attack_audit
        else output / "audits" / "v21_attack.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    attack = json.loads(attack_path.read_text(encoding="utf-8"))

    ideal_rows = build_ideal_rows(config)
    separation_rows = build_separation_rows(
        config,
        certificate,
        attack,
    )
    ideal_path = output / "tables" / "ideal_rates.csv"
    separation_path = output / "tables" / "strict_separation.csv"
    write_csv(ideal_path, ideal_rows)
    write_csv(separation_path, separation_rows)

    manifest = {
        "schema": "routed-dicka-reference-data-v1",
        "created_utc": utc_now(),
        "passed": True,
        "ideal_symmetric_threshold": ideal_symmetric_threshold(),
        "strict_separation_ready": (
            separation_rows[0]["guaranteed_key_rate"] == 0.0
            and float(separation_rows[1]["guaranteed_key_rate"]) > 0.0
        ),
        "inputs": {
            portable_input_key(config_path, output): file_sha256(config_path),
            portable_input_key(certificate_path, output): file_sha256(certificate_path),
            portable_input_key(attack_path, output): file_sha256(attack_path),
        },
        "files": {
            ideal_path.relative_to(output).as_posix(): file_sha256(ideal_path),
            separation_path.relative_to(output).as_posix(): file_sha256(separation_path),
        },
        "rows": {
            "ideal_rates": len(ideal_rows),
            "strict_separation": len(separation_rows),
        },
    }
    manifest_path = output / "audits" / "reference_data.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "passed": True,
                "ideal_rows": len(ideal_rows),
                "strict_separation": separation_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
