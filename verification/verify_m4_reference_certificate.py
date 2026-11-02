"""Verify the archived m=4 routed certificate at the strict-separation point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routed_dicka.io import file_sha256, utc_now, write_json  # noqa: E402
from routed_dicka.protocols import TargetPoint  # noqa: E402
from routed_dicka.reduced_models import build_reduced_model  # noqa: E402
from routed_dicka.target import build_target  # noqa: E402
from routed_dicka.witness_bank import (  # noqa: E402
    evaluate_witness_envelope,
    load_verified_witness,
)
from verify_reduced_certificate import (  # noqa: E402
    certified_entropy,
    verify_node_certificate,
)


def verify_affine_matches_certificate(
    certificate_path: Path,
    affine_path: Path,
) -> dict[str, object]:
    """Regenerate the affine witness and require exact JSON equality."""

    certificate_path = Path(certificate_path)
    affine_path = Path(affine_path)
    with tempfile.TemporaryDirectory() as temporary:
        regenerated = Path(temporary) / affine_path.name
        audit = verify_node_certificate(certificate_path, affine_export=regenerated)
        expected = json.loads(regenerated.read_text(encoding="utf-8"))
    archived = json.loads(affine_path.read_text(encoding="utf-8"))
    if archived != expected:
        raise AssertionError(
            f"Archived affine witness does not match {certificate_path.name}"
        )
    return audit


def verify_reference_certificate(config_path: Path) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    archives = config["external_archives"]
    root = config_path.parents[1]
    certificate_paths = [
        root / archives[f"v21_routed_m4_node{index}_certificate"]
        for index in range(3)
    ]
    affine_paths = [
        root / archives[f"v21_routed_m4_node{index}_affine"]
        for index in range(3)
    ]

    node_audits = [
        verify_affine_matches_certificate(certificate, affine)
        for certificate, affine in zip(certificate_paths, affine_paths)
    ]
    exact_entropy = certified_entropy(certificate_paths, m=4)
    witnesses = [
        load_verified_witness(certificate, affine)
        for certificate, affine in zip(certificate_paths, affine_paths)
    ]

    reference = config["reference_points"]["S1"]
    point = TargetPoint(
        nu=float(reference["nu"]),
        eta_s=float(reference["eta_s"]),
        eta_b=float(reference["eta_l"]),
        eta_c=float(reference["eta_l"]),
        theta=float(reference["theta"]),
    )
    behavior = build_target(point)
    model = build_reduced_model(
        behavior,
        "rPhase",
        statistics_recording="ternary",
        include_near=True,
    )
    transported = evaluate_witness_envelope(witnesses, model)
    if abs(float(exact_entropy["entropy_lower"]) - float(transported["entropy_lower"])) > 2e-13:
        raise AssertionError("The exact and affine evaluations of the m=4 family disagree")

    h_ab, h_ac = behavior.key_entropies(recording="ternary")
    ec_upper = min(1.0, max(h_ab, h_ac) + 1e-12)
    entropy_lower = float(transported["entropy_lower"])
    key_rate = entropy_lower - ec_upper
    certificate_records = []
    for index, (certificate, affine, audit) in enumerate(
        zip(certificate_paths, affine_paths, node_audits)
    ):
        certificate_records.append(
            {
                "node_index": index,
                "certificate_file": certificate.relative_to(root).as_posix(),
                "certificate_sha256": file_sha256(certificate),
                "affine_file": affine.relative_to(root).as_posix(),
                "affine_sha256": file_sha256(affine),
                "exact_identity_verified": bool(audit["verified"]),
                "node_lower": audit["node_lower"],
            }
        )
    return {
        "schema": "routed-dicka-m4-reference-certificate-audit-v1",
        "created_utc": utc_now(),
        "passed": key_rate > float(config["scientific_gates"]["minimum_positive_rate"]),
        "config_sha256": file_sha256(config_path),
        "reference_point": {
            "protocol": "rPhase",
            "nu": point.nu,
            "eta_s": point.eta_s,
            "eta_b": point.eta_b,
            "eta_c": point.eta_c,
            "theta": point.far_theta,
            "m": 4,
            "H_lower": entropy_lower,
            "H_lower_exact": transported["entropy_lower_exact"],
            "EC_upper": ec_upper,
            "K_lower": key_rate,
            "positive": key_rate > 0.0,
        },
        "exact_same_point_entropy": exact_entropy["entropy_lower"],
        "affine_same_point_entropy": transported["entropy_lower"],
        "certificates": certificate_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "paper_suite.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "audits" / "m4_same_point_certificate.json",
    )
    args = parser.parse_args()
    result = verify_reference_certificate(args.config)
    write_json(args.output.resolve(), result)
    print(json.dumps({"passed": result["passed"], "reference_point": result["reference_point"]}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
