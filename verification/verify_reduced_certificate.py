"""Independent exact checker for generated or archived reduced BFF certificates."""

from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routed_dicka.reduced_certificate import radau_interval_rule  # noqa: E402
from verify_v21_certificate import RewriteOracle, _accumulate, _log_bounds  # noqa: E402


def verify_node_certificate(
    certificate_path: Path,
    *,
    affine_export: Path | None = None,
) -> dict[str, object]:
    """Rebuild the SOS identity using only exact integers and rational arithmetic."""

    certificate_path = Path(certificate_path)
    payload = json.loads(certificate_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    oracle = RewriteOracle(payload["symbols"], payload["commuting_parties"])
    bits = int(summary["dyadic_bits"])
    denominator = 1 << bits
    alpha = int(summary["alpha"])
    t_lower, t_upper = map(F, summary["node_interval"])
    if not 0 < t_lower <= t_upper < 1:
        raise AssertionError("Invalid BFF node interval")
    if F(alpha) < F(3, 2) * max(1 / t_lower, 1 / (1 - t_upper)):
        raise AssertionError("Certificate norm bound is too small")

    basis = [tuple(word) for word in payload["word_basis"]]
    localizer_basis = [()] + [
        (index,)
        for index, symbol in enumerate(payload["symbols"])
        if symbol["kind"] == "projector"
    ]
    p0: dict[tuple[int, ...], F] = {}
    p1: dict[tuple[int, ...], F] = {}
    key = int(payload["akey"])
    zs = [int(value) for value in payload["zs"]]
    for outcome, z in enumerate(zs):
        z_dagger = oracle.star[z]
        key_terms = [((key,), 1)] if outcome == 0 else [((), 1), ((key,), -1)]
        for key_word, sign in key_terms:
            for suffix in ((z,), (z_dagger,), (z_dagger, z)):
                _accumulate(p0, oracle, key_word + suffix, sign)
            _accumulate(p1, oracle, key_word + (z_dagger, z), -sign)
        _accumulate(p1, oracle, (z, z_dagger), 1)

    patterns = [None] + [
        pattern
        for z in zs
        for pattern in ((oracle.star[z], z), (z, oracle.star[z]))
    ]
    factors = payload["gram_factors"]
    scales = payload["block_scales"]
    if len(patterns) != len(factors):
        raise AssertionError("Certificate has the wrong number of Gram blocks")
    gram_summaries = []
    for block_index, (factor_rows, scale, inner) in enumerate(
        zip(factors, scales, patterns)
    ):
        block_basis = basis if block_index == 0 else localizer_basis
        expected_scale = 1 if block_index == 0 else alpha * alpha
        if int(scale) != expected_scale:
            raise AssertionError("Certificate localizer scale mismatch")
        factor = np.asarray(factor_rows, dtype=object)
        if factor.ndim != 2 or factor.shape[0] != len(block_basis):
            raise AssertionError("Gram factor and basis have different dimensions")
        if not all(isinstance(value, int) for value in factor.flat):
            raise AssertionError("Gram factor is not integral")
        gram = factor @ factor.T
        for column, right in enumerate(block_basis):
            for row in range(column, len(block_basis)):
                left = block_basis[row]
                coefficient = F(
                    int(gram[row, column]) * (1 if row == column else 2),
                    denominator * denominator * expected_scale,
                )
                if inner is None:
                    _accumulate(
                        p0,
                        oracle,
                        oracle.adjoint(left) + right,
                        -coefficient,
                    )
                else:
                    _accumulate(
                        p0,
                        oracle,
                        oracle.adjoint(left) + right,
                        -coefficient * alpha * alpha,
                    )
                    _accumulate(
                        p0,
                        oracle,
                        oracle.adjoint(left) + inner + right,
                        coefficient,
                    )
        gram_summaries.append(
            {"size": int(factor.shape[0]), "rank_bound": int(factor.shape[1])}
        )

    fixed = {
        tuple(row["word"]): F.from_float(float.fromhex(row["value_hex"]))
        for row in payload["fixed"]
    }
    if fixed.get(()) != 1:
        raise AssertionError("Certificate normalization is missing")
    radius = F.from_float(float(summary["data_radius"]))
    fixed_part = F(0)
    penalty = F(0)
    for word in set(p0) | set(p1):
        constant = p0.get(word, F(0))
        slope = p1.get(word, F(0))
        lower = constant + slope * (t_lower if slope >= 0 else t_upper)
        upper = constant + slope * (t_upper if slope >= 0 else t_lower)
        if word in fixed:
            value = fixed[word]
            value_lower = value if word == () else value - radius
            value_upper = value if word == () else value + radius
            fixed_part += min(
                lower * value_lower,
                lower * value_upper,
                upper * value_lower,
                upper * value_upper,
            )
        else:
            z_degree = sum(
                payload["symbols"][index]["kind"] != "projector" for index in word
            )
            penalty += max(abs(lower), abs(upper)) * alpha**z_degree
    certified_lower = fixed_part - penalty
    expected = F(int(summary["exact_numerator"]), int(summary["exact_denominator"]))
    if certified_lower != expected:
        raise AssertionError("Exact certificate identity failed")

    affine = None
    if affine_export is not None:
        if any(p1.get(word, 0) != 0 for word in fixed):
            raise AssertionError("Observed affine coefficients depend on the node")
        coefficients = {
            word: p0.get(word, F(0)) + t_lower * p1.get(word, F(0))
            for word in fixed
        }
        affine = {
            "schema": "routed-dicka-verified-node-witness-v1",
            "source_certificate": certificate_path.name,
            "metadata": payload.get("metadata", {}),
            "node": str(t_lower),
            "node_interval": [str(t_lower), str(t_upper)],
            "alpha": alpha,
            "coefficients": [
                {"word": list(word), "value": str(value)}
                for word, value in coefficients.items()
            ],
            "residual_penalty": str(penalty),
            "interpretation": (
                "Global affine lower bound in the encoded operator algebra; "
                "the anchor optimizer value is not used as interpolation data."
            ),
        }
        affine_export = Path(affine_export)
        affine_export.parent.mkdir(parents=True, exist_ok=True)
        affine_export.write_text(json.dumps(affine, indent=2) + "\n", encoding="utf-8")
    return {
        "verified": True,
        "certificate_file": str(certificate_path),
        "node_lower": float(certified_lower),
        "exact_node_lower": str(certified_lower),
        "residual_penalty": float(penalty),
        "node_interval": [str(t_lower), str(t_upper)],
        "word_count": len(basis),
        "gram_blocks": gram_summaries,
        "metadata": payload.get("metadata", {}),
        "affine_export": str(affine_export) if affine_export is not None else None,
    }


def certified_entropy(
    certificate_paths: list[Path],
    *,
    m: int,
) -> dict[str, object]:
    rule = radau_interval_rule(m)
    if len(certificate_paths) != len(rule):
        raise ValueError("One independently verified certificate is required per node")
    log_interval = _log_bounds(F(2))
    total = F(0)
    node_results = []
    for path, ((t_lower, t_upper), (w_lower, w_upper)) in zip(
        certificate_paths,
        rule,
    ):
        result = verify_node_certificate(path)
        certificate_lower, certificate_upper = map(F, result["node_interval"])
        if certificate_lower > t_lower or certificate_upper < t_upper:
            raise AssertionError("Certificate interval does not cover the quadrature node")
        node_value = F(result["exact_node_lower"])
        factor_lower = w_lower / t_upper / log_interval.upper
        factor_upper = w_upper / t_lower / log_interval.lower
        total += (1 + node_value) * (
            factor_lower if node_value >= -1 else factor_upper
        )
        node_results.append(result)
    return {
        "verified": True,
        "m": m,
        "entropy_lower": float(total),
        "entropy_lower_exact": str(total),
        "nodes": node_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificates", nargs="+", type=Path)
    parser.add_argument("--m", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.m is None:
        results = [verify_node_certificate(path) for path in args.certificates]
        result: object = results[0] if len(results) == 1 else results
    else:
        result = certified_entropy(args.certificates, m=args.m)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
