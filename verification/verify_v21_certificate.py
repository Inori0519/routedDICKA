"""Independently verify and evaluate the archived V2.1 routed certificate."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class RewriteOracle:
    """Independent adjacent-rewrite normalizer for the certificate algebra."""

    def __init__(self, symbols, commuting_parties):
        self.symbols = symbols
        self.commuting = {frozenset(pair) for pair in commuting_parties}
        self.star = [index if symbol["adj"] < 0 else symbol["adj"] for index, symbol in enumerate(symbols)]

    def commute(self, first: int, second: int) -> bool:
        parties = frozenset((self.symbols[first]["party"], self.symbols[second]["party"]))
        return first == second or parties in self.commuting

    @lru_cache(maxsize=300_000)
    def reduce(self, word: tuple[int, ...]):
        seen = {word}
        pending = deque([word])
        while pending:
            current = pending.pop()
            for index in range(len(current) - 1):
                left = self.symbols[current[index]]
                right = self.symbols[current[index + 1]]
                rewritten = None
                if (
                    left["kind"] == right["kind"] == "projector"
                    and (left["party"], left["setting"])
                    == (right["party"], right["setting"])
                ):
                    if left["outcome"] != right["outcome"]:
                        return None
                    rewritten = current[:index] + current[index + 1 :]
                elif current[index] != current[index + 1] and self.commute(
                    current[index], current[index + 1]
                ):
                    rewritten = (
                        current[:index]
                        + (current[index + 1], current[index])
                        + current[index + 2 :]
                    )
                if rewritten is not None and rewritten not in seen:
                    seen.add(rewritten)
                    pending.append(rewritten)
        return min(seen, key=lambda item: (len(item), item))

    def adjoint(self, word: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(self.star[index] for index in reversed(word))

    def moment(self, word: tuple[int, ...]):
        reduced = self.reduce(word)
        if reduced is None:
            return None
        return min(reduced, self.reduce(self.adjoint(reduced)))


def _accumulate(polynomial: dict[tuple[int, ...], F], oracle, word, value) -> None:
    reduced = oracle.moment(tuple(word))
    if reduced is not None:
        polynomial[reduced] = polynomial.get(reduced, F(0)) + F(value)


def _certificate_affine(payload: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
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
        if factor.shape[0] != len(block_basis):
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
                    _accumulate(p0, oracle, oracle.adjoint(left) + right, -coefficient)
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
    if t_lower != t_upper:
        raise AssertionError("Affine export requires an exact Radau node")
    coefficients = {
        word: p0.get(word, F(0)) + t_lower * p1.get(word, F(0)) for word in fixed
    }
    affine = {
        "node": str(t_lower),
        "alpha": alpha,
        "coefficients": [
            {"word": list(word), "value": str(value)}
            for word, value in coefficients.items()
        ],
        "residual_penalty": str(penalty),
    }
    result = {
        "verified": True,
        "quadrature_order": 2,
        "word_count": len(basis),
        "fixed_moments": len(fixed),
        "node_lower": float(certified_lower),
        "exact_node_lower": str(certified_lower),
        "residual_penalty": float(penalty),
        "gram_blocks": gram_summaries,
    }
    return affine, result


def verify_certificate_bundle(certificate_path: Path, affine_path: Path) -> dict[str, object]:
    certificate_path = Path(certificate_path)
    affine_path = Path(affine_path)
    generated, result = _certificate_affine(
        json.loads(certificate_path.read_text(encoding="utf-8"))
    )
    archived = json.loads(affine_path.read_text(encoding="utf-8"))
    comparable = {
        key: archived[key]
        for key in ("node", "alpha", "coefficients", "residual_penalty")
    }
    result.update(
        {
            "affine_matches_certificate": generated == comparable,
            "certificate_file": str(certificate_path),
            "certificate_sha256": sha256(certificate_path.read_bytes()).hexdigest(),
            "affine_file": str(affine_path),
            "affine_sha256": sha256(affine_path.read_bytes()).hexdigest(),
        }
    )
    if not result["affine_matches_certificate"]:
        raise AssertionError("Archived affine witness does not match the certificate")
    return result


@dataclass(frozen=True)
class Interval:
    lower: F
    upper: F

    def __post_init__(self):
        if self.lower > self.upper:
            raise ValueError("Empty interval")

    def __add__(self, other):
        other = _interval(other)
        return Interval(self.lower + other.lower, self.upper + other.upper)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.upper, -self.lower)

    def __sub__(self, other):
        return self + (-_interval(other))

    def __rsub__(self, other):
        return _interval(other) + (-self)

    def __mul__(self, other):
        other = _interval(other)
        products = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return Interval(min(products), max(products))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _interval(other)
        if other.lower <= 0 <= other.upper:
            raise ZeroDivisionError("Interval division across zero")
        return self * Interval(1 / other.upper, 1 / other.lower)

    def __pow__(self, exponent: int):
        if exponent < 0:
            raise ValueError("Negative interval powers are unsupported")
        result = _interval(1)
        for _ in range(exponent):
            result *= self
        return result


def _interval(value) -> Interval:
    return value if isinstance(value, Interval) else Interval(F(value), F(value))


def _sqrt_bounds(value: F, bits: int = 150) -> Interval:
    value = F(value)
    denominator = 1 << bits
    root_floor = math.isqrt(value.numerator * denominator * denominator // value.denominator)
    return Interval(F(root_floor, denominator), F(root_floor + 1, denominator))


def _inverse_sqrt_bounds(value) -> Interval:
    value = _interval(value)
    return Interval(
        1 / _sqrt_bounds(value.upper).upper,
        1 / _sqrt_bounds(value.lower).lower,
    )


def _log_bounds(value: F, terms: int = 75) -> Interval:
    value = F(value)
    exponent = 0
    while value < 1:
        value *= 2
        exponent -= 1
    while value > 2:
        value /= 2
        exponent += 1

    def atanh_bound(argument: F) -> Interval:
        lower = 2 * sum(
            argument ** (2 * index + 1) / F(2 * index + 1)
            for index in range(terms)
        )
        tail = (
            2
            * argument ** (2 * terms + 1)
            / F(2 * terms + 1)
            / (1 - argument * argument)
        )
        return Interval(lower, lower + tail)

    return atanh_bound((value - 1) / (value + 1)) + exponent * atanh_bound(F(1, 3))


def _binary_entropy_bounds(probability: F) -> Interval:
    probability = F(probability)
    if probability in (0, 1):
        return _interval(0)
    return -(
        probability * _log_bounds(probability)
        + (1 - probability) * _log_bounds(1 - probability)
    ) / _log_bounds(2)


def _target_moments(nu: F, eta_s: F, eta_b: F, eta_c: F):
    nu, eta_s, eta_b, eta_c = map(_interval, (nu, eta_s, eta_b, eta_c))
    inverse_norm = _inverse_sqrt_bounds(1 + eta_s**2)
    moments = {(): _interval(1), (0,): _interval(F(1, 2)), (1,): _interval(F(1, 2))}
    near_marginal = 1 - eta_s / 2
    for index in (2, 3):
        sign = 1 if index == 2 else -1
        moments[(index,)] = near_marginal
        moments[(0, index)] = near_marginal / 2 + nu * eta_s * inverse_norm / 4
        moments[(1, index)] = (
            near_marginal / 2 + sign * nu * eta_s**3 * inverse_norm / 4
        )
    joint_detection = eta_b * eta_c
    for index in (4, 5):
        sign = 1 if index == 4 else -1
        moments[(index,)] = joint_detection / 2
        moments[(0, index)] = joint_detection / 4
        moments[(1, index)] = (1 + sign * nu) * joint_detection / 4
    return moments


def evaluate_reference_point(
    affine_path: Path,
    *,
    nu: F,
    eta_s: F,
    eta_b: F,
    eta_c: F,
) -> dict[str, object]:
    affine = json.loads(Path(affine_path).read_text(encoding="utf-8"))
    if F(affine["node"]) != F(1, 3):
        raise AssertionError("The reference evaluator expects the m=2 Radau node")
    moments = _target_moments(nu, eta_s, eta_b, eta_c)
    node_value = _interval(-F(affine["residual_penalty"]))
    for row in affine["coefficients"]:
        node_value += F(row["value"]) * moments[tuple(row["word"])]
    entropy = F(9, 4) * (1 + node_value) / _log_bounds(2)
    error_probability = (1 - F(nu)) / 2
    error_correction = 1 - min(F(eta_b), F(eta_c)) * (
        1 - _binary_entropy_bounds(error_probability)
    )
    key_rate = entropy - error_correction
    return {
        "nu": float(nu),
        "eta_s": float(eta_s),
        "eta_b": float(eta_b),
        "eta_c": float(eta_c),
        "H_lower": float(entropy.lower),
        "H_upper": float(entropy.upper),
        "EC_upper": float(error_correction.upper),
        "K_lower": float(key_rate.lower),
        "positive": key_rate.lower > 0,
        "H_lower_exact": str(entropy.lower),
        "EC_upper_exact": str(error_correction.upper),
        "K_lower_exact": str(key_rate.lower),
        "evidence_kind": "verified_affine_certificate",
        "result_scope": "fixed_raw_data_alice_forward",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "paper_suite.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "audits" / "v21_routed_certificate.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    archives = config["external_archives"]
    certificate = (ROOT / archives["v21_routed_certificate"]).resolve()
    affine = (ROOT / archives["v21_routed_affine"]).resolve()
    point = config["reference_points"]["S1"]
    verification = verify_certificate_bundle(certificate, affine)
    evaluation = evaluate_reference_point(
        affine,
        nu=F(str(point["nu"])),
        eta_s=F(str(point["eta_s"])),
        eta_b=F(str(point["eta_l"])),
        eta_c=F(str(point["eta_l"])),
    )
    minimum = F(str(config["scientific_gates"]["minimum_positive_rate"]))
    if F(evaluation["K_lower_exact"]) <= minimum:
        raise AssertionError("Archived routed certificate does not pass the positive-rate gate")
    result = {
        "schema": "routed-dicka-archived-certificate-audit-v1",
        "passed": True,
        "certificate": verification,
        "reference_point": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": True,
                "word_count": verification["word_count"],
                "H_lower": evaluation["H_lower"],
                "EC_upper": evaluation["EC_upper"],
                "K_lower": evaluation["K_lower"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
