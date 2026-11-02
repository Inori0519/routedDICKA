"""Exact dyadic certificates for reduced BFF search iterates."""

from __future__ import annotations

from fractions import Fraction
from math import isqrt
import json
from pathlib import Path
import time

import numpy as np

from .reduced_solver import ReducedNPA


F = Fraction


def radau_interval_rule(
    m: int,
    *,
    bits: int = 100,
) -> list[tuple[tuple[F, F], tuple[F, F]]]:
    """Exact rational enclosures for the m=2 and m=3 internal nodes."""

    if m == 2:
        return [((F(1, 3), F(1, 3)), (F(3, 4), F(3, 4)))]
    if m == 3:
        denominator = 1 << bits
        root_floor = isqrt(6 * denominator * denominator)
        root_lower = F(root_floor, denominator)
        root_upper = F(root_floor + 1, denominator)
        return [
            (
                ((4 - root_upper) / 10, (4 - root_lower) / 10),
                ((16 - root_upper) / 36, (16 - root_lower) / 36),
            ),
            (
                ((4 + root_lower) / 10, (4 + root_upper) / 10),
                ((16 + root_lower) / 36, (16 + root_upper) / 36),
            ),
        ]
    if m != 4:
        raise ValueError("Exact reduced certificates currently support m=2, m=3, or m=4")

    # The internal m=4 nodes are the three roots of
    # 35 t^3 - 45 t^2 + 15 t - 1.  The decimal centers below merely locate
    # dyadic brackets; exact sign and derivative checks certify each bracket.
    centers = (
        F("0.08858795951270394739554614376945619688573"),
        F("0.4094668644407347108649262520688298940519"),
        F("0.7876594617608470560252418898759996233481"),
    )
    denominator = 1 << bits

    def bracket(center: F) -> tuple[F, F]:
        scaled = center * denominator
        lower = F(scaled.numerator // scaled.denominator, denominator)
        return lower, lower + F(1, denominator)

    def node_polynomial(value: F) -> F:
        return 35 * value**3 - 45 * value**2 + 15 * value - 1

    def node_derivative(value: F) -> F:
        return 105 * value**2 - 90 * value + 15

    def multiply(
        left: tuple[F, F], right: tuple[F, F]
    ) -> tuple[F, F]:
        products = (
            left[0] * right[0],
            left[0] * right[1],
            left[1] * right[0],
            left[1] * right[1],
        )
        return min(products), max(products)

    def add(left: tuple[F, F], right: tuple[F, F]) -> tuple[F, F]:
        return left[0] + right[0], left[1] + right[1]

    def scale(value: int, interval: tuple[F, F]) -> tuple[F, F]:
        endpoints = (value * interval[0], value * interval[1])
        return min(endpoints), max(endpoints)

    result = []
    for center in centers:
        t_interval = bracket(center)
        lower, upper = t_interval
        if node_polynomial(lower) * node_polynomial(upper) > 0:
            raise AssertionError("m=4 Radau root bracket does not change sign")
        derivative_endpoints = (node_derivative(lower), node_derivative(upper))
        if min(derivative_endpoints) <= 0 <= max(derivative_endpoints):
            raise AssertionError("m=4 Radau root bracket is not monotone")

        t2 = multiply(t_interval, t_interval)
        t3 = multiply(t2, t_interval)
        legendre = add(
            add(scale(20, t3), scale(-30, t2)),
            add(scale(12, t_interval), (F(-1), F(-1))),
        )
        if legendre[0] <= 0 <= legendre[1]:
            raise AssertionError("Legendre denominator crosses zero")
        squares = (legendre[0] ** 2, legendre[1] ** 2)
        square_interval = min(squares), max(squares)
        weight_interval = (
            lower / (16 * square_interval[1]),
            upper / (16 * square_interval[0]),
        )
        result.append((t_interval, weight_interval))
    return result


def certify_reduced_node(
    problem: ReducedNPA,
    *,
    m: int,
    node_index: int,
    data_radius: float = 1e-12,
    dyadic_bits: int = 28,
    export: Path | None = None,
) -> dict[str, object]:
    """Turn numerical dual Gram matrices into an exact PSD certificate."""

    if problem.solution is None:
        raise RuntimeError("The reduced SDP must be solved before certification")
    started = time.perf_counter()
    rule = radau_interval_rule(m)
    if not 0 <= node_index < len(rule):
        raise IndexError("Radau node index is outside the certified rule")
    (rule_lower, rule_upper), _ = rule[node_index]
    numeric_t = F.from_float(problem.t)
    rounding_pad = F(1, 1 << 48)
    t_lower = min(rule_lower, numeric_t - rounding_pad)
    t_upper = max(rule_upper, numeric_t + rounding_pad)
    if F(problem.alpha) < F(3, 2) * max(
        1 / t_lower,
        1 / (1 - t_upper),
    ):
        raise AssertionError("The BFF norm bound is insufficient on this node interval")

    denominator = 1 << dyadic_bits
    residual = {word: F(coefficient) for word, coefficient in problem.p0.items()}
    gram_summaries = []
    factors = []
    for block, numerical_gram in zip(problem.blocks, problem.dual_grams()):
        eigenvalues, eigenvectors = np.linalg.eigh(
            (numerical_gram + numerical_gram.T) / 2.0
        )
        positive = eigenvalues > 1e-13
        roots = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
        integer_factor = np.rint(roots * denominator).astype(np.int64)
        exact_factor = integer_factor.astype(object)
        exact_gram = exact_factor @ exact_factor.T
        packed_index = 0
        for column in range(block.size):
            for row in range(column, block.size):
                coefficient = F(
                    int(exact_gram[row, column]) * (1 if row == column else 2),
                    denominator * denominator * block.scale,
                )
                for word, value in block.entries[packed_index].items():
                    residual[word] = residual.get(word, F(0)) - coefficient * value
                packed_index += 1
        gram_summaries.append(
            {
                "size": block.size,
                "rank": int(integer_factor.shape[1]),
                "raw_min_eigenvalue": float(eigenvalues.min()),
                "max_factor": float(np.max(np.abs(roots), initial=0.0)),
            }
        )
        factors.append(integer_factor.tolist())

    for word in problem.p1:
        residual.setdefault(word, F(0))
    radius = F.from_float(float(data_radius))
    fixed_part = F(0)
    penalty = F(0)
    free_residual_l1 = F(0)
    for word, constant in residual.items():
        slope = F(problem.p1.get(word, 0))
        lower = constant + slope * (t_lower if slope >= 0 else t_upper)
        upper = constant + slope * (t_upper if slope >= 0 else t_lower)
        if word in problem.fixed:
            value = F.from_float(float(problem.fixed[word]))
            value_lower = value if word == () else value - radius
            value_upper = value if word == () else value + radius
            fixed_part += min(
                lower * value_lower,
                lower * value_upper,
                upper * value_lower,
                upper * value_upper,
            )
        else:
            magnitude = max(abs(lower), abs(upper))
            free_residual_l1 += magnitude
            penalty += magnitude * problem.alpha ** problem.algebra.z_degree(word)
    certified_lower = fixed_part - penalty
    displayed_lower = float(np.nextafter(float(certified_lower), -np.inf))
    summary = {
        "certified_node_lower": displayed_lower,
        "exact_numerator": str(certified_lower.numerator),
        "exact_denominator": str(certified_lower.denominator),
        "fixed_part": float(fixed_part),
        "residual_penalty": float(penalty),
        "free_residual_l1": float(free_residual_l1),
        "data_radius": data_radius,
        "dyadic_bits": dyadic_bits,
        "alpha": problem.alpha,
        "node_interval": [str(t_lower), str(t_upper)],
        "grams": gram_summaries,
        "seconds": time.perf_counter() - started,
    }
    if export is not None:
        payload = {
            "schema": "routed-dicka-reduced-node-certificate-v1",
            "metadata": {
                "protocol": problem.model.protocol,
                "statistics_recording": problem.model.statistics_recording,
                "required_features": sorted(problem.features),
                "m": m,
                "node_index": node_index,
                "solver_backend": problem.solution.get("backend"),
                "solver_status": problem.solution.get("status"),
            },
            "summary": summary,
            "word_basis": [list(word) for word in problem.word_basis],
            "symbols": [symbol.__dict__ for symbol in problem.algebra.symbols],
            "commuting_parties": [sorted(pair) for pair in problem.algebra.cp],
            "fixed": [
                {"word": list(word), "value_hex": float(value).hex()}
                for word, value in problem.fixed.items()
            ],
            "p0": [(list(word), coefficient) for word, coefficient in problem.p0.items()],
            "p1": [(list(word), coefficient) for word, coefficient in problem.p1.items()],
            "gram_factors": factors,
            "block_scales": [block.scale for block in problem.blocks],
            "zs": list(problem.eve_operators),
            "akey": problem.alice_key_projector,
        }
        export = Path(export)
        export.parent.mkdir(parents=True, exist_ok=True)
        export.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return summary
