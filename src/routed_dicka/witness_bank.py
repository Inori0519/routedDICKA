"""Exact evaluation of reusable reduced BFF affine certificates."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

from .reduced_certificate import radau_interval_rule
from .reduced_models import ReducedModelData


def _log_two_bounds(terms: int = 75) -> tuple[F, F]:
    argument = F(1, 3)
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
    return lower, lower + tail


def evaluate_affine_witness(
    witness: dict[str, object],
    model: ReducedModelData,
) -> F:
    """Evaluate one globally feasible node witness on a target behavior."""

    fixed = model.fixed_moments()
    radius = F(str(witness.get("data_radius", "1/1000000000000")))
    total = F(0)
    fixed_uncertainty = F(0)
    for row in witness["coefficients"]:
        word = tuple(int(value) for value in row["word"])
        if word not in fixed:
            raise ValueError(f"Witness requests unavailable fixed moment {word}")
        coefficient = F(str(row["value"]))
        total += coefficient * F.from_float(float(fixed[word]))
        if word:
            fixed_uncertainty += abs(coefficient) * radius
    return total - F(str(witness["residual_penalty"])) - fixed_uncertainty


def combine_radau_entropy(m: int, node_values: Iterable[F]) -> F:
    """Combine certified node values with outward rational quadrature bounds."""

    values = list(node_values)
    rule = radau_interval_rule(int(m))
    if len(values) != len(rule):
        raise ValueError("One certified value is required for every internal node")
    log_lower, log_upper = _log_two_bounds()
    total = F(0)
    for value, ((t_lower, t_upper), (weight_lower, weight_upper)) in zip(
        values, rule
    ):
        factor_lower = weight_lower / t_upper / log_upper
        factor_upper = weight_upper / t_lower / log_lower
        total += (1 + value) * (
            factor_lower if value >= -1 else factor_upper
        )
    return total


def load_verified_witness(
    certificate_path: Path,
    affine_path: Path | None = None,
) -> dict[str, object]:
    """Load the exact affine part of a certificate after external verification."""

    certificate_path = Path(certificate_path)
    affine_path = affine_path or certificate_path.with_name(
        certificate_path.name.replace(".certificate.json", ".affine.json")
    )
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    affine = json.loads(Path(affine_path).read_text(encoding="utf-8"))
    metadata = certificate["metadata"]
    if affine.get("metadata") != metadata:
        raise ValueError("Affine witness metadata differs from its certificate")
    affine["data_radius"] = str(certificate["summary"]["data_radius"])
    affine["symbols"] = certificate["symbols"]
    affine["protocol"] = metadata["protocol"]
    affine["statistics_recording"] = metadata["statistics_recording"]
    affine["required_features"] = metadata["required_features"]
    affine["m"] = int(metadata["m"])
    affine["node_index"] = int(metadata["node_index"])
    affine["certificate_file"] = str(certificate_path)
    affine["certificate_sha256"] = sha256(certificate_path.read_bytes()).hexdigest()
    affine["affine_file"] = str(affine_path)
    affine["affine_sha256"] = sha256(Path(affine_path).read_bytes()).hexdigest()
    return affine


def _assert_compatible(witness: dict[str, object], model: ReducedModelData) -> None:
    if witness["protocol"] != model.protocol:
        raise ValueError("Witness and target use different protocols")
    if witness["statistics_recording"] != model.statistics_recording:
        raise ValueError("Witness and target use different recording alphabets")
    expected_symbols = [symbol.__dict__ for symbol in model.algebra.symbols]
    if witness["symbols"] != expected_symbols:
        raise ValueError("Witness and target use different operator encodings")
    if set(witness["required_features"]) != set(model.required_features):
        raise ValueError("Witness and target use different information features")


def evaluate_witness_envelope(
    witnesses: Iterable[dict[str, object]],
    model: ReducedModelData,
) -> dict[str, object]:
    """Take the best valid node witness, then the best supported Radau order."""

    grouped: dict[int, dict[int, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for witness in witnesses:
        _assert_compatible(witness, model)
        grouped[int(witness["m"])][int(witness["node_index"])].append(witness)
    candidates = []
    for m, by_node in sorted(grouped.items()):
        expected_nodes = len(radau_interval_rule(m))
        if set(by_node) != set(range(expected_nodes)):
            continue
        selected = []
        node_values = []
        for node_index in range(expected_nodes):
            evaluated = [
                (evaluate_affine_witness(witness, model), witness)
                for witness in by_node[node_index]
            ]
            value, witness = max(evaluated, key=lambda pair: pair[0])
            node_values.append(value)
            selected.append(
                {
                    "node_index": node_index,
                    "node_lower": str(value),
                    "certificate_sha256": witness["certificate_sha256"],
                    "affine_sha256": witness["affine_sha256"],
                }
            )
        entropy = combine_radau_entropy(m, node_values)
        candidates.append((entropy, m, selected))
    if not candidates:
        raise ValueError("No complete Radau witness family is available")
    entropy, m, selected = max(candidates, key=lambda item: item[0])
    return {
        "entropy_lower": float(entropy),
        "entropy_lower_exact": str(entropy),
        "m": m,
        "nodes": selected,
        "candidate_orders": [item[1] for item in candidates],
    }
