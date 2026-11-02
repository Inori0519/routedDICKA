"""Independent exact check of the archived eta=0.78 long-only attack."""

from __future__ import annotations

import argparse
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Field:
    """Elements a+b*r with r^2=d, represented by exact rational pairs."""

    def __init__(self, d: F) -> None:
        self.d = F(d)

    def add(self, first, second):
        return first[0] + second[0], first[1] + second[1]

    def neg(self, value):
        return -value[0], -value[1]

    def mul(self, first, second):
        return (
            first[0] * second[0] + self.d * first[1] * second[1],
            first[0] * second[1] + first[1] * second[0],
        )

    def div(self, numerator, denominator):
        norm = denominator[0] ** 2 - self.d * denominator[1] ** 2
        return self.mul(numerator, (denominator[0] / norm, -denominator[1] / norm))

    def const(self, value):
        return F(value), F(0)

    def interval(self, value, bits: int = 180):
        denominator = 1 << bits
        root_floor = math.isqrt(
            self.d.numerator * denominator * denominator // self.d.denominator
        )
        lower = F(root_floor, denominator)
        upper = F(root_floor + 1, denominator)
        return (
            value[0] + value[1] * (lower if value[1] >= 0 else upper),
            value[0] + value[1] * (upper if value[1] >= 0 else lower),
        )

    def target(self, nu: F, eta: F, ideal: bool = False):
        zero, one = self.const(0), self.const(1)
        radical = (F(0), F(1))
        efficiency = F(1) if ideal else eta
        visibility = F(1) if ideal else nu
        observables = {
            "Z": (zero, one),
            "X": (one, zero),
            "T0": (self.mul(self.const(eta), radical), radical),
            "T1": (self.mul(self.const(-eta), radical), radical),
        }
        probabilities = []
        for x, y, z in product(("Z", "X"), ("Z", "X", "T0", "T1"), ("Z", "X")):
            for a, b, c in product(range(2), range(3), range(3)):
                effects = []
                for label, outcome, local_eta in zip(
                    (x, y, z),
                    (a, b, c),
                    (F(1), efficiency, efficiency),
                ):
                    if outcome == 2:
                        effects.append((self.const(1 - local_eta), zero, zero))
                    else:
                        xx, zz = observables[label]
                        sign = 1 if outcome == 0 else -1
                        effects.append(
                            (
                                self.const(local_eta / 2),
                                self.mul(self.const(sign * local_eta / 2), xx),
                                self.mul(self.const(sign * local_eta / 2), zz),
                            )
                        )
                (a0, ax, az), (b0, bx, bz), (c0, cx, cz) = effects
                correlations = (
                    self.mul(self.mul(az, bz), c0),
                    self.mul(self.mul(az, b0), cz),
                    self.mul(self.mul(a0, bz), cz),
                    self.mul(self.mul(ax, bx), cx),
                )
                correlation = zero
                for term in correlations:
                    correlation = self.add(correlation, term)
                probabilities.append(
                    self.add(
                        self.mul(self.mul(a0, b0), c0),
                        self.mul(self.const(visibility), correlation),
                    )
                )
        return probabilities


def _pair(value):
    return tuple(map(F, value))


def _serial(value):
    return [str(entry) for entry in value]


def verify_attack(payload: dict[str, object]) -> dict[str, object]:
    nu, eta = F(payload["nu"]), F(payload["eta"])
    field = Field(F(1, 1) / (1 + eta * eta))
    target = field.target(nu, eta)
    ideal = field.target(nu, eta, ideal=True)
    quantum_weight = _pair(payload["quantum_weight"])
    weights = [_pair(weight) for weight in payload["local_weights"]]
    strategies = payload["local_strategies"]
    if len(weights) != len(strategies):
        raise AssertionError("Attack weights and strategies have different lengths")

    total = quantum_weight
    minimum_weight = field.interval(quantum_weight)[0]
    reproduced = [field.mul(quantum_weight, probability) for probability in ideal]
    for weight, (alice, bob, charlie) in zip(weights, strategies):
        lower, _ = field.interval(weight)
        if lower < 0:
            raise AssertionError(f"Negative attack weight: {weight}")
        minimum_weight = min(minimum_weight, lower)
        total = field.add(total, weight)
        for context_index, (x, y, z) in enumerate(product(range(2), range(4), range(2))):
            row = 18 * context_index + 9 * alice[x] + 3 * bob[y] + charlie[z]
            reproduced[row] = field.add(reproduced[row], weight)

    if total != field.const(1):
        raise AssertionError(f"Attack weights do not normalize: {total}")
    if reproduced != target:
        mismatch = next(i for i, (left, right) in enumerate(zip(reproduced, target)) if left != right)
        raise AssertionError(f"Attack does not reproduce probability index {mismatch}")
    _, quantum_upper = field.interval(quantum_weight)
    return {
        "exactly_verified": True,
        "full_contexts": 16,
        "probabilities": 288,
        "local_atoms": len(weights),
        "minimum_weight_lower": float(minimum_weight),
        "quantum_weight_upper": float(quantum_upper),
        "quantum_weight_upper_rational": str(quantum_upper),
    }


def verify_converse(payload: dict[str, object]) -> dict[str, object]:
    attack = verify_attack(payload)
    nu, eta = F(payload["nu"]), F(payload["eta"])
    field = Field(F(1, 1) / (1 + eta * eta))
    quantum_weight = _pair(payload["quantum_weight"])
    one, zero = field.const(1), field.const(0)
    known_probability = field.add(one, field.neg(quantum_weight))
    known_lower, _ = field.interval(known_probability)
    if known_lower < eta:
        raise AssertionError("The receiver-simulation converse does not apply")

    local_mass = [zero, zero]
    for weight, (alice, _, _) in zip(payload["local_weights"], payload["local_strategies"]):
        local_mass[alice[0]] = field.add(local_mass[alice[0]], _pair(weight))
    expected_mass = field.mul(known_probability, field.const(F(1, 2)))
    if local_mass != [expected_mass, expected_mass]:
        raise AssertionError("Local attack mass is not balanced over Alice's key bit")

    keep = field.div(field.const(eta), known_probability)
    bit_error = (1 - nu) / 2
    degrading = []
    for eve in range(3):
        if eve == 2:
            degrading.append([zero, zero, one])
        else:
            degrading.append(
                [
                    field.mul(
                        keep,
                        field.const(1 - bit_error if receiver == eve else bit_error),
                    )
                    for receiver in range(2)
                ]
                + [field.add(one, field.neg(keep))]
            )

    for row in degrading:
        total = zero
        for entry in row:
            if field.interval(entry)[0] < 0:
                raise AssertionError("Negative degrading-channel entry")
            total = field.add(total, entry)
        if total != one:
            raise AssertionError("Degrading-channel row does not normalize")

    for alice in range(2):
        eve_distribution = [
            known_probability if eve == alice else (quantum_weight if eve == 2 else zero)
            for eve in range(3)
        ]
        for receiver in range(3):
            value = zero
            for eve in range(3):
                value = field.add(
                    value,
                    field.mul(eve_distribution[eve], degrading[eve][receiver]),
                )
            target = field.const(
                1 - eta
                if receiver == 2
                else eta * (1 - bit_error if receiver == alice else bit_error)
            )
            if value != target:
                raise AssertionError("Receiver degrading-channel equality failed")

    return {
        **attack,
        "exact_degrading_channel_verified": True,
        "nu": str(nu),
        "eta": str(eta),
        "eve_known_probability_lower": float(known_lower),
        "receiver_efficiency": float(eta),
        "degrading_keep_probability_upper": float(field.interval(keep)[1]),
        "bit_flip_probability": str(bit_error),
        "no_key_class": (
            "Fixed ZZZ raw data with Alice-to-receivers public communication; "
            "Alice stochastic or block preprocessing is allowed, receiver feedback and "
            "key-round selection are excluded."
        ),
        "degrading_channel": [[_serial(entry) for entry in row] for row in degrading],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "paper_suite.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "audits" / "v21_attack.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    separation = config["reference_points"]["S1"]
    attack_path = (ROOT / config["external_archives"]["v21_attack"]).resolve()
    payload = json.loads(attack_path.read_text(encoding="utf-8"))
    expected_nu = F(str(separation["nu"]))
    expected_eta = F(str(separation["eta_l"]))
    if F(payload["nu"]) != expected_nu or F(payload["eta"]) != expected_eta:
        raise AssertionError(
            "Archived attack parameters do not match the configured separation point"
        )
    configured_theta = float(separation["theta"])
    if not math.isclose(
        configured_theta,
        math.atan(float(expected_eta)),
        rel_tol=0.0,
        abs_tol=2e-15,
    ):
        raise AssertionError("Configured separation angle is not theta=atan(eta_L)")
    result = verify_converse(payload)
    result.update(
        {
            "schema": "routed-dicka-exact-attack-audit-v1",
            "source_file": str(attack_path),
            "source_sha256": sha256(attack_path.read_bytes()).hexdigest(),
            "configured_nu": str(expected_nu),
            "configured_eta": str(expected_eta),
            "configured_theta": configured_theta,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("exactly_verified", "exact_degrading_channel_verified", "probabilities", "eta")}, indent=2))


if __name__ == "__main__":
    main()
