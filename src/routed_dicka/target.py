"""Physical GHZ target tables with binned near and ternary far outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from math import cos, log2, sin
from typing import Iterable

import numpy as np

from .protocols import (
    ALICE_SETTINGS,
    FAR_BOB_SETTINGS,
    FAR_CHARLIE_SETTINGS,
    KEY_SETTING,
    NEAR_BOB_SETTINGS,
    NEAR_CHARLIE_SETTINGS,
    TargetPoint,
)


I2 = np.eye(2, dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _observable(label: str, theta: float) -> np.ndarray:
    if label == "Z":
        return Z
    if label == "X":
        return X
    if label == "T0":
        return cos(theta) * Z + sin(theta) * X
    if label == "T1":
        return cos(theta) * Z - sin(theta) * X
    raise ValueError(f"Unknown observable label: {label}")


def _source_state(nu: float, dephased: bool = False) -> np.ndarray:
    if dephased:
        signal = np.zeros((8, 8), dtype=complex)
        signal[0, 0] = 0.5
        signal[7, 7] = 0.5
    else:
        ghz = np.zeros(8, dtype=complex)
        ghz[0] = ghz[7] = 1.0 / np.sqrt(2.0)
        signal = np.outer(ghz, ghz.conj())
    return nu * signal + (1.0 - nu) * np.eye(8, dtype=complex) / 8.0


def _binary_povm(observable: np.ndarray, efficiency: float) -> tuple[np.ndarray, ...]:
    clicked_plus = efficiency * (I2 + observable) / 2.0
    clicked_minus = efficiency * (I2 - observable) / 2.0
    return clicked_plus + (1.0 - efficiency) * I2, clicked_minus


def _ternary_povm(observable: np.ndarray, efficiency: float) -> tuple[np.ndarray, ...]:
    return (
        efficiency * (I2 + observable) / 2.0,
        efficiency * (I2 - observable) / 2.0,
        (1.0 - efficiency) * I2,
    )


def _context_probabilities(
    rho: np.ndarray,
    alice: Iterable[np.ndarray],
    bob: Iterable[np.ndarray],
    charlie: Iterable[np.ndarray],
) -> np.ndarray:
    alice = tuple(alice)
    bob = tuple(bob)
    charlie = tuple(charlie)
    table = np.zeros((len(alice), len(bob), len(charlie)), dtype=float)
    for a, b, c in product(range(len(alice)), range(len(bob)), range(len(charlie))):
        operator = np.kron(np.kron(alice[a], bob[b]), charlie[c])
        value = float(np.trace(rho @ operator).real)
        table[a, b, c] = 0.0 if abs(value) < 1e-15 else value
    return table


@dataclass(frozen=True)
class TargetBehavior:
    point: TargetPoint
    near: dict[tuple[str, str, str], np.ndarray]
    far: dict[tuple[str, str, str], np.ndarray]
    construction: str = "honest"

    def validate(self, tolerance: float = 2e-12) -> dict[str, float]:
        all_tables = [*self.near.values(), *self.far.values()]
        minimum = min(float(np.min(table)) for table in all_tables)
        normalization_error = max(
            abs(float(np.sum(table)) - 1.0) for table in all_tables
        )
        if minimum < -tolerance:
            raise ValueError(f"Negative target probability: {minimum}")
        if normalization_error > tolerance:
            raise ValueError(
                f"Target normalization error {normalization_error} exceeds {tolerance}"
            )
        return {
            "minimum_probability": minimum,
            "maximum_normalization_error": normalization_error,
        }

    def key_table(self) -> np.ndarray:
        return self.far[KEY_SETTING]

    def key_entropies(self, recording: str = "ternary") -> tuple[float, float]:
        table = self.key_table()
        if recording == "binary":
            binned = np.zeros((2, 2, 2), dtype=float)
            for a, b, c in product(range(2), range(3), range(3)):
                bb = 0 if b in (0, 2) else 1
                cc = 0 if c in (0, 2) else 1
                binned[a, bb, cc] += table[a, b, c]
            table = binned
        elif recording != "ternary":
            raise ValueError(f"Unknown key recording: {recording}")
        return (
            _conditional_entropy(np.sum(table, axis=2)),
            _conditional_entropy(np.sum(table, axis=1)),
        )

    def to_record(self) -> dict[str, object]:
        def rows(route: dict[tuple[str, str, str], np.ndarray]) -> list[dict[str, object]]:
            return [
                {"x": x, "y": y, "z": z, "probabilities": table.tolist()}
                for (x, y, z), table in sorted(route.items())
            ]

        payload = {
            "schema": "routed-dicka-target-v1",
            "point": self.point.as_dict(),
            "construction": self.construction,
            "outcomes": {
                "near": ["+1", "-1"],
                "far_alice": ["+1", "-1"],
                "far_receivers": ["+1", "-1", "perp"],
            },
            "near": rows(self.near),
            "far": rows(self.far),
            "validation": self.validate(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["sha256"] = sha256(encoded).hexdigest()
        return payload


def _entropy(probabilities: np.ndarray) -> float:
    flat = np.asarray(probabilities, dtype=float).ravel()
    return -sum(float(p) * log2(float(p)) for p in flat if p > 0.0)


def _conditional_entropy(joint_ab: np.ndarray) -> float:
    return _entropy(joint_ab) - _entropy(np.sum(joint_ab, axis=0))


def build_target(point: TargetPoint, dephased: bool = False) -> TargetBehavior:
    rho = _source_state(point.nu, dephased=dephased)
    alice = {
        label: _binary_povm(_observable(label, point.far_theta), point.eta_a)
        for label in ALICE_SETTINGS
    }
    near_bob = {
        label: _binary_povm(_observable(label, point.near_theta), point.eta_s)
        for label in NEAR_BOB_SETTINGS
    }
    near_charlie = {
        label: _binary_povm(_observable(label, point.near_theta), point.eta_s)
        for label in NEAR_CHARLIE_SETTINGS
    }
    far_bob = {
        label: _ternary_povm(_observable(label, point.far_theta), point.eta_b)
        for label in FAR_BOB_SETTINGS
    }
    far_charlie = {
        label: _ternary_povm(_observable(label, point.far_theta), point.eta_c)
        for label in FAR_CHARLIE_SETTINGS
    }

    near = {
        (x, y, z): _context_probabilities(rho, alice[x], near_bob[y], near_charlie[z])
        for x, y, z in product(
            ALICE_SETTINGS, NEAR_BOB_SETTINGS, NEAR_CHARLIE_SETTINGS
        )
    }
    far = {
        (x, y, z): _context_probabilities(rho, alice[x], far_bob[y], far_charlie[z])
        for x, y, z in product(ALICE_SETTINGS, FAR_BOB_SETTINGS, FAR_CHARLIE_SETTINGS)
    }
    behavior = TargetBehavior(
        point=point,
        near=near,
        far=far,
        construction="globally-dephased-source" if dephased else "honest",
    )
    behavior.validate()
    return behavior


def build_route_selective_dephasing_attack(point: TargetPoint) -> TargetBehavior:
    """Keep the honest near route and dephase only the remote routed systems.

    Operationally, the remote devices perform the computational-basis parent
    measurement after the route is selected and Eve retains its record.  This
    preserves the near test and every remote marginal/key statistic while
    removing the three GHZ phase-sensitive remote triples.
    """

    honest = build_target(point)
    globally_dephased = build_target(point, dephased=True)
    attack = TargetBehavior(
        point=point,
        near={context: table.copy() for context, table in honest.near.items()},
        far={
            context: table.copy()
            for context, table in globally_dephased.far.items()
        },
        construction="honest-near/dephased-far-parent-attack",
    )
    attack.validate()
    return attack
