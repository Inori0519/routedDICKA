"""Canonical observed-coordinate sets for protocols and M/W/R/F ablations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from typing import Iterable, Mapping

import numpy as np

from .protocols import (
    ALICE_SETTINGS,
    FAR_CHARLIE_SETTINGS,
    KEY_SETTING,
    NEAR_BOB_SETTINGS,
    NEAR_CHARLIE_SETTINGS,
    Protocol,
    get_protocol,
)
from .target import TargetBehavior


PARTY_ORDER = {"A": 0, "BS": 1, "CS": 2, "BL": 3, "CL": 4}


@dataclass(frozen=True, order=True)
class Factor:
    party: str
    setting: str
    component: str

    @property
    def name(self) -> str:
        return f"{self.party}:{self.setting}:{self.component}"


@dataclass(frozen=True, order=True)
class Coordinate:
    factors: tuple[Factor, ...]

    @property
    def name(self) -> str:
        return "*".join(factor.name for factor in self.factors)

    @property
    def route(self) -> str:
        parties = {factor.party for factor in self.factors}
        if parties & {"BS", "CS"}:
            return "near"
        if parties & {"BL", "CL"}:
            return "far"
        return "shared"

    @property
    def order(self) -> int:
        return len(self.factors)


def make_coordinate(*factors: Factor) -> Coordinate:
    if not factors:
        raise ValueError("The constant coordinate is not stored")
    parties = [factor.party for factor in factors]
    if len(parties) != len(set(parties)):
        raise ValueError(f"At most one factor per party is allowed: {factors}")
    ordered = tuple(
        sorted(factors, key=lambda factor: (PARTY_ORDER[factor.party], factor.setting, factor.component))
    )
    return Coordinate(ordered)


def _context_coordinates(
    x: str,
    y: str,
    z: str,
    bob_party: str,
    charlie_party: str,
    receiver_components: tuple[str, ...],
) -> set[Coordinate]:
    local = (
        (Factor("A", x, "V"),),
        tuple(Factor(bob_party, y, component) for component in receiver_components),
        tuple(Factor(charlie_party, z, component) for component in receiver_components),
    )
    coordinates: set[Coordinate] = set()
    for choices in product(*[(None, *factors) for factors in local]):
        chosen = tuple(factor for factor in choices if factor is not None)
        if chosen:
            coordinates.add(make_coordinate(*chosen))
    return coordinates


def near_full_coordinates() -> set[Coordinate]:
    coordinates: set[Coordinate] = set()
    for x, y, z in product(ALICE_SETTINGS, NEAR_BOB_SETTINGS, NEAR_CHARLIE_SETTINGS):
        coordinates |= _context_coordinates(x, y, z, "BS", "CS", ("V",))
    return coordinates


def far_full_coordinates(
    protocol: Protocol | str,
    recording: str = "ternary",
) -> set[Coordinate]:
    protocol = get_protocol(protocol) if isinstance(protocol, str) else protocol
    if recording == "ternary":
        components = ("V", "D")
    elif recording == "binary":
        components = ("BIN",)
    else:
        raise ValueError(f"Unknown statistics recording: {recording}")
    coordinates: set[Coordinate] = set()
    for x, y, z in product(
        ALICE_SETTINGS,
        protocol.bob_far_settings,
        protocol.charlie_far_settings,
    ):
        coordinates |= _context_coordinates(x, y, z, "BL", "CL", components)
    return coordinates


def protocol_coordinates(
    protocol: Protocol | str,
    recording: str = "ternary",
    include_near: bool = True,
) -> set[Coordinate]:
    result = far_full_coordinates(protocol, recording=recording)
    if include_near:
        result |= near_full_coordinates()
    return result


def coherent_coordinates() -> set[Coordinate]:
    return {
        make_coordinate(
            Factor("A", "X", "V"),
            Factor("BL", "T0", "V"),
            Factor("CL", "X", "V"),
        ),
        make_coordinate(
            Factor("A", "X", "V"),
            Factor("BL", "T1", "V"),
            Factor("CL", "X", "V"),
        ),
        make_coordinate(
            Factor("A", "X", "V"),
            Factor("BL", "X", "V"),
            Factor("CL", "X", "V"),
        ),
    }


def _key_triple_coordinates() -> set[Coordinate]:
    x, y, z = KEY_SETTING
    return {
        make_coordinate(
            Factor("A", x, "V"),
            Factor("BL", y, bob_component),
            Factor("CL", z, charlie_component),
        )
        for bob_component, charlie_component in product(("V", "D"), repeat=2)
    }


def mwrf_coordinate_sets() -> dict[str, set[Coordinate]]:
    near = near_full_coordinates()
    far = far_full_coordinates("rHybrid", recording="ternary")
    lower_order = {coordinate for coordinate in far if coordinate.order <= 2}
    witness = coherent_coordinates()
    marginal = near | lower_order | _key_triple_coordinates()
    full = near | far
    sets = {
        "M": marginal,
        "W": marginal | witness,
        "R": full - witness,
        "F": full,
    }
    validate_mwrf_lattice(sets)
    return sets


def validate_mwrf_lattice(
    sets: Mapping[str, set[Coordinate]] | None = None,
) -> dict[str, object]:
    sets = dict(sets or mwrf_coordinate_sets())
    checks = {
        "M_subset_W": sets["M"] < sets["W"],
        "M_subset_R": sets["M"] < sets["R"],
        "W_subset_F": sets["W"] < sets["F"],
        "R_subset_F": sets["R"] < sets["F"],
        "W_R_incomparable": not (
            sets["W"] <= sets["R"] or sets["R"] <= sets["W"]
        ),
        "W_union_R_equals_F": (sets["W"] | sets["R"]) == sets["F"],
    }
    if not all(checks.values()):
        raise AssertionError(f"Invalid M/W/R/F lattice: {checks}")
    return {
        "checks": checks,
        "counts": {model: len(coordinates) for model, coordinates in sets.items()},
    }


def _outcome_weights(factor: Factor) -> np.ndarray:
    if factor.party in {"A", "BS", "CS"}:
        if factor.component != "V":
            raise ValueError(f"Binary party only supports V: {factor}")
        return np.array([1.0, -1.0])
    if factor.component == "V":
        return np.array([1.0, -1.0, 0.0])
    if factor.component == "D":
        return np.array([1.0, 1.0, 0.0])
    if factor.component == "BIN":
        return np.array([1.0, -1.0, 1.0])
    raise ValueError(f"Unknown coordinate component: {factor.component}")


def coordinate_value(behavior: TargetBehavior, coordinate: Coordinate) -> float:
    factors = {factor.party: factor for factor in coordinate.factors}
    if coordinate.route == "near":
        x = factors.get("A", Factor("A", "Z", "V")).setting
        y = factors.get("BS", Factor("BS", NEAR_BOB_SETTINGS[0], "V")).setting
        z = factors.get("CS", Factor("CS", NEAR_CHARLIE_SETTINGS[0], "V")).setting
        table = behavior.near[(x, y, z)]
        parties = ("A", "BS", "CS")
    else:
        x = factors.get("A", Factor("A", "Z", "V")).setting
        y = factors.get("BL", Factor("BL", "Z", "V")).setting
        z = factors.get("CL", Factor("CL", "Z", "V")).setting
        table = behavior.far[(x, y, z)]
        parties = ("A", "BL", "CL")

    value = 0.0
    for outcomes in product(*[range(size) for size in table.shape]):
        weight = 1.0
        for axis, party in enumerate(parties):
            factor = factors.get(party)
            if factor is not None:
                weight *= _outcome_weights(factor)[outcomes[axis]]
        value += weight * float(table[outcomes])
    return float(value)


def constraint_manifest(
    behavior: TargetBehavior,
    coordinates: Iterable[Coordinate],
) -> dict[str, object]:
    rows = [
        {"name": coordinate.name, "value": coordinate_value(behavior, coordinate)}
        for coordinate in sorted(set(coordinates), key=lambda item: item.name)
    ]
    identifier = sha256(
        "\n".join(row["name"] for row in rows).encode("ascii")
    ).hexdigest()
    value_hash = sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "constraint_id": identifier,
        "constraint_value_sha256": value_hash,
        "constraint_count": len(rows),
        "coordinates": rows,
    }


def dephasing_difference_report(
    honest: TargetBehavior,
    attack: TargetBehavior,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    sets = mwrf_coordinate_sets()
    full = sets["F"]
    differences = {
        coordinate.name: coordinate_value(honest, coordinate)
        - coordinate_value(attack, coordinate)
        for coordinate in full
    }
    changed = {
        name: value for name, value in differences.items() if abs(value) > tolerance
    }
    expected = {coordinate.name for coordinate in coherent_coordinates()}
    return {
        "changed_coordinates": dict(sorted(changed.items())),
        "expected_changed_coordinates": sorted(expected),
        "only_expected_coordinates_changed": set(changed) == expected,
        "accepted_models": {
            model: all(
                abs(differences[coordinate.name]) <= tolerance
                for coordinate in coordinates
            )
            for model, coordinates in sets.items()
        },
        "tolerance": tolerance,
        "honest_construction": honest.construction,
        "attack_construction": attack.construction,
    }
