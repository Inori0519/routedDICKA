"""Protocol-specific composite-receiver reductions from complete target tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .protocols import Protocol, get_protocol
from .reduced_algebra import Algebra, Symbol, Word
from .target import TargetBehavior


PROTOCOL_FEATURES = {
    "rPCHSH": frozenset(("near", "tilted")),
    "rPhase": frozenset(("near", "phase")),
    "rHybrid": frozenset(("near", "phase", "tilted")),
}


def protocol_features(
    protocol: Protocol | str,
    *,
    include_near: bool = True,
) -> frozenset[str]:
    identifier = get_protocol(protocol).identifier if isinstance(protocol, str) else protocol.identifier
    features = PROTOCOL_FEATURES[identifier]
    return features if include_near else features - {"near"}


@dataclass(frozen=True)
class ReducedModelData:
    protocol: str
    statistics_recording: str
    include_near: bool
    algebra: Algebra
    groups: dict[str, dict[Word, float]]
    alice_key_projector: int
    eve_operators: tuple[int, int]
    required_features: frozenset[str]
    symbol_indices: dict[str, int]

    def fixed_moments(self, features: Iterable[str] | None = None) -> dict[Word, float]:
        selected = self.required_features if features is None else frozenset(features)
        if not selected <= self.required_features:
            raise ValueError("Requested moments are outside this reduced model")
        result = dict(self.groups["alice"])
        for feature in sorted(selected):
            for word, value in self.groups[feature].items():
                if word in result and abs(result[word] - value) > 2e-12:
                    raise AssertionError(f"Inconsistent reduced moment {word}")
                result[word] = value
        return result


def _linear_effect(
    behavior: TargetBehavior,
    route: str,
    alice_setting: str,
    terms: Iterable[tuple[float, str, set[int] | None, str, set[int] | None]],
    *,
    alice_plus: bool,
) -> float:
    tables = behavior.near if route == "near" else behavior.far
    total = 0.0
    for coefficient, bob_setting, bob_outcomes, charlie_setting, charlie_outcomes in terms:
        table = tables[(alice_setting, bob_setting, charlie_setting)]
        alice_range = (0,) if alice_plus else range(table.shape[0])
        bob_range = range(table.shape[1]) if bob_outcomes is None else bob_outcomes
        charlie_range = (
            range(table.shape[2]) if charlie_outcomes is None else charlie_outcomes
        )
        total += coefficient * sum(
            float(table[alice, bob, charlie])
            for alice in alice_range
            for bob in bob_range
            for charlie in charlie_range
        )
    return total


def _alice_marginal(behavior: TargetBehavior, setting: str) -> float:
    table = behavior.far[(setting, "Z", "Z")]
    return float(table[0, :, :].sum())


def _add_effect_moments(
    group: dict[Word, float],
    model: ReducedModelData,
    symbol: int,
    behavior: TargetBehavior,
    route: str,
    terms,
) -> None:
    group[(symbol,)] = _linear_effect(
        behavior, route, "Z", terms, alice_plus=False
    )
    for alice_setting in ("Z", "X"):
        alice = model.symbol_indices[f"A.{alice_setting}.0"]
        word = model.algebra.moment((alice, symbol))
        if word is None:
            raise AssertionError("A physical composite moment reduced to zero")
        group[word] = _linear_effect(
            behavior, route, alice_setting, terms, alice_plus=True
        )


def build_reduced_model(
    behavior: TargetBehavior,
    protocol: Protocol | str,
    *,
    statistics_recording: str = "ternary",
    include_near: bool = True,
) -> ReducedModelData:
    if statistics_recording not in ("binary", "ternary"):
        raise ValueError("statistics_recording must be 'binary' or 'ternary'")
    protocol_object = get_protocol(protocol) if isinstance(protocol, str) else protocol
    features = protocol_features(protocol_object, include_near=include_near)
    specifications = [
        ("A", "Z", 0),
        ("A", "X", 0),
    ]
    if "near" in features:
        specifications.extend((("S", "0", 0), ("S", "1", 0)))
    if "phase" in features:
        specifications.extend((("L", "phase", 0), ("L", "phase", 1)))
    if "tilted" in features:
        specifications.extend((("L", "test0", 0), ("L", "test1", 0)))
    symbols = [
        Symbol(f"{party}.{setting}.{outcome}", party, setting, outcome)
        for party, setting, outcome in specifications
    ]
    physical_count = len(symbols)
    for outcome in range(2):
        z = len(symbols)
        symbols.extend(
            (
                Symbol(f"Z{outcome}", "E", str(outcome), 0, "aux", z + 1),
                Symbol(f"Z{outcome}d", "E", str(outcome), 0, "aux", z),
            )
        )
    commuting_pairs = [("A", "L"), ("A", "E"), ("L", "E")]
    if include_near:
        commuting_pairs.insert(0, ("A", "S"))
    algebra = Algebra(symbols, tuple(commuting_pairs))
    indices = {symbol.name: index for index, symbol in enumerate(symbols)}
    groups: dict[str, dict[Word, float]] = {
        "alice": {
            (): 1.0,
            (indices["A.Z.0"],): _alice_marginal(behavior, "Z"),
            (indices["A.X.0"],): _alice_marginal(behavior, "X"),
        }
    }
    model = ReducedModelData(
        protocol=protocol_object.identifier,
        statistics_recording=statistics_recording,
        include_near=include_near,
        algebra=algebra,
        groups=groups,
        alice_key_projector=indices["A.Z.0"],
        eve_operators=(physical_count, physical_count + 2),
        required_features=features,
        symbol_indices=indices,
    )

    near_terms = (
        (
            (1.0, "T1", {0}, "X", None),
            (1.0, "T0", {0}, "X", {0}),
            (-1.0, "T1", {0}, "X", {0}),
        ),
        (
            (1.0, "T0", {0}, "X", None),
            (1.0, "T1", {0}, "X", {0}),
            (-1.0, "T0", {0}, "X", {0}),
        ),
    )
    if "near" in features:
        groups["near"] = {}
        for setting, terms in enumerate(near_terms):
            _add_effect_moments(
                groups["near"],
                model,
                indices[f"S.{setting}.0"],
                behavior,
                "near",
                terms,
            )

    if "phase" in features:
        plus = {0, 2} if statistics_recording == "binary" else {0}
        phase_terms = (
            (
                (1.0, "X", plus, "X", plus),
                (1.0, "X", {1}, "X", {1}),
            ),
            (
                (1.0, "X", plus, "X", {1}),
                (1.0, "X", {1}, "X", plus),
            ),
        )
        groups["phase"] = {}
        for outcome, terms in enumerate(phase_terms):
            _add_effect_moments(
                groups["phase"],
                model,
                indices[f"L.phase.{outcome}"],
                behavior,
                "far",
                terms,
            )

    if "tilted" in features:
        tilted_terms = (
            (
                (1.0, "T0", {0, 2}, "X", None),
                (1.0, "T0", {1}, "X", {1}),
                (-1.0, "T1", {1}, "X", {1}),
            ),
            (
                (1.0, "T1", {0, 2}, "X", None),
                (1.0, "T1", {1}, "X", {1}),
                (-1.0, "T0", {1}, "X", {1}),
            ),
        )
        groups["tilted"] = {}
        for setting, terms in enumerate(tilted_terms):
            _add_effect_moments(
                groups["tilted"],
                model,
                indices[f"L.test{setting}.0"],
                behavior,
                "far",
                terms,
            )
    return model
