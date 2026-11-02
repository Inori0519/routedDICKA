"""One-point entropy and conference-key-rate calculation."""

from __future__ import annotations

from typing import Iterable

from .protocols import Protocol, get_protocol
from .sdp import SolverSettings, entropy_lower_bound
from .statistics import Coordinate, constraint_manifest, protocol_coordinates
from .target import TargetBehavior


RECORDING_MODES = {
    "binary_statistics_binary_reconciliation": ("binary", "binary"),
    "binary_statistics_ternary_reconciliation": ("binary", "ternary"),
    "ternary_statistics_ternary_reconciliation": ("ternary", "ternary"),
}


def compute_point(
    behavior: TargetBehavior,
    coordinates: Iterable[Coordinate],
    m: int,
    settings: SolverSettings,
    reconciliation_recording: str = "ternary",
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    coordinates = set(coordinates)
    manifest = constraint_manifest(behavior, coordinates)
    entropy = entropy_lower_bound(behavior, coordinates, m=m, settings=settings)
    h_ab, h_ac = behavior.key_entropies(recording=reconciliation_recording)
    ec = max(h_ab, h_ac)
    raw_rate = float(entropy["entropy_lower"]) - ec
    return {
        "schema": "routed-dicka-bff-point-v1",
        "metadata": dict(metadata or {}),
        "physical_point": behavior.point.as_dict(),
        "target_construction": behavior.construction,
        "constraint_manifest": manifest,
        "solver": {
            "name": "mosek",
            "npa_level": settings.level,
            "relative_gap_tolerance": settings.relative_gap_tolerance,
            "primal_feasibility_tolerance": settings.primal_feasibility_tolerance,
            "dual_feasibility_tolerance": settings.dual_feasibility_tolerance,
            "max_iterations": settings.max_iterations,
            "threads": settings.threads,
            "numerical_guard": settings.numerical_guard,
            "max_accepted_primal_dual_gap": settings.max_accepted_primal_dual_gap,
        },
        "entropy": entropy,
        "reconciliation_recording": reconciliation_recording,
        "H_A_given_B": h_ab,
        "H_A_given_C": h_ac,
        "EC_cost": ec,
        "raw_key_rate_lower": raw_rate,
        "key_rate_lower": max(0.0, raw_rate),
        "claim_eligible": bool(entropy["all_nodes_claim_eligible"]),
    }


def compute_protocol_point(
    behavior: TargetBehavior,
    protocol: Protocol | str,
    m: int,
    settings: SolverSettings,
    statistics_recording: str = "ternary",
    reconciliation_recording: str = "ternary",
    include_near: bool = True,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    protocol = get_protocol(protocol) if isinstance(protocol, str) else protocol
    coordinates = protocol_coordinates(
        protocol,
        recording=statistics_recording,
        include_near=include_near,
    )
    combined_metadata = {
        "protocol": protocol.identifier,
        "statistics_recording": statistics_recording,
        "include_near": include_near,
        **(metadata or {}),
    }
    return compute_point(
        behavior,
        coordinates,
        m=m,
        settings=settings,
        reconciliation_recording=reconciliation_recording,
        metadata=combined_metadata,
    )


def recordings_for_mode(mode: str) -> tuple[str, str]:
    try:
        return RECORDING_MODES[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unknown recording ablation mode {mode!r}; choose from {sorted(RECORDING_MODES)}"
        ) from exc
