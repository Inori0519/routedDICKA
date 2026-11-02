"""Analytic reference rates for the fixed Alice-forward raw-key task."""

from __future__ import annotations

from math import sqrt


def ideal_loss_rate(eta_b: float, eta_c: float) -> float:
    """Exact ideal-loss rate for rPhase/rHybrid in the stated protocol class."""

    eta_b = float(eta_b)
    eta_c = float(eta_c)
    if not 0.0 <= eta_b <= 1.0 or not 0.0 <= eta_c <= 1.0:
        raise ValueError("Receiver efficiencies must lie in [0, 1]")
    return max(0.0, eta_b * eta_c + min(eta_b, eta_c) - 1.0)


def ideal_symmetric_threshold() -> float:
    """Positive-rate threshold for eta_B=eta_C=eta."""

    return (sqrt(5.0) - 1.0) / 2.0


def symmetric_ideal_rows(
    *,
    points: int = 101,
    eta_min: float = 0.5,
    eta_max: float = 1.0,
) -> list[dict[str, object]]:
    """Dense analytic rows; no row represents a fresh SDP optimization."""

    if points < 2:
        raise ValueError("At least two plotting points are required")
    if not 0.0 <= eta_min < eta_max <= 1.0:
        raise ValueError("Require 0 <= eta_min < eta_max <= 1")
    etas = [eta_min + (eta_max - eta_min) * index / (points - 1) for index in range(points)]
    rows = []
    for eta in etas:
        rate = ideal_loss_rate(eta, eta)
        for protocol in ("rPhase", "rHybrid"):
            rows.append(
                {
                    "protocol": protocol,
                    "eta": eta,
                    "eta_b": eta,
                    "eta_c": eta,
                    "key_rate_lower": rate,
                    "evidence_kind": "analytic",
                    "result_scope": "fixed_raw_data_alice_forward",
                }
            )
    return rows
