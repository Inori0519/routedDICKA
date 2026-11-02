from __future__ import annotations

from math import isclose
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routed_dicka.protocols import TargetPoint  # noqa: E402
from routed_dicka.statistics import dephasing_difference_report  # noqa: E402
from routed_dicka.target import (  # noqa: E402
    build_route_selective_dephasing_attack,
    build_target,
)


class TargetTests(unittest.TestCase):
    def test_complete_far_behavior_is_independent_of_near_efficiency(self):
        low = build_target(TargetPoint(nu=0.99, eta_s=0.97, eta_b=0.8, eta_c=0.8))
        high = build_target(TargetPoint(nu=0.99, eta_s=1.0, eta_b=0.8, eta_c=0.8))
        self.assertEqual(set(low.far), set(high.far))
        for context in low.far:
            self.assertTrue((low.far[context] == high.far[context]).all())

    def test_ternary_key_entropy_tracks_erasure_probability(self) -> None:
        point = TargetPoint(nu=1.0, eta_s=1.0, eta_b=0.78, eta_c=0.83)
        behavior = build_target(point)
        h_ab, h_ac = behavior.key_entropies(recording="ternary")
        self.assertTrue(isclose(h_ab, 1.0 - point.eta_b, abs_tol=2e-12))
        self.assertTrue(isclose(h_ac, 1.0 - point.eta_c, abs_tol=2e-12))

    def test_target_contexts_are_normalized(self) -> None:
        point = TargetPoint(nu=0.99, eta_s=0.99, eta_b=0.78, eta_c=0.78)
        report = build_target(point).validate()
        self.assertGreaterEqual(report["minimum_probability"], -2e-12)
        self.assertLessEqual(report["maximum_normalization_error"], 2e-12)

    def test_noisy_ternary_reconciliation_formula(self) -> None:
        from math import log2

        nu = 0.99
        point = TargetPoint(nu=nu, eta_s=0.99, eta_b=0.78, eta_c=0.83)
        h_ab, h_ac = build_target(point).key_entropies(recording="ternary")
        q = (1.0 - nu) / 2.0
        binary_entropy = -q * log2(q) - (1.0 - q) * log2(1.0 - q)
        expected_ab = 1.0 - point.eta_b * (1.0 - binary_entropy)
        expected_ac = 1.0 - point.eta_c * (1.0 - binary_entropy)
        self.assertTrue(isclose(h_ab, expected_ab, abs_tol=2e-12))
        self.assertTrue(isclose(h_ac, expected_ac, abs_tol=2e-12))

    def test_dephasing_is_detected_only_by_coherent_triples(self) -> None:
        point = TargetPoint(nu=1.0, eta_s=1.0, eta_b=0.9, eta_c=0.9)
        report = dephasing_difference_report(
            build_target(point), build_route_selective_dephasing_attack(point)
        )
        self.assertTrue(report["only_expected_coordinates_changed"])
        self.assertEqual(
            report["accepted_models"],
            {"M": True, "W": False, "R": True, "F": False},
        )


if __name__ == "__main__":
    unittest.main()
