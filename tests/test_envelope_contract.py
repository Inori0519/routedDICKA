from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main"
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))
VERIFY = ROOT / "verification"
if str(VERIFY) not in sys.path:
    sys.path.insert(0, str(VERIFY))

from evaluate_witness_envelopes import _evaluate_row, build_rows, validate_curve_matrix_rows  # noqa: E402
from verify_witness_envelopes import route_order_errors  # noqa: E402


class EnvelopeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (ROOT / "config" / "paper_suite.json").read_text(encoding="utf-8")
        )
        slices = self.config["experiments"]["protocol_curve_anchors"]["slices"]
        points = int(self.config["analytic_curves"]["certificate_plot_grid"]["plot_points"])
        self.rows = [
            {"slice_id": curve_slice["slice_id"], "protocol": protocol}
            for curve_slice in slices
            for protocol in curve_slice["protocols"]
            for _ in range(points)
        ]

    def test_complete_curve_matrix_is_accepted(self) -> None:
        summary = validate_curve_matrix_rows(self.config, self.rows)

        self.assertEqual(summary["families"], 16)
        self.assertEqual(summary["rows"], 16 * 201)

    def test_missing_curve_family_is_rejected(self) -> None:
        incomplete = [
            row
            for row in self.rows
            if not (
                row["slice_id"] == "routed_nu1_s0p97"
                and row["protocol"] == "rHybrid"
            )
        ]

        with self.assertRaisesRegex(ValueError, "Curve matrix row contract failed"):
            validate_curve_matrix_rows(self.config, incomplete)

    def test_routed_bound_can_inherit_a_stronger_far_only_certificate(self) -> None:
        def bound(value, label):
            return {
                "entropy_lower": float(value), "entropy_lower_exact": str(value),
                "m": 4, "nodes": [{"certificate_sha256": label}],
            }

        with patch("evaluate_witness_envelopes.evaluate_witness_envelope") as evaluate:
            evaluate.side_effect = [bound(Fraction(4, 5), "near"), bound(Fraction(9, 10), "far")]
            row = _evaluate_row(
                self.config, [{}], protocol="rHybrid", slice_id="test",
                nu=1.0, eta_s=0.97, eta=0.98, include_near=True,
                far_witnesses=[{}],
            )
        self.assertEqual(row["entropy_lower_exact"], "9/10")
        self.assertEqual(row["bound_source_scope"], "far_only")
        self.assertEqual(json.loads(row["selected_nodes"])[0]["certificate_sha256"], "far")
        self.assertAlmostEqual(row["raw_key_rate_lower"], 0.88, places=10)

    def test_far_only_certificates_are_not_given_nearby_observations(self) -> None:
        def evaluate(witnesses, model):
            expected = {"near", "tilted"} if witnesses == ["routed"] else {"tilted"}
            self.assertEqual(set(model.required_features), expected)
            value = Fraction(9, 10) if "near" in expected else Fraction(4, 5)
            return {"entropy_lower": float(value), "entropy_lower_exact": str(value), "m": 4, "nodes": []}

        with patch("evaluate_witness_envelopes.evaluate_witness_envelope", side_effect=evaluate):
            row = _evaluate_row(
                self.config, ["routed"], protocol="rPCHSH", slice_id="test",
                nu=1.0, eta_s=0.97, eta=0.98, include_near=True,
                far_witnesses=["far"],
            )
        self.assertEqual(row["entropy_lower_exact"], "9/10")
        self.assertEqual(row["bound_source_scope"], "routed")

    def test_hybrid_preserves_far_only_provenance_inherited_through_a_child(self) -> None:
        self.config["analytic_curves"]["certificate_plot_grid"]["plot_points"] = 2
        self.config["experiments"]["protocol_curve_anchors"]["slices"] = [{
            "slice_id": "test", "protocols": ["rPCHSH", "rHybrid"],
            "nu": 1.0, "eta_s": 0.97, "include_near": True,
        }]

        def evaluate(config, witnesses, *, protocol, eta, **kwargs):
            value = Fraction(9, 10) if protocol == "rPCHSH" else Fraction(4, 5)
            return {
                "slice_id": "test", "protocol": protocol, "eta_b": eta,
                "entropy_lower": float(value), "entropy_lower_exact": str(value),
                "raw_key_rate_lower": float(value), "key_rate_lower": float(value),
                "m": 4, "selected_nodes": "[]", "bound_source_protocol": protocol,
                "bound_source_scope": "far_only" if protocol == "rPCHSH" else "routed",
            }

        banks = {(protocol, near): [{}] for protocol in ("rPCHSH", "rHybrid") for near in (False, True)}
        with patch("evaluate_witness_envelopes._evaluate_row", side_effect=evaluate):
            rows = build_rows(self.config, banks)
        hybrid = [row for row in rows if row["protocol"] == "rHybrid"]
        self.assertTrue(all(row["entropy_lower_exact"] == "9/10" for row in hybrid))
        self.assertTrue(all(row["bound_source_scope"] == "far_only" for row in hybrid))
        self.assertTrue(all(row["bound_source_protocol"] == "rPCHSH" for row in hybrid))

    def test_independent_order_check_rejects_the_original_reversal(self) -> None:
        common = {
            "protocol": "rHybrid", "nu": 1, "eta_b": 0.98, "eta_c": 0.98,
            "theta": 0.775, "statistics_recording": "ternary",
            "reconciliation_recording": "ternary", "EC_cost": 0.02,
        }
        far = {**common, "data_scope": "far_only", "entropy_lower_exact": "21/25", "key_rate_lower": 0.82}
        routed = {**common, "data_scope": "routed", "entropy_lower_exact": "41/50", "key_rate_lower": 0.80}
        self.assertTrue(route_order_errors([far, routed]))
        routed.update(entropy_lower_exact="21/25", key_rate_lower=0.82)
        self.assertEqual(route_order_errors([far, routed]), [])


if __name__ == "__main__":
    unittest.main()
