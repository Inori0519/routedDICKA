import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "verification"
if str(VERIFY) not in sys.path:
    sys.path.insert(0, str(VERIFY))

from verify_model_contract import verify


class ExperimentProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (ROOT / "config" / "paper_suite.json").read_text(encoding="utf-8")
        )

    def test_profiles_are_layered_by_cost(self):
        profiles = self.config["profiles"]
        self.assertEqual(
            set(profiles),
            {"smoke0", "smoke1", "smoke2", "core", "paper"},
        )
        self.assertEqual(profiles["smoke0"], [])

    def test_paper_has_no_cartesian_full_grid_sweep(self):
        experiments = self.config["experiments"]
        selected = [experiments[name] for name in self.config["profiles"]["paper"]]
        self.assertFalse(any(spec["kind"] == "protocol_curves" for spec in selected))
        self.assertEqual(
            [spec["kind"] for spec in selected].count("reduced_curve_matrix"), 1
        )

    def test_main_separation_uses_exact_attack_and_archived_certificate(self):
        separation = self.config["reference_points"]["S1"]
        self.assertEqual(separation["protocol"], "rHybrid")
        self.assertEqual(separation["nu"], 0.99)
        self.assertEqual(separation["eta_s"], 0.99)
        self.assertEqual(separation["eta_l"], 0.78)
        self.assertEqual(separation["routed_evidence"], "verified_affine_certificate")
        self.assertEqual(separation["long_only_evidence"], "exact_receiver_simulation_attack")

    def test_mwrf_numerics_are_limited_to_w_and_f(self):
        experiment = self.config["experiments"]["mwrf_primary"]
        self.assertEqual(experiment["models"], ["W", "F"])
        self.assertEqual(experiment["attack_models"], ["M", "R"])
        self.assertEqual(experiment["m"], 2)

    def test_quadrature_order_is_per_experiment_not_global_gate(self):
        orders = {
            int(spec["m"])
            for spec in self.config["experiments"].values()
            if "m" in spec
        }
        self.assertIn(2, orders)
        self.assertTrue(all(order >= 2 for order in orders))

    def test_model_contract_counts_the_reduced_paper_plan(self):
        result = verify(self.config)
        self.assertTrue(result["passed"])
        self.assertEqual(result["profiles"]["smoke0"]["sdp_node_solves"], 0)
        self.assertEqual(result["profiles"]["smoke1"]["sdp_node_solves"], 2)
        self.assertEqual(result["profiles"]["paper"]["entropy_points"], 155)
        self.assertEqual(result["profiles"]["paper"]["sdp_node_solves"], 446)

    def test_curve_matrix_uses_m4_without_promoting_secondary_experiments(self):
        curves = self.config["experiments"]["protocol_curve_anchors"]
        self.assertEqual(curves["m"], 4)
        self.assertFalse(curves["allow_m_override"])
        self.assertEqual(curves["eta_l"], [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
        self.assertEqual(len(curves["slices"]), 6)
        self.assertTrue(any(not item["include_near"] for item in curves["slices"]))
        self.assertEqual(self.config["experiments"]["mwrf_primary"]["m"], 2)
        self.assertEqual(self.config["experiments"]["recording_anchors"]["m"], 2)

    def test_s1_rphase_sensitivity_audit_covers_m2_to_m4(self):
        experiment = self.config["experiments"]["reduced_radau_audit"]
        self.assertEqual(experiment["protocol"], "rPhase")
        self.assertEqual(experiment["eta_l"], 0.78)
        self.assertEqual(experiment["m_values"], [2, 3, 4])
        self.assertNotIn("optional_m", experiment)
        self.assertIn("reduced_radau_audit", self.config["profiles"]["core"])


if __name__ == "__main__":
    unittest.main()
