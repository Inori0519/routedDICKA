import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main"
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from build_reference_data import build_ideal_rows, build_separation_rows, portable_input_key


class ReferenceDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (ROOT / "config" / "paper_suite.json").read_text(encoding="utf-8")
        )

    def test_ideal_rows_are_dense_analytic_evaluations(self):
        rows = build_ideal_rows(self.config)
        self.assertEqual(len(rows), 2 * 201)
        self.assertEqual({row["protocol"] for row in rows}, {"rPhase", "rHybrid"})
        self.assertTrue(all(row["evidence_kind"] == "analytic" for row in rows))
        self.assertTrue(all(row["sdp_optimized"] is False for row in rows))

    def test_reference_audit_input_keys_are_portable(self):
        output = ROOT / "outputs"

        config_key = portable_input_key(ROOT / "config" / "paper_suite.json", output)
        audit_key = portable_input_key(
            output / "audits" / "m4_same_point_certificate.json", output
        )

        self.assertEqual(config_key, "../config/paper_suite.json")
        self.assertEqual(audit_key, "audits/m4_same_point_certificate.json")
        self.assertNotIn(":", config_key)

    def test_strict_separation_combines_two_independent_evidence_lanes(self):
        certificate = {
            "passed": True,
            "reference_point": {
                "nu": 0.99,
                "eta_s": 0.99,
                "eta_b": 0.78,
                "eta_c": 0.78,
                "H_lower": 0.2780100576,
                "EC_upper": 0.2554234600,
                "K_lower": 0.0225865976,
                "positive": True,
            },
        }
        attack = {
            "exactly_verified": True,
            "exact_degrading_channel_verified": True,
            "configured_nu": "99/100",
            "configured_eta": "39/50",
            "probabilities": 288,
        }
        rows = build_separation_rows(self.config, certificate, attack)
        self.assertEqual([row["data_scope"] for row in rows], ["far_only", "routed"])
        self.assertEqual(rows[0]["guaranteed_key_rate"], 0.0)
        self.assertGreater(rows[1]["guaranteed_key_rate"], 0.0)
        self.assertEqual(rows[0]["evidence_kind"], "exact_receiver_simulation_attack")
        self.assertEqual(rows[1]["evidence_kind"], "verified_affine_certificate")

    def test_m4_archive_is_the_primary_routed_separation_evidence(self):
        certificate = {
            "passed": True,
            "reference_point": {
                "nu": 0.99,
                "eta_s": 0.99,
                "eta_b": 0.78,
                "eta_c": 0.78,
                "m": 4,
                "H_lower": 0.3327199023192324,
                "EC_upper": 0.2554234600213591,
                "K_lower": 0.0772964422978733,
                "positive": True,
            },
        }
        attack = {
            "exactly_verified": True,
            "exact_degrading_channel_verified": True,
            "configured_nu": "99/100",
            "configured_eta": "39/50",
            "probabilities": 288,
        }
        rows = build_separation_rows(self.config, certificate, attack)
        self.assertEqual(rows[1]["m"], 4)
        self.assertEqual(
            rows[1]["certificate_source_file"], "source_data/certificates/m4"
        )
        self.assertAlmostEqual(rows[1]["guaranteed_key_rate"], 0.0772964422978733)

    def test_strict_separation_rejects_point_mismatch(self):
        certificate = {
            "passed": True,
            "reference_point": {
                "nu": 0.99,
                "eta_s": 0.99,
                "eta_b": 0.78,
                "eta_c": 0.78,
                "H_lower": 0.28,
                "EC_upper": 0.25,
                "K_lower": 0.03,
                "positive": True,
            },
        }
        attack = {
            "exactly_verified": True,
            "exact_degrading_channel_verified": True,
            "configured_nu": "99/100",
            "configured_eta": "4/5",
            "probabilities": 288,
        }
        with self.assertRaises(AssertionError):
            build_separation_rows(self.config, certificate, attack)

if __name__ == "__main__":
    unittest.main()
