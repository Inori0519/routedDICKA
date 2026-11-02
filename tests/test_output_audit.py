import importlib.util
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SPEC = importlib.util.spec_from_file_location(
    "verify_outputs", ROOT / "verification" / "verify_outputs.py"
)
verify_outputs = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_outputs)


class OutputAuditTests(unittest.TestCase):
    def test_raw_rate_uses_unclipped_entropy_lower_bound(self):
        record = {
            "entropy": {
                "raw_entropy_lower": -0.0025,
                "entropy_lower": 0.0,
            },
            "H_A_given_B": 0.4,
            "H_A_given_C": 0.4,
            "EC_cost": 0.4,
            "raw_key_rate_lower": -0.4025,
            "key_rate_lower": 0.0,
            "claim_eligible": True,
        }

        summary = verify_outputs._validate_rate_identity(record)

        self.assertAlmostEqual(summary["raw_key_rate_lower"], -0.4025)
        self.assertEqual(summary["entropy_lower"], 0.0)

    def test_reference_input_paths_resolve_relative_to_output(self):
        output = ROOT / "outputs"

        resolved = verify_outputs.resolve_audit_input(
            output, "../config/paper_suite.json"
        )

        self.assertEqual(resolved, (ROOT / "config" / "paper_suite.json").resolve())

    def test_exact_certificate_aggregation_is_checked_before_entropy_clipping(self):
        combined = {"entropy_lower": -0.0025}
        recorded = {"raw_entropy_lower": -0.0025, "entropy_lower": 0.0}

        summary = verify_outputs.validate_certified_entropy_identity(combined, recorded)

        self.assertAlmostEqual(summary["raw_entropy_lower"], -0.0025)
        self.assertEqual(summary["entropy_lower"], 0.0)

    def test_m2_radau_rule_is_a_valid_manuscript_order(self):
        entropy = {
            "m": 2,
            "radau_rule": [
                {"index": 0, "t": 1.0 / 3.0, "weight": 0.75, "endpoint": False},
                {"index": 1, "t": 1.0, "weight": 0.25, "endpoint": True},
            ],
        }

        summary = verify_outputs.validate_radau_rule(entropy)

        self.assertEqual(summary["m"], 2)
        self.assertLess(summary["max_moment_error"], 1e-14)

    def test_explicit_zero_entropy_attack_does_not_require_sdp_nodes(self):
        record = {
            "evidence_kind": "explicit_compatible_zero_entropy_attack",
            "entropy": {
                "m": 0,
                "nodes": [],
                "raw_entropy_lower": 0.0,
                "entropy_lower": 0.0,
                "compatible_attack_entropy_upper": 0.0,
                "all_nodes_claim_eligible": True,
            },
            "H_A_given_B": 0.2,
            "H_A_given_C": 0.2,
            "EC_cost": 0.2,
            "raw_key_rate_lower": -0.2,
            "key_rate_lower": 0.0,
            "guaranteed_key_rate_upper": 0.0,
            "claim_eligible": True,
        }

        summary = verify_outputs.validate_evidence_identity(record)

        self.assertEqual(summary["evidence_kind"], record["evidence_kind"])
        self.assertEqual(summary["sdp_nodes"], 0)
        self.assertEqual(summary["guaranteed_key_rate_upper"], 0.0)

    def test_raw_bff_identity_is_recomputed_at_m2(self):
        coefficient = 0.75 / ((1.0 / 3.0) * math.log(2.0))
        guarded_dual = -0.9 - 1e-7
        entropy_lower = coefficient * (1.0 + guarded_dual)
        record = {
            "entropy": {
                "m": 2,
                "radau_rule": [
                    {"index": 0, "t": 1.0 / 3.0, "weight": 0.75, "endpoint": False},
                    {"index": 1, "t": 1.0, "weight": 0.25, "endpoint": True},
                ],
                "nodes": [
                    {
                        "index": 0,
                        "coefficient": coefficient,
                        "dual": -0.9,
                        "numerical_guard": 1e-7,
                        "guarded_dual": guarded_dual,
                        "status_optimal": True,
                        "gap_accepted": True,
                        "claim_eligible": True,
                    }
                ],
                "raw_entropy_lower": entropy_lower,
                "entropy_lower": entropy_lower,
                "all_nodes_claim_eligible": True,
            },
            "H_A_given_B": 0.1,
            "H_A_given_C": 0.1,
            "EC_cost": 0.1,
            "raw_key_rate_lower": entropy_lower - 0.1,
            "key_rate_lower": max(0.0, entropy_lower - 0.1),
            "claim_eligible": True,
        }

        summary = verify_outputs.validate_evidence_identity(record)

        self.assertEqual(summary["evidence_kind"], "raw_bff_dual")
        self.assertEqual(summary["m"], 2)
        self.assertAlmostEqual(summary["entropy_lower"], entropy_lower)


if __name__ == "__main__":
    unittest.main()
