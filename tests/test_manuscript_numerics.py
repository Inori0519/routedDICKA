from pathlib import Path
import math
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main"
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from export_manuscript_numerics import _directed, representative_rows, radau_rows, write_latex


class ManuscriptNumericsTests(unittest.TestCase):
    def test_score_calibration_at_reference_is_below_reconciliation(self):
        nu = eta_s = 0.99
        eta = 0.78
        q = (1 - nu) / 2
        binary_entropy = -q * math.log2(q) - (1 - q) * math.log2(1 - q)
        beta = nu * eta_s * math.sqrt(1 + eta_s**2)
        calibration = 1 - math.log2(1 + beta * math.sqrt(2 - beta**2))
        phase_entropy = 1 - eta**2 * (1 - binary_entropy)
        score_bound = calibration - phase_entropy
        reconciliation = 1 - eta * (1 - binary_entropy)
        self.assertAlmostEqual(score_bound, 0.0631645525, places=9)
        self.assertLess(score_bound, reconciliation)
        self.assertGreater(0.3327199 - 0.2554235, 0.07729)

    def test_latex_writer_is_python39_compatible_and_uses_lf(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.tex"

            write_latex(path, "first\nsecond\n")

            self.assertEqual(path.read_bytes(), b"first\nsecond\n")

    def test_displayed_bounds_are_rounded_in_the_conservative_direction(self):
        self.assertEqual(_directed(0.0772964422978733, 10), "0.0772964422")
        self.assertEqual(
            _directed(0.2554234600213591, 10, upper=True), "0.2554234601"
        )

    def test_representative_rows_select_the_m4_envelope_at_s2(self):
        rows = [
            {
                "slice_id": "routed_nu0p99_s0p99",
                "protocol": protocol,
                "eta_b": "0.8",
                "m": "4",
                "entropy_lower": "0.3",
                "EC_cost": "0.2",
                "key_rate_lower": "0.1",
            }
            for protocol in ("rPCHSH", "rPhase", "rHybrid")
        ]
        selected = representative_rows(rows)
        self.assertEqual([row["protocol"] for row in selected], ["rPCHSH", "rPhase", "rHybrid"])
        self.assertTrue(all(row["m"] == 4 for row in selected))

    def test_radau_rows_replace_m4_with_the_primary_archive(self):
        numerical = [
            {
                "m": str(m),
                "entropy_lower": str(0.1 * m),
                "EC_cost": "0.2",
                "raw_key_rate_lower": str(0.1 * m - 0.2),
                "claim_eligible": "True",
            }
            for m in (2, 3, 4)
        ]
        audit = {
            "passed": True,
            "reference_point": {
                "m": 4,
                "H_lower": 0.3327199023192324,
                "EC_upper": 0.2554234600213591,
                "K_lower": 0.0772964422978733,
            },
        }
        selected = radau_rows(numerical, audit)
        self.assertEqual([row["m"] for row in selected], [2, 3, 4])
        self.assertAlmostEqual(selected[-1]["key_rate_lower"], 0.0772964422978733)
        self.assertEqual(selected[-1]["source"], "archived_m4")


if __name__ == "__main__":
    unittest.main()
