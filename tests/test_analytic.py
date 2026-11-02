import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routed_dicka.analytic import (
    ideal_loss_rate,
    ideal_symmetric_threshold,
    symmetric_ideal_rows,
)


class AnalyticRateTests(unittest.TestCase):
    def test_symmetric_threshold_is_golden_ratio_conjugate(self):
        expected = (math.sqrt(5.0) - 1.0) / 2.0
        self.assertAlmostEqual(ideal_symmetric_threshold(), expected, places=15)
        self.assertEqual(ideal_loss_rate(expected, expected), 0.0)

    def test_asymmetric_rate_uses_product_and_weaker_receiver(self):
        self.assertAlmostEqual(ideal_loss_rate(0.9, 0.8), 0.52, places=15)
        self.assertEqual(ideal_loss_rate(0.5, 0.5), 0.0)

    def test_dense_rows_are_analytic_and_cover_both_protocols(self):
        rows = symmetric_ideal_rows(points=3, eta_min=0.5, eta_max=1.0)
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["protocol"] for row in rows}, {"rPhase", "rHybrid"})
        self.assertTrue(all(row["evidence_kind"] == "analytic" for row in rows))
        self.assertEqual({row["eta"] for row in rows}, {0.5, 0.75, 1.0})


if __name__ == "__main__":
    unittest.main()
