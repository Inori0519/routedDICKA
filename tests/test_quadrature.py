from __future__ import annotations

from math import isclose
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routed_dicka.quadrature import right_radau_rule, validate_rule  # noqa: E402


class RightRadauTests(unittest.TestCase):
    def test_primary_and_sensitivity_orders(self) -> None:
        for m in (4, 5):
            with self.subTest(m=m):
                nodes = right_radau_rule(m)
                self.assertEqual(len(nodes), m)
                self.assertEqual(sum(node.endpoint for node in nodes), 1)
                self.assertTrue(nodes[-1].endpoint)
                self.assertEqual(nodes[-1].t, 1.0)
                self.assertTrue(isclose(nodes[-1].weight, 1.0 / m**2))
                report = validate_rule(nodes)
                self.assertEqual(report["degree_checked"], 2 * m - 2)
                self.assertLess(report["max_moment_error"], 2e-12)

    def test_invalid_order_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            right_radau_rule(1)


if __name__ == "__main__":
    unittest.main()
