from fractions import Fraction
from math import sqrt
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routed_dicka.reduced_certificate import radau_interval_rule


class ReducedCertificateTests(unittest.TestCase):
    def test_m2_rule_is_exact(self):
        rule = radau_interval_rule(2)
        self.assertEqual(rule, [
            ((Fraction(1, 3), Fraction(1, 3)),
             (Fraction(3, 4), Fraction(3, 4)))
        ])

    def test_m3_intervals_enclose_nodes_and_weights(self):
        rule = radau_interval_rule(3)
        self.assertEqual(len(rule), 2)
        numeric = [
            ((4.0 - sqrt(6.0)) / 10.0, (16.0 - sqrt(6.0)) / 36.0),
            ((4.0 + sqrt(6.0)) / 10.0, (16.0 + sqrt(6.0)) / 36.0),
        ]
        for ((t_lower, t_upper), (w_lower, w_upper)), (t, weight) in zip(rule, numeric):
            self.assertLessEqual(float(t_lower) - 1e-15, t)
            self.assertGreaterEqual(float(t_upper) + 1e-15, t)
            self.assertLessEqual(float(w_lower) - 1e-15, weight)
            self.assertGreaterEqual(float(w_upper) + 1e-15, weight)

    def test_m4_intervals_enclose_all_internal_nodes_and_weights(self):
        rule = radau_interval_rule(4)
        numeric = [
            (0.08858795951270395, 0.22046221117676834),
            (0.4094668644407347, 0.38819346884317185),
            (0.7876594617608471, 0.3288443199800599),
        ]
        self.assertEqual(len(rule), 3)
        for ((t_lower, t_upper), (w_lower, w_upper)), (t, weight) in zip(rule, numeric):
            self.assertLessEqual(float(t_lower) - 1e-15, t)
            self.assertGreaterEqual(float(t_upper) + 1e-15, t)
            self.assertLessEqual(float(w_lower) - 1e-15, weight)
            self.assertGreaterEqual(float(w_upper) + 1e-15, weight)
        for degree in range(7):
            lower = sum(w[0] * t[0] ** degree for t, w in rule) + Fraction(1, 16)
            upper = sum(w[1] * t[1] ** degree for t, w in rule) + Fraction(1, 16)
            exact = Fraction(1, degree + 1)
            self.assertLessEqual(lower, exact)
            self.assertGreaterEqual(upper, exact)

    def test_unsupported_order_is_explicit(self):
        with self.assertRaises(ValueError):
            radau_interval_rule(5)


if __name__ == "__main__":
    unittest.main()
