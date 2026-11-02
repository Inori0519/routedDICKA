from fractions import Fraction
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routed_dicka.protocols import TargetPoint  # noqa: E402
from routed_dicka.reduced_models import build_reduced_model  # noqa: E402
from routed_dicka.target import build_target  # noqa: E402
from routed_dicka.witness_bank import (  # noqa: E402
    combine_radau_entropy,
    evaluate_affine_witness,
)


class WitnessBankTests(unittest.TestCase):
    def setUp(self) -> None:
        behavior = build_target(
            TargetPoint(nu=0.99, eta_s=0.99, eta_b=0.8, eta_c=0.8)
        )
        self.model = build_reduced_model(behavior, "rPhase")

    def test_affine_witness_is_evaluated_on_new_target_moments(self) -> None:
        az = self.model.symbol_indices["A.Z.0"]
        witness = {
            "coefficients": [
                {"word": [], "value": "1/4"},
                {"word": [az], "value": "3/2"},
            ],
            "residual_penalty": "1/10",
            "data_radius": "0",
        }

        value = evaluate_affine_witness(witness, self.model)

        expected = Fraction(1, 4) + Fraction(3, 2) * Fraction.from_float(
            self.model.fixed_moments()[(az,)]
        ) - Fraction(1, 10)
        self.assertEqual(value, expected)

    def test_m2_entropy_combination_is_conservative(self) -> None:
        entropy = combine_radau_entropy(2, [Fraction(-9, 10)])
        floating_reference = 0.75 / ((1.0 / 3.0) * __import__("math").log(2.0)) * 0.1

        self.assertLessEqual(float(entropy), floating_reference)
        self.assertAlmostEqual(float(entropy), floating_reference, places=12)


if __name__ == "__main__":
    unittest.main()
