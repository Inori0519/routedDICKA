from fractions import Fraction
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main"))

from export_supplement_tables import _fraction_tex


class SupplementTableTests(unittest.TestCase):
    def test_single_line_exact_values_survive_table_formatting(self):
        values = (
            "0", "1", "-3", "7/8", "-17/1024", "39/800000",
            "500622691152273507/288230376151711744",
            "429397180060949/216172782113783808",
        )
        for value in values:
            with self.subTest(value=value):
                rendered = _fraction_tex(value)
                self.assertNotIn(r"\frac", rendered)
                match = re.fullmatch(r"(-?\d+)\\,2\^\{-(\d+)\}", rendered)
                reconstructed = (
                    Fraction(int(match[1]), 2 ** int(match[2]))
                    if match else Fraction(rendered)
                )
                self.assertEqual(reconstructed, Fraction(value))


if __name__ == "__main__":
    unittest.main()
