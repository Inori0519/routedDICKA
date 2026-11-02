from fractions import Fraction as F
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "verification"
if str(VERIFY) not in sys.path:
    sys.path.insert(0, str(VERIFY))

from verify_v21_certificate import evaluate_reference_point, verify_certificate_bundle


class V21CertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads(
            (ROOT / "config" / "paper_suite.json").read_text(encoding="utf-8")
        )
        archives = config["external_archives"]
        cls.certificate = (ROOT / archives["v21_routed_certificate"]).resolve()
        cls.affine = (ROOT / archives["v21_routed_affine"]).resolve()

    def test_integer_gram_certificate_matches_affine_export(self):
        result = verify_certificate_bundle(self.certificate, self.affine)
        self.assertTrue(result["verified"])
        self.assertEqual(result["quadrature_order"], 2)
        self.assertEqual(result["word_count"], 127)
        self.assertTrue(result["affine_matches_certificate"])

    def test_reference_point_has_positive_guarded_rate(self):
        result = evaluate_reference_point(
            self.affine,
            nu=F(99, 100),
            eta_s=F(99, 100),
            eta_b=F(39, 50),
            eta_c=F(39, 50),
        )
        self.assertGreater(result["K_lower"], 0.0225)
        self.assertLess(result["K_lower"], 0.0227)
        self.assertGreater(result["H_lower"], result["EC_upper"])


if __name__ == "__main__":
    unittest.main()
