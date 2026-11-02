import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "verification"
if str(VERIFY) not in sys.path:
    sys.path.insert(0, str(VERIFY))

from verify_reduced_certificate import certified_entropy, verify_node_certificate


class ReducedCertificateVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads(
            (ROOT / "config" / "paper_suite.json").read_text(encoding="utf-8")
        )
        cls.certificate = (ROOT / config["external_archives"]["v21_routed_certificate"]).resolve()
        cls.archived_affine = (ROOT / config["external_archives"]["v21_routed_affine"]).resolve()

    def test_generic_checker_reproduces_archived_affine(self):
        with tempfile.TemporaryDirectory() as directory:
            affine = Path(directory) / "verified.affine.json"
            result = verify_node_certificate(self.certificate, affine_export=affine)
            self.assertTrue(result["verified"])
            generated = json.loads(affine.read_text(encoding="utf-8"))
            archived = json.loads(self.archived_affine.read_text(encoding="utf-8"))
            for key in ("node", "alpha", "coefficients", "residual_penalty"):
                self.assertEqual(generated[key], archived[key])

    def test_certified_entropy_uses_verified_node_not_solver_status(self):
        result = certified_entropy([self.certificate], m=2)
        self.assertTrue(result["verified"])
        self.assertGreater(result["entropy_lower"], 0.29)
        self.assertLess(result["entropy_lower"], 0.31)


if __name__ == "__main__":
    unittest.main()
