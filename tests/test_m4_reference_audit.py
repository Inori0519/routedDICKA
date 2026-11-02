from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "verification"
if str(VERIFY) not in sys.path:
    sys.path.insert(0, str(VERIFY))

from verify_m4_reference_certificate import (
    verify_affine_matches_certificate,
    verify_reference_certificate,
)


class M4ReferenceAuditTests(unittest.TestCase):
    def test_archived_m4_family_is_positive_at_the_attack_point(self):
        audit = verify_reference_certificate(ROOT / "config" / "paper_suite.json")
        self.assertTrue(audit["passed"])
        point = audit["reference_point"]
        self.assertEqual(point["m"], 4)
        self.assertAlmostEqual(point["H_lower"], 0.3327199023192324, places=13)
        self.assertAlmostEqual(point["EC_upper"], 0.2554234600213591, places=13)
        self.assertAlmostEqual(point["K_lower"], 0.0772964422978733, places=13)
        self.assertTrue(point["positive"])
        self.assertEqual(len(audit["certificates"]), 3)

    def test_tampered_affine_is_rejected(self):
        source = ROOT / "source_data" / "certificates" / "m4"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            certificate = directory / "node_0.certificate.json"
            affine = directory / "node_0.affine.json"
            shutil.copy2(source / certificate.name, certificate)
            shutil.copy2(source / affine.name, affine)
            payload = json.loads(affine.read_text(encoding="utf-8"))
            payload["coefficients"][0]["value"] = "0"
            affine.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(AssertionError):
                verify_affine_matches_certificate(certificate, affine)


if __name__ == "__main__":
    unittest.main()
