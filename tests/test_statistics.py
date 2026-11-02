from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routed_dicka.protocols import PROTOCOLS  # noqa: E402
from routed_dicka.statistics import (  # noqa: E402
    coherent_coordinates,
    mwrf_coordinate_sets,
    protocol_coordinates,
    validate_mwrf_lattice,
)


class CoordinateTests(unittest.TestCase):
    def test_protocols_have_the_intended_far_settings(self) -> None:
        self.assertEqual(PROTOCOLS["rPCHSH"].bob_far_settings, ("Z", "T0", "T1"))
        self.assertEqual(PROTOCOLS["rPhase"].bob_far_settings, ("Z", "X"))
        self.assertEqual(PROTOCOLS["rHybrid"].bob_far_settings, ("Z", "X", "T0", "T1"))

    def test_protocol_information_is_nested_in_hybrid(self) -> None:
        hybrid = protocol_coordinates("rHybrid", recording="ternary", include_near=True)
        for protocol in ("rPCHSH", "rPhase"):
            with self.subTest(protocol=protocol):
                coordinates = protocol_coordinates(
                    protocol, recording="ternary", include_near=True
                )
                self.assertLess(coordinates, hybrid)

    def test_mwrf_lattice_and_counts(self) -> None:
        sets = mwrf_coordinate_sets()
        report = validate_mwrf_lattice(sets)
        self.assertEqual(report["counts"], {"M": 89, "W": 92, "R": 146, "F": 149})
        self.assertEqual(sets["W"] - sets["M"], coherent_coordinates())
        self.assertEqual(sets["F"] - sets["R"], coherent_coordinates())


if __name__ == "__main__":
    unittest.main()
