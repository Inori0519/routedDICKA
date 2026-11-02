import math
from pathlib import Path
import sys
import unittest

import numpy as np
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import mosek  # noqa: F401
except ImportError:
    mosek = None

from routed_dicka.reduced_solver import solve_free_sdp


@unittest.skipIf(mosek is None, "MOSEK is not installed")
class MosekConicTests(unittest.TestCase):
    def test_two_by_two_psd_epigraph_and_dual_scaling(self):
        # min x subject to [[x, 1], [1, x]] >= 0.
        matrix = sp.csc_matrix([[-1.0], [0.0], [-1.0]])
        result = solve_free_sdp(
            matrix,
            np.asarray([0.0, math.sqrt(2.0), 0.0]),
            np.asarray([1.0]),
            block_sizes=[2],
        )
        self.assertTrue(result["status_optimal"])
        self.assertAlmostEqual(result["x"][0], 1.0, places=7)
        self.assertAlmostEqual(result["primal_objective"], 1.0, places=7)
        self.assertAlmostEqual(result["dual_objective"], 1.0, places=7)

        dual = result["dual"]
        gram = np.array(
            [[dual[0], dual[1] / math.sqrt(2.0)],
             [dual[1] / math.sqrt(2.0), dual[2]]]
        )
        self.assertGreaterEqual(np.linalg.eigvalsh(gram).min(), -1e-7)


if __name__ == "__main__":
    unittest.main()
