from math import sqrt
from pathlib import Path
import random
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routed_dicka.protocols import TargetPoint
from routed_dicka.reduced_models import build_reduced_model, protocol_features
from routed_dicka.reduced_solver import ReducedNPA
from routed_dicka.target import build_target


class ReducedModelTests(unittest.TestCase):
    def test_protocol_features_and_operator_sets_are_cropped(self):
        expected = {
            "rPCHSH": frozenset({"near", "tilted"}),
            "rPhase": frozenset({"near", "phase"}),
            "rHybrid": frozenset({"near", "phase", "tilted"}),
        }
        point = TargetPoint(nu=0.99, eta_s=0.99, eta_b=0.8, eta_c=0.8)
        behavior = build_target(point)
        for protocol, features in expected.items():
            self.assertEqual(protocol_features(protocol), features)
            model = build_reduced_model(behavior, protocol)
            self.assertEqual(model.required_features, features)
            labels = {symbol.name for symbol in model.algebra.symbols}
            self.assertEqual("L.phase.0" in labels, "phase" in features)
            self.assertEqual("L.test0.0" in labels, "tilted" in features)
        self.assertLess(
            len(build_reduced_model(behavior, "rPhase").algebra.symbols),
            len(build_reduced_model(behavior, "rHybrid").algebra.symbols),
        )

    def test_long_only_models_remove_every_near_constraint(self):
        first = build_target(
            TargetPoint(nu=0.99, eta_s=0.97, eta_b=0.8, eta_c=0.8)
        )
        second = build_target(
            TargetPoint(nu=0.99, eta_s=1.0, eta_b=0.8, eta_c=0.8)
        )
        for protocol in ("rPCHSH", "rPhase", "rHybrid"):
            model = build_reduced_model(first, protocol, include_near=False)
            reference = build_reduced_model(second, protocol, include_near=False)
            self.assertNotIn("near", model.required_features)
            self.assertNotIn("near", model.groups)
            self.assertFalse(any(symbol.name.startswith("S.") for symbol in model.algebra.symbols))
            self.assertEqual(model.fixed_moments(), reference.fixed_moments())
            self.assertFalse(model.include_near)

    def test_far_observations_are_a_restriction_of_the_routed_model(self):
        for nu in (1.0, 0.99):
            behavior = build_target(
                TargetPoint(nu=nu, eta_s=0.97, eta_b=0.98, eta_c=0.98)
            )
            for protocol in ("rPCHSH", "rPhase", "rHybrid"):
                routed = build_reduced_model(behavior, protocol, include_near=True)
                far = build_reduced_model(behavior, protocol, include_near=False)
                indices = {
                    index: routed.symbol_indices[symbol.name]
                    for index, symbol in enumerate(far.algebra.symbols)
                }
                available = routed.fixed_moments()
                for word, value in far.fixed_moments().items():
                    renamed = routed.algebra.moment(tuple(indices[index] for index in word))
                    self.assertIn(renamed, available)
                    self.assertAlmostEqual(available[renamed], value, places=12)

    def test_full_probability_reduction_matches_closed_form_targets(self):
        rng = random.Random(28263)
        for _ in range(12):
            nu = rng.uniform(0.8, 1.0)
            eta_s = rng.uniform(0.8, 1.0)
            eta_b = rng.uniform(0.75, 1.0)
            eta_c = rng.uniform(0.75, 1.0)
            point = TargetPoint(nu=nu, eta_s=eta_s, eta_b=eta_b, eta_c=eta_c)
            model = build_reduced_model(build_target(point), "rHybrid")
            indices = model.symbol_indices
            groups = model.groups

            near_norm = sqrt(1.0 + eta_s * eta_s)
            for setting in range(2):
                symbol = indices[f"S.{setting}.0"]
                sign = 1.0 if setting == 0 else -1.0
                marginal = 1.0 - eta_s / 2.0
                self.assertAlmostEqual(groups["near"][(symbol,)], marginal, places=12)
                self.assertAlmostEqual(
                    groups["near"][model.algebra.moment((indices["A.Z.0"], symbol))],
                    marginal / 2.0 + nu * eta_s / (4.0 * near_norm),
                    places=12,
                )
                self.assertAlmostEqual(
                    groups["near"][model.algebra.moment((indices["A.X.0"], symbol))],
                    marginal / 2.0 + sign * nu * eta_s**3 / (4.0 * near_norm),
                    places=12,
                )

            joint_detection = eta_b * eta_c
            for outcome in range(2):
                symbol = indices[f"L.phase.{outcome}"]
                sign = 1.0 if outcome == 0 else -1.0
                self.assertAlmostEqual(
                    groups["phase"][(symbol,)], joint_detection / 2.0, places=12
                )
                self.assertAlmostEqual(
                    groups["phase"][model.algebra.moment((indices["A.Z.0"], symbol))],
                    joint_detection / 4.0,
                    places=12,
                )
                self.assertAlmostEqual(
                    groups["phase"][model.algebra.moment((indices["A.X.0"], symbol))],
                    joint_detection * (1.0 + sign * nu) / 4.0,
                    places=12,
                )

            far_norm = sqrt(1.0 + eta_c * eta_c)
            for setting in range(2):
                symbol = indices[f"L.test{setting}.0"]
                sign = 1.0 if setting == 0 else -1.0
                marginal = 1.0 - eta_b / 2.0
                self.assertAlmostEqual(groups["tilted"][(symbol,)], marginal, places=12)
                self.assertAlmostEqual(
                    groups["tilted"][model.algebra.moment((indices["A.Z.0"], symbol))],
                    marginal / 2.0 + nu * eta_b / (4.0 * far_norm),
                    places=12,
                )
                self.assertAlmostEqual(
                    groups["tilted"][model.algebra.moment((indices["A.X.0"], symbol))],
                    marginal / 2.0
                    + sign * nu * eta_b * eta_c**2 / (4.0 * far_norm),
                    places=12,
                )

    def test_fixed_moments_include_only_protocol_features(self):
        behavior = build_target(
            TargetPoint(nu=0.99, eta_s=0.99, eta_b=0.8, eta_c=0.8)
        )
        phase = build_reduced_model(behavior, "rPhase")
        pchsh = build_reduced_model(behavior, "rPCHSH")
        self.assertEqual(set(phase.groups), {"alice", "near", "phase"})
        self.assertEqual(set(pchsh.groups), {"alice", "near", "tilted"})
        self.assertEqual(phase.fixed_moments(), {
            **phase.groups["alice"],
            **phase.groups["near"],
            **phase.groups["phase"],
        })

    def test_reduced_relaxations_are_protocol_sized_and_use_localizers(self):
        behavior = build_target(
            TargetPoint(nu=0.99, eta_s=0.99, eta_b=0.8, eta_c=0.8)
        )
        phase = ReducedNPA(build_reduced_model(behavior, "rPhase"), t=1.0 / 3.0)
        pchsh = ReducedNPA(build_reduced_model(behavior, "rPCHSH"), t=1.0 / 3.0)
        hybrid = ReducedNPA(build_reduced_model(behavior, "rHybrid"), t=1.0 / 3.0)
        self.assertLess(phase.metadata["variables"], hybrid.metadata["variables"])
        self.assertLess(pchsh.metadata["variables"], hybrid.metadata["variables"])
        for problem in (phase, pchsh, hybrid):
            self.assertEqual(len(problem.blocks), 5)
            self.assertEqual(problem.blocks[0].label, "moment")
            self.assertTrue(all(block.scale == problem.alpha**2 for block in problem.blocks[1:]))
            self.assertEqual(problem.matrix.shape[0], sum(
                block.size * (block.size + 1) // 2 for block in problem.blocks
            ))

    def test_binary_phase_statistics_bin_no_clicks_without_changing_ec(self):
        behavior = build_target(
            TargetPoint(nu=0.99, eta_s=0.99, eta_b=0.8, eta_c=0.8)
        )
        ternary = build_reduced_model(
            behavior, "rPhase", statistics_recording="ternary"
        )
        binary = build_reduced_model(
            behavior, "rPhase", statistics_recording="binary"
        )
        ternary_mass = sum(
            ternary.groups["phase"][(ternary.symbol_indices[f"L.phase.{outcome}"],)]
            for outcome in range(2)
        )
        binary_mass = sum(
            binary.groups["phase"][(binary.symbol_indices[f"L.phase.{outcome}"],)]
            for outcome in range(2)
        )
        self.assertAlmostEqual(ternary_mass, 0.8**2, places=12)
        self.assertAlmostEqual(binary_mass, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
