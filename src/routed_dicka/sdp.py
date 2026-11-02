"""BFF/NPA relaxation for the unified routed model, solved with MOSEK."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite
import time
from typing import Iterable, Mapping

import ncpol2sdpa as ncp
import numpy as np
from sympy.physics.quantum.dagger import Dagger

from .protocols import (
    ALICE_SETTINGS,
    FAR_BOB_SETTINGS,
    FAR_CHARLIE_SETTINGS,
    NEAR_BOB_SETTINGS,
    NEAR_CHARLIE_SETTINGS,
)
from .quadrature import RadauNode, bff_coefficient, right_radau_rule
from .statistics import Coordinate, Factor, coordinate_value
from .target import TargetBehavior


def _flatten(measurements: Iterable[object]) -> list[object]:
    return list(ncp.flatten(measurements))


def _projectors(setting: Iterable[object]) -> tuple[object, ...]:
    explicit = tuple(setting)
    return (*explicit, 1 - sum(explicit))


def _add_commutation(
    substitutions: dict[object, object],
    left: Iterable[object],
    right: Iterable[object],
) -> None:
    for left_op in left:
        for right_op in right:
            substitutions[right_op * left_op] = left_op * right_op


@dataclass(frozen=True)
class SolverSettings:
    level: int = 2
    relative_gap_tolerance: float = 1e-7
    primal_feasibility_tolerance: float = 1e-7
    dual_feasibility_tolerance: float = 1e-7
    max_iterations: int = 500
    threads: int = 0
    numerical_guard: float = 1e-7
    max_accepted_primal_dual_gap: float = 5e-5

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
        level: int | None = None,
    ) -> "SolverSettings":
        return cls(
            level=int(values.get("level", 2) if level is None else level),
            relative_gap_tolerance=float(values.get("relative_gap_tolerance", 1e-7)),
            primal_feasibility_tolerance=float(
                values.get("primal_feasibility_tolerance", 1e-7)
            ),
            dual_feasibility_tolerance=float(
                values.get("dual_feasibility_tolerance", 1e-7)
            ),
            max_iterations=int(values.get("max_iterations", 500)),
            threads=int(values.get("threads", 0)),
            numerical_guard=float(values.get("numerical_guard", 1e-7)),
            max_accepted_primal_dual_gap=float(
                values.get("max_accepted_primal_dual_gap", 5e-5)
            ),
        )

    def mosek_parameters(self) -> dict[str, int | float]:
        parameters: dict[str, int | float] = {
            "dparam.intpnt_co_tol_rel_gap": self.relative_gap_tolerance,
            "dparam.intpnt_co_tol_pfeas": self.primal_feasibility_tolerance,
            "dparam.intpnt_co_tol_dfeas": self.dual_feasibility_tolerance,
            "iparam.intpnt_max_iterations": self.max_iterations,
        }
        if self.threads > 0:
            parameters["iparam.num_threads"] = self.threads
        return parameters


class RoutedOperatorModel:
    """One operator model shared by all protocols and information views."""

    def __init__(self) -> None:
        self.A = ncp.generate_measurements([2, 2], "A")
        self.BS = ncp.generate_measurements([2, 2], "BS")
        self.CS = ncp.generate_measurements([2], "CS")
        self.BL = ncp.generate_measurements([3, 3, 3, 3], "BL")
        self.CL = ncp.generate_measurements([3, 3], "CL")
        self.Z = ncp.generate_operators("E", 2, hermitian=False)

        self._families = {
            "A": (ALICE_SETTINGS, self.A),
            "BS": (NEAR_BOB_SETTINGS, self.BS),
            "CS": (NEAR_CHARLIE_SETTINGS, self.CS),
            "BL": (FAR_BOB_SETTINGS, self.BL),
            "CL": (FAR_CHARLIE_SETTINGS, self.CL),
        }
        self._projector_map: dict[tuple[str, str], tuple[object, ...]] = {}
        for party, (labels, measurements) in self._families.items():
            for label, setting in zip(labels, measurements):
                self._projector_map[(party, label)] = _projectors(setting)

        self.substitutions = self._build_substitutions()
        self.extra_monomials = self._build_extra_monomials()
        self.operators = _flatten([self.A, self.BS, self.CS, self.BL, self.CL]) + list(self.Z)

    def factor_expression(self, factor: Factor) -> object:
        projectors = self._projector_map[(factor.party, factor.setting)]
        if factor.component == "V":
            return projectors[0] - projectors[1]
        if factor.component == "D":
            if len(projectors) != 3:
                raise ValueError(f"D is only defined for a ternary receiver: {factor}")
            return projectors[0] + projectors[1]
        if factor.component == "BIN":
            if len(projectors) != 3:
                raise ValueError(f"BIN is only defined for a ternary receiver: {factor}")
            return projectors[0] - projectors[1] + projectors[2]
        raise ValueError(f"Unknown factor component: {factor.component}")

    def coordinate_expression(self, coordinate: Coordinate) -> object:
        expression: object = 1
        for factor in coordinate.factors:
            expression *= self.factor_expression(factor)
        return expression

    def key_projectors(self) -> tuple[object, object]:
        projectors = self._projector_map[("A", "Z")]
        return projectors[0], projectors[1]

    def _build_substitutions(self) -> dict[object, object]:
        substitutions: dict[object, object] = {}
        for family in (self.A, self.BS, self.CS, self.BL, self.CL):
            substitutions.update(ncp.projective_measurement_constraints(family))

        a_flat = _flatten(self.A)
        bs_flat = _flatten(self.BS)
        cs_flat = _flatten(self.CS)
        bl_flat = _flatten(self.BL)
        cl_flat = _flatten(self.CL)

        for family in (bs_flat, cs_flat, bl_flat, cl_flat):
            _add_commutation(substitutions, a_flat, family)
        _add_commutation(substitutions, bs_flat, cs_flat)
        _add_commutation(substitutions, bl_flat, cl_flat)

        for z in self.Z:
            for operator in [*a_flat, *bl_flat, *cl_flat]:
                substitutions[z * operator] = operator * z
                substitutions[Dagger(z) * operator] = operator * Dagger(z)
        return substitutions

    def _build_extra_monomials(self) -> list[object]:
        """Add the tripartite data and BFF couplings missing at base Level 2.

        Each route is enriched independently.  In particular, no near/far
        product and no four-factor ABCZ word is introduced.  The A-measurement-Z
        products make the cubic BFF terms visible as Gram entries, while the ABC
        words carry the observed three-party coordinates.
        """

        extras: list[object] = []
        seen: set[object] = set()

        def add(expression: object) -> None:
            if expression != 1 and expression not in seen:
                seen.add(expression)
                extras.append(expression)

        a_flat = _flatten(self.A)
        route_families = [
            (_flatten(self.BS), _flatten(self.CS)),
            (_flatten(self.BL), _flatten(self.CL)),
        ]
        z_terms = [1, *self.Z, *(Dagger(z) for z in self.Z)]

        for bob_flat, charlie_flat in route_families:
            for a_op, b_op, c_op in product(a_flat, bob_flat, charlie_flat):
                add(a_op * b_op * c_op)
            measurement_terms = [1, *bob_flat, *charlie_flat]
            for a_op, measurement_op, z_op in product(
                [1, *a_flat],
                measurement_terms,
                z_terms,
            ):
                add(a_op * measurement_op * z_op)
        return extras

    def build_relaxation(
        self,
        behavior: TargetBehavior,
        coordinates: Iterable[Coordinate],
        t: float,
        level: int,
    ) -> ncp.SdpRelaxation:
        equalities = [
            self.coordinate_expression(coordinate)
            - coordinate_value(behavior, coordinate)
            for coordinate in sorted(set(coordinates), key=lambda item: item.name)
        ]
        key_projectors = self.key_projectors()

        objective = 0
        for outcome, z in enumerate(self.Z):
            key_effect = key_projectors[outcome]
            objective += key_effect * (
                z + Dagger(z) + (1.0 - t) * Dagger(z) * z
            ) + t * z * Dagger(z)

        alpha = 1.5 * max(1.0 / t, 1.0 / (1.0 - t))
        alpha_squared = alpha**2
        inequalities = []
        for z in self.Z:
            inequalities.append(alpha_squared - z * Dagger(z))
            inequalities.append(alpha_squared - Dagger(z) * z)

        relaxation = ncp.SdpRelaxation(self.operators, verbose=False, normalized=True)
        relaxation.get_relaxation(
            level=level,
            equalities=[],
            inequalities=inequalities,
            momentequalities=equalities,
            momentinequalities=[],
            substitutions=self.substitutions,
            extramonomials=self.extra_monomials,
            objective=objective,
        )
        return relaxation


def _optimal_status(status: object) -> bool:
    normalized = str(status).lower()
    return "optimal" in normalized and "unknown" not in normalized


def solve_bff_node(
    model: RoutedOperatorModel,
    behavior: TargetBehavior,
    coordinates: set[Coordinate],
    node: RadauNode,
    settings: SolverSettings,
) -> dict[str, object]:
    if node.endpoint:
        raise ValueError("The t=1 Radau endpoint is not an SDP node")
    started = time.perf_counter()
    relaxation = model.build_relaxation(
        behavior=behavior,
        coordinates=coordinates,
        t=node.t,
        level=settings.level,
    )
    build_seconds = time.perf_counter() - started
    solve_started = time.perf_counter()
    relaxation.solve("mosek", solverparameters=settings.mosek_parameters())
    solve_seconds = time.perf_counter() - solve_started

    primal = float(relaxation.primal)
    dual = float(relaxation.dual)
    gap = primal - dual
    guard = settings.numerical_guard * (1.0 + abs(dual))
    guarded_dual = dual - guard
    status = str(relaxation.status)
    status_optimal = _optimal_status(status)
    finite = all(isfinite(value) for value in (primal, dual, guarded_dual, gap))
    gap_accepted = gap >= -settings.max_accepted_primal_dual_gap and abs(gap) <= settings.max_accepted_primal_dual_gap
    claim_eligible = status_optimal and finite and gap_accepted

    return {
        "index": node.index,
        "t": node.t,
        "weight": node.weight,
        "coefficient": bff_coefficient(node),
        "alpha": 1.5 * max(1.0 / node.t, 1.0 / (1.0 - node.t)),
        "status": status,
        "status_optimal": status_optimal,
        "primal": primal,
        "dual": dual,
        "numerical_guard": guard,
        "guarded_dual": guarded_dual,
        "primal_dual_gap": gap,
        "gap_accepted": gap_accepted,
        "claim_eligible": claim_eligible,
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "block_structure": [int(value) for value in relaxation.block_struct],
        "n_variables": int(relaxation.n_vars),
    }


def entropy_lower_bound(
    behavior: TargetBehavior,
    coordinates: set[Coordinate],
    m: int,
    settings: SolverSettings,
) -> dict[str, object]:
    rule = right_radau_rule(m)
    model = RoutedOperatorModel()
    node_results = [
        solve_bff_node(model, behavior, coordinates, node, settings)
        for node in rule
        if not node.endpoint
    ]
    raw_entropy = sum(
        float(node["coefficient"]) * (1.0 + float(node["guarded_dual"]))
        for node in node_results
    )
    entropy = min(1.0, max(0.0, raw_entropy))
    return {
        "m": m,
        "radau_rule": [node.as_dict() for node in rule],
        "nodes": node_results,
        "raw_entropy_lower": raw_entropy,
        "entropy_lower": entropy,
        "all_nodes_claim_eligible": all(
            bool(node["claim_eligible"]) for node in node_results
        ),
        "total_build_seconds": sum(float(node["build_seconds"]) for node in node_results),
        "total_solve_seconds": sum(float(node["solve_seconds"]) for node in node_results),
    }
