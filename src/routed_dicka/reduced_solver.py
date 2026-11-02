"""Sparse MOSEK backend for the reduced real-part BFF relaxations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil, sqrt
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from .reduced_algebra import Word, add_term, bff_polynomials
from .reduced_models import ReducedModelData


@dataclass(frozen=True)
class MomentBlock:
    size: int
    entries: list[dict[Word, int]]
    scale: int = 1
    label: str = "moment"


class ReducedNPA:
    """Sparse Level-2 BFF relaxation for one reduced protocol data set."""

    def __init__(
        self,
        model: ReducedModelData,
        *,
        t: float,
        features: Sequence[str] | None = None,
        level: int = 2,
        enrich: bool = True,
        localizers: bool = True,
    ) -> None:
        if not 0.0 < float(t) < 1.0:
            raise ValueError("A BFF quadrature node must lie in (0,1)")
        self.model = model
        self.algebra = model.algebra
        self.t = float(t)
        self.features = frozenset(features or model.required_features)
        if not self.features <= model.required_features:
            raise ValueError("Reduced SDP requests unavailable protocol features")
        self.alpha = int(ceil(1.5 * max(1.0 / self.t, 1.0 / (1.0 - self.t))))
        self.fixed = model.fixed_moments(self.features)
        self.alice_key_projector = model.alice_key_projector
        self.eve_operators = model.eve_operators

        basis = [
            word
            for word in self.algebra.basis(level)
            if self.algebra.z_degree(word) <= 1
        ]
        if enrich:
            extra: set[Word] = {
                word for word in self.fixed if len(word) >= 3
            }
            alice = [
                index
                for index, symbol in enumerate(self.algebra.symbols)
                if symbol.party == "A"
            ]
            receivers = [
                index
                for index, symbol in enumerate(self.algebra.symbols)
                if symbol.party == "L"
            ]
            z_terms = [
                operator
                for z in self.eve_operators
                for operator in (z, self.algebra.star[z])
            ]
            for a, receiver, z in product(alice, receivers, z_terms):
                word = self.algebra.word((a, receiver, z))
                if word is not None:
                    extra.add(word)
            basis = sorted(set(basis) | extra, key=lambda word: (len(word), word))
        self.word_basis = basis

        def gram_entry(left: Word, right: Word) -> dict[Word, int]:
            word = self.algebra.moment(self.algebra.adjoint(left) + right)
            return {} if word is None else {word: 1}

        blocks = [
            MomentBlock(
                len(basis),
                [
                    gram_entry(basis[row], basis[column])
                    for column in range(len(basis))
                    for row in range(column, len(basis))
                ],
            )
        ]
        if localizers:
            localizer_basis = [()] + [
                (index,)
                for index, symbol in enumerate(self.algebra.symbols)
                if symbol.kind == "projector"
            ]
            for outcome, z in enumerate(self.eve_operators):
                z_dagger = self.algebra.star[z]
                for order, inner in enumerate(((z_dagger, z), (z, z_dagger))):
                    entries: list[dict[Word, int]] = []
                    for column in range(len(localizer_basis)):
                        for row in range(column, len(localizer_basis)):
                            polynomial: dict[Word, int] = {}
                            left = localizer_basis[row]
                            right = localizer_basis[column]
                            add_term(
                                polynomial,
                                self.algebra.moment(
                                    self.algebra.adjoint(left) + right
                                ),
                                self.alpha**2,
                            )
                            add_term(
                                polynomial,
                                self.algebra.moment(
                                    self.algebra.adjoint(left) + inner + right
                                ),
                                -1,
                            )
                            entries.append(polynomial)
                    blocks.append(
                        MomentBlock(
                            len(localizer_basis),
                            entries,
                            self.alpha**2,
                            f"norm-{outcome}-{order}",
                        )
                    )
        self.blocks = blocks

        self.p0, self.p1 = bff_polynomials(
            self.algebra,
            self.alice_key_projector,
            self.eve_operators,
        )
        moment_set = set(self.p0) | set(self.p1) | set(self.fixed)
        for block in blocks:
            for polynomial in block.entries:
                moment_set.update(polynomial)
        self.moments = sorted(moment_set, key=lambda word: (len(word), word))
        self.free = [word for word in self.moments if word not in self.fixed]
        free_index = {word: index for index, word in enumerate(self.free)}

        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        rhs: list[float] = []
        row_index = 0
        self.block_slices: list[tuple[int, int]] = []
        for block in blocks:
            start = row_index
            packed_index = 0
            for column in range(block.size):
                for row in range(column, block.size):
                    factor = (1.0 if row == column else sqrt(2.0)) / block.scale
                    fixed_value = 0.0
                    for word, coefficient in block.entries[packed_index].items():
                        if word in self.fixed:
                            fixed_value += factor * coefficient * self.fixed[word]
                        else:
                            rows.append(row_index)
                            columns.append(free_index[word])
                            values.append(-factor * coefficient)
                    rhs.append(fixed_value)
                    row_index += 1
                    packed_index += 1
            self.block_slices.append((start, row_index))
        self.matrix = sp.csc_matrix(
            (values, (rows, columns)),
            shape=(row_index, len(self.free)),
        )
        self.rhs = np.asarray(rhs, dtype=float)
        self.objective = np.asarray(
            [self.p0.get(word, 0) + self.t * self.p1.get(word, 0) for word in self.free],
            dtype=float,
        )
        self.objective_offset = float(
            sum(
                (self.p0.get(word, 0) + self.t * self.p1.get(word, 0)) * value
                for word, value in self.fixed.items()
            )
        )
        self.solution: dict[str, object] | None = None
        self.metadata = {
            "protocol": model.protocol,
            "statistics_recording": model.statistics_recording,
            "required_features": sorted(self.features),
            "t": self.t,
            "alpha": self.alpha,
            "word_count": len(self.word_basis),
            "variables": len(self.free),
            "moments": len(self.moments),
            "fixed_moments": len(self.fixed),
            "matrix_shape": list(self.matrix.shape),
            "matrix_nonzeros": int(self.matrix.nnz),
            "blocks": [block.size for block in blocks],
            "enriched": enrich,
            "localizers": localizers,
        }

    def solve(
        self,
        *,
        parameters: Mapping[str, int | float] | None = None,
        verbose: bool = False,
    ) -> dict[str, object]:
        self.solution = solve_free_sdp(
            self.matrix,
            self.rhs,
            self.objective,
            block_sizes=[block.size for block in self.blocks],
            objective_offset=self.objective_offset,
            parameters=parameters,
            verbose=verbose,
        )
        return self.solution

    def dual_grams(self) -> list[np.ndarray]:
        if self.solution is None:
            raise RuntimeError("The reduced SDP has not been solved")
        dual = self.solution["dual"]
        result = []
        for block, (start, stop) in zip(self.blocks, self.block_slices):
            packed = dual[start:stop]
            gram = np.zeros((block.size, block.size), dtype=float)
            index = 0
            for column in range(block.size):
                for row in range(column, block.size):
                    value = packed[index] / (1.0 if row == column else sqrt(2.0))
                    gram[row, column] = gram[column, row] = value
                    index += 1
            result.append(gram)
        return result


def _stream_printer(message: str) -> None:
    print(message, end="")


def solve_free_sdp(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    objective: np.ndarray,
    *,
    block_sizes: Sequence[int],
    objective_offset: float = 0.0,
    parameters: Mapping[str, int | float] | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    """Solve ``min c.x`` with ``A x + svec(X) = b`` and PSD blocks ``X``.

    The packed cone convention matches the V2.1 certificate generator: diagonal
    entries are unscaled and off-diagonal entries are multiplied by ``sqrt(2)``.
    MOSEK's equality multipliers have the opposite sign from the conic dual used
    by that generator, so the returned ``dual`` vector is negated explicitly.
    """

    import mosek

    matrix = sp.coo_matrix(matrix, dtype=np.float64)
    matrix.sum_duplicates()
    rhs = np.ascontiguousarray(rhs, dtype=np.float64)
    objective = np.ascontiguousarray(objective, dtype=np.float64)
    block_sizes = [int(size) for size in block_sizes]
    rows, columns = matrix.shape
    if rhs.shape != (rows,):
        raise ValueError("The right-hand side length does not match A")
    if objective.shape != (columns,):
        raise ValueError("The objective length does not match A")
    packed = sum(size * (size + 1) // 2 for size in block_sizes)
    if packed != rows:
        raise ValueError(f"PSD blocks contain {packed} packed rows, expected {rows}")

    with mosek.Env() as environment:
        with environment.Task(0, 0) as task:
            if verbose:
                task.set_Stream(mosek.streamtype.log, _stream_printer)
            task.appendvars(columns)
            task.appendcons(rows)
            task.appendbarvars(block_sizes)
            if columns:
                task.putvarboundsliceconst(
                    0,
                    columns,
                    mosek.boundkey.fr,
                    0.0,
                    0.0,
                )
                task.putclist(list(range(columns)), objective.tolist())
            task.putcfix(float(objective_offset))
            task.putconboundslice(
                0,
                rows,
                [mosek.boundkey.fx] * rows,
                rhs.tolist(),
                rhs.tolist(),
            )
            if matrix.nnz:
                task.putaijlist(
                    matrix.row.astype(np.int32).tolist(),
                    matrix.col.astype(np.int32).tolist(),
                    matrix.data.tolist(),
                )

            constraint_indices: list[int] = []
            block_indices: list[int] = []
            row_indices: list[int] = []
            column_indices: list[int] = []
            values: list[float] = []
            constraint = 0
            for block_index, size in enumerate(block_sizes):
                for column in range(size):
                    for row in range(column, size):
                        constraint_indices.append(constraint)
                        block_indices.append(block_index)
                        row_indices.append(row)
                        column_indices.append(column)
                        values.append(1.0 if row == column else 1.0 / sqrt(2.0))
                        constraint += 1
            task.putbarablocktriplet(
                constraint_indices,
                block_indices,
                row_indices,
                column_indices,
                values,
            )
            task.putobjsense(mosek.objsense.minimize)

            settings = dict(parameters or {})
            double_parameters = {
                "relative_gap_tolerance": mosek.dparam.intpnt_co_tol_rel_gap,
                "primal_feasibility_tolerance": mosek.dparam.intpnt_co_tol_pfeas,
                "dual_feasibility_tolerance": mosek.dparam.intpnt_co_tol_dfeas,
                "max_time_seconds": mosek.dparam.optimizer_max_time,
            }
            integer_parameters = {
                "max_iterations": mosek.iparam.intpnt_max_iterations,
                "threads": mosek.iparam.num_threads,
            }
            for name, parameter in double_parameters.items():
                if name in settings and float(settings[name]) > 0:
                    task.putdouparam(parameter, float(settings[name]))
            for name, parameter in integer_parameters.items():
                if name in settings and int(settings[name]) > 0:
                    task.putintparam(parameter, int(settings[name]))

            task.optimize()
            solution_type = mosek.soltype.itr
            status = task.getsolsta(solution_type)
            problem_status = task.getprosta(solution_type)
            x = [0.0] * columns
            dual_mosek = [0.0] * rows
            if columns:
                task.getxx(solution_type, x)
            task.gety(solution_type, dual_mosek)
            dual = [-value for value in dual_mosek]
            primal_objective = float(task.getprimalobj(solution_type))
            dual_objective = float(task.getdualobj(solution_type))
            relative_gap = abs(primal_objective - dual_objective) / (
                1.0 + abs(primal_objective) + abs(dual_objective)
            )
            return {
                "backend": "mosek-low-level-sdp",
                "mosek_version": ".".join(map(str, mosek.Env.getversion())),
                "status": str(status),
                "problem_status": str(problem_status),
                "status_optimal": status == mosek.solsta.optimal,
                "x": x,
                "dual": dual,
                "primal_objective": primal_objective,
                "dual_objective": dual_objective,
                "relative_gap": relative_gap,
                "primal_feasibility": float(
                    task.getdouinf(mosek.dinfitem.intpnt_primal_feas)
                ),
                "dual_feasibility": float(
                    task.getdouinf(mosek.dinfitem.intpnt_dual_feas)
                ),
                "iterations": int(task.getintinf(mosek.iinfitem.intpnt_iter)),
                "solve_seconds": float(task.getdouinf(mosek.dinfitem.intpnt_time)),
            }
