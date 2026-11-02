"""Right Gauss-Radau rules for the BFF entropy decomposition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose

import numpy as np
from scipy.special import roots_jacobi


@dataclass(frozen=True)
class RadauNode:
    index: int
    t: float
    weight: float
    endpoint: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def right_radau_rule(m: int) -> list[RadauNode]:
    """Return the m-point rule on [0,1] whose fixed endpoint is t=1.

    The m-1 interior nodes are Gaussian nodes for the weight 1-t.  SciPy's
    Jacobi rule is defined on [-1,1], so its weights acquire a factor 1/4
    under t=(x+1)/2.  Dividing by 1-t converts the weighted Gaussian rule to
    the Radau weights.
    """

    if int(m) != m or m < 2:
        raise ValueError("m must be an integer at least 2")
    x, jacobi_weights = roots_jacobi(m - 1, 1.0, 0.0)
    t = (x + 1.0) / 2.0
    beta = jacobi_weights / 4.0
    interior_weights = beta / (1.0 - t)
    order = np.argsort(t)
    nodes = [
        RadauNode(i, float(t[j]), float(interior_weights[j]), False)
        for i, j in enumerate(order)
    ]
    endpoint_weight = 1.0 - float(np.sum(interior_weights))
    exact_endpoint_weight = 1.0 / (m * m)
    if not isclose(endpoint_weight, exact_endpoint_weight, rel_tol=2e-13, abs_tol=2e-15):
        raise ArithmeticError(
            f"Radau endpoint weight mismatch: {endpoint_weight} != {exact_endpoint_weight}"
        )
    nodes.append(RadauNode(m - 1, 1.0, exact_endpoint_weight, True))
    validate_rule(nodes)
    return nodes


def validate_rule(nodes: list[RadauNode], tolerance: float = 2e-12) -> dict[str, object]:
    m = len(nodes)
    if m < 2 or not nodes[-1].endpoint or nodes[-1].t != 1.0:
        raise ValueError("Expected a right Radau rule with endpoint t=1")
    if any(node.weight <= 0 for node in nodes):
        raise ValueError("All Radau weights must be positive")
    errors = []
    for degree in range(2 * m - 1):
        approximation = sum(node.weight * node.t**degree for node in nodes)
        exact = 1.0 / (degree + 1)
        errors.append(abs(approximation - exact))
    max_error = max(errors, default=0.0)
    if max_error > tolerance:
        raise ArithmeticError(
            f"Radau rule fails degree {2*m-2} moment check: {max_error}"
        )
    return {
        "m": m,
        "degree_checked": 2 * m - 2,
        "max_moment_error": max_error,
        "endpoint_weight": nodes[-1].weight,
    }


def bff_coefficient(node: RadauNode) -> float:
    if node.endpoint:
        raise ValueError("The BFF endpoint is not optimized")
    return node.weight / (node.t * np.log(2.0))

