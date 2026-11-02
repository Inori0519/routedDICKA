"""Protocol labels and physical parameter points shared by every experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import atan, pi
from typing import Mapping


ALICE_SETTINGS = ("Z", "X")
NEAR_BOB_SETTINGS = ("T0", "T1")
NEAR_CHARLIE_SETTINGS = ("X",)
FAR_BOB_SETTINGS = ("Z", "X", "T0", "T1")
FAR_CHARLIE_SETTINGS = ("Z", "X")
KEY_SETTING = ("Z", "Z", "Z")


@dataclass(frozen=True)
class Protocol:
    identifier: str
    label: str
    bob_far_settings: tuple[str, ...]
    charlie_far_settings: tuple[str, ...] = FAR_CHARLIE_SETTINGS

    @property
    def has_phase_test(self) -> bool:
        return "X" in self.bob_far_settings

    @property
    def has_tilted_test(self) -> bool:
        return "T0" in self.bob_far_settings and "T1" in self.bob_far_settings


PROTOCOLS = {
    "rPCHSH": Protocol("rPCHSH", "rPCHSH", ("Z", "T0", "T1")),
    "rPhase": Protocol("rPhase", "rPhase", ("Z", "X")),
    "rHybrid": Protocol("rHybrid", "rHybrid", FAR_BOB_SETTINGS),
}


@dataclass(frozen=True)
class TargetPoint:
    nu: float
    eta_s: float
    eta_b: float
    eta_c: float
    eta_a: float = 1.0
    theta: float | None = None
    far_angle_policy: str = "matched"
    near_angle_policy: str = "matched"

    def __post_init__(self) -> None:
        for name in ("nu", "eta_s", "eta_b", "eta_c", "eta_a"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1], got {value}")
        if self.theta is not None and not 0.0 <= float(self.theta) <= pi / 2:
            raise ValueError("theta must lie in [0, pi/2]")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
        common: Mapping[str, object] | None = None,
    ) -> "TargetPoint":
        common = common or {}
        eta_l = float(values.get("eta_l", values.get("eta_b", 1.0)))
        eta_b = float(values.get("eta_b", eta_l))
        eta_c = float(values.get("eta_c", eta_l))
        theta_value = values.get("theta")
        return cls(
            nu=float(values.get("nu", 1.0)),
            eta_s=float(values.get("eta_s", 1.0)),
            eta_b=eta_b,
            eta_c=eta_c,
            eta_a=float(values.get("eta_a", common.get("eta_a", 1.0))),
            theta=None if theta_value is None else float(theta_value),
            far_angle_policy=str(
                values.get(
                    "far_angle_policy",
                    common.get("far_angle_policy", "matched"),
                )
            ),
            near_angle_policy=str(
                values.get(
                    "near_angle_policy",
                    common.get("near_angle_policy", "matched"),
                )
            ),
        )

    @property
    def far_theta(self) -> float:
        if self.theta is not None:
            return float(self.theta)
        if self.far_angle_policy == "matched":
            return atan(self.eta_c)
        if self.far_angle_policy == "pi_over_4":
            return pi / 4
        raise ValueError(f"Unknown far angle policy: {self.far_angle_policy}")

    @property
    def near_theta(self) -> float:
        if self.near_angle_policy == "matched":
            return atan(self.eta_s)
        if self.near_angle_policy == "pi_over_4":
            return pi / 4
        raise ValueError(f"Unknown near angle policy: {self.near_angle_policy}")

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["far_theta"] = self.far_theta
        result["near_theta"] = self.near_theta
        return result

    @property
    def slug(self) -> str:
        def token(value: float) -> str:
            return f"{value:.6g}".replace(".", "p")

        return (
            f"nu{token(self.nu)}_s{token(self.eta_s)}_"
            f"eb{token(self.eta_b)}_ec{token(self.eta_c)}_"
            f"th{token(self.far_theta)}"
        )


def get_protocol(identifier: str) -> Protocol:
    try:
        return PROTOCOLS[identifier]
    except KeyError as exc:
        raise ValueError(
            f"Unknown protocol {identifier!r}; choose from {sorted(PROTOCOLS)}"
        ) from exc

