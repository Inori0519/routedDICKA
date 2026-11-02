"""Publication figures for the routed DICKA protocol comparison."""

from __future__ import annotations

import csv
import json
from math import floor, isclose
from pathlib import Path
from typing import Callable, Iterable, Mapping

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, MaxNLocator

from .io import write_json


PROTOCOL_ORDER = ("rPCHSH", "rPhase", "rHybrid")
PROTOCOL_LABELS = {
    "rPCHSH": "rPCHSH",
    "rPhase": "rPhase",
    "rHybrid": "rHybrid",
}
PROTOCOL_COLORS = {
    "rPCHSH": "#2878AD",
    "rPhase": "#159581",
    "rHybrid": "#DC7826",
}
EFFICIENCY_COLORS = {1.0: "#DC7826", 0.99: "#2878AD", 0.98: "#159581", 0.97: "#99669A"}
PROTOCOL_MARKERS = {"rPCHSH": "s", "rPhase": "o", "rHybrid": "^"}
EFFICIENCY_MARKERS = {1.0: "o", 0.99: "s", 0.98: "^", 0.97: "D"}
ETA_S_ORDER = (1.0, 0.99, 0.98, 0.97)
CROSS_PROTOCOL_ETA_S = (1.0, 0.98, 0.97)
VISIBILITY_ORDER = (1.0, 0.99)
VISIBILITY_ETA_S = 0.99
ETA_S_STYLES = {
    1.0: "-",
    0.99: "--",
    0.98: "-.",
    0.97: ":",
}
VISIBILITY_STYLES = {1.0: "-", 0.99: "--"}


def configure_curve_contract(config: Mapping[str, object]) -> None:
    """Load the displayed curve families from the manuscript configuration."""

    global ETA_S_ORDER, CROSS_PROTOCOL_ETA_S, VISIBILITY_ORDER, VISIBILITY_ETA_S
    specification = config["curve_plots"]
    ETA_S_ORDER = tuple(float(value) for value in specification["per_protocol_eta_s"])
    CROSS_PROTOCOL_ETA_S = tuple(
        float(value) for value in specification["cross_protocol_eta_s"]
    )
    VISIBILITY_ORDER = tuple(
        float(value) for value in specification["visibility_values"]
    )
    VISIBILITY_ETA_S = float(specification["visibility_eta_s"])


def configure_matplotlib() -> None:
    """Apply one restrained, journal-friendly style to every figure."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 11.0,
            "axes.labelsize": 11.5,
            "axes.labelpad": 8.0,
            "axes.titlesize": 12.0,
            "axes.titlepad": 10.0,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#404040",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 9.0,
            "legend.frameon": False,
            "lines.linewidth": 1.15,
            "lines.antialiased": True,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.dash_capstyle": "round",
            "lines.dash_joinstyle": "round",
            "path.simplify": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_results(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_plot_tables(output: Path) -> tuple[list[dict[str, str]], dict[str, Path]]:
    """Merge numerical and analytic curve tables."""

    table_dir = Path(output) / "tables"
    specifications = (
        ("results.csv", None, True),
        ("ideal_rates.csv", "ideal_loss_analytic", False),
        ("protocol_envelopes.csv", "certificate_envelope", False),
    )
    rows: list[dict[str, str]] = []
    inputs: dict[str, Path] = {}
    for name, default_kind, required in specifications:
        path = table_dir / name
        if not path.is_file():
            if required:
                raise FileNotFoundError(f"Missing numerical table: {path}")
            continue
        inputs[name] = path
        for row in read_results(path):
            if default_kind and not row.get("experiment_kind"):
                row["experiment_kind"] = default_kind
            rows.append(row)
    return rows, inputs


def as_float(row: Mapping[str, object], key: str) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        raise ValueError(f"Missing numeric field {key!r}")
    return float(value)


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _same(left: object, right: float, tolerance: float = 5e-10) -> bool:
    try:
        return abs(float(left) - right) <= tolerance
    except (TypeError, ValueError):
        return False


def _curve_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        kind = row.get("experiment_kind")
        if kind == "ideal_loss_analytic":
            normalized.append(
                {
                    **row,
                    "slice_id": "routed_nu1_s1",
                    "data_scope": "routed",
                    "include_near": True,
                    "nu": 1.0,
                    "eta_s": 1.0,
                    "displayed_rate": as_float(row, "key_rate_lower"),
                    "bound_source": "exact_formula",
                }
            )
        elif kind == "certificate_envelope" and is_true(row.get("claim_eligible")):
            normalized.append(
                {
                    **row,
                    "include_near": is_true(row.get("include_near")),
                    "displayed_rate": as_float(row, "key_rate_lower"),
                    "bound_source": row.get(
                        "bound_source_protocol", row.get("evidence_kind", "certificate")
                    ),
                }
            )
    return normalized


def _finish_axis(axis: Axes, series: list[dict[str, object]]) -> None:
    points = [point for item in series for point in item["points"]]
    minimum = min(point["eta_l"] for point in points)
    maximum = max(point["eta_l"] for point in points)
    positive = [point["eta_l"] for point in points if point["key_rate_lower"] > 1e-12]
    left = max(minimum, floor((min(positive) - 0.012 + 1e-10) * 100) / 100) if positive else minimum
    if left >= maximum:
        left = max(0.0, maximum - 0.02)
        maximum = max(maximum, left + 0.02)
    maximum_rate = max(point["key_rate_lower"] for point in points)
    top = 1.045 if maximum_rate >= 0.85 else max(0.1, maximum_rate * 1.045)
    axis.grid(color="#CDD3D8", linewidth=0.5, linestyle=(0, (2, 4)), alpha=0.8)
    axis.set_axisbelow(True)
    axis.set_xlim(left, maximum)
    axis.set_ylim(0.0, top)
    step = 0.05 if maximum - left <= 0.36 else 0.1
    ticks = [left] + [index * step for index in range(21) if left + 0.015 < index * step < maximum - 0.01] + [maximum]
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    axis.set_xlabel(r"Long-path detection efficiency $\eta_L$")
    axis.set_ylabel(r"Key rate $K_{\rm cert}$ (bits/round)")


def save_figure(
    figure: Figure,
    figure_dir: Path,
    stem: str,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = figure_dir / f"{stem}.{extension}"
        kwargs = {"dpi": 600} if extension == "png" else {}
        figure.savefig(path, **kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


def _point_payload(row: Mapping[str, object]) -> dict[str, object]:
    selected_nodes = row.get("selected_nodes")
    if isinstance(selected_nodes, str) and selected_nodes:
        selected_nodes = json.loads(selected_nodes)
    return {
        "eta_l": as_float(row, "eta_b"),
        "key_rate_lower": float(row["displayed_rate"]),
        "entropy_lower": (
            None if row.get("entropy_lower") in (None, "") else as_float(row, "entropy_lower")
        ),
        "ec_cost": None if row.get("EC_cost") in (None, "") else as_float(row, "EC_cost"),
        "m": None if row.get("m") in (None, "") else int(float(row["m"])),
        "bound_source": str(row.get("bound_source", "")),
        "bound_source_scope": str(row.get("bound_source_scope", "")),
        "evidence_kind": str(row.get("evidence_kind", "")),
        "selected_nodes": selected_nodes or None,
    }


def _make_series(
    series_id: str,
    label: str,
    rows: Iterable[dict[str, object]],
    **metadata: object,
) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: as_float(row, "eta_b"))
    return {
        "series_id": series_id,
        "label": label,
        **metadata,
        "points": [_point_payload(row) for row in ordered],
    }


def _write_figure_data(
    data_dir: Path,
    stem: str,
    series: list[dict[str, object]],
    axis: Axes,
) -> Path:
    path = data_dir / f"{stem}.json"
    write_json(
        path,
        {
            "schema": "routed-dicka-figure-data-v2",
            "figure": stem,
            "presentation": {
                "x_limits": list(axis.get_xlim()),
                "y_limits": list(axis.get_ylim()),
                "figure_size_inches": list(axis.figure.get_size_inches()),
                "png_dpi": 600,
                "interpolation": "none",
                "zero_rate_prefix": "only the last zero before the positive branch is drawn",
            },
            "series": series,
        },
    )
    return path


def _draw_series(
    axis: Axes,
    series: dict[str, object],
    *,
    color: str,
    linestyle: object,
    linewidth: float = 1.15,
    marker: str | None = None,
    marker_phase: int = 0,
) -> None:
    points = series["points"]
    positive_indices = [index for index, point in enumerate(points) if point["key_rate_lower"] > 1e-12]
    if positive_indices:
        drawn = points[max(0, positive_indices[0] - 1):]
    elif series.get("data_scope") == "far_only":
        drawn = points
        marker = None
    else:
        drawn = []
    stride = max(18, len(drawn) // 6)
    offset = (5 + marker_phase * 7) % stride
    series["style"] = {
        "color": color,
        "linestyle": linestyle,
        "linewidth": linewidth,
        "marker": marker,
        "marker_every": [offset, stride],
        "marker_size": 3.7,
    }
    axis.plot(
        [point["eta_l"] for point in drawn],
        [point["key_rate_lower"] for point in drawn],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markevery=(offset, stride),
        markersize=3.7,
        markerfacecolor="white",
        markeredgewidth=0.85,
        antialiased=True,
        label=series["label"],
    )


def _draw_protocol_comparison(axis: Axes, series: list[dict[str, object]], condition: str) -> None:
    previous: list[dict[str, object]] = []
    for index, item in enumerate(series):
        styles = VISIBILITY_STYLES if condition == "nu" else ETA_S_STYLES
        linestyle = styles[float(item[condition])]
        for other in previous:
            if len(item["points"]) == len(other["points"]) and all(
                isclose(point["eta_l"], reference["eta_l"], abs_tol=1e-12)
                and isclose(point["key_rate_lower"], reference["key_rate_lower"], rel_tol=0.0, abs_tol=1e-12)
                for point, reference in zip(item["points"], other["points"])
            ):
                item["coincident_with"] = other["series_id"]
                linestyle = "None"
                break
        _draw_series(
            axis,
            item,
            color=PROTOCOL_COLORS[str(item["protocol"])],
            linestyle=linestyle,
            marker=PROTOCOL_MARKERS[str(item["protocol"])],
            marker_phase=index,
        )
        previous.append(item)
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=3,
        columnspacing=1.4, handlelength=2.8, handletextpad=0.65, labelspacing=0.8,
    )


def _protocol_plotter(protocol: str) -> Callable:
    def plotter(
        rows: list[dict[str, str]],
        figure_dir: Path,
        data_dir: Path,
    ) -> tuple[list[Path], list[Path]]:
        curve_rows = _curve_rows(rows)
        series: list[dict[str, object]] = []
        for eta_s in ETA_S_ORDER:
            selected = [
                row
                for row in curve_rows
                if row.get("protocol") == protocol
                and bool(row.get("include_near"))
                and _same(row.get("nu"), 1.0)
                and _same(row.get("eta_s"), eta_s)
            ]
            if selected:
                series.append(
                    _make_series(
                        f"{protocol}_routed_nu1_s{eta_s:.2f}",
                        rf"routed, $\eta_S={eta_s:g}$",
                        selected,
                        protocol=protocol,
                        data_scope="routed",
                        nu=1.0,
                        eta_s=eta_s,
                    )
                )
        long_only = [
            row
            for row in curve_rows
            if row.get("protocol") == protocol
            and not bool(row.get("include_near"))
            and _same(row.get("nu"), 1.0)
        ]
        if long_only:
            series.append(
                _make_series(
                    f"{protocol}_long_only_nu1",
                    "long-only (non-routed)",
                    long_only,
                    protocol=protocol,
                    data_scope="far_only",
                    nu=1.0,
                    eta_s=None,
                )
            )
        if not series:
            return [], []

        figure, axis = plt.subplots(figsize=(6.4, 4.3))
        for index, item in enumerate(series):
            if item["data_scope"] == "far_only":
                _draw_series(axis, item, color="#404040", linestyle=(0, (5, 2.5)), marker="x", marker_phase=index)
            else:
                _draw_series(
                    axis,
                    item,
                    color=EFFICIENCY_COLORS[float(item["eta_s"])],
                    linestyle=ETA_S_STYLES[float(item["eta_s"])],
                    marker=EFFICIENCY_MARKERS[float(item["eta_s"])],
                    marker_phase=index,
                )
        axis.set_title(PROTOCOL_LABELS[protocol], loc="left")
        axis.legend(loc="upper left", handlelength=3.0, labelspacing=0.75, borderaxespad=0.7)
        _finish_axis(axis, series)
        figure.tight_layout()
        stem = f"routed_vs_long_only_{protocol.lower()}"
        data_path = _write_figure_data(data_dir, stem, series, axis)
        return save_figure(figure, figure_dir, stem), [data_path]

    plotter.__name__ = f"plot_routed_vs_long_only_{protocol.lower()}"
    return plotter


plot_routed_vs_long_only_rpchsh = _protocol_plotter("rPCHSH")
plot_routed_vs_long_only_rphase = _protocol_plotter("rPhase")
plot_routed_vs_long_only_rhybrid = _protocol_plotter("rHybrid")


def plot_routed_protocols_near_efficiency(
    rows: list[dict[str, str]],
    figure_dir: Path,
    data_dir: Path,
) -> tuple[list[Path], list[Path]]:
    curve_rows = _curve_rows(rows)
    series: list[dict[str, object]] = []
    for protocol in PROTOCOL_ORDER:
        for eta_s in CROSS_PROTOCOL_ETA_S:
            selected = [
                row
                for row in curve_rows
                if row.get("protocol") == protocol
                and bool(row.get("include_near"))
                and _same(row.get("nu"), 1.0)
                and _same(row.get("eta_s"), eta_s)
            ]
            if selected:
                series.append(
                    _make_series(
                        f"{protocol}_nu1_s{eta_s:.2f}",
                        rf"{PROTOCOL_LABELS[protocol]}, $\eta_S={eta_s:g}$",
                        selected,
                        protocol=protocol,
                        data_scope="routed",
                        nu=1.0,
                        eta_s=eta_s,
                    )
                )
    if not series:
        return [], []
    figure, axis = plt.subplots(figsize=(6.6, 5.0))
    _draw_protocol_comparison(axis, series, "eta_s")
    _finish_axis(axis, series)
    figure.tight_layout()
    stem = "routed_protocols_near_efficiency"
    data_path = _write_figure_data(data_dir, stem, series, axis)
    return save_figure(figure, figure_dir, stem), [data_path]


def plot_routed_protocols_visibility(
    rows: list[dict[str, str]],
    figure_dir: Path,
    data_dir: Path,
) -> tuple[list[Path], list[Path]]:
    curve_rows = _curve_rows(rows)
    series: list[dict[str, object]] = []
    for protocol in PROTOCOL_ORDER:
        for nu in VISIBILITY_ORDER:
            selected = [
                row
                for row in curve_rows
                if row.get("protocol") == protocol
                and bool(row.get("include_near"))
                and _same(row.get("nu"), nu)
                and _same(row.get("eta_s"), VISIBILITY_ETA_S)
            ]
            if selected:
                series.append(
                    _make_series(
                        f"{protocol}_nu{nu:.2f}_s0.99",
                        rf"{PROTOCOL_LABELS[protocol]}, $\nu={nu:g}$",
                        selected,
                        protocol=protocol,
                        data_scope="routed",
                        nu=nu,
                        eta_s=VISIBILITY_ETA_S,
                    )
                )
    if not series:
        return [], []
    figure, axis = plt.subplots(figsize=(6.6, 4.8))
    _draw_protocol_comparison(axis, series, "nu")
    _finish_axis(axis, series)
    figure.tight_layout()
    stem = "routed_protocols_visibility"
    data_path = _write_figure_data(data_dir, stem, series, axis)
    return save_figure(figure, figure_dir, stem), [data_path]


PLOTTERS: tuple[
    Callable[
        [list[dict[str, str]], Path, Path],
        tuple[list[Path], list[Path]],
    ],
    ...,
] = (
    plot_routed_vs_long_only_rpchsh,
    plot_routed_vs_long_only_rphase,
    plot_routed_vs_long_only_rhybrid,
    plot_routed_protocols_near_efficiency,
    plot_routed_protocols_visibility,
)
