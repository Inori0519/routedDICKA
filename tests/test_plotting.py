from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN = ROOT / "main"
for path in (SRC, MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from plot_figures import _prepare_presentation_manifest, generate_figures  # noqa: E402
from routed_dicka.io import file_sha256, source_hashes, write_csv, write_json  # noqa: E402
from routed_dicka.plotting import (  # noqa: E402
    PLOTTERS,
    _draw_protocol_comparison,
    _draw_series,
    _finish_axis,
    configure_matplotlib,
    configure_curve_contract,
    read_plot_tables,
)
import matplotlib.pyplot as plt


def _row(**updates: object) -> dict[str, str]:
    values = {
        "claim_eligible": "True",
        "experiment_kind": "certificate_envelope",
        "slice_id": "routed_nu1_s1",
        "protocol": "rHybrid",
        "data_scope": "routed",
        "include_near": "True",
        "nu": "1.0",
        "eta_s": "1.0",
        "eta_b": "0.8",
        "eta_c": "0.8",
        "entropy_lower": "0.50",
        "EC_cost": "0.22",
        "raw_key_rate_lower": "0.28",
        "key_rate_lower": "0.28",
        "evidence_kind": "verified_affine_certificate_envelope",
        "m": "4",
    }
    values.update({key: str(value) for key, value in updates.items()})
    return values


def _curve_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for eta in (0.6, 0.8, 1.0):
        for protocol in ("rPhase", "rHybrid"):
            rows.append(
                _row(
                    experiment_kind="ideal_loss_analytic",
                    slice_id="",
                    protocol=protocol,
                    eta_b=eta,
                    eta_c=eta,
                    key_rate_lower=max(0.0, eta * eta + eta - 1.0),
                    evidence_kind="analytic",
                    m="",
                )
            )
        rows.append(_row(protocol="rPCHSH", eta_b=eta, eta_c=eta))
        for eta_s in (0.99, 0.98, 0.97):
            slice_id = f"routed_nu1_s{str(eta_s).replace('.', 'p')}"
            for protocol in ("rPCHSH", "rPhase", "rHybrid"):
                rows.append(
                    _row(
                        slice_id=slice_id,
                        protocol=protocol,
                        eta_s=eta_s,
                        eta_b=eta,
                        eta_c=eta,
                    )
                )
        for protocol in ("rPCHSH", "rPhase", "rHybrid"):
            rows.append(
                _row(
                    slice_id="routed_nu0p99_s0p99",
                    protocol=protocol,
                    nu=0.99,
                    eta_s=0.99,
                    eta_b=eta,
                    eta_c=eta,
                )
            )
            rows.append(
                _row(
                    slice_id="long_only_nu1",
                    protocol=protocol,
                    data_scope="far_only",
                    include_near="False",
                    eta_b=eta,
                    eta_c=eta,
                )
            )
    return rows


class PlottingTests(unittest.TestCase):
    def test_presentation_refresh_preserves_the_numerical_run_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            hashes = source_hashes(ROOT)
            hashes["src/routed_dicka/plotting.py"] = "previous-renderer-hash"
            original = {"source_hashes": hashes, "run_contract_sha256": "numerical-contract"}
            write_json(output / "manifest.json", original)
            refreshed = _prepare_presentation_manifest(output)
            self.assertEqual(refreshed["run_contract_sha256"], "numerical-contract")
            self.assertEqual(refreshed["source_hashes_before_presentation"], hashes)
            self.assertEqual(refreshed["source_hashes"], source_hashes(ROOT))

            original["source_hashes"]["src/routed_dicka/target.py"] = "changed-physical-model"
            write_json(output / "manifest.json", original)
            with self.assertRaisesRegex(RuntimeError, "Numerical source changed"):
                _prepare_presentation_manifest(output)

    def test_presentation_refresh_rejects_changed_numerical_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            raw = output / "raw.json"
            write_json(raw, {"rate": 0.1})
            write_json(output / "manifest.json", {
                "source_hashes": source_hashes(ROOT),
                "point_hashes": {"raw.json": file_sha256(raw)},
            })
            write_json(raw, {"rate": 0.2})
            with self.assertRaisesRegex(RuntimeError, "Numerical artifact changed"):
                _prepare_presentation_manifest(output)

    def test_redraw_crops_zero_prefix_without_changing_data_or_threshold(self) -> None:
        series = {
            "series_id": "test", "label": "test", "data_scope": "routed",
            "points": [
                {"eta_l": 0.6, "key_rate_lower": 0.0},
                {"eta_l": 0.772, "key_rate_lower": 0.0},
                {"eta_l": 0.774, "key_rate_lower": 0.0016},
                {"eta_l": 1.0, "key_rate_lower": 0.95},
            ],
        }
        original = deepcopy(series["points"])
        figure, axis = plt.subplots()
        try:
            _draw_series(axis, series, color="black", linestyle="-")
            _finish_axis(axis, [series])
            self.assertEqual(list(axis.lines[0].get_xdata()), [0.772, 0.774, 1.0])
            self.assertEqual(list(axis.lines[0].get_ydata()), [0.0, 0.0016, 0.95])
            self.assertAlmostEqual(axis.get_xlim()[0], 0.76)
            self.assertEqual(series["points"], original)
        finally:
            plt.close(figure)

    def test_coincident_protocols_remain_visible_without_coordinate_offsets(self) -> None:
        points = [
            {"eta_l": 0.6, "key_rate_lower": 0.0},
            {"eta_l": 0.8, "key_rate_lower": 0.44},
            {"eta_l": 1.0, "key_rate_lower": 1.0},
        ]
        series = [
            {"series_id": protocol, "protocol": protocol, "label": protocol,
             "data_scope": "routed", "eta_s": 1.0, "points": deepcopy(points)}
            for protocol in ("rPhase", "rHybrid")
        ]
        figure, axis = plt.subplots()
        try:
            _draw_protocol_comparison(axis, series, "eta_s")
            self.assertEqual(list(axis.lines[0].get_xdata()), list(axis.lines[1].get_xdata()))
            self.assertEqual(list(axis.lines[0].get_ydata()), list(axis.lines[1].get_ydata()))
            self.assertEqual(series[1]["coincident_with"], "rPhase")
            self.assertNotEqual(axis.lines[0].get_marker(), axis.lines[1].get_marker())
            self.assertEqual(axis.lines[1].get_linestyle(), "None")
            self.assertTrue(all(item["points"] == points for item in series))
        finally:
            plt.close(figure)

    def test_plot_inputs_merge_numerical_and_reference_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            tables = output / "tables"
            write_csv(tables / "results.csv", [_row(experiment_kind="certificate_anchor")])
            write_csv(tables / "ideal_rates.csv", [_row(experiment_kind="ideal_loss_analytic")])
            write_csv(tables / "protocol_envelopes.csv", [_row()])

            rows, inputs = read_plot_tables(output)

            self.assertEqual(len(rows), 3)
            self.assertEqual(
                {row["experiment_kind"] for row in rows},
                {"certificate_anchor", "ideal_loss_analytic", "certificate_envelope"},
            )
            self.assertEqual(
                set(inputs), {"results.csv", "ideal_rates.csv", "protocol_envelopes.csv"}
            )

    def test_five_manuscript_plotters_export_pdf_png_and_json(self) -> None:
        self.assertEqual(len(PLOTTERS), 5)
        configure_matplotlib()
        configure_curve_contract(
            json.loads((ROOT / "config" / "paper_suite.json").read_text(encoding="utf-8"))
        )
        expected_series = (5, 5, 5, 9, 6)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for plotter, series_count in zip(PLOTTERS, expected_series):
                figures, data = plotter(_curve_rows(), root / "figures", root / "data")
                self.assertEqual({path.suffix for path in figures}, {".pdf", ".png"})
                self.assertEqual(len(data), 1)
                self.assertEqual(data[0].suffix, ".json")
                self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in figures))
                payload = json.loads(data[0].read_text(encoding="utf-8"))
                self.assertEqual(len(payload["series"]), series_count)
                self.assertTrue(all(series["points"] for series in payload["series"]))
                self.assertTrue(
                    all(series["points"][0]["eta_l"] == 0.6 for series in payload["series"])
                )
                self.assertTrue(
                    all(series["points"][-1]["eta_l"] == 1.0 for series in payload["series"])
                )

    def test_generation_replaces_stale_artifacts_and_requires_all_five_figures(self) -> None:
        config_path = ROOT / "config" / "paper_suite.json"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            tables = output / "tables"
            figures = output / "figures"
            data = output / "figure_data"
            figures.mkdir(parents=True)
            data.mkdir(parents=True)
            (figures / "protocol_rates.svg").write_text("stale", encoding="utf-8")
            (data / "protocol_rates.csv").write_text("stale", encoding="utf-8")

            rows = _curve_rows()
            write_csv(
                tables / "results.csv",
                [row for row in rows if row["experiment_kind"] == "certificate_anchor"],
            )
            write_csv(
                tables / "ideal_rates.csv",
                [row for row in rows if row["experiment_kind"] == "ideal_loss_analytic"],
            )
            write_csv(
                tables / "protocol_envelopes.csv",
                [row for row in rows if row["experiment_kind"] == "certificate_envelope"],
            )

            manifest = generate_figures(config_path, output)

            expected = set(
                json.loads(config_path.read_text(encoding="utf-8"))["curve_plots"][
                    "figure_stems"
                ]
            )
            self.assertEqual(
                {path.stem for path in figures.glob("*.png")},
                expected,
            )
            self.assertEqual(
                {path.stem for path in figures.glob("*.pdf")},
                expected,
            )
            self.assertEqual({path.stem for path in data.glob("*.json")}, expected)
            self.assertFalse((figures / "protocol_rates.svg").exists())
            self.assertFalse((data / "protocol_rates.csv").exists())
            self.assertEqual(manifest["schema"], "routed-dicka-figure-manifest-v2")

    def test_generation_rejects_incomplete_curve_families_without_replacing_old_figures(self) -> None:
        config_path = ROOT / "config" / "paper_suite.json"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            tables = output / "tables"
            figures = output / "figures"
            figures.mkdir(parents=True)
            stale = figures / "existing.png"
            stale.write_bytes(b"stale")
            write_csv(tables / "results.csv", [_row(experiment_kind="certificate_anchor")])
            write_csv(
                tables / "ideal_rates.csv",
                [_row(experiment_kind="ideal_loss_analytic", protocol="rPhase")],
            )
            write_csv(tables / "protocol_envelopes.csv", [_row()])

            with self.assertRaisesRegex(RuntimeError, "Missing manuscript figures"):
                generate_figures(config_path, output)

            self.assertTrue(stale.is_file())


if __name__ == "__main__":
    unittest.main()
