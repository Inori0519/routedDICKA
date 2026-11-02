"""Generate every available manuscript figure from the audited CSV table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routed_dicka.io import file_sha256, source_hashes, utc_now, write_json  # noqa: E402
from routed_dicka.plotting import (  # noqa: E402
    PLOTTERS,
    configure_matplotlib,
    configure_curve_contract,
    read_plot_tables,
)


def _prepare_presentation_manifest(output: Path) -> dict[str, object] | None:
    path = output / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    current = source_hashes(ROOT)
    previous = manifest["source_hashes"]
    presentation_sources = {"main/plot_figures.py", "src/routed_dicka/plotting.py"}
    changed = {
        name for name in set(current) | set(previous)
        if current.get(name) != previous.get(name)
    }
    unexpected = changed - presentation_sources
    if unexpected:
        raise RuntimeError(f"Numerical source changed; figure-only refresh refused: {sorted(unexpected)}")
    for group in ("point_hashes", "auxiliary_hashes", "table_hashes"):
        for relative, expected in manifest.get(group, {}).items():
            artifact = output / relative
            if not artifact.is_file() or file_sha256(artifact) != expected:
                raise RuntimeError(f"Numerical artifact changed; figure-only refresh refused: {relative}")
    manifest.setdefault("source_hashes_before_presentation", dict(previous))
    manifest["source_hashes"] = current
    manifest["presentation_update"] = {
        "updated_utc": utc_now(),
        "source_hashes": {name: current[name] for name in sorted(presentation_sources)},
        "numerical_artifacts_unchanged": True,
    }
    return manifest


def generate_figures(config_path: Path, output: Path) -> dict[str, object]:
    config_path = config_path.resolve()
    output = output.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    presentation_manifest = _prepare_presentation_manifest(output)
    configure_curve_contract(config)
    configure_matplotlib()
    try:
        rows, input_tables = read_plot_tables(output)
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".figure-staging-", dir=output) as temporary:
        staging = Path(temporary)
        staged_figure_dir = staging / "figures"
        staged_data_dir = staging / "figure_data"
        generated: list[Path] = []
        data_files: list[Path] = []
        for plotter in PLOTTERS:
            figures, data = plotter(rows, staged_figure_dir, staged_data_dir)
            generated.extend(figures)
            data_files.extend(data)

        expected_stems = set(config["curve_plots"]["figure_stems"])
        expected_figures = {
            (stem, suffix) for stem in expected_stems for suffix in (".pdf", ".png")
        }
        actual_figures = {(path.stem, path.suffix.lower()) for path in generated}
        actual_data = {path.stem for path in data_files if path.suffix.lower() == ".json"}
        missing_figures = sorted(expected_figures - actual_figures)
        unexpected_figures = sorted(actual_figures - expected_figures)
        missing_data = sorted(expected_stems - actual_data)
        unexpected_data = sorted(actual_data - expected_stems)
        if missing_figures or unexpected_figures or missing_data or unexpected_data:
            raise RuntimeError(
                "Missing manuscript figures or data: "
                f"figures={missing_figures}, data={missing_data}; "
                f"unexpected_figures={unexpected_figures}, unexpected_data={unexpected_data}"
            )

        figure_dir = output / "figures"
        data_dir = output / "figure_data"
        for destination in (figure_dir, data_dir):
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True)
        # Fresh destination directories retain the workspace ACL on Windows.
        for source_dir, destination in ((staged_figure_dir, figure_dir), (staged_data_dir, data_dir)):
            for source in source_dir.iterdir():
                shutil.copy2(source, destination / source.name)

    generated = [figure_dir / path.name for path in generated]
    data_files = [data_dir / path.name for path in data_files]
    manifest_path = figure_dir / "figure_manifest.json"
    manifest = {
        "schema": "routed-dicka-figure-manifest-v2",
        "created_utc": utc_now(),
        "config_sha256": file_sha256(config_path),
        "input_tables": {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in input_tables.values()
        },
        "files": {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in sorted(generated)
        },
        "figure_data": {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in sorted(data_files)
        },
    }
    write_json(manifest_path, manifest)
    if presentation_manifest is not None:
        if source_hashes(ROOT) != presentation_manifest["source_hashes"]:
            raise RuntimeError("Source files changed while figures were being rendered")
        presentation_manifest["presentation_update"]["figure_manifest_sha256"] = file_sha256(manifest_path)
        write_json(output / "manifest.json", presentation_manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "paper_suite.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        generate_figures(args.config, output)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    manifest_path = output / "figures" / "figure_manifest.json"
    print(f"WROTE {manifest_path}")


if __name__ == "__main__":
    main()
