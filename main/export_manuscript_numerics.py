"""Export the manuscript's selected numerical tables from audited artifacts."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import json
from math import isclose
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routed_dicka.io import file_sha256, utc_now, write_json  # noqa: E402


PROTOCOLS = ("rPCHSH", "rPhase", "rHybrid")
PROTOCOL_TEX = {protocol: "\\" + protocol for protocol in PROTOCOLS}
MODELS = ("M", "W", "R", "F")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_latex(path: Path, content: str) -> None:
    """Write deterministic LF text without relying on newer Path APIs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _same(value: object, expected: float) -> bool:
    return isclose(float(value), expected, rel_tol=0.0, abs_tol=2e-12)


def _directed(value: object, places: int, *, upper: bool = False) -> str:
    quantum = Decimal(1).scaleb(-places)
    rounding = ROUND_CEILING if upper else ROUND_FLOOR
    return format(Decimal(str(value)).quantize(quantum, rounding=rounding), f".{places}f")


def representative_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    selected = {}
    for row in rows:
        if (
            row.get("slice_id") == "routed_nu0p99_s0p99"
            and row.get("protocol") in PROTOCOLS
            and _same(row["eta_b"], 0.8)
        ):
            selected[str(row["protocol"])] = {
                "protocol": str(row["protocol"]),
                "m": int(float(row["m"])),
                "entropy_lower": float(row["entropy_lower"]),
                "ec_cost": float(row["EC_cost"]),
                "key_rate_lower": float(row["key_rate_lower"]),
                "bound_source_protocol": row.get("bound_source_protocol", row["protocol"]),
            }
    if set(selected) != set(PROTOCOLS):
        raise AssertionError("The S2 envelope does not contain all three protocols")
    return [selected[protocol] for protocol in PROTOCOLS]


def mwrf_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    selected = {str(row["info_model"]): row for row in rows}
    if set(selected) != set(MODELS):
        raise AssertionError("The M/W/R/F table is incomplete")
    result = []
    for model in MODELS:
        row = selected[model]
        attack = row.get("evidence_kind") == "explicit_compatible_zero_entropy_attack"
        result.append(
            {
                "model": model,
                "evidence": "explicit compatible attack" if attack else "guarded numerical dual bound",
                "entropy_statement": 0.0 if attack else float(row["entropy_lower"]),
                "key_rate_statement": (
                    float(row["guaranteed_key_rate_upper"])
                    if attack
                    else float(row["key_rate_lower"])
                ),
                "relation": "upper" if attack else "lower",
                "m": None if attack else int(float(row["m"])),
            }
        )
    return result


def radau_rows(
    numerical_rows: list[dict[str, str]],
    m4_audit: dict[str, object],
) -> list[dict[str, object]]:
    by_order = {}
    for row in numerical_rows:
        if str(row.get("claim_eligible", "")).lower() not in {"true", "1"}:
            continue
        m = int(float(row["m"]))
        by_order[m] = {
            "m": m,
            "optimized_nodes": m - 1,
            "entropy_lower": float(row["entropy_lower"]),
            "ec_cost": float(row["EC_cost"]),
            "key_rate_lower": float(row["raw_key_rate_lower"]),
            "source": "generated_exact_certificate",
        }
    if m4_audit.get("passed") is not True:
        raise AssertionError("The primary m=4 certificate audit did not pass")
    primary = m4_audit["reference_point"]
    by_order[4] = {
        "m": 4,
        "optimized_nodes": 3,
        "entropy_lower": float(primary["H_lower"]),
        "ec_cost": float(primary["EC_upper"]),
        "key_rate_lower": float(primary["K_lower"]),
        "source": "archived_m4",
    }
    if set(by_order) != {2, 3, 4}:
        raise AssertionError("The Radau sensitivity table requires m=2,3,4")
    return [by_order[m] for m in (2, 3, 4)]


def _representative_tex(rows: list[dict[str, object]]) -> str:
    lines = []
    for row in rows:
        rate = float(row["key_rate_lower"])
        displayed_rate = _directed(rate, 6) if rate > 0.0 else "0 (not certified)"
        lines.append(
            f"\\({PROTOCOL_TEX[str(row['protocol'])]}\\) & {_directed(row['entropy_lower'], 6)} & "
            f"{_directed(row['ec_cost'], 6, upper=True)} & {displayed_rate} \\\\"
        )
    return "\n".join(lines) + "\n"


def _mwrf_tex(rows: list[dict[str, object]]) -> str:
    lines = []
    for row in rows:
        if row["relation"] == "upper":
            statement = r"compatible \(H(A_Z|E)=0\) attack"
            rate = r"\(K^\rightarrow\leq0\)"
        else:
            statement = rf"\(\underline H(A_Z|E)={_directed(row['entropy_statement'], 6)}\)"
            rate = rf"\(\underline K^\rightarrow={_directed(row['key_rate_statement'], 6)}\)"
        lines.append(f"{row['model']} & {statement} & {rate} \\\\")
    return "\n".join(lines) + "\n"


def _radau_tex(rows: list[dict[str, object]]) -> str:
    return "\n".join(
        f"{row['m']} & {row['optimized_nodes']} & {_directed(row['entropy_lower'], 10)} & "
        f"{_directed(row['key_rate_lower'], 10)} & exact reconstruction passed \\\\"
        for row in rows
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--paper", type=Path, default=ROOT / "paper")
    args = parser.parse_args()
    output = args.output.resolve()
    paper = args.paper.resolve()
    envelope_path = output / "tables" / "protocol_envelopes.csv"
    mwrf_path = output / "tables" / "mwrf.csv"
    radau_path = output / "tables" / "radau_sensitivity.csv"
    audit_path = output / "audits" / "m4_same_point_certificate.json"

    representative = representative_rows(_read_csv(envelope_path))
    mwrf = mwrf_rows(_read_csv(mwrf_path))
    radau = radau_rows(
        _read_csv(radau_path),
        json.loads(audit_path.read_text(encoding="utf-8")),
    )
    selected = {
        "schema": "routed-dicka-manuscript-numerics-v1",
        "created_utc": utc_now(),
        "representative_point": representative,
        "mwrf": mwrf,
        "radau_sensitivity": radau,
        "inputs": {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in (envelope_path, mwrf_path, radau_path, audit_path)
        },
    }
    write_json(output / "tables" / "manuscript_numerics.json", selected)
    generated = paper / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    write_latex(
        generated / "representative_point_rows.tex",
        _representative_tex(representative),
    )
    write_latex(generated / "mwrf_rows.tex", _mwrf_tex(mwrf))
    write_latex(generated / "radau_sensitivity_rows.tex", _radau_tex(radau))
    print(json.dumps({"passed": True, "output": "tables/manuscript_numerics.json"}, indent=2))


if __name__ == "__main__":
    main()
