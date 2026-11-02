# Routed DICKA

Source code, figure data and LaTeX source for **Device-Independent Conference
Key Agreement via Tripartite Routed Bell Tests**.

This is the implementation used for the manuscript's existing efficiency and
noise figures. Remote receivers keep the outcomes `+1`, `-1` and `perp`.
To reduce the optimization size, secrecy is certified from low-dimensional
coarse-grained composite statistics; error correction uses each receiver's
full ternary record. Alice is ideal in the reported scans, and nearby
nondetections are assigned to `+1`.

The code supports `rPCHSH`, `rPhase` and `rHybrid`. In particular, the phase
reduction retains positive parity, negative parity and erasure as distinct
composite outcomes. It is not a complete-table optimization over separate
ternary remote stations. The later full-ternary scanning implementation is
not included here.

## Contents

| Directory | Contents |
| --- | --- |
| `paper/` | Current manuscript with inline appendices, bibliography and figures |
| `src/routed_dicka/` | Target model, entropy optimization and plotting functions |
| `main/` | Experiment, curve evaluation and figure entry points |
| `config/paper_suite.json` | Parameters and experiment profiles used for the figures |
| `outputs/` | Published figures, plotting tables and their numerical records |
| `source_data/` | Frozen attack and reference entropy certificates |
| `verification/` | Independent checks of attacks, entropy bounds and outputs |
| `tests/` | Numerical and manuscript regression tests |
| `scripts/` | Windows workflow and figure-only reproduction |

The computational source files match the hashes in `outputs/manifest.json`.
The manuscript is the current combined version, not the older manuscript
that accompanied the original numerical run.

## Installation

Use a separate Python environment. The dependencies are pinned to the versions
used by the numerical implementation:

```powershell
python -m pip install -r requirements.txt
```

New SDP searches require a working MOSEK installation and your own MOSEK
license. No solver license or credentials are distributed. Checking the saved
proofs and redrawing the figures does not require a new optimization or a
MOSEK license.

The code and tests have been checked with Python 3.9.20 and the pinned
numerical dependencies.

## Check the Included Results

Run from the repository root:

```powershell
python -B -m unittest discover -s tests -p "test_*.py"
python -B verification/verify_outputs.py --config config/paper_suite.json --output outputs --require-figures --require-scientific-gates
```

These commands check the saved data; they do not scan new efficiency points.
The exact ideal rates, certified numerical lower bounds and attack-proved
zero-key statements have different roles. A zero numerical lower bound is
not, by itself, a proof that key is impossible.

## Redraw the Five Rate Figures

On Windows PowerShell:

```powershell
$Python = 'python'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\replot.ps1 -Python $Python
```

Use the path to your environment's `python.exe` instead of `python` when
needed. The script uses the shipped plotting tables and unchanged plotting
style, writes to `reproduction/plots/`, and leaves the published outputs and
paper figures untouched. It does not run BFF/NPA optimization.

## Run New Calculations

First inspect the commands without executing them:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -Python $Python -Profile paper -Output reproduction/run -Resume -DryRun
```

Remove `-DryRun` to run the full calculation. This is expensive and is not
needed to use the included figures. The `paper` profile fixes the curve
quadrature order at four; the recording and M/W/R/F controls use their own
configured orders. `smoke1` runs two small composite-model checks, and
`smoke2` checks the physical-station M/W/R/F calculation.

For a platform-independent experiment entry point:

```bash
python main/run_experiments.py --config config/paper_suite.json --profile smoke1 --output reproduction/smoke1 --resume
```

## Manuscript

`paper/main.tex` contains the article and Appendices A-E. All external images
included by the manuscript are PNG files. The PDF companions of the rate
figures are retained as original figure exports, not compiled manuscripts.
Compile the LaTeX source from `paper/`; new plots under `reproduction/` do not
automatically replace the manuscript's frozen figure copies.

No manuscript PDF, development history, full-ternary scan, or nested release
package is included. See [RELEASE_NOTES.md](RELEASE_NOTES.md) for the scope of
this source snapshot.
