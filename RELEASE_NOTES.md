# Publication Snapshot

Prepared on 2026-11-02 for the routed DICKA manuscript.

The numerical implementation is the composite-statistics version recorded by
the published run manifest. Every Python source file in `src/`, `main/` and
`verification/` is preserved byte for byte from that snapshot. The supplied
configuration, solver parameters, plotting style and numerical results are
unchanged. Later complete-table, full-ternary extensions are excluded.

The LaTeX article is the revised combined manuscript of 2026-11-02. It includes
the SRQ and information-ablation proofs in its appendices, with no separate
supplementary manuscript. Supporting generated tables remain for compatibility
with the original exports; the article does not include them automatically.

The published output directory contains files referenced by the run and
figure manifests, their point-level target dependencies, the reference tables
and the relevant saved audits. Probe runs, duplicate
submission/verification packages, old papers, editor caches and machine-local
solver configuration are omitted.

Packaging adds documentation, ignore rules, a figure-only PowerShell wrapper
and publication-layout checks. It does not change a scientific calculation.
Checks of the existing results do not mean the SDP searches were rerun.
The Windows workflow no longer invokes the historical nested-package exporter,
whose schema required a separate supplementary manuscript. Its original Python
helpers are retained unchanged as part of the recorded source snapshot, not as
the publication entry point.

No software license has been selected on the authors' behalf. Dependency
licenses remain those of their respective projects; MOSEK must be licensed
separately. This directory has not been uploaded or pushed to a remote service
by the packaging operation.
