# Published Numerical Results

This directory preserves the results underlying the manuscript figures.

- `figures/`: five rate figures, their PNG/PDF exports and figure manifest.
- `figure_data/`: plotted numerical series and the selected bound sources.
- `tables/`: plotting tables and original numerical summary exports.
- `raw/`, `targets/`, `certificates/`: the records named by the run manifest
  and their complete target/certificate dependencies.
- `audits/`: saved checks relevant to this numerical run.
- `manifest.json`: the original run description and source/data hashes.

The ideal phase and hybrid curves are analytical. Other curves are certified
lower bounds, evaluated on a dense grid from verified affine bounds. The first
positive sample need not be an optimal detector threshold.

No new optimization was performed to assemble this directory. Use
`reproduction/` for new plots or calculations so the published files remain
unchanged. The figure-only command in the root README uses that location.
