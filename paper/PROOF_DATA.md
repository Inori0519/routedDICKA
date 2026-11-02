# Result Sources

The article's exact ideal-rate and M/W/R/F conclusions are analytical. The
same-target noisy separation combines the physical attack in
`source_data/attack/` with the three reference entropy certificates in
`source_data/certificates/m4/`. Their checks are in `verification/`.

The efficiency and source-noise figures use `config/paper_suite.json`, the
saved records in `outputs/`, and the composite-statistics implementation in
`src/routed_dicka/`. The curve tables evaluate verified affine entropy bounds;
their dense grids are not independent SDP optimizations at every plotted
point. Both the plotted values and their selected sources are retained in
`outputs/figure_data/`.

The SRQ appendix image uses the earlier binary witness scan preserved under
`source_data/srq/`. Those illustrative scan values are not used as ternary
secrecy certificates. The article's attacks match complete remote behavior;
the smaller observed set used by the entropy optimizer is stated separately
in Appendix D.
