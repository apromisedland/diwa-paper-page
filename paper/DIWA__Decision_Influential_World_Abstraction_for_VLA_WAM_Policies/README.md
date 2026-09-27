# DIWA: manuscript source and numerical summaries

This archive contains the anonymous ICLR 2027 manuscript, its figures and tables, the unmodified conference style, and the inputs needed to regenerate numerical summaries.

## Build the manuscript

Open `main.tex` in Overleaf and select pdfLaTeX, or run `bash scripts/build.sh` locally with a TeX installation that provides `latexmk`.

## Regenerate tables and plots

Run `python3 scripts/make_results.py` to regenerate the nine result tables and the numerical ledger. This requires the Python standard library. The input `evidence/reported_measurements.json` records the experiment summaries without editing their numerical values.

Run `python3 scripts/make_figures.py` to regenerate the two result plots. This additionally requires matplotlib. The method diagram is the original JPEG in `figures/architecture.jpg`; the plotting script leaves it unchanged.

The table generator checks 160 arithmetic relationships, including seed means, sample standard deviations, standard errors, physical success counts, and budget counts. Within-task physical confidence intervals are derived from the recorded counts under the assumptions stated in the paper.

## Scope

These scripts reproduce the manuscript and its numerical summaries. They do not retrain a policy or rerun robot evaluations. Run-level hardware, task, checkpoint, supervision, and diagnostic records are separate from this manuscript source; their current coverage is described in the paper's evaluation appendix. The ledger retains the separate phase-retention measurements for traceability and marks them as excluded from the manuscript because their configuration is not identified.
