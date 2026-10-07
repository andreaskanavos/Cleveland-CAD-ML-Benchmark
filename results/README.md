# Reference Results

This directory contains the reference outputs associated with the analysis reported in the manuscript.

The files are included for two reasons:

1. to make the numerical results easy to inspect without rerunning the complete workflow; and
2. to provide reference values against which a fresh execution of `src/analysis_pipeline.py` can be checked.

A new run should be written to a separate directory, for example `reproduced_outputs/`, which is ignored by Git.

The reference outputs include cross-validation summaries, fold-level accuracy data, held-out performance and confidence intervals, Friedman/Wilcoxon-Holm results, the repeated MLP analysis, the `ca`-exclusion sensitivity analysis, and the main figures.
