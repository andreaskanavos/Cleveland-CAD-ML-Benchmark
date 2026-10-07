# Cleveland CAD Machine-Learning Benchmark

Code, data, and reference outputs accompanying the manuscript **“Comparative and Explainable Machine Learning for Coronary Artery Disease Classification.”**

The project compares eight supervised-learning approaches on the Cleveland Heart Disease dataset under a common preprocessing and evaluation framework. It includes the held-out analysis, stratified 10-fold cross-validation, nested k-Nearest-Neighbors selection, uncertainty estimation, statistical comparisons, repeated Multilayer-Perceptron training, a sensitivity analysis excluding the fluoroscopy-derived `ca` predictor, and model-interpretability analyses.

## Repository Structure

```text
Cleveland-CAD-ML-Benchmark/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   └── processed.cleveland.data
├── src/
│   └── analysis_pipeline.py
└── results/
    ├── README.md
    ├── cross_validation_summary.csv
    ├── fold_accuracy.csv
    ├── holdout_results.csv
    ├── holdout_95ci.csv
    ├── statistical_tests.txt
    ├── wilcoxon_holm.csv
    ├── mlp_repeated_runs_seeds_43_62.csv
    ├── ca_exclusion_cv_results.csv
    ├── ca_exclusion_holdout_results.csv
    └── figures and additional reference outputs
```

## Dataset

The analysis uses the processed Cleveland subset of the **UCI Heart Disease dataset**. The source dataset contains 303 records. Six records with missing values are excluded by complete-case analysis, yielding an analytic sample of 297 observations.

Dataset citation:

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C52P4X.

The UCI dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. See `data/README.md` for details.

## Models

The workflow evaluates:

- Logistic Regression
- Decision Tree
- k-Nearest Neighbors
- Random Forest
- Gradient Boosting
- Support Vector Machine
- Equal-weight Soft-Voting Ensemble
- Multilayer Perceptron implemented in PyTorch

## Main Analysis Settings

The workflow follows the analysis specification reported in the manuscript:

- stratified 80/20 held-out split with random seed 42;
- stratified 10-fold cross-validation with shuffling and random seed 42;
- training-only five-fold stratified selection of `k` over `1,...,20` for the held-out k-NN model;
- nested five-fold k-NN selection within each outer cross-validation training fold;
- model-specific one-hot encoding and scaling fitted only on the relevant training data;
- PyTorch MLP with hidden layers of 64 and 32 ReLU units, dropout 0.30 and 0.20, a single linear output logit, Adam (`lr=0.001`), and `BCEWithLogitsLoss`;
- early stopping on an internal stratified 20% validation subset with patience 10 and maximum 200 epochs;
- 20 additional MLP runs using seeds 43–62 on the fixed train/test partition;
- 5,000 stratified bootstrap resamples for confidence intervals of threshold-dependent held-out metrics;
- Hanley–McNeil confidence intervals for ROC AUC;
- Friedman testing on paired fold-level accuracy scores, followed by 28 paired Wilcoxon signed-rank comparisons with Holm correction;
- sensitivity analysis excluding the fluoroscopy-derived `ca` predictor;
- Random-Forest and Gradient-Boosting impurity-based feature importance;
- Gradient-Boosting SHAP analysis using an interventional background sample of 100 training observations with seed 42.

## Software Environment

The manuscript reports:

- Python 3.13.5
- scikit-learn 1.8.0
- PyTorch 2.10.0
- SHAP 0.50.0

The remaining Python dependencies are listed in `requirements.txt`.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```text
.venv\Scripts\activate
```

## Running the Analysis

From the repository root:

```bash
python src/analysis_pipeline.py \
  --data data/processed.cleveland.data \
  --out reproduced_outputs
```

The generated files will be written to `reproduced_outputs/`. Compare these with the archived files in `results/`.

PyTorch results can show small platform-dependent numerical differences across hardware and backend versions even when deterministic seeds are used. The `results/` directory therefore provides the reference values associated with the manuscript.

## Results Included in the Repository

Yes—the repository intentionally includes the reference results. They are not required for executing the code, but they make the repository much more useful for readers and reviewers because the reported values can be inspected immediately and compared with a fresh run.

The repository includes numerical summaries and the principal figures; raw temporary/intermediate files are not included.

## Clinical Interpretation

This repository accompanies an internal comparative benchmark using a small historical dataset. The models are not presented as calibrated clinical decision-support tools, validated screening models, or evidence of clinical deployment readiness. The `ca`-exclusion analysis is a separate sensitivity analysis and should not be interpreted as a validated earlier-stage prediction model.

## License

The original source code and documentation in this repository are released under the **MIT License**. The Cleveland Heart Disease dataset is third-party material distributed by the UCI Machine Learning Repository under **CC BY 4.0** and must be attributed separately.
