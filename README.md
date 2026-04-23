# Telco Customer Churn Analysis (Sections 2/3/4)

This project is based on `WA_Fn-UseC_-Telco-Customer-Churn.csv` and covers the following parts:

- Part 2: Data analysis and opportunity identification (including statistical tables and visualizations)
- Part 3: Data preprocessing (cleaning, encoding, scaling, and feature engineering)
- Part 4: Statistical and advanced modeling (baseline model + advanced models + tuning + comparative evaluation + neural network)

## File Overview

- `section2_analysis.py`: EDA and key trend visualizations
- `section3_preprocessing.py`: preprocessing pipeline and intermediate artifact export
- `section4_modeling.py`: model training, hyperparameter tuning, and comprehensive evaluation
- `使用手册.md`: Chinese user manual (with chapter text that can be directly used in reports)

## Quick Start

```powershell
pip install -r requirements.txt
python section2_analysis.py
python section3_preprocessing.py
python section4_modeling.py
```

## Main Outputs

- Part 2 outputs: `img/data_feathers/section2_*`
- Part 3 outputs: `img/data_feathers/section3_*`
- Part 4 outputs: `img/ai_data_img/section4_*`

## Model Results Snapshot (Current Run)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 0.7299 | 0.4950 | 0.7888 | 0.6082 | 0.8370 |
| DecisionTree | 0.7065 | 0.4697 | 0.8075 | 0.5939 | 0.8229 |
| RandomForest | 0.7598 | 0.5335 | 0.7674 | 0.6294 | 0.8342 |
| GradientBoosting | 0.7953 | 0.6433 | 0.5160 | 0.5727 | 0.8389 |
| NeuralNetwork (MLP) | 0.7747 | 0.5922 | 0.4893 | 0.5359 | 0.8162 |

Note: The best advanced model is still `GradientBoosting`; `NeuralNetwork` has been included as a supplementary experiment in the full comparison.

## FAQ

1. Plot scripts hang in debug environments or raise Tk-related errors:
	- The scripts are configured to use the `Agg` backend for `matplotlib`. Please re-run using the latest scripts in this project.
2. Preprocessing export reports `ndarray` has no attribute `format`:
	- The script now consistently converts outputs to `csr_matrix` before saving. Please use the latest `section3_preprocessing.py`.
