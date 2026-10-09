# Credit Scoring Model Using Machine Learning

This project builds a credit-risk classification pipeline for the German Credit dataset.

## Key note

The CSV in the `credit/` folder is a feature-only version without the target `Risk` column. The script therefore checks for a target column first and falls back to the official UCI German Credit dataset via the `ucimlrepo` package when needed.

## Files

- `credit_scoring_project.py`: full training, evaluation, and prediction pipeline
- `requirements.txt`: package requirements
- `credit/german_credit_data (1).csv`: provided dataset snapshot

## Run the project

```bash
c:/python313/python.exe -m pip install -r requirements.txt
c:/python313/python.exe credit_scoring_project.py
```

## Outputs

The script saves evaluation results and charts into the `output/` folder, including:

- `model_results.csv`
- `roc_curve.png`
- `feature_importance.png`
- `random_forest_confusion_matrix.png`

## Models used

- Logistic Regression
- Decision Tree
- Random Forest

## Target encoding

- `Good` = 1
- `Bad` = 0

## Project goal

Build a working prototype credit risk model for educational purposes, not for real-world lending decisions.
