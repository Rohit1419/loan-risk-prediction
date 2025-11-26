# loan-risk-prediction

# Loan Default Risk Prediction

A machine learning system for predicting loan default risk using logistic regression and decision tree models.

## Project Structure

```
LoanPrediction/
├── src/
│   ├── preprocessing.py      # Data cleaning and feature engineering
│   ├── model_train.py       # Model training pipeline
│   ├── evaluation.py        # Model evaluation and analysis
│
├── data/
│   └── 2007_to_2018Q4.csv
├── models/
│   ├── logistic_regression_model.joblib
│   ├── decision_tree_model.joblib
│   ├── scaler.joblib
│   └── model_metrics.csv
├── reports/
│   ├── confusion_matrices.png
│   ├── precision_recall_curves.png
│   ├── calibration_curves.png
│   ├── risk_segmentation.png
│   ├── roc_comparison.png
│   ├── feature_importance.png
│   ├── risk_segmentation.csv
│   └── model_comparison.csv
└── README.md
```

## Installation

```bash
git clone <repository-url>
cd LoanPrediction
pip install pandas numpy scikit-learn matplotlib seaborn joblib fastapi uvicorn
```

## Usage

### 1. Train Models

```bash
python src/model_train.py
```

### 2. Evaluate Models

```bash
python src/evaluation.py
```

### 3. Run Web App

```bash
python src/app.py
# Visit: http://localhost:8000
```

## Model Performance

| Model               | ROC-AUC | Precision | Recall | F1-Score |
| ------------------- | ------- | --------- | ------ | -------- |
| Logistic Regression | 0.9951  | 0.9166    | 0.9553 | 0.9355   |
| Decision Tree       | 0.9253  | 0.4722    | 0.8522 | 0.6077   |

**Best Model:** Logistic Regression (99.5% ROC-AUC)

## Features

- **Data Processing:** Missing value handling, feature engineering
- **Model Training:** Logistic Regression and Decision Tree
- **Evaluation:** ROC curves, confusion matrices, risk segmentation
- **Risk Scoring:** Low/Medium/High/Very High risk categories
- **Web Interface:** Real-time loan risk assessment

## Risk Segmentation

| Risk Level     | Default Probability | Recommendation |
| -------------- | ------------------- | -------------- |
| Low Risk       | < 10%               | Approve        |
| Medium Risk    | 10-30%              | Review         |
| High Risk      | 30-70%              | Reject         |
| Very High Risk | > 70%               | Reject         |

## Key Files

- `preprocessing.py` - Data cleaning and feature engineering
- `model_train.py` - Model training pipeline
- `evaluation.py` - Performance evaluation and visualization
- `app.py` - Web application for predictions

## Dataset

- **Source:** Lending Club loan data (2007-2018)
- **URL:** https://www.kaggle.com/datasets/wordsforthewise/lending-club/data
- **Size:** 2.2M+ loan records
- **Features:** 151 original features, engineered to 175+ features
- **Target:** Binary classification (default vs non-default)

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- joblib
