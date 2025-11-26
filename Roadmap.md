<<<<<<< HEAD
You are my coding assistant. I am building a Loan Default Risk Prediction project.  
I already have a preprocessing.py file with data cleaning and feature engineering.

Now help me generate the next files in this project.  
Follow this roadmap and generate code step by step:

1. Create a "model_training.py" file that:

   - Imports the processed data from preprocessing.py
   - Trains Logistic Regression and Decision Tree models
   - Calculates accuracy, ROC-AUC, KS-statistic
   - Saves trained models using joblib
   - Prints performance summary in clean format

2. Create a "evaluation.py" file that:

   - Loads saved models
   - Plots ROC curve, Precision-Recall curve
   - Calculates confusion matrix
   - Generates risk segmentation buckets (low, medium, high)
   - Saves evaluation plots under /reports/

3. Create "app.py" (optional):

   - A simple FastAPI or Flask API
   - Accepts JSON borrower data
   - Returns predicted default probability and risk level

4. Create "README.md":

   - Installation steps
   - Explanation of preprocessing, feature engineering
   - Model performance results
   - Link to Power BI dashboard
   - How to run the project

5. Create folder structure automatically:
   /src
   preprocessing.py  
    model_training.py  
    evaluation.py  
    app.py  
   /data
   /models
   /reports
   /notebooks
   README.md

Please generate code only for the file currently open.  
Ask me which file to generate next after completing one.
=======
You are my coding assistant for building a complete Loan Default Risk Prediction project.
Follow this roadmap and generate code, explanations, and files step-by-step.
Keep outputs clean, structured, and beginner-friendly.

---

## PROJECT: Loan Default Risk Prediction

## GOAL

Build a complete end-to-end machine learning project that predicts whether a borrower will default on a loan using the LendingClub dataset.

## TECH STACK

Python, Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn, Power BI (for dashboard)

## CORE DELIVERABLES

1. Data cleaning & preprocessing script
2. EDA notebook with major insights
3. Feature engineering (income-to-loan, credit utilization, repayment consistency)
4. Model building (Logistic Regression + Decision Tree)
5. Evaluate models using ROC-AUC and KS-Statistic
6. Risk segmentation (Low, Medium, High)
7. Final prediction pipeline script
8. README.md for GitHub

## FOLDER STRUCTURE (Generate these folders)

loan_default_project/
data/
notebooks/
01_EDA.ipynb
02_Modeling.ipynb
src/
preprocessing.py
model_train.py
risk_scoring.py
powerbi/
README.md

---

## FILE-BY-FILE INSTRUCTIONS FOR COPILOT

---

### 1. preprocessing.py

- Load dataset
- Handle missing values
- Create engineered features:
  income_to_loan = annual_inc / loan_amnt
  credit_util = revol_bal / loan_amnt
  repayment_consistency (basic: on-time indicators)
- Encode categorical variables using OneHotEncoder
- Standardize numeric features
- Output clean X, y, and train-test split

### 2. 01_EDA.ipynb

Generate visuals:

- Distribution of loan amount, income, credit utilization
- Default rate by loan grade, term, purpose
- Boxplots for engineered features vs default
- Correlation heatmap
- Insight markdown sections

### 3. model_train.py

- Train Logistic Regression model
- Train Decision Tree classifier
- Calculate metrics:
  accuracy
  ROC-AUC
  KS-statistic (use scipy.stats.ks_2samp)
- Save models using joblib

### 4. 02_Modeling.ipynb

- Compare LR and Decision Tree visually using ROC curves
- Create risk tiers:
  low: prob < 0.25
  medium: 0.25–0.6
  high: > 0.6
- Export predictions CSV

### 5. risk_scoring.py

- Load trained model
- Accept new borrower features
- Output predicted probability + risk class

### 6. README.md (Copilot generates)

Sections:

- Project intro
- Dataset used
- How logistic regression works (simple explanation)
- How decision tree works
- Steps to run the project
- AUC and KS results
- Future improvements

---

## DEVELOPMENT STYLE

- Write clean, readable Python
- Add comments explaining important steps
- Keep functions modular
- Use try/except where helpful
- Make the project beginner-friendly

---

Start with creating the folder structure and generating preprocessing.py.
I will ask you for next files step-by-step.
>>>>>>> 8e2afad8415a6a34849a58480fae602d6202a7f3
