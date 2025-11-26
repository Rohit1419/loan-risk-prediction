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
