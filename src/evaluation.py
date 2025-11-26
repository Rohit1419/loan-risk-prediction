import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve
import warnings

warnings.filterwarnings('ignore')

class LoanModelEvaluator:
    def __init__(self, models_dir='models', reports_dir='reports'):
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.load_models()
        self.load_test_data()
        
    def load_models(self):
        self.lr_model = joblib.load(f'{self.models_dir}/logistic_regression_model.joblib')
        self.dt_model = joblib.load(f'{self.models_dir}/decision_tree_model.joblib')
        self.scaler = joblib.load(f'{self.models_dir}/scaler.joblib')
        self.metrics_df = pd.read_csv(f'{self.models_dir}/model_metrics.csv', index_col=0)
        
    def load_test_data(self):
        from preprocessing import LoanDataProcessor
        
        processor = LoanDataProcessor('data/accepted_2007_to_2018Q4.csv')
        processor.handle_missing_values()
        processor.engineer_features()
        
        if len(processor.df) > 500000:
            processor.df = processor.df.sample(n=500000, random_state=42)
        
        _, X_test, _, y_test, _ = processor.prepare_features_target()
        self.X_test = X_test
        self.y_test = y_test
        
    def generate_predictions(self):
        self.lr_pred = self.lr_model.predict(self.X_test)
        self.lr_pred_proba = self.lr_model.predict_proba(self.X_test)[:, 1]
        self.dt_pred = self.dt_model.predict(self.X_test)
        self.dt_pred_proba = self.dt_model.predict_proba(self.X_test)[:, 1]
        
    def plot_confusion_matrices(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        models = [('Logistic Regression', self.lr_pred), ('Decision Tree', self.dt_pred)]
        
        for i, (name, pred) in enumerate(models):
            cm = confusion_matrix(self.y_test, pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(name)
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{self.reports_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_precision_recall_curves(self):
        plt.figure(figsize=(8, 6))
        models = [('Logistic Regression', self.lr_pred_proba), ('Decision Tree', self.dt_pred_proba)]
        
        for name, pred_proba in models:
            precision, recall, _ = precision_recall_curve(self.y_test, pred_proba)
            plt.plot(recall, precision, linewidth=2, label=name)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{self.reports_dir}/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_calibration_curves(self):
        plt.figure(figsize=(8, 6))
        models = [('Logistic Regression', self.lr_pred_proba), ('Decision Tree', self.dt_pred_proba)]
        
        for name, pred_proba in models:
            fraction_pos, mean_pred = calibration_curve(self.y_test, pred_proba, n_bins=10)
            plt.plot(mean_pred, fraction_pos, "s-", linewidth=2, label=name)
        
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{self.reports_dir}/calibration_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_risk_segmentation(self):
        pred_proba = self.lr_pred_proba
        risk_segments = pd.cut(pred_proba, bins=[0, 0.1, 0.3, 0.7, 1.0], 
                              labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])
        
        analysis_df = pd.DataFrame({
            'actual_default': self.y_test.values,
            'predicted_prob': pred_proba,
            'risk_segment': risk_segments
        })
        
        segment_stats = analysis_df.groupby('risk_segment').agg({
            'actual_default': ['count', 'sum', 'mean'],
            'predicted_prob': 'mean'
        }).round(4)
        
        segment_stats.columns = ['Count', 'Defaults', 'Default_Rate', 'Avg_Pred_Prob']
        segment_stats.to_csv(f'{self.reports_dir}/risk_segmentation.csv')
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        segment_stats['Default_Rate'].plot(kind='bar', ax=axes[0])
        segment_stats['Count'].plot(kind='bar', ax=axes[1])
        axes[0].set_title('Default Rate by Risk Segment')
        axes[1].set_title('Volume by Risk Segment')
        
        plt.tight_layout()
        plt.savefig(f'{self.reports_dir}/risk_segmentation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return segment_stats
        
    def generate_report(self):
        lr_report = classification_report(self.y_test, self.lr_pred, output_dict=True)
        dt_report = classification_report(self.y_test, self.dt_pred, output_dict=True)
        
        comparison = {
            'Logistic Regression': {
                'ROC-AUC': self.metrics_df.loc['Logistic Regression', 'roc_auc'],
                'Precision': lr_report['1']['precision'],
                'Recall': lr_report['1']['recall'],
                'F1-Score': lr_report['1']['f1-score']
            },
            'Decision Tree': {
                'ROC-AUC': self.metrics_df.loc['Decision Tree', 'roc_auc'],
                'Precision': dt_report['1']['precision'],
                'Recall': dt_report['1']['recall'],
                'F1-Score': dt_report['1']['f1-score']
            }
        }
        
        results_df = pd.DataFrame(comparison).T
        results_df.to_csv(f'{self.reports_dir}/model_comparison.csv')
        return results_df
        
    def run_evaluation(self):
        print("Running model evaluation...")
        
        self.generate_predictions()
        self.plot_confusion_matrices()
        self.plot_precision_recall_curves()
        self.plot_calibration_curves()
        
        risk_stats = self.create_risk_segmentation()
        results = self.generate_report()
        
        print("\nEvaluation complete!")
        print("Files saved in reports/ directory:")
        print("- confusion_matrices.png")
        print("- precision_recall_curves.png")
        print("- calibration_curves.png")
        print("- risk_segmentation.png")
        print("- risk_segmentation.csv")
        print("- model_comparison.csv")
        
        return results


if __name__ == "__main__":
    evaluator = LoanModelEvaluator()
    results = evaluator.run_evaluation()
    print("\nModel Comparison Results:")
    print(results.round(4))