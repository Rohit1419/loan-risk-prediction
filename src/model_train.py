import pandas as pd
import numpy as np
import joblib
import warnings
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import LoanDataProcessor

warnings.filterwarnings('ignore')

class LoanModelTrainer:
    """
    Train and evaluate loan default prediction models
    """
    
    def __init__(self, data_path, sample_size=None):
        """Initialize with data path and optional sampling"""
        print("🚀 Starting Model Training Pipeline...")
        self.data_path = data_path
        self.sample_size = sample_size
        self.models = {}
        self.metrics = {}
        
    def prepare_data(self):
        """Load and prepare data using preprocessing pipeline"""
        print("\n📊 Loading and preprocessing data...")
        
        # Use our preprocessing pipeline
        processor = LoanDataProcessor(self.data_path)
        processor.handle_missing_values()
        processor.engineer_features()
        
        # Optional: Sample data for faster training
        if self.sample_size and len(processor.df) > self.sample_size:
            print(f"🔄 Sampling {self.sample_size:,} records for training...")
            processor.df = processor.df.sample(n=self.sample_size, random_state=42)
        
        # Get train/test splits
        X_train, X_test, y_train, y_test, scaler = processor.prepare_features_target()
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = scaler
        
        print(f"✅ Data prepared:")
        print(f"   Train set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        print(f"   Test set: {X_test.shape[0]:,} samples")
        print(f"   Default rate (train): {y_train.mean():.2%}")
        print(f"   Default rate (test): {y_test.mean():.2%}")
        
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n🔄 Training Logistic Regression...")
        print("   This may take a few minutes...")
        
        # Train model with optimized parameters
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=200,  # Reduced iterations
            class_weight='balanced',
            solver='liblinear',  # Faster solver for large datasets
            n_jobs=-1  # Use all CPU cores
        )
        
        lr_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        print("   Making predictions...")
        y_pred = lr_model.predict(self.X_test)
        y_pred_proba = lr_model.predict_proba(self.X_test)[:, 1]
        
        # Store model and predictions
        self.models['logistic_regression'] = lr_model
        self.predictions_lr = {'y_pred': y_pred, 'y_pred_proba': y_pred_proba}
        
        print("✅ Logistic Regression trained successfully")
        
    def train_decision_tree(self):
        """Train Decision Tree model"""
        print("\n🌲 Training Decision Tree...")
        print("   This may take a few minutes...")
        
        # Train model with conservative parameters
        dt_model = DecisionTreeClassifier(
            random_state=42,
            max_depth=8,  # Reduced depth for faster training
            min_samples_split=2000,  # Increased to speed up
            min_samples_leaf=1000,   # Increased to speed up
            class_weight='balanced',
            max_features='sqrt'  # Use subset of features for speed
        )
        
        dt_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        print("   Making predictions...")
        y_pred = dt_model.predict(self.X_test)
        y_pred_proba = dt_model.predict_proba(self.X_test)[:, 1]
        
        # Store model and predictions
        self.models['decision_tree'] = dt_model
        self.predictions_dt = {'y_pred': y_pred, 'y_pred_proba': y_pred_proba}
        
        print("✅ Decision Tree trained successfully")
        
    def calculate_ks_statistic(self, y_true, y_pred_proba):
        """Calculate KS statistic for model evaluation"""
        # Sample data if too large for KS test
        if len(y_true) > 50000:
            sample_idx = np.random.choice(len(y_true), 50000, replace=False)
            y_true_sample = y_true.iloc[sample_idx] if hasattr(y_true, 'iloc') else y_true[sample_idx]
            y_pred_sample = y_pred_proba[sample_idx]
        else:
            y_true_sample = y_true
            y_pred_sample = y_pred_proba
        
        # Separate predictions by actual class
        defaults = y_pred_sample[y_true_sample == 1]
        non_defaults = y_pred_sample[y_true_sample == 0]
        
        # Calculate KS statistic
        ks_stat, p_value = ks_2samp(defaults, non_defaults)
        return ks_stat
        
    def evaluate_models(self):
        """Evaluate both models and compare performance"""
        print("\n📈 Evaluating Model Performance...")
        
        models_to_eval = [
            ('Logistic Regression', self.predictions_lr),
            ('Decision Tree', self.predictions_dt)
        ]
        
        for model_name, predictions in models_to_eval:
            print(f"   Evaluating {model_name}...")
            y_pred = predictions['y_pred']
            y_pred_proba = predictions['y_pred_proba']
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            ks_stat = self.calculate_ks_statistic(self.y_test, y_pred_proba)
            
            # Store metrics
            self.metrics[model_name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'ks_statistic': ks_stat
            }
            
            print(f"\n📊 {model_name} Performance:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   ROC-AUC: {roc_auc:.4f}")
            print(f"   KS-Statistic: {ks_stat:.4f}")
            
    def plot_roc_curves(self):
        """Plot ROC curves for both models"""
        print("\n📊 Generating ROC Curves...")
        
        # Create reports directory
        os.makedirs('reports', exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curves for both models
        colors = ['blue', 'red']
        models_to_plot = [
            ('Logistic Regression', self.predictions_lr),
            ('Decision Tree', self.predictions_dt)
        ]
        
        for i, (model_name, predictions) in enumerate(models_to_plot):
            y_pred_proba = predictions['y_pred_proba']
            
            # Sample data if too large for plotting
            if len(self.y_test) > 100000:
                sample_idx = np.random.choice(len(self.y_test), 100000, replace=False)
                y_test_sample = self.y_test.iloc[sample_idx] if hasattr(self.y_test, 'iloc') else self.y_test[sample_idx]
                y_proba_sample = y_pred_proba[sample_idx]
            else:
                y_test_sample = self.y_test
                y_proba_sample = y_pred_proba
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test_sample, y_proba_sample)
            auc_score = self.metrics[model_name]['roc_auc']
            
            # Plot
            plt.plot(fpr, tpr, color=colors[i], linewidth=2,
                    label=f'{model_name} (AUC = {auc_score:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
        
        # Formatting
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison - Loan Default Prediction')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/roc_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def feature_importance_analysis(self):
        """Analyze feature importance for both models"""
        print("\n🔍 Feature Importance Analysis...")
        
        # Get feature names
        feature_names = self.X_train.columns.tolist()
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Logistic Regression coefficients
        lr_coef = abs(self.models['logistic_regression'].coef_[0])
        top_lr_idx = np.argsort(lr_coef)[-15:]  # Top 15 features
        
        axes[0].barh(range(15), lr_coef[top_lr_idx], color='skyblue')
        axes[0].set_yticks(range(15))
        axes[0].set_yticklabels([feature_names[i] for i in top_lr_idx])
        axes[0].set_title('Logistic Regression - Top 15 Feature Importance')
        axes[0].set_xlabel('Absolute Coefficient Value')
        
        # Decision Tree feature importance
        dt_importance = self.models['decision_tree'].feature_importances_
        top_dt_idx = np.argsort(dt_importance)[-15:]  # Top 15 features
        
        axes[1].barh(range(15), dt_importance[top_dt_idx], color='lightcoral')
        axes[1].set_yticks(range(15))
        axes[1].set_yticklabels([feature_names[i] for i in top_dt_idx])
        axes[1].set_title('Decision Tree - Top 15 Feature Importance')
        axes[1].set_xlabel('Feature Importance')
        
        plt.tight_layout()
        plt.savefig('reports/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_models(self):
        """Save trained models and scaler"""
        print("\n💾 Saving models...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save models
        joblib.dump(self.models['logistic_regression'], 'models/logistic_regression_model.joblib')
        joblib.dump(self.models['decision_tree'], 'models/decision_tree_model.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
        
        # Save metrics
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.to_csv('models/model_metrics.csv')
        
        print("✅ Models saved successfully:")
        print("   - models/logistic_regression_model.joblib")
        print("   - models/decision_tree_model.joblib")
        print("   - models/scaler.joblib")
        print("   - models/model_metrics.csv")
        
    def generate_performance_summary(self):
        """Generate comprehensive performance summary"""
        print("\n" + "="*60)
        print("📋 MODEL TRAINING SUMMARY")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.metrics).T
        comparison_df = comparison_df.round(4)
        
        print("\n📊 Performance Metrics Comparison:")
        print(comparison_df.to_string())
        
        # Determine best model
        best_model_auc = comparison_df['roc_auc'].idxmax()
        best_model_ks = comparison_df['ks_statistic'].idxmax()
        
        print(f"\n🏆 Best Model by ROC-AUC: {best_model_auc} ({comparison_df.loc[best_model_auc, 'roc_auc']:.4f})")
        print(f"🏆 Best Model by KS-Statistic: {best_model_ks} ({comparison_df.loc[best_model_ks, 'ks_statistic']:.4f})")
        
        # Performance interpretation
        min_auc = comparison_df['roc_auc'].min()
        min_ks = comparison_df['ks_statistic'].min()
        
        print(f"\n💡 Model Performance Interpretation:")
        print(f"   • ROC-AUC Range: {comparison_df['roc_auc'].min():.3f} - {comparison_df['roc_auc'].max():.3f}")
        print(f"   • KS-Statistic Range: {comparison_df['ks_statistic'].min():.3f} - {comparison_df['ks_statistic'].max():.3f}")
        print(f"   • Overall Assessment: {'Excellent' if min_auc > 0.8 else 'Good' if min_auc > 0.7 else 'Fair'} predictive power")
        
        print(f"\n📁 Generated Files:")
        print(f"   • ROC Curves: reports/roc_comparison.png")
        print(f"   • Feature Importance: reports/feature_importance.png")
        print(f"   • Model Files: models/ directory")
        
        print("\n🎉 Model Training Pipeline Complete!")
        print("✅ Ready for next step: Evaluation & Risk Scoring!")
        
    def run_full_pipeline(self):
        """Run the complete model training pipeline"""
        self.prepare_data()
        self.train_logistic_regression()
        self.train_decision_tree()
        self.evaluate_models()
        self.plot_roc_curves()
        self.feature_importance_analysis()
        self.save_models()
        self.generate_performance_summary()


if __name__ == "__main__":
    # Initialize trainer with sampling for faster training
    trainer = LoanModelTrainer('data/2007_to_2018Q4.csv', sample_size=500000)
    
    # Run complete pipeline
    trainer.run_full_pipeline()