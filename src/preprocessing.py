import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LoanDataProcessor:
    """
    Handles data loading, cleaning, and feature engineering for loan default prediction.
    """
    
    def __init__(self, filepath):
        print("Loading dataset...")
        self.df = pd.read_csv(filepath)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
    
    def handle_missing_values(self):
        """Handle missing values by dropping or filling"""
        # Drop columns with >50% missing values
        missing_pct = (self.df.isnull().sum() / len(self.df)) * 100
        cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
        self.df.drop(columns=cols_to_drop, inplace=True)
        
        # Fill numeric columns with median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        
        # Fill categorical columns with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        print("✓ Missing values handled")
        return self
    
    def engineer_features(self):
        """Create domain-specific features for credit risk"""
        try:
            # Income-to-loan ratio
            if 'annual_inc' in self.df.columns and 'loan_amnt' in self.df.columns:
                self.df['income_to_loan'] = self.df['annual_inc'] / (self.df['loan_amnt'] + 1)
            
            # Credit utilization ratio
            if 'revol_bal' in self.df.columns and 'revol_credit_limit' in self.df.columns:
                self.df['credit_util'] = self.df['revol_bal'] / (self.df['revol_credit_limit'] + 1)
                self.df['credit_util'] = self.df['credit_util'].clip(0, 1)  # Cap at 100%
            
            # Repayment consistency (based on delinquent accounts)
            if 'delinq_2yrs' in self.df.columns:
                self.df['repayment_consistency'] = 1 / (self.df['delinq_2yrs'] + 1)
            
            print("✓ Features engineered")
        except Exception as e:
            print(f"⚠ Feature engineering warning: {e}")
        
        return self
    
    def create_binary_target(self, target_col='loan_status'):
   
        print(f"🎯 Converting {target_col} to binary target...")
        
        # Check unique values
        unique_values = self.df[target_col].unique()
        print(f"   Original categories: {len(unique_values)} unique values")
        
        # Define default categories (bad loans)
        default_categories = [
            'Charged Off', 
            'Default', 
            'Late (31-120 days)', 
            'Late (16-30 days)',
            'Does not meet the credit policy. Status:Charged Off'
        ]
    
        # Create binary target
        self.df['is_default'] = self.df[target_col].isin(default_categories).astype(int)
        
        # Print distribution
        default_count = self.df['is_default'].sum()
        total_count = len(self.df)
        default_rate = default_count / total_count
        
        print(f"✅ Binary target created:")
        print(f"   Default (1): {default_count:,} ({default_rate:.2%})")
        print(f"   Non-Default (0): {total_count - default_count:,} ({1-default_rate:.2%})")
        
        return self
    
    def prepare_features_target(self, target_col='loan_status', test_size=0.2, random_state=42):
        """
        Separating the features and target, encode categoricals, standardize numerics.
        Returns X_train, X_test, y_train, y_test, scaler
        """
        
        # Convert target to binary first
        self.create_binary_target(target_col)
        
        # Use binary target
        y = self.df['is_default']
        X = self.df.drop(columns=[target_col, 'is_default'])  # Drop both original and binary target
        
        # Drop non-numeric, non-categorical columns
        X = X.drop(columns=[col for col in X.columns if col.lower() in ['id', 'member_id', 'url']], errors='ignore')
        
        # Separate numeric and categorical features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # MEMORY-EFFICIENT categorical encoding: Only keep low-cardinality features
        if categorical_features:
            # Keep only categorical features with < 50 unique values
            low_cardinality_cats = []
            for col in categorical_features:
                if X[col].nunique() < 50:
                    low_cardinality_cats.append(col)
                else:
                    print(f"Dropping high-cardinality feature: {col} ({X[col].nunique()} unique values)")
            
            if low_cardinality_cats:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_cats = encoder.fit_transform(X[low_cardinality_cats])
                encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(low_cardinality_cats), index=X.index)
                X = pd.concat([X[numeric_features].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
                print(f"✓ Encoded {len(low_cardinality_cats)} categorical features")
            else:
                X = X[numeric_features]
                print("✓ Using only numeric features")
        else:
            X = X[numeric_features]
        
        # Standardize numeric features
        scaler = StandardScaler()
        if len(numeric_features) > 0:
            X[numeric_features] = scaler.fit_transform(X[numeric_features])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        print(f"✓ Features prepared: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        print(f"  Final features: {X_train.shape[1]} total")
        
        return X_train, X_test, y_train, y_test, scaler
# Example usage
if __name__ == "__main__":
    # NOTE: Replace with your actual dataset path
    processor = LoanDataProcessor('data/2007_to_2018Q4.csv')
    processor.handle_missing_values()
    processor.engineer_features()
    X_train, X_test, y_train, y_test, scaler = processor.prepare_features_target()
    print("\n✓ Preprocessing complete! Ready for modeling.")