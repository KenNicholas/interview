import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. QUICK-CHECK EDA FUNCTION
def quick_eda(df, target_col=None):
    print("=== DATA OVERVIEW ===")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Check Missing & Duplicates
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    dup_count = df.duplicated().sum()
    
    info_df = pd.DataFrame({
        'Dtype': df.dtypes,
        'Missing Values': missing,
        'Missing (%)': missing_pct.round(2),
        'Unique Values': df.nunique()
    })
    print(info_df)
    print(f"\nTotal Duplicate Rows: {dup_count}")
    
    # Check Target Distribution
    if target_col and target_col in df.columns:
        print(f"\n=== TARGET DISTRIBUTION ({target_col}) ===")
        counts = df[target_col].value_counts()
        pcts = df[target_col].value_counts(normalize=True) * 100
        target_df = pd.DataFrame({'Count': counts, 'Percentage (%)': pcts.round(2)})
        print(target_df)
        
        # Plot Target
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=target_col)
        plt.title(f"Target Distribution: {target_col}")
        plt.tight_layout()
        plt.show()

# 2. PREPROCESSING PIPELINE GENERATOR
def build_preprocessing_pipeline(X):

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    print(f"Numeric Columns ({len(num_cols)}): {num_cols}")
    print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}")
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )
    return preprocessor

# NLP
def build_nlp_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)))
    ])