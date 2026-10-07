import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# 1. MODEL BENCHMARKING LOOP
def benchmark_models(X_train, y_train, X_test, y_test):
    """
    Melatih dan mengevaluasi beberapa model sekaligus.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        
        # Hitung ROC-AUC jika prediksi probabilitas tersedia
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = np.nan
            
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'ROC-AUC': auc,
            'Train Time (s)': round(fit_time, 3)
        })
        trained_models[name] = model
        
    summary_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    return summary_df, trained_models

# 2. DETAILED EVALUATION & PLOT
def evaluate_best_model(model, X_test, y_test, model_name="Best Model"):
    y_pred = model.predict(X_test)
    
    print(f"=== CLASSIFICATION REPORT: {model_name} ===")
    print(classification_report(y_test, y_pred))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()