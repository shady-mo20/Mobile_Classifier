import logging
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

def drop_column_importance(model: Any, X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 0) -> pd.DataFrame:
    model_clone = clone(model)
    if hasattr(model_clone, 'random_state'):
        model_clone.random_state = random_state
    model_clone.fit(X_train, y_train)
    baseline_score = cross_val_score(model_clone, X_train, y_train, cv=5, scoring='accuracy').mean()
    importances = []
    for col in X_train.columns:
        model_temp = clone(model)
        if hasattr(model_temp, 'random_state'):
            model_temp.random_state = random_state
        model_temp.fit(X_train.drop(col, axis=1), y_train)
        new_score = cross_val_score(model_temp, X_train.drop(col, axis=1), y_train, cv=5, scoring='accuracy').mean()
        importance = baseline_score - new_score
        importances.append(importance)
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    return importance_df

def permutation_importance_wrapper(model: Any, X_train: pd.DataFrame, y_train: pd.Series, n_repeats: int = 10,
                                    random_state: int = 0, scoring: str = 'accuracy') -> pd.DataFrame:
    result = permutation_importance(model, X_train, y_train, n_repeats=n_repeats,
                                    random_state=random_state, scoring=scoring, n_jobs=-1)
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance Mean': result.importances_mean,
        'Importance Std': result.importances_std
    }).sort_values(by='Importance Mean', ascending=False).reset_index(drop=True)
    return importance_df

def shap_importance(model: Any, X_train: pd.DataFrame, y_train: pd.Series, sample: Optional[int] = 100) -> pd.DataFrame:
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train.sample(n=sample, random_state=0) if sample else X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    image_path = os.path.join('output/plots/', 'shap_summary_plot.png')
    plt.savefig(image_path)
    plt.close()
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': shap_importance
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    return importance_df

def save_feature_importance(importance_df: pd.DataFrame, method: str, output_path: str = 'output/feature_importance/') -> None:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_path = os.path.join(output_path, f'{method}_feature_importance.csv')
    importance_df.to_csv(file_path, index=False)

def load_feature_importance(file_path: str) -> pd.DataFrame:
    importance_df = pd.read_csv(file_path)
    return importance_df

def visualize_feature_importance(importance_df: pd.DataFrame, method: str, output_path: str = 'output/feature_importance/') -> None:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette='viridis')
    plt.title(f'Top 20 Feature Importances ({method})')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plot_path = os.path.join(output_path, f'{method}_feature_importance.png')
    plt.savefig(plot_path)
    plt.close()

def shap_summary_plot(model: Any, X_train: pd.DataFrame, output_dir: str = 'output/plots/') -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    image_path = os.path.join(output_dir, 'shap_summary_plot.png')
    plt.savefig(image_path)
    plt.close()

def main_feature_importance(model: Any, X_train: pd.DataFrame, y_train: pd.Series, methods: List[str] = ['Drop', 'Permutation', 'SHAP'], 
                            random_state: int = 0, sample: int = 100) -> None:
    for method in methods:
        if method.lower() == 'drop':
            importance_df = drop_column_importance(model, X_train, y_train, random_state=random_state)
            save_feature_importance(importance_df, method='Drop')
            visualize_feature_importance(importance_df, method='Drop')
        elif method.lower() == 'permutation':
            importance_df = permutation_importance_wrapper(model, X_train, y_train, n_repeats=10, random_state=random_state, scoring='accuracy')
            save_feature_importance(importance_df, method='Permutation')
            visualize_feature_importance(importance_df, method='Permutation')
        elif method.lower() == 'shap':
            importance_df = shap_importance(model, X_train, y_train, sample=sample)
            save_feature_importance(importance_df, method='SHAP')
            visualize_feature_importance(importance_df, method='SHAP')
        else:
            continue
