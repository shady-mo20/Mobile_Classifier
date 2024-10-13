import joblib
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from typing import Dict, Any, Optional

def get_models() -> Dict[str, Any]:
    models = {
        'SVM': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=15, min_samples_split=5, min_samples_leaf=2),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=2),
        'Logistic Regression L1': LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
        'Logistic Regression L2': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_estimators=300, learning_rate=0.1),
        'LightGBM': LGBMClassifier(random_state=42, n_estimators=300, learning_rate=0.1)
    }
    return models

def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    model.fit(X_train, y_train)
    return model

def perform_hyperparameter_tuning(model: Any, param_grid: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series,
                                  cv: int = 5, scoring: str = 'roc_auc', n_iter: int = 50, random_state: int = 42) -> Any:
    if len(param_grid) > 10:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
            verbose=1,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
            verbose=1,
            n_jobs=-1
        )
    search.fit(X_train, y_train)
    return search.best_estimator_

def save_model(model: Any, model_name: str, output_dir: str = 'output/models/') -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    joblib.dump(model, os.path.join(output_dir, f'{model_name}.pkl'))

def load_model(model_path: str) -> Any:
    return joblib.load(model_path)

def train_and_save_models(models: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, hyperparameter_tuning: bool = False) -> Dict[str, Any]:
    trained_models = {}
    for name, model in models.items():
        if hyperparameter_tuning:
            param_grids = {
                'SVM': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear']
                },
                'Decision Tree': {
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Random Forest': {
                    'n_estimators': [100, 300, 500],
                    'max_depth': [20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Logistic Regression L1': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l1']
                },
                'Logistic Regression L2': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l2']
                },
                'K-Nearest Neighbors': {
                    'n_neighbors': [5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                'XGBoost': {
                    'n_estimators': [100, 300, 500],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1],
                    'colsample_bytree': [0.8, 1]
                },
                'LightGBM': {
                    'n_estimators': [100, 300, 500],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 70],
                    'subsample': [0.8, 1],
                    'colsample_bytree': [0.8, 1]
                }
            }
            if name in param_grids:
                model = perform_hyperparameter_tuning(
                    model, param_grids[name], X_train, y_train
                )
        model = train_model(model, X_train, y_train)
        save_model(model, name)
        trained_models[name] = model
    return trained_models
