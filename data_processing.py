import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from typing import Tuple, Dict, Any, List, Optional

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df

def check_missing_values(df: pd.DataFrame) -> pd.Series:
    missing = df.isnull().sum()
    return missing

def handle_missing_values(df: pd.DataFrame, strategy: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    df_cleaned = df.copy()
    numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if strategy:
        for col, strat in strategy.items():
            if strat['method'] == 'knn':
                imputer = KNNImputer(n_neighbors=strat.get('n_neighbors', 5))
                df_cleaned[col] = imputer.fit_transform(df_cleaned[[col]])
            elif strat['method'] == 'mean':
                imputer = SimpleImputer(strategy='mean')
                df_cleaned[col] = imputer.fit_transform(df_cleaned[[col]])
            elif strat['method'] == 'median':
                imputer = SimpleImputer(strategy='median')
                df_cleaned[col] = imputer.fit_transform(df_cleaned[[col]])
            elif strat['method'] == 'most_frequent':
                imputer = SimpleImputer(strategy='most_frequent')
                df_cleaned[col] = imputer.fit_transform(df_cleaned[[col]])
            elif strat['method'] == 'constant':
                imputer = SimpleImputer(strategy='constant', fill_value=strat.get('fill_value', 'Unknown'))
                df_cleaned[col] = imputer.fit_transform(df_cleaned[[col]])
    else:
        imputer = KNNImputer(n_neighbors=5)
        df_cleaned[numerical_cols] = imputer.fit_transform(df_cleaned[numerical_cols])
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_cleaned[categorical_cols] = imputer_cat.fit_transform(df_cleaned[categorical_cols])
    
    return df_cleaned

def check_duplicate_rows(df: pd.DataFrame) -> int:
    duplicate_rows = df.duplicated().sum()
    return duplicate_rows

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.drop_duplicates()
    numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned[col] = np.where(df_cleaned[col] < lower_bound, lower_bound,
                                  np.where(df_cleaned[col] > upper_bound, upper_bound, df_cleaned[col]))
    return df_cleaned

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df_converted = df.copy()
    int_cols = df_converted.select_dtypes(include=['float64']).columns.tolist()
    for col in int_cols:
        if df_converted[col].dropna().apply(float.is_integer).all():
            df_converted[col] = df_converted[col].astype('int64')
    
    object_cols = df_converted.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        num_unique = df_converted[col].nunique()
        if num_unique / len(df_converted) < 0.5:
            df_converted[col] = df_converted[col].astype('category')
    
    return df_converted

def validate_data(df: pd.DataFrame, expected_shape: Tuple[int, int], required_columns: List[str]) -> bool:
    if df.shape != expected_shape:
        return False
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False
    return True

def preprocess_data(filepath: str, strategy: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    df = load_data(filepath)
    missing = check_missing_values(df)
    if missing.any():
        df = handle_missing_values(df, strategy)
    duplicate_count = check_duplicate_rows(df)
    if duplicate_count > 0:
        df = clean_data(df)
    df = convert_data_types(df)
    return df
