import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from typing import Tuple, Any, Dict, Optional, List

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['screen_area'] = df['px_height'] * df['px_width']
    df['battery_power_per_wt'] = df['battery_power'] / df['mobile_wt']
    df['ram_to_int_memory_ratio'] = df['ram'] / df['int_memory']
    df['total_camera_pixels'] = df['fc'] + df['pc']
    df['battery_ram_interaction'] = df['battery_power'] * df['ram']
    df['screen_ram_interaction'] = df['screen_area'] * df['ram']
    df['battery_int_memory_interaction'] = df['battery_power'] * df['int_memory']
    skewed_features = ['battery_power', 'mobile_wt', 'px_height', 'px_width', 'ram', 'int_memory', 'screen_area']
    for feature in skewed_features:
        df[f'log_{feature}'] = np.log1p(df[feature])
    df['ram_squared'] = df['ram'] ** 2
    df['int_memory_squared'] = df['int_memory'] ** 2
    df['ram_category'] = pd.cut(df['ram'],
                                 bins=[0, 1000, 2000, 4000, np.inf],
                                 labels=['Low', 'Medium', 'High', 'Very High'])
    df['dual_sim_battery_interaction'] = df['dual_sim'] * df['battery_power']
    df['network_type'] = df[['three_g', 'four_g']].astype(str).agg('_'.join, axis=1)
    df['network_type'] = df['network_type'].replace({'0_0': 'No_3G_No_4G',
                                                    '0_1': 'No_3G_4G',
                                                    '1_0': '3G_No_4G',
                                                    '1_1': '3G_4G'})
    return df

def encode_features(df: pd.DataFrame, categorical_features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    if categorical_features is None:
        categorical_features = ['ram_category', 'network_type']
    X = pd.get_dummies(df.drop('price_range', axis=1), columns=categorical_features, drop_first=True)
    y = df['price_range']
    return X, y

def scale_features(X: pd.DataFrame, scaling_method: str = 'standard') -> pd.DataFrame:
    X = X.copy()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        scaler = None
    if scaler:
        X_scaled = scaler.fit_transform(X[numerical_features])
        X_scaled = pd.DataFrame(X_scaled, columns=numerical_features, index=X.index)
        X.update(X_scaled)
    variance_filter = VarianceThreshold(threshold=0.01)
    X_filtered = variance_filter.fit_transform(X)
    selected_features = X.columns[variance_filter.get_support()]
    X = pd.DataFrame(X_filtered, columns=selected_features, index=X.index)
    return X

def reduce_dimensionality(X: pd.DataFrame, n_components: int = 10, random_state: int = 42) -> pd.DataFrame:
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    pca_columns = [f'PC_{i+1}' for i in range(n_components)]
    X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
    return X_pca_df

def select_features(X: pd.DataFrame, y: pd.Series, k: int = 20) -> pd.DataFrame:
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    X_selected = pd.DataFrame(X_new, columns=selected_features, index=X.index)
    return X_selected

def feature_engineering_pipeline(df: pd.DataFrame, scaling_method: str = 'standard') -> Tuple[pd.DataFrame, pd.Series]:
    df = create_features(df)
    X, y = encode_features(df)
    X = scale_features(X, scaling_method)
    X = select_features(X, y, k=20)
    return X, y
