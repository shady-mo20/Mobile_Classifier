import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.model_selection import learning_curve
from typing import Optional, List, Any

def create_output_directory(output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def plot_numerical_features_distribution(df_imputed: pd.DataFrame, output_dir: str = 'output/plots/') -> None:
    create_output_directory(output_dir)
    numerical_cols = df_imputed.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        fig = px.histogram(df_imputed, x=col, nbins=30, title=f'Distribution of {col.capitalize()}',
                           marginal='box', opacity=0.75, labels={col: col.capitalize()}, 
                           color_discrete_sequence=['lightblue'])
        image_path = os.path.join(output_dir, f"{col}_distribution.html")
        fig.write_html(image_path)

def plot_correlation_heatmap(df_imputed: pd.DataFrame, output_dir: str = 'output/plots/') -> None:
    plt.figure(figsize=(18, 12))
    corr_matrix = df_imputed.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Features')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

def plot_interactive_correlation_heatmap(df_imputed: pd.DataFrame, output_dir: str = 'output/plots/') -> None:
    create_output_directory(output_dir)
    corr_matrix = df_imputed.corr()
    fig = px.imshow(corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu',
                    title='Interactive Correlation Matrix of Features')
    image_path = os.path.join(output_dir, 'interactive_correlation_matrix.html')
    fig.write_html(image_path)

def plot_pairplot(df_imputed: pd.DataFrame, output_dir: str = 'output/plots/') -> None:
    important_features = ['battery_power', 'ram', 'price_range']
    g = sns.pairplot(df_imputed[important_features], hue='price_range', palette='Set2')
    g.fig.suptitle('Pairplot of Key Features Colored by Price Range', y=1.02)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'pairplot.png'))
    plt.close()

def plot_boxplots(df_imputed: pd.DataFrame, output_dir: str = 'output/plots/') -> None:
    numerical_cols = df_imputed.select_dtypes(include=np.number).columns
    num_cols = len(numerical_cols)
    cols = 3
    rows = (num_cols // cols) + 1
    plt.figure(figsize=(cols * 6, rows * 4))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(rows, cols, i)
        sns.boxplot(y=df_imputed[col], color='lightgreen')
        plt.title(f'Boxplot of {col.capitalize()}')
    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'boxplots.png'))
    plt.close()

def plot_violin_plots(df_imputed: pd.DataFrame, output_dir: str = 'output/plots/') -> None:
    numerical_cols = df_imputed.select_dtypes(include=np.number).columns
    num_cols = len(numerical_cols)
    cols = 3
    rows = (num_cols // cols) + 1
    plt.figure(figsize=(cols * 6, rows * 4))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(rows, cols, i)
        sns.violinplot(y=df_imputed[col], color='lightblue')
        plt.title(f'Violin Plot of {col.capitalize()}')
    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'violin_plots.png'))
    plt.close()

def plot_pca_visualization(df_imputed: pd.DataFrame, output_dir: str = 'output/plots/') -> None:
    X = df_imputed.drop('price_range', axis=1)
    y = df_imputed['price_range']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, y.reset_index(drop=True)], axis=1)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='price_range', data=pca_df, palette='Set2', alpha=0.7)
    plt.title('PCA of Mobile Price Classification')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} Variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} Variance)')
    plt.legend(title='Price Range')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_visualization.png'))
    plt.close()

def plot_learning_curve(estimator: Any, title: str, X: pd.DataFrame, y: pd.Series,
                        cv: int = 5, n_jobs: int = -1, train_sizes: np.ndarray = np.linspace(.1, 1.0, 5),
                        output_dir: str = 'output/plots/') -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.figure(figsize=(10, 6))
    plt.title(title)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy'
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.tight_layout()
    image_path = os.path.join(output_dir, f'learning_curve_{title.replace(" ", "_")}.png')
    plt.savefig(image_path)
    plt.close()

def plot_performance_comparison(performance_df: pd.DataFrame, metrics: List[str] = ['Accuracy', 'Precision', 'Recall', 'F1_Score'], output_dir: str = 'output/plots/') -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Model', y=metric, data=performance_df, palette='Set2')
        plt.title(f'Comparison of {metric} Across Models')
        plt.ylim(0, 1)
        plt.ylabel(metric)
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        image_path = os.path.join(output_dir, f'performance_comparison_{metric}.png')
        plt.savefig(image_path)
        plt.close()
