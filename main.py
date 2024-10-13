import pandas as pd
import os
from data_processing import preprocess_data
from feature_engineering import feature_engineering_pipeline
from models import get_models, train_and_save_models
from evaluation import evaluate_model
from data_visualization import plot_numerical_features_distribution, plot_correlation_heatmap, plot_learning_curve

DATA_PATH = 'data/train.csv'

OUTPUT_DIR = 'output/'
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'models/')
PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'plots/')
REPORTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'reports/')

def create_output_directories():
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_OUTPUT_DIR, exist_ok=True)

def main():
    print("Loading and preprocessing data...")
    df = preprocess_data(DATA_PATH)

    print("Performing feature engineering...")
    X, y = feature_engineering_pipeline(df)

    print("Plotting numerical feature distributions...")
    plot_numerical_features_distribution(df, PLOTS_OUTPUT_DIR)

    print("Plotting correlation heatmap...")
    plot_correlation_heatmap(df, PLOTS_OUTPUT_DIR)

    print("Training models...")
    models = get_models()
    trained_models = train_and_save_models(models, X, y, hyperparameter_tuning=False)

    print("Evaluating models...")
    results = []
    for model_name, model in trained_models.items():
        print(f"Evaluating {model_name}...")
        metrics = evaluate_model(model, X, y, model_name)
        results.append(metrics)
    
    print("Generating final reports...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(REPORTS_OUTPUT_DIR, 'model_performance.csv'), index=False)
    
    print("Plotting learning curves for models...")
    for model_name, model in trained_models.items():
        plot_learning_curve(model, model_name, X, y, output_dir=PLOTS_OUTPUT_DIR)
    
    print("All tasks completed successfully!")

if __name__ == '__main__':
    create_output_directories()
    main()
