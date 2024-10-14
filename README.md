Telco Customer Churn Prediction

Project Overview

This project aims to predict customer churn for a telecommunications company using machine learning models. The dataset used contains information about customer demographics, services subscribed to, account details, and whether the customer churned or not. The goal is to build and evaluate models that can accurately predict whether a customer will churn.

Project Files

1. **main.py**
This is the central script that orchestrates the entire process of data loading, preprocessing, model training, and evaluation.

- **Steps included**:
  1. **Data Loading**: Calls the `preprocess_data()` function from `data_processing.py` to load and clean the dataset.
  2. **Feature Engineering**: Executes the `feature_engineering_pipeline()` to create new features and prepare data for modeling.
  3. **Visualization**: Generates visualizations like feature distributions and correlation matrices using functions from `data_visualization.py`.
  4. **Model Training**: Fetches models from `models.py` and trains them on the processed data.
  5. **Evaluation**: Evaluates the models and stores the results in CSV and graphical format.
  6. **Learning Curves**: Plots learning curves for all models to observe performance over the training process.

2. **data_processing.py**
This file contains the logic for preprocessing the dataset.

- **Functions**:
  - `load_data()`: Loads the dataset from a CSV file.
  - `handle_missing_values()`: Imputes missing values using various methods (KNN, mean, etc.).
  - `check_duplicate_rows()`: Identifies and removes any duplicate rows.
  - `clean_data()`: Cleans data by removing outliers and duplicates.
  - `convert_data_types()`: Converts the types of the columns for better memory and performance.

3. **feature_engineering.py**
Contains functions for feature engineering and feature selection.

- **Key Features**:
  - `create_features()`: Generates new features like `screen_area`, `ram_to_int_memory_ratio`, etc.
  - `scale_features()`: Standardizes or normalizes numerical features to bring them to a similar scale.
  - `encode_features()`: Encodes categorical variables using One-Hot Encoding.
  - `select_features()`: Selects the most relevant features using statistical methods like `SelectKBest`.

4. **models.py**
Defines and trains machine learning models for churn prediction.

- **Functions**:
  - `get_models()`: Initializes and returns a dictionary of machine learning models (SVM, Random Forest, XGBoost, etc.).
  - `train_and_save_models()`: Trains the models on the provided dataset and saves them for future use.
  - `perform_hyperparameter_tuning()`: Performs hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.

5. **evaluation.py**
Handles the evaluation of the trained models.

- **Functions**:
  - `evaluate_model()`: Computes metrics like accuracy, precision, recall, F1-score, and ROC AUC for each model.
  - `plot_confusion_matrix()`: Visualizes the confusion matrix for each model.
  - `plot_roc_curve()`: Plots ROC curves to measure the model’s performance in distinguishing between churn and non-churn customers.

6. **data_visualization.py**
This file provides several functions for visualizing the data and model results.

- **Functions**:
  - `plot_numerical_features_distribution()`: Plots the distribution of numerical features to explore the data.
  - `plot_correlation_heatmap()`: Creates a heatmap to visualize correlations between different features.
  - `plot_learning_curve()`: Plots learning curves to show how the model's performance improves with more data.
  - `plot_performance_comparison()`: Visualizes model performance across different evaluation metrics.

7. **utils.py**
This file provides utility functions, mainly for calculating feature importance.

- **Functions**:
  - `shap_importance()`: Uses SHAP values to calculate and plot feature importance for each model.
  - `permutation_importance_wrapper()`: Calculates feature importance by measuring the decrease in model accuracy after shuffling each feature.

8. **data_info.py**
Provides a function to generate and save comprehensive information about the dataset.

- **Functions**:
  - `display_data_info()`: Saves DataFrame info, shape, descriptive statistics, and first 5 rows to a text file.

Data Description

### **Dataset Overview**

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2000 entries, 0 to 1999
Data columns (total 21 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   battery_power  2000 non-null   int64  
 1   blue           2000 non-null   int64  
 2   clock_speed    2000 non-null   float64
 3   dual_sim       2000 non-null   int64  
 4   fc             2000 non-null   int64  
 5   four_g         2000 non-null   int64  
 6   int_memory     2000 non-null   int64  
 7   m_dep          2000 non-null   float64
 8   mobile_wt      2000 non-null   int64  
 9   n_cores        2000 non-null   int64  
 10  pc             2000 non-null   int64  
 11  px_height      2000 non-null   int64  
 12  px_width       2000 non-null   int64  
 13  ram            2000 non-null   int64  
 14  sc_h           2000 non-null   int64  
 15  sc_w           2000 non-null   int64  
 16  talk_time      2000 non-null   int64  
 17  three_g        2000 non-null   int64  
 18  touch_screen   2000 non-null   int64  
 19  wifi           2000 non-null   int64  
 20  price_range    2000 non-null   int64  
dtypes: float64(2), int64(19)
memory usage: 328.3 KB
