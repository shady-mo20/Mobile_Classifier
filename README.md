```markdown
# Telco Customer Churn Prediction

## Project Overview

This project aims to predict customer churn for a telecommunications company using machine learning models. The dataset used contains information about customer demographics, services subscribed to, account details, and whether the customer churned or not. The goal is to build and evaluate models that can accurately predict whether a customer will churn.

## Project Files

### 1. **main.py**
This is the central script that orchestrates the entire process of data loading, preprocessing, model training, and evaluation.

- **Steps included**:
  1. **Data Loading**: Calls the `preprocess_data()` function from `data_processing.py` to load and clean the dataset.
  2. **Feature Engineering**: Executes the `feature_engineering_pipeline()` to create new features and prepare data for modeling.
  3. **Visualization**: Generates visualizations like feature distributions and correlation matrices using functions from `data_visualization.py`.
  4. **Model Training**: Fetches models from `models.py` and trains them on the processed data.
  5. **Evaluation**: Evaluates the models and stores the results in CSV and graphical format.
  6. **Learning Curves**: Plots learning curves for all models to observe performance over the training process.

### 2. **data_processing.py**
This file contains the logic for preprocessing the dataset.

- **Functions**:
  - `load_data()`: Loads the dataset from a CSV file.
  - `handle_missing_values()`: Imputes missing values using various methods (KNN, mean, etc.).
  - `check_duplicate_rows()`: Identifies and removes any duplicate rows.
  - `clean_data()`: Cleans data by removing outliers and duplicates.
  - `convert_data_types()`: Converts the types of the columns for better memory and performance.

### 3. **feature_engineering.py**
Contains functions for feature engineering and feature selection.

- **Key Features**:
  - `create_features()`: Generates new features like `screen_area`, `ram_to_int_memory_ratio`, etc.
  - `scale_features()`: Standardizes or normalizes numerical features to bring them to a similar scale.
  - `encode_features()`: Encodes categorical variables using One-Hot Encoding.
  - `select_features()`: Selects the most relevant features using statistical methods like `SelectKBest`.

### 4. **models.py**
Defines and trains machine learning models for churn prediction.

- **Functions**:
  - `get_models()`: Initializes and returns a dictionary of machine learning models (SVM, Random Forest, XGBoost, etc.).
  - `train_and_save_models()`: Trains the models on the provided dataset and saves them for future use.
  - `perform_hyperparameter_tuning()`: Performs hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.

### 5. **evaluation.py**
Handles the evaluation of the trained models.

- **Functions**:
  - `evaluate_model()`: Computes metrics like accuracy, precision, recall, F1-score, and ROC AUC for each model.
  - `plot_confusion_matrix()`: Visualizes the confusion matrix for each model.
  - `plot_roc_curve()`: Plots ROC curves to measure the model’s performance in distinguishing between churn and non-churn customers.

### 6. **data_visualization.py**
This file provides several functions for visualizing the data and model results.

- **Functions**:
  - `plot_numerical_features_distribution()`: Plots the distribution of numerical features to explore the data.
  - `plot_correlation_heatmap()`: Creates a heatmap to visualize correlations between different features.
  - `plot_learning_curve()`: Plots learning curves to show how the model's performance improves with more data.
  - `plot_performance_comparison()`: Visualizes model performance across different evaluation metrics.

### 7. **utils.py**
This file provides utility functions, mainly for calculating feature importance.

- **Functions**:
  - `shap_importance()`: Uses SHAP values to calculate and plot feature importance for each model.
  - `permutation_importance_wrapper()`: Calculates feature importance by measuring the decrease in model accuracy after shuffling each feature.

### 8. **data_info.py**
Provides a function to generate and save comprehensive information about the dataset.

- **Functions**:
  - `display_data_info()`: Saves DataFrame info, shape, descriptive statistics, and first 5 rows to a text file.

## Data Description

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
```

### **DataFrame Shape**

```
Shape: (2000, 21)
```

### **Descriptive Statistics**

```python
       battery_power       blue  clock_speed     dual_sim  ...      three_g  touch_screen         wifi  price_range
count    2000.000000  2000.0000  2000.000000  2000.000000  ...  2000.000000   2000.000000  2000.000000  2000.000000
mean     1238.518500     0.4950     1.522250     0.509500  ...     0.761500      0.503000     0.507000     1.500000
std       439.418206     0.5001     0.816004     0.500035  ...     0.426273      0.500116     0.500076     1.118314
min       501.000000     0.0000     0.500000     0.000000  ...     0.000000      0.000000     0.000000     0.000000
25%       851.750000     0.0000     0.700000     0.000000  ...     1.000000      0.000000     0.000000     0.750000
50%      1226.000000     0.0000     1.500000     1.000000  ...     1.000000      1.000000     1.000000     1.500000
75%      1615.250000     1.0000     2.200000     1.000000  ...     1.000000      1.000000     1.000000     2.250000
max      1998.000000     1.0000     3.000000     1.000000  ...     1.000000      1.000000     1.000000     3.000000

[8 rows x 21 columns]
```

### **First 5 Rows**

|   | battery_power | blue | clock_speed | dual_sim | fc | four_g | int_memory | m_dep | mobile_wt | n_cores | pc | px_height | px_width | ram | sc_h | sc_w | talk_time | three_g | touch_screen | wifi | price_range |
|---|---------------|------|-------------|----------|----|--------|------------|-------|-----------|---------|----|-----------|----------|-----|------|------|-----------|---------|--------------|------|-------------|
| 0 | 842           | 0    | 2.2         | 0        | 1  | 0      | 64         | 0.6   | 86        | 2       | 16 | 9         | 67       | 4   | 7    | 7    | 19        | 0       | 0            | 1    | 1           |
| 1 | 1021          | 1    | 0.5         | 1        | 0  | 1      | 32         | 0.0   | 58        | 2       | 16 | 12        | 76       | 3   | 3    | 3    | 7         | 1       | 1            | 0    | 2           |
| 2 | 563           | 1    | 0.5         | 1        | 2  | 1      | 16         | 0.3   | 73        | 4       | 16 | 8         | 57       | 2   | 2    | 2    | 9         | 1       | 1            | 0    | 2           |
| 3 | 615           | 1    | 2.5         | 0        | 0  | 0      | 64         | 0.4   | 89        | 4       | 16 | 11        | 68       | 4   | 8    | 8    | 11        | 1       | 0            | 0    | 2           |
| 4 | 1821          | 1    | 1.2         | 0        | 13 | 1      | 256        | 0.7   | 86        | 2       | 16 | 7         | 63       | 8   | 2    | 2    | 15        | 1       | 1            | 0    | 1           |

## Project Workflow

1. **Preprocessing**: 
   - Data is loaded, cleaned, and prepared using functions from `data_processing.py`.
   - Missing values are handled, outliers are removed, and data types are converted to optimize memory usage.

2. **Feature Engineering**: 
   - New features are created and selected using `feature_engineering.py`.
   - Categorical features are encoded, and numerical features are scaled for better performance.

3. **Model Training**: 
   - Models like SVM, Random Forest, XGBoost, and others are initialized in `models.py`.
   - The models are trained on the processed data, and the best-performing model is selected based on evaluation metrics.

4. **Evaluation**: 
   - Model performance is evaluated using metrics like accuracy, precision, recall, F1-score, and ROC AUC.
   - Confusion matrices and ROC curves are generated to better understand model predictions.

5. **Visualization**:
   - Data distributions, feature correlations, and learning curves are visualized to help better understand the model's performance and the dataset.

6. **Reporting**: 
   - The results of the model evaluations are stored in CSV and PNG formats for further analysis and reporting.

## Running the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shady-mo20/telco-churn-prediction.git
   cd telco-churn-prediction
   ```

2. **Install Dependencies**:
   Ensure all required libraries are installed. You can use a `requirements.txt` file if available, or install libraries manually:
   ```bash
   pip install -r requirements.txt
   ```
   or
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. **Run the Main Script**:
   ```bash
   python main.py
   ```

4. **View Model Performance**:
   - **Model performance metrics** will be saved in `output/reports/`.
   - **Confusion matrices and ROC curves** will be saved in `output/plots/`.

## Contact

For questions or feedback, contact:

Shady Mohammed: [shasdymohammed@gmail.com](mailto:shasdymohammed@gmail.com)  
GitHub: [https://github.com/shady-mo20](https://github.com/shady-mo20)
