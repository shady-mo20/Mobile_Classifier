import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import learning_curve
import numpy as np
import pandas as pd
import os
from typing import Any, Dict, Optional, List

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    if y_proba is not None:
        try:
            if len(np.unique(y_test)) == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                plot_roc_curve(y_test, y_proba[:, 1], model_name)
            else:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                plot_multiclass_roc_curve(y_test, y_proba, model_name)
        except ValueError as e:
            print(f"Error calculating ROC AUC: {e}")
            roc_auc = None
    else:
        roc_auc = None
    
    if y_proba is not None:
        if len(np.unique(y_test)) > 2:
            plot_multiclass_precision_recall_curve(y_test, y_proba, model_name)
        else:
            plot_precision_recall_curve(y_test, y_proba[:, 1], model_name)
    
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc if roc_auc is not None else np.nan,
        'classification_report': report,
        'confusion_matrix': cm
    }
    plot_confusion_matrix(cm, model_name)
    
    return metrics

def plot_confusion_matrix(cm: np.ndarray, model_name: str, output_dir: str = 'output/plots/') -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    image_path = os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(image_path)
    plt.close()

def plot_roc_curve(y_test: pd.Series, y_proba: np.ndarray, model_name: str, output_dir: str = 'output/plots/') -> None:
    os.makedirs(output_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve for {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    image_path = os.path.join(output_dir, f'roc_curve_{model_name}.png')
    plt.savefig(image_path)
    plt.close()

def plot_multiclass_roc_curve(y_test: pd.Series, y_proba: np.ndarray, model_name: str, output_dir: str = 'output/plots/') -> None:
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    import itertools

    os.makedirs(output_dir, exist_ok=True)
    
    classes = np.unique(y_test)
    y_test_binarized = label_binarize(y_test, classes=classes)
    n_classes = y_test_binarized.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(8, 6))
    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    image_path = os.path.join(output_dir, f'roc_curve_{model_name}.png')
    plt.savefig(image_path)
    plt.close()

def plot_precision_recall_curve(y_test: pd.Series, y_proba: np.ndarray, model_name: str, output_dir: str = 'output/plots/') -> None:
    os.makedirs(output_dir, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.tight_layout()
    image_path = os.path.join(output_dir, f'precision_recall_curve_{model_name}.png')
    plt.savefig(image_path)
    plt.close()

def plot_multiclass_precision_recall_curve(y_test: pd.Series, y_proba: np.ndarray, model_name: str, output_dir: str = 'output/plots/') -> None:
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import precision_recall_curve, auc
    import itertools

    os.makedirs(output_dir, exist_ok=True)
    
    classes = np.unique(y_test)
    y_test_binarized = label_binarize(y_test, classes=classes)
    n_classes = y_test_binarized.shape[1]
    
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_proba[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
    
    plt.figure(figsize=(8, 6))
    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'PR curve of class {classes[i]} (area = {pr_auc[i]:0.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves for {model_name}')
    plt.legend(loc="lower left")
    plt.tight_layout()
    image_path = os.path.join(output_dir, f'precision_recall_curve_{model_name}.png')
    plt.savefig(image_path)
    plt.close()

def generate_text_report(results_df: pd.DataFrame, output_dir: str = 'output/reports/') -> None:
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write("Model Performance Summary:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nDetailed Classification Reports:\n")
        for model in results_df['Model']:
            classification_report_path = os.path.join(output_dir, f"classification_report_{model}.csv")
            if os.path.exists(classification_report_path):
                f.write(f"\nClassification Report for {model}:\n")
                report_df = pd.read_csv(classification_report_path)
                f.write(report_df.to_string(index=False))
                f.write("\n")
        f.write("\nGenerated Plots are saved in 'output/plots/' directory.\n")
    print(f"Report generated at {report_path}")
