"""
Visualization utilities for training metrics and confusion matrices
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def plot_training_history(history: Dict, save_path: str = None):
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        history: Training history dictionary from model.fit()
        save_path: Optional path to save the figure
    """
    # Extract metric names
    loss_metrics = [key for key in history.keys() if 'loss' in key and 'val' not in key]
    val_loss_metrics = [key for key in history.keys() if 'val' in key and 'loss' in key]
    accuracy_metrics = [key for key in history.keys() if 'accuracy' in key and 'val' not in key]
    val_accuracy_metrics = [key for key in history.keys() if 'val' in key and 'accuracy' in key]
    
    epochs = range(1, len(history[loss_metrics[0]]) + 1)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot losses
    for loss_name, val_loss_name in zip(loss_metrics, val_loss_metrics):
        ax1.plot(epochs, history[loss_name], label=f'Train {loss_name}', linewidth=2)
        ax1.plot(epochs, history[val_loss_name], label=f'Val {val_loss_name}', 
                linestyle='--', linewidth=2)
    
    ax1.set_title('Loss az Epoch-ok függvényében', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    for acc_name, val_acc_name in zip(accuracy_metrics, val_accuracy_metrics):
        ax2.plot(epochs, history[acc_name], label=f'Train {acc_name}', linewidth=2)
        ax2.plot(epochs, history[val_acc_name], label=f'Val {val_acc_name}', 
                linestyle='--', linewidth=2)
    
    ax2.set_title('Pontosság az Epoch-ok függvényében', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"?? Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    title: str = 'Confusion Matrix',
    save_path: str = None,
    normalize: bool = True
):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        save_path: Optional path to save figure
        normalize: Whether to normalize the matrix
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"?? Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_category_distribution(data_df, column_name: str, title: str = None, top_n: int = 20):
    """
    Plot distribution of categories
    
    Args:
        data_df: DataFrame containing the data
        column_name: Column name to plot
        title: Plot title
        top_n: Number of top categories to show
    """
    counts = data_df[column_name].value_counts().head(top_n)
    
    plt.figure(figsize=(14, 6))
    counts.plot(kind='bar', color='steelblue', edgecolor='black')
    
    if title is None:
        title = f'Top {top_n} {column_name} Distribution'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
