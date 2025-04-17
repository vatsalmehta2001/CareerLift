import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)


class ModelEvaluator:
    """
    Utility class for evaluating machine learning models for the income prediction task.
    """
    
    def __init__(self, model=None):
        """
        Initialize the ModelEvaluator.
        
        Parameters:
        -----------
        model : object
            Trained machine learning model with predict and predict_proba methods.
        """
        self.model = model
        self.y_true = None
        self.y_pred = None
        self.y_proba = None
    
    def set_model(self, model):
        """
        Set the model to evaluate.
        
        Parameters:
        -----------
        model : object
            Trained machine learning model with predict and predict_proba methods.
        """
        self.model = model
    
    def predict(self, X, y_true):
        """
        Make predictions using the model and store results for evaluation.
        
        Parameters:
        -----------
        X : array-like
            Features to make predictions on.
        y_true : array-like
            True labels for the data.
            
        Returns:
        --------
        tuple
            (y_pred, y_proba)
        """
        if self.model is None:
            raise ValueError("Model is not set. Use set_model() first.")
        
        self.y_true = y_true
        self.y_pred = self.model.predict(X)
        
        # Check if the model supports probability predictions
        if hasattr(self.model, 'predict_proba'):
            self.y_proba = self.model.predict_proba(X)[:, 1]
        else:
            self.y_proba = None
        
        return self.y_pred, self.y_proba
    
    def calculate_metrics(self):
        """
        Calculate classification metrics.
        
        Returns:
        --------
        dict
            Dictionary containing calculated metrics.
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Predictions are not available. Use predict() first.")
        
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred),
            'recall': recall_score(self.y_true, self.y_pred),
            'f1': f1_score(self.y_true, self.y_pred)
        }
        
        if self.y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_proba)
        
        return metrics
    
    def print_classification_report(self):
        """
        Print the classification report.
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Predictions are not available. Use predict() first.")
        
        print("\nClassification Report:")
        print(classification_report(self.y_true, self.y_pred, target_names=['<=50K', '>50K']))
    
    def plot_confusion_matrix(self, figsize=(8, 6), save_path=None):
        """
        Plot the confusion matrix.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size.
        save_path : str
            Path to save the plot. If None, the plot is not saved.
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Predictions are not available. Use predict() first.")
        
        plt.figure(figsize=figsize)
        cm = confusion_matrix(self.y_true, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['<=50K', '>50K'],
                    yticklabels=['<=50K', '>50K'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_roc_curve(self, figsize=(8, 6), save_path=None):
        """
        Plot the ROC curve.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size.
        save_path : str
            Path to save the plot. If None, the plot is not saved.
        """
        if self.y_true is None or self.y_proba is None:
            raise ValueError("Probability predictions are not available. Use predict() first with a model that supports predict_proba.")
        
        plt.figure(figsize=figsize)
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        roc_auc = roc_auc_score(self.y_true, self.y_proba)
        
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_precision_recall_curve(self, figsize=(8, 6), save_path=None):
        """
        Plot the precision-recall curve.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size.
        save_path : str
            Path to save the plot. If None, the plot is not saved.
        """
        if self.y_true is None or self.y_proba is None:
            raise ValueError("Probability predictions are not available. Use predict() first with a model that supports predict_proba.")
        
        plt.figure(figsize=figsize)
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_proba)
        
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def evaluate_and_visualize(self, X, y_true, output_dir='plots'):
        """
        Perform comprehensive evaluation with visualization.
        
        Parameters:
        -----------
        X : array-like
            Features to make predictions on.
        y_true : array-like
            True labels for the data.
        output_dir : str
            Directory to save plots.
            
        Returns:
        --------
        dict
            Dictionary containing calculated metrics.
        """
        import os
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Make predictions
        self.predict(X, y_true)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Print classification report
        self.print_classification_report()
        
        # Plot and save confusion matrix
        self.plot_confusion_matrix(save_path=os.path.join(output_dir, 'confusion_matrix.png'))
        
        # Plot and save ROC curve if probability predictions are available
        if self.y_proba is not None:
            self.plot_roc_curve(save_path=os.path.join(output_dir, 'roc_curve.png'))
            self.plot_precision_recall_curve(save_path=os.path.join(output_dir, 'precision_recall_curve.png'))
        
        return metrics 