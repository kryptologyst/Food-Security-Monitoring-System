"""Evaluation and metrics for food security monitoring models."""

import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for food security monitoring."""
    
    def __init__(self, config: Dict):
        """Initialize evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.results = {}
        
    def evaluate_all_models(self, models: Dict[str, Any], X_test: np.ndarray, 
                          y_test: np.ndarray) -> pd.DataFrame:
        """Evaluate all models and create leaderboard.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with model performance metrics
        """
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}")
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_proba)
            metrics['model'] = model_name
            results.append(metrics)
            
            # Store detailed results
            self.results[model_name] = {
                'y_true': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'metrics': metrics
            }
        
        # Create leaderboard
        leaderboard = pd.DataFrame(results)
        leaderboard = leaderboard.sort_values('roc_auc', ascending=False)
        
        logger.info("Model evaluation completed")
        return leaderboard
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = self._accuracy_score(y_true, y_pred)
        metrics['precision'] = self._precision_score(y_true, y_pred)
        metrics['recall'] = self._recall_score(y_true, y_pred)
        metrics['f1_score'] = self._f1_score(y_true, y_pred)
        
        # Probability-based metrics
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        metrics['average_precision'] = average_precision_score(y_true, y_proba[:, 1])
        metrics['brier_score'] = self._brier_score(y_true, y_proba[:, 1])
        
        # Food security specific metrics
        metrics['sensitivity'] = self._sensitivity_score(y_true, y_pred)
        metrics['specificity'] = self._specificity_score(y_true, y_pred)
        metrics['false_positive_rate'] = self._false_positive_rate(y_true, y_pred)
        metrics['false_negative_rate'] = self._false_negative_rate(y_true, y_pred)
        
        return metrics
    
    def _accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score."""
        return np.mean(y_true == y_pred)
    
    def _precision_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision score."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _recall_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall score."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score."""
        precision = self._precision_score(y_true, y_pred)
        recall = self._recall_score(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _brier_score(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate Brier score."""
        return np.mean((y_proba - y_true) ** 2)
    
    def _sensitivity_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate sensitivity (same as recall)."""
        return self._recall_score(y_true, y_pred)
    
    def _specificity_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _false_positive_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate false positive rate."""
        return 1 - self._specificity_score(y_true, y_pred)
    
    def _false_negative_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate false negative rate."""
        return 1 - self._recall_score(y_true, y_pred)
    
    def create_confusion_matrix_plot(self, model_name: str, save_path: Optional[str] = None) -> None:
        """Create confusion matrix visualization.
        
        Args:
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if model_name not in self.results:
            raise ValueError(f"No results found for model: {model_name}")
        
        y_true = self.results[model_name]['y_true']
        y_pred = self.results[model_name]['y_pred']
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Food Secure', 'Food Insecure'],
                   yticklabels=['Food Secure', 'Food Insecure'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_roc_curve_plot(self, models: List[str], save_path: Optional[str] = None) -> None:
        """Create ROC curve comparison plot.
        
        Args:
            models: List of model names to compare
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name in models:
            if model_name not in self.results:
                continue
                
            y_true = self.results[model_name]['y_true']
            y_proba = self.results[model_name]['y_proba'][:, 1]
            
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_precision_recall_plot(self, models: List[str], save_path: Optional[str] = None) -> None:
        """Create precision-recall curve comparison plot.
        
        Args:
            models: List of model names to compare
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name in models:
            if model_name not in self.results:
                continue
                
            y_true = self.results[model_name]['y_true']
            y_proba = self.results[model_name]['y_proba'][:, 1]
            
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            ap = average_precision_score(y_true, y_proba)
            
            plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_leaderboard(self, leaderboard: pd.DataFrame) -> go.Figure:
        """Create interactive leaderboard visualization.
        
        Args:
            leaderboard: Model performance DataFrame
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROC AUC', 'F1 Score', 'Precision', 'Recall'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # ROC AUC
        fig.add_trace(
            go.Bar(x=leaderboard['model'], y=leaderboard['roc_auc'],
                  name='ROC AUC', marker_color='lightblue'),
            row=1, col=1
        )
        
        # F1 Score
        fig.add_trace(
            go.Bar(x=leaderboard['model'], y=leaderboard['f1_score'],
                  name='F1 Score', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Precision
        fig.add_trace(
            go.Bar(x=leaderboard['model'], y=leaderboard['precision'],
                  name='Precision', marker_color='lightcoral'),
            row=2, col=1
        )
        
        # Recall
        fig.add_trace(
            go.Bar(x=leaderboard['model'], y=leaderboard['recall'],
                  name='Recall', marker_color='lightyellow'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Model Performance Leaderboard",
            showlegend=False,
            height=600
        )
        
        return fig
    
    def generate_evaluation_report(self, leaderboard: pd.DataFrame, 
                                 output_dir: str = 'assets') -> None:
        """Generate comprehensive evaluation report.
        
        Args:
            leaderboard: Model performance DataFrame
            output_dir: Output directory for reports
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save leaderboard
        leaderboard.to_csv(output_path / 'model_leaderboard.csv', index=False)
        
        # Create visualizations
        model_names = leaderboard['model'].tolist()
        
        # Confusion matrices for top 3 models
        for i, model_name in enumerate(model_names[:3]):
            self.create_confusion_matrix_plot(
                model_name, 
                str(output_path / f'confusion_matrix_{model_name}.png')
            )
        
        # ROC curves
        self.create_roc_curve_plot(
            model_names, 
            str(output_path / 'roc_curves.png')
        )
        
        # Precision-recall curves
        self.create_precision_recall_plot(
            model_names, 
            str(output_path / 'precision_recall_curves.png')
        )
        
        # Interactive leaderboard
        fig = self.create_interactive_leaderboard(leaderboard)
        fig.write_html(str(output_path / 'interactive_leaderboard.html'))
        
        logger.info(f"Evaluation report saved to {output_path}")


def main():
    """Main function for evaluation."""
    # This would be called from the main evaluation script
    pass


if __name__ == "__main__":
    main()
