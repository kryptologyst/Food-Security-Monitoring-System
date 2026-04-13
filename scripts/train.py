"""Main training script for food security monitoring system."""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.pipeline import FoodSecurityDataGenerator, DataProcessor, set_seed
from src.models.trainer import create_model, ModelEvaluator
from src.eval.metrics import ModelEvaluator as EvalMetrics
from src.viz.plots import FoodSecurityVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_configs() -> Dict[str, Any]:
    """Load all configuration files.
    
    Returns:
        Combined configuration dictionary
    """
    config_dir = Path('configs')
    
    # Load individual configs
    data_config = OmegaConf.load(config_dir / 'data_config.yaml')
    model_config = OmegaConf.load(config_dir / 'model_config.yaml')
    geo_config = OmegaConf.load(config_dir / 'geo_config.yaml')
    
    # Combine configs
    config = OmegaConf.merge(data_config, model_config, geo_config)
    
    return config


def train_models(config: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
    """Train all models.
    
    Args:
        config: Configuration dictionary
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Dictionary of trained models
    """
    models = {}
    model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'neural_network']
    
    logger.info("Starting model training...")
    
    for model_name in model_names:
        logger.info(f"Training {model_name}...")
        
        try:
            # Create model
            if model_name == 'neural_network':
                model = create_model(model_name, config, input_dim=X_train.shape[1])
            else:
                model = create_model(model_name, config)
            
            # Train model
            model.fit(X_train, y_train)
            models[model_name] = model
            
            logger.info(f"{model_name} training completed")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    logger.info(f"Successfully trained {len(models)} models")
    return models


def evaluate_models(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray,
                   config: Dict[str, Any]) -> pd.DataFrame:
    """Evaluate all models.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        config: Configuration dictionary
        
    Returns:
        Model performance leaderboard
    """
    logger.info("Starting model evaluation...")
    
    # Create evaluator
    evaluator = EvalMetrics(config)
    
    # Evaluate all models
    leaderboard = evaluator.evaluate_all_models(models, X_test, y_test)
    
    # Generate evaluation report
    evaluator.generate_evaluation_report(leaderboard)
    
    logger.info("Model evaluation completed")
    return leaderboard


def create_visualizations(data: pd.DataFrame, geo_data: Any, models: Dict[str, Any],
                         X_test: np.ndarray, y_test: np.ndarray, config: Dict[str, Any]) -> None:
    """Create visualizations.
    
    Args:
        data: Feature data
        geo_data: Geographic data
        models: Trained models
        X_test: Test features
        y_test: Test labels
        config: Configuration dictionary
    """
    logger.info("Creating visualizations...")
    
    # Create visualizer
    visualizer = FoodSecurityVisualizer(config)
    
    # Get predictions from best model (highest ROC AUC)
    best_model_name = list(models.keys())[0]  # Assume first model for now
    best_model = models[best_model_name]
    
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]
    
    # Create visualizations
    visualizer.save_all_visualizations(
        data, geo_data, predictions, probabilities
    )
    
    logger.info("Visualizations created")


def main():
    """Main training pipeline."""
    logger.info("Starting Food Security Monitoring Training Pipeline")
    
    # Load configurations
    config = load_configs()
    logger.info("Configurations loaded")
    
    # Set random seed for reproducibility
    set_seed(config['training']['random_state'])
    
    # Generate data
    logger.info("Generating synthetic dataset...")
    generator = FoodSecurityDataGenerator(config)
    features_df, geo_df = generator.generate_dataset()
    
    # Process data
    logger.info("Processing data...")
    processor = DataProcessor(config)
    X, y = processor.prepare_features(features_df)
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
    X_train_scaled, X_val_scaled, X_test_scaled = processor.scale_features(
        X_train, X_val, X_test
    )
    
    logger.info(f"Data processed - Train: {len(X_train_scaled)}, "
               f"Val: {len(X_val_scaled)}, Test: {len(X_test_scaled)}")
    
    # Train models
    models = train_models(config, X_train_scaled, y_train, X_val_scaled, y_val)
    
    if not models:
        logger.error("No models were successfully trained. Exiting.")
        return
    
    # Evaluate models
    leaderboard = evaluate_models(models, X_test_scaled, y_test, config)
    
    # Print results
    logger.info("Model Performance Leaderboard:")
    print(leaderboard.to_string(index=False))
    
    # Create visualizations
    create_visualizations(features_df, geo_df, models, X_test_scaled, y_test, config)
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
