"""Unit tests for food security monitoring system."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.pipeline import FoodSecurityDataGenerator, DataProcessor, set_seed
from src.models.trainer import create_model, ModelEvaluator
from src.eval.metrics import ModelEvaluator as EvalMetrics


class TestDataPipeline:
    """Test data pipeline functionality."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        # This should not raise an exception
        assert True
    
    def test_data_generator_initialization(self):
        """Test data generator initialization."""
        config = {
            'data': {
                'n_samples': 100,
                'features': {
                    'crop_yield': {'mean': 2.5, 'std': 0.8, 'min': 0.5, 'max': 5.0},
                    'rainfall': {'mean': 100, 'std': 30, 'min': 20, 'max': 300},
                    'market_access_score': {'min': 0.0, 'max': 1.0},
                    'poverty_rate': {'mean': 0.3, 'std': 0.1, 'min': 0.05, 'max': 0.8},
                    'food_price_index': {'mean': 120, 'std': 20, 'min': 80, 'max': 200},
                    'population_density': {'mean': 150, 'std': 100, 'min': 10, 'max': 1000},
                    'conflict_index': {'mean': 0.2, 'std': 0.15, 'min': 0.0, 'max': 1.0},
                    'infrastructure_score': {'mean': 0.6, 'std': 0.2, 'min': 0.1, 'max': 1.0}
                },
                'thresholds': {
                    'crop_yield_min': 2.0,
                    'poverty_rate_max': 0.35,
                    'market_access_min': 0.4,
                    'rainfall_min': 50,
                    'food_price_max': 150
                }
            },
            'geographic': {
                'lat_range': [10, 50],
                'lon_range': [-120, -70],
                'crs': 'EPSG:4326'
            }
        }
        
        generator = FoodSecurityDataGenerator(config)
        assert generator is not None
    
    def test_data_generation(self):
        """Test data generation."""
        config = {
            'data': {
                'n_samples': 100,
                'features': {
                    'crop_yield': {'mean': 2.5, 'std': 0.8, 'min': 0.5, 'max': 5.0},
                    'rainfall': {'mean': 100, 'std': 30, 'min': 20, 'max': 300},
                    'market_access_score': {'min': 0.0, 'max': 1.0},
                    'poverty_rate': {'mean': 0.3, 'std': 0.1, 'min': 0.05, 'max': 0.8},
                    'food_price_index': {'mean': 120, 'std': 20, 'min': 80, 'max': 200},
                    'population_density': {'mean': 150, 'std': 100, 'min': 10, 'max': 1000},
                    'conflict_index': {'mean': 0.2, 'std': 0.15, 'min': 0.0, 'max': 1.0},
                    'infrastructure_score': {'mean': 0.6, 'std': 0.2, 'min': 0.1, 'max': 1.0}
                },
                'thresholds': {
                    'crop_yield_min': 2.0,
                    'poverty_rate_max': 0.35,
                    'market_access_min': 0.4,
                    'rainfall_min': 50,
                    'food_price_max': 150
                }
            },
            'geographic': {
                'lat_range': [10, 50],
                'lon_range': [-120, -70],
                'crs': 'EPSG:4326'
            }
        }
        
        generator = FoodSecurityDataGenerator(config)
        features_df, geo_df = generator.generate_dataset(50)
        
        assert len(features_df) == 50
        assert len(geo_df) == 50
        assert 'food_insecure' in features_df.columns
        assert 'latitude' in geo_df.columns
        assert 'longitude' in geo_df.columns
    
    def test_data_processor(self):
        """Test data processor."""
        config = {
            'training': {
                'test_size': 0.2,
                'validation_size': 0.2,
                'random_state': 42,
                'stratify': True
            }
        }
        
        processor = DataProcessor(config)
        
        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
        
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert len(y_train) + len(y_val) + len(y_test) == len(y)


class TestModels:
    """Test model functionality."""
    
    def test_model_creation(self):
        """Test model creation."""
        config = {
            'models': {
                'logistic_regression': {
                    'C': 1.0,
                    'max_iter': 1000,
                    'random_state': 42
                }
            }
        }
        
        model = create_model('logistic_regression', config)
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_model_training(self):
        """Test model training."""
        config = {
            'models': {
                'logistic_regression': {
                    'C': 1.0,
                    'max_iter': 1000,
                    'random_state': 42
                }
            }
        }
        
        model = create_model('logistic_regression', config)
        
        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Train model
        model.fit(X, y)
        
        # Test predictions
        predictions = model.predict(X[:10])
        probabilities = model.predict_proba(X[:10])
        
        assert len(predictions) == 10
        assert probabilities.shape == (10, 2)
        assert model.is_fitted is True


class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        config = {
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
            'cv_folds': 5,
            'cv_strategy': 'stratified'
        }
        
        evaluator = EvalMetrics(config)
        assert evaluator is not None
        assert 'accuracy' in evaluator.config['metrics']
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        config = {
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
            'cv_folds': 5,
            'cv_strategy': 'stratified'
        }
        
        evaluator = EvalMetrics(config)
        
        # Create sample data
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
        
        metrics = evaluator._calculate_metrics(y_true, y_pred, y_proba)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert all(isinstance(v, float) for v in metrics.values())


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline."""
        config = {
            'data': {
                'n_samples': 100,
                'features': {
                    'crop_yield': {'mean': 2.5, 'std': 0.8, 'min': 0.5, 'max': 5.0},
                    'rainfall': {'mean': 100, 'std': 30, 'min': 20, 'max': 300},
                    'market_access_score': {'min': 0.0, 'max': 1.0},
                    'poverty_rate': {'mean': 0.3, 'std': 0.1, 'min': 0.05, 'max': 0.8},
                    'food_price_index': {'mean': 120, 'std': 20, 'min': 80, 'max': 200},
                    'population_density': {'mean': 150, 'std': 100, 'min': 10, 'max': 1000},
                    'conflict_index': {'mean': 0.2, 'std': 0.15, 'min': 0.0, 'max': 1.0},
                    'infrastructure_score': {'mean': 0.6, 'std': 0.2, 'min': 0.1, 'max': 1.0}
                },
                'thresholds': {
                    'crop_yield_min': 2.0,
                    'poverty_rate_max': 0.35,
                    'market_access_min': 0.4,
                    'rainfall_min': 50,
                    'food_price_max': 150
                }
            },
            'geographic': {
                'lat_range': [10, 50],
                'lon_range': [-120, -70],
                'crs': 'EPSG:4326'
            },
            'training': {
                'test_size': 0.2,
                'validation_size': 0.2,
                'random_state': 42,
                'stratify': True
            },
            'models': {
                'logistic_regression': {
                    'C': 1.0,
                    'max_iter': 1000,
                    'random_state': 42
                }
            },
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
            'cv_folds': 5,
            'cv_strategy': 'stratified'
        }
        
        # Set seed
        set_seed(42)
        
        # Generate data
        generator = FoodSecurityDataGenerator(config)
        features_df, geo_df = generator.generate_dataset(100)
        
        # Process data
        processor = DataProcessor(config)
        X, y = processor.prepare_features(features_df)
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
        X_train_scaled, X_val_scaled, X_test_scaled = processor.scale_features(
            X_train, X_val, X_test
        )
        
        # Train model
        model = create_model('logistic_regression', config)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        evaluator = EvalMetrics(config)
        scores = evaluator.evaluate_all_models({'test_model': model}, X_test_scaled, y_test)
        
        # Assertions
        assert len(features_df) == 100
        assert len(geo_df) == 100
        assert X_train_scaled.shape[0] + X_val_scaled.shape[0] + X_test_scaled.shape[0] == 100
        assert model.is_fitted is True
        assert len(scores) > 0
        assert 'accuracy' in scores.columns
        assert scores['accuracy'].iloc[0] >= 0.0
        assert scores['accuracy'].iloc[0] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
