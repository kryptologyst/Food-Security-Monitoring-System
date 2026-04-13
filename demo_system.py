#!/usr/bin/env python3
"""Demonstration script for Food Security Monitoring System."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.pipeline import FoodSecurityDataGenerator, DataProcessor, set_seed
from src.models.trainer import create_model
from src.eval.metrics import ModelEvaluator
from src.viz.plots import FoodSecurityVisualizer
from omegaconf import OmegaConf


def main():
    """Run a complete demonstration of the food security monitoring system."""
    print("🌾 Food Security Monitoring System - Demonstration")
    print("=" * 60)
    
    # Load configuration
    print("📋 Loading configuration...")
    data_config = OmegaConf.load('configs/data_config.yaml')
    model_config = OmegaConf.load('configs/model_config.yaml')
    geo_config = OmegaConf.load('configs/geo_config.yaml')
    config = OmegaConf.merge(data_config, model_config, geo_config)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Generate synthetic dataset
    print("📊 Generating synthetic dataset...")
    generator = FoodSecurityDataGenerator(config)
    features_df, geo_df = generator.generate_dataset(1000)
    
    print(f"   Generated {len(features_df)} samples")
    print(f"   Food insecurity rate: {features_df['food_insecure'].mean():.3f}")
    print(f"   Average crop yield: {features_df['crop_yield'].mean():.2f} tons/ha")
    print(f"   Average poverty rate: {features_df['poverty_rate'].mean():.3f}")
    
    # Process data
    print("\n🔧 Processing data...")
    processor = DataProcessor(config)
    X, y = processor.prepare_features(features_df)
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
    X_train_scaled, X_val_scaled, X_test_scaled = processor.scale_features(
        X_train, X_val, X_test
    )
    
    print(f"   Training set: {len(X_train_scaled)} samples")
    print(f"   Validation set: {len(X_val_scaled)} samples")
    print(f"   Test set: {len(X_test_scaled)} samples")
    
    # Train multiple models
    print("\n🤖 Training models...")
    models = {}
    model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    
    for model_name in model_names:
        print(f"   Training {model_name}...")
        try:
            model = create_model(model_name, config)
            model.fit(X_train_scaled, y_train)
            models[model_name] = model
            print(f"   ✅ {model_name} trained successfully")
        except Exception as e:
            print(f"   ❌ {model_name} failed: {str(e)}")
    
    if not models:
        print("   ❌ No models were successfully trained!")
        return
    
    # Evaluate models
    print("\n📈 Evaluating models...")
    evaluator = ModelEvaluator(config)
    leaderboard = evaluator.evaluate_all_models(models, X_test_scaled, y_test)
    
    print("\n🏆 Model Performance Leaderboard:")
    print("-" * 50)
    for idx, row in leaderboard.iterrows():
        print(f"{row['model']:20} | Accuracy: {row['accuracy']:.3f} | "
              f"F1: {row['f1_score']:.3f} | ROC-AUC: {row['roc_auc']:.3f}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    visualizer = FoodSecurityVisualizer(config)
    
    # Get predictions from best model
    best_model_name = leaderboard.iloc[0]['model']
    best_model = models[best_model_name]
    predictions = best_model.predict(X_test_scaled)
    probabilities = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Save visualizations
    visualizer.save_all_visualizations(
        features_df, geo_df, predictions, probabilities
    )
    
    print(f"   ✅ Visualizations saved to assets/ directory")
    print(f"   📊 Best model: {best_model_name}")
    print(f"   🎯 Test accuracy: {leaderboard.iloc[0]['accuracy']:.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Demonstration completed successfully!")
    print("\n📁 Generated files:")
    print("   - assets/food_security_features.csv")
    print("   - assets/food_security_geography.geojson")
    print("   - assets/model_leaderboard.csv")
    print("   - assets/interactive_dashboard.html")
    print("   - assets/food_security_map.html")
    print("   - assets/risk_heatmap.html")
    print("\n🚀 To run the interactive demo:")
    print("   streamlit run demo/app.py")
    print("\n⚠️  Remember: This is a research demonstration tool.")
    print("   Data is synthetic and for educational purposes only.")


if __name__ == "__main__":
    main()
