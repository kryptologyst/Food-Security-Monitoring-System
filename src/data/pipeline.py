"""Data pipeline for food security monitoring system."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class FoodSecurityDataGenerator:
    """Generate synthetic food security monitoring data."""
    
    def __init__(self, config: Dict):
        """Initialize data generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        self.geo_config = config['geographic']
        
    def generate_features(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic feature data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with generated features
        """
        features = self.data_config['features']
        
        # Generate crop yield data
        crop_yield = np.random.normal(
            features['crop_yield']['mean'],
            features['crop_yield']['std'],
            n_samples
        )
        crop_yield = np.clip(crop_yield, features['crop_yield']['min'], 
                            features['crop_yield']['max'])
        
        # Generate rainfall data
        rainfall = np.random.normal(
            features['rainfall']['mean'],
            features['rainfall']['std'],
            n_samples
        )
        rainfall = np.clip(rainfall, features['rainfall']['min'], 
                          features['rainfall']['max'])
        
        # Generate market access score
        market_access_score = np.random.uniform(
            features['market_access_score']['min'],
            features['market_access_score']['max'],
            n_samples
        )
        
        # Generate poverty rate
        poverty_rate = np.random.normal(
            features['poverty_rate']['mean'],
            features['poverty_rate']['std'],
            n_samples
        )
        poverty_rate = np.clip(poverty_rate, features['poverty_rate']['min'], 
                              features['poverty_rate']['max'])
        
        # Generate food price index
        food_price_index = np.random.normal(
            features['food_price_index']['mean'],
            features['food_price_index']['std'],
            n_samples
        )
        food_price_index = np.clip(food_price_index, 
                                  features['food_price_index']['min'],
                                  features['food_price_index']['max'])
        
        # Generate additional features
        population_density = np.random.lognormal(
            np.log(features['population_density']['mean']),
            features['population_density']['std'] / features['population_density']['mean'],
            n_samples
        )
        population_density = np.clip(population_density, 
                                    features['population_density']['min'],
                                    features['population_density']['max'])
        
        conflict_index = np.random.beta(
            features['conflict_index']['mean'] * 10,
            (1 - features['conflict_index']['mean']) * 10,
            n_samples
        )
        
        infrastructure_score = np.random.beta(
            features['infrastructure_score']['mean'] * 10,
            (1 - features['infrastructure_score']['mean']) * 10,
            n_samples
        )
        
        # Create DataFrame
        data = pd.DataFrame({
            'crop_yield': crop_yield,
            'rainfall': rainfall,
            'market_access_score': market_access_score,
            'poverty_rate': poverty_rate,
            'food_price_index': food_price_index,
            'population_density': population_density,
            'conflict_index': conflict_index,
            'infrastructure_score': infrastructure_score
        })
        
        return data
    
    def generate_geographic_data(self, n_samples: int) -> gpd.GeoDataFrame:
        """Generate geographic coordinates.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            GeoDataFrame with geographic information
        """
        lat_range = self.geo_config['lat_range']
        lon_range = self.geo_config['lon_range']
        
        # Generate random coordinates
        latitudes = np.random.uniform(lat_range[0], lat_range[1], n_samples)
        longitudes = np.random.uniform(lon_range[0], lon_range[1], n_samples)
        
        # Create geometry points
        geometry = [Point(lon, lat) for lon, lat in zip(longitudes, latitudes)]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'latitude': latitudes,
            'longitude': longitudes,
            'region_id': range(n_samples)
        }, geometry=geometry, crs=self.geo_config['crs'])
        
        return gdf
    
    def generate_labels(self, data: pd.DataFrame) -> pd.Series:
        """Generate food security labels based on thresholds.
        
        Args:
            data: Feature DataFrame
            
        Returns:
            Binary labels (1 = food insecure, 0 = food secure)
        """
        thresholds = self.data_config['thresholds']
        
        # Define food insecurity conditions
        food_insecure = (
            (data['crop_yield'] < thresholds['crop_yield_min']) |
            (data['poverty_rate'] > thresholds['poverty_rate_max']) |
            (data['market_access_score'] < thresholds['market_access_min']) |
            (data['rainfall'] < thresholds['rainfall_min']) |
            (data['food_price_index'] > thresholds['food_price_max'])
        ).astype(int)
        
        return food_insecure
    
    def generate_dataset(self, n_samples: Optional[int] = None) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """Generate complete dataset.
        
        Args:
            n_samples: Number of samples (uses config default if None)
            
        Returns:
            Tuple of (features DataFrame, geographic DataFrame)
        """
        if n_samples is None:
            n_samples = self.data_config['n_samples']
        
        logger.info(f"Generating {n_samples} samples")
        
        # Generate features
        features_df = self.generate_features(n_samples)
        
        # Generate geographic data
        geo_df = self.generate_geographic_data(n_samples)
        
        # Generate labels
        labels = self.generate_labels(features_df)
        features_df['food_insecure'] = labels
        
        # Add region information
        features_df['region_id'] = geo_df['region_id']
        
        logger.info(f"Generated dataset with {len(features_df)} samples")
        logger.info(f"Food insecurity rate: {labels.mean():.3f}")
        
        return features_df, geo_df


class DataProcessor:
    """Process and prepare data for modeling."""
    
    def __init__(self, config: Dict):
        """Initialize data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scaler = StandardScaler()
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for modeling.
        
        Args:
            data: Input DataFrame with features and labels
            
        Returns:
            Tuple of (features array, labels array)
        """
        # Define feature columns (exclude labels and metadata)
        feature_cols = [
            'crop_yield', 'rainfall', 'market_access_score', 
            'poverty_rate', 'food_price_index', 'population_density',
            'conflict_index', 'infrastructure_score'
        ]
        
        X = data[feature_cols].values
        y = data['food_insecure'].values
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data into train/validation/test sets.
        
        Args:
            X: Feature array
            y: Label array
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        train_config = self.config['training']
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=train_config['test_size'],
            random_state=train_config['random_state'],
            stratify=y if train_config['stratify'] else None
        )
        
        # Second split: train vs val
        val_size = train_config['validation_size'] / (1 - train_config['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=train_config['random_state'],
            stratify=y_temp if train_config['stratify'] else None
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train: np.ndarray, X_val: np.ndarray, 
                       X_test: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Tuple of scaled feature arrays
        """
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Features scaled using StandardScaler")
        
        return X_train_scaled, X_val_scaled, X_test_scaled


def main():
    """Main function to generate and save dataset."""
    # Load configuration
    config = load_config('configs/data_config.yaml')
    
    # Set random seed
    set_seed(42)
    
    # Generate data
    generator = FoodSecurityDataGenerator(config)
    features_df, geo_df = generator.generate_dataset()
    
    # Save data
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_df.to_csv(output_dir / 'food_security_features.csv', index=False)
    geo_df.to_file(output_dir / 'food_security_geography.geojson', driver='GeoJSON')
    
    logger.info(f"Dataset saved to {output_dir}")
    logger.info(f"Features shape: {features_df.shape}")
    logger.info(f"Geography shape: {geo_df.shape}")


if __name__ == "__main__":
    main()
