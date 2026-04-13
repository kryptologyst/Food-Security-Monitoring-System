# Food Security Monitoring System

A comprehensive machine learning system for monitoring food security using environmental and socioeconomic indicators. This project demonstrates how to build, train, and deploy models for identifying regions at risk of food insecurity.

## Overview

Food security monitoring is critical for identifying regions at risk of hunger or malnutrition due to environmental, economic, or supply factors. This system uses machine learning to analyze various indicators and classify regions as food secure or insecure.

### Key Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, and Neural Networks
- **Comprehensive Evaluation**: ROC-AUC, Precision-Recall, F1-Score, and domain-specific metrics
- **Interactive Visualizations**: Geographic maps, risk heatmaps, and time series analysis
- **Streamlit Demo**: User-friendly web interface for exploration and analysis
- **Synthetic Data Generation**: Realistic food security datasets for research and education

## Project Structure

```
food-security-monitoring/
├── src/                    # Source code
│   ├── data/              # Data pipeline and processing
│   ├── models/            # ML model implementations
│   ├── eval/              # Evaluation metrics and tools
│   └── viz/               # Visualization components
├── configs/               # Configuration files
│   ├── data_config.yaml   # Data generation parameters
│   ├── model_config.yaml  # Model hyperparameters
│   └── geo_config.yaml    # Geographic settings
├── scripts/               # Training and evaluation scripts
├── demo/                  # Streamlit demo application
├── tests/                 # Unit tests
├── data/                  # Data storage (raw/processed/external)
├── assets/                # Generated visualizations and reports
├── notebooks/             # Jupyter notebooks for exploration
└── requirements.txt       # Python dependencies
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Food-Security-Monitoring-System.git
cd Food-Security-Monitoring-System
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Generation and Training

1. Generate synthetic dataset:
```bash
python scripts/train.py
```

2. Run the interactive demo:
```bash
streamlit run demo/app.py
```

## Data Schema

The system generates synthetic data with the following features:

### Environmental Indicators
- **Crop Yield** (tons/hectare): Agricultural productivity measure
- **Rainfall** (mm/month): Precipitation levels
- **Market Access Score** (0-1): Infrastructure and connectivity

### Socioeconomic Indicators
- **Poverty Rate** (0-1): Percentage of population below poverty line
- **Food Price Index** (relative): Cost of food relative to baseline
- **Population Density** (people/km²): Demographic pressure
- **Conflict Index** (0-1): Political instability measure
- **Infrastructure Score** (0-1): Development level

### Geographic Data
- **Latitude/Longitude**: Spatial coordinates
- **Region ID**: Unique identifier for each region

### Target Variable
- **Food Insecure** (0/1): Binary classification label

## Model Performance

The system evaluates models using multiple metrics:

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity to food insecurity
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Average Precision**: Area under precision-recall curve
- **Brier Score**: Calibration quality

### Domain-Specific Metrics
- **Sensitivity**: Ability to detect food insecurity
- **Specificity**: Ability to identify food security
- **False Positive Rate**: Incorrect food insecurity alerts
- **False Negative Rate**: Missed food insecurity cases

## Usage Examples

### Training Models

```python
from src.data.pipeline import FoodSecurityDataGenerator, DataProcessor
from src.models.trainer import create_model, ModelEvaluator

# Load configuration
config = load_configs()

# Generate data
generator = FoodSecurityDataGenerator(config)
features_df, geo_df = generator.generate_dataset()

# Process data
processor = DataProcessor(config)
X, y = processor.prepare_features(features_df)
X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)

# Train model
model = create_model('xgboost', config)
model.fit(X_train, y_train)

# Evaluate
evaluator = ModelEvaluator(config)
scores = evaluator.evaluate_model(model, X_test, y_test)
```

### Creating Visualizations

```python
from src.viz.plots import FoodSecurityVisualizer

# Create visualizer
visualizer = FoodSecurityVisualizer(config)

# Generate maps and plots
visualizer.create_food_security_map(geo_df, predictions, probabilities)
visualizer.create_feature_distribution_plots(features_df)
visualizer.create_correlation_heatmap(features_df)
```

## Configuration

### Data Configuration (`configs/data_config.yaml`)
- Dataset size and parameters
- Feature distributions and thresholds
- Geographic boundaries

### Model Configuration (`configs/model_config.yaml`)
- Model hyperparameters
- Training settings
- Evaluation metrics

### Geographic Configuration (`configs/geo_config.yaml`)
- Coordinate reference systems
- Map settings and styling
- Regional boundaries

## Demo Application

The Streamlit demo provides an interactive interface for:

- **Data Exploration**: View dataset statistics and distributions
- **Geographic Analysis**: Interactive maps showing food security status
- **Model Training**: Train models with different parameters
- **Performance Evaluation**: View model metrics and confusion matrices
- **Risk Assessment**: Analyze food security risk scores

Access the demo by running:
```bash
streamlit run demo/app.py
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

**Important**: This is a research demonstration tool. The data and predictions shown are synthetic and for educational purposes only. Do not use for operational decision-making or real-world food security assessments.

## Author

**kryptologyst** - [GitHub](https://github.com/kryptologyst)

## Acknowledgments

- Built for educational and research purposes
- Demonstrates best practices in ML for environmental and social applications
- Uses modern Python data science stack
- Implements comprehensive evaluation and visualization

## Issues

For questions, bug reports, or feature requests, please visit the [GitHub Issues](https://github.com/kryptologyst) page.

---

*This project is part of the Environmental & Social Applications series, demonstrating how machine learning can be applied to critical global challenges.*
# Food-Security-Monitoring-System
