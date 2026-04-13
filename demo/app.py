"""Streamlit demo for Food Security Monitoring System."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.pipeline import FoodSecurityDataGenerator, DataProcessor, set_seed
from src.models.trainer import create_model
from src.viz.plots import FoodSecurityVisualizer

# Page configuration
st.set_page_config(
    page_title="Food Security Monitoring System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False


def load_config():
    """Load configuration."""
    import yaml
    from omegaconf import OmegaConf
    
    config_dir = Path('configs')
    data_config = OmegaConf.load(config_dir / 'data_config.yaml')
    model_config = OmegaConf.load(config_dir / 'model_config.yaml')
    geo_config = OmegaConf.load(config_dir / 'geo_config.yaml')
    
    config = OmegaConf.merge(data_config, model_config, geo_config)
    return config


def generate_sample_data(n_samples: int = 1000):
    """Generate sample data for demonstration."""
    config = load_config()
    set_seed(42)
    
    generator = FoodSecurityDataGenerator(config)
    features_df, geo_df = generator.generate_dataset(n_samples)
    
    return features_df, geo_df


def train_sample_models(features_df: pd.DataFrame):
    """Train sample models for demonstration."""
    config = load_config()
    
    processor = DataProcessor(config)
    X, y = processor.prepare_features(features_df)
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
    X_train_scaled, X_val_scaled, X_test_scaled = processor.scale_features(
        X_train, X_val, X_test
    )
    
    # Train a simple logistic regression model for demo
    model = create_model('logistic_regression', config)
    model.fit(X_train_scaled, y_train)
    
    # Get predictions
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    
    return model, predictions, probabilities, X_test_scaled, y_test


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🌾 Food Security Monitoring System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Disclaimer:</strong> This is a research demonstration tool. 
    The data and predictions shown are synthetic and for educational purposes only. 
    Do not use for operational decision-making.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Data generation parameters
    st.sidebar.subheader("Data Parameters")
    n_samples = st.sidebar.slider("Number of Regions", 100, 5000, 1000)
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "Neural Network"]
    )
    
    # Generate data button
    if st.sidebar.button("Generate Sample Data", type="primary"):
        with st.spinner("Generating sample data..."):
            features_df, geo_df = generate_sample_data(n_samples)
            st.session_state.features_df = features_df
            st.session_state.geo_df = geo_df
            st.session_state.data_generated = True
        st.sidebar.success("Data generated successfully!")
    
    # Main content
    if st.session_state.data_generated:
        features_df = st.session_state.features_df
        geo_df = st.session_state.geo_df
        
        # Overview metrics
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Regions", len(features_df))
        
        with col2:
            insecure_rate = features_df['food_insecure'].mean()
            st.metric("Food Insecurity Rate", f"{insecure_rate:.1%}")
        
        with col3:
            avg_yield = features_df['crop_yield'].mean()
            st.metric("Average Crop Yield", f"{avg_yield:.2f} tons/ha")
        
        with col4:
            avg_poverty = features_df['poverty_rate'].mean()
            st.metric("Average Poverty Rate", f"{avg_poverty:.1%}")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Geographic View", "📈 Analytics", "🤖 Model Predictions", "📋 Data Explorer"])
        
        with tab1:
            st.subheader("Geographic Distribution")
            
            # Create map
            center_lat = geo_df['latitude'].mean()
            center_lon = geo_df['longitude'].mean()
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=4,
                tiles='OpenStreetMap'
            )
            
            # Add markers
            for idx, row in geo_df.iterrows():
                lat = row['latitude']
                lon = row['longitude']
                status = features_df.loc[idx, 'food_insecure']
                
                color = '#DC143C' if status == 1 else '#2E8B57'
                status_text = 'Food Insecure' if status == 1 else 'Food Secure'
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    popup=f"Region {idx}: {status_text}",
                    color='black',
                    weight=1,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
            
            # Display map
            st_folium(m, width=700, height=500)
        
        with tab2:
            st.subheader("Feature Analysis")
            
            # Feature distributions
            feature_cols = ['crop_yield', 'rainfall', 'poverty_rate', 'market_access_score']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[col.replace('_', ' ').title() for col in feature_cols]
            )
            
            for i, feature in enumerate(feature_cols):
                row = i // 2 + 1
                col = i % 2 + 1
                
                fig.add_trace(
                    go.Histogram(x=features_df[feature], name=feature),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Feature Correlations")
            corr_matrix = features_df[feature_cols + ['food_insecure']].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab3:
            st.subheader("Model Predictions")
            
            if st.button("Train Model", type="primary"):
                with st.spinner("Training model..."):
                    model, predictions, probabilities, X_test, y_test = train_sample_models(features_df)
                    st.session_state.model = model
                    st.session_state.predictions = predictions
                    st.session_state.probabilities = probabilities
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.models_trained = True
                st.success("Model trained successfully!")
            
            if st.session_state.models_trained:
                predictions = st.session_state.predictions
                probabilities = st.session_state.probabilities
                y_test = st.session_state.y_test
                
                # Model performance
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    accuracy = accuracy_score(y_test, predictions)
                    st.metric("Accuracy", f"{accuracy:.3f}")
                
                with col2:
                    precision = precision_score(y_test, predictions, zero_division=0)
                    st.metric("Precision", f"{precision:.3f}")
                
                with col3:
                    recall = recall_score(y_test, predictions, zero_division=0)
                    st.metric("Recall", f"{recall:.3f}")
                
                with col4:
                    f1 = f1_score(y_test, predictions, zero_division=0)
                    st.metric("F1 Score", f"{f1:.3f}")
                
                # Risk distribution
                st.subheader("Risk Score Distribution")
                fig_risk = px.histogram(
                    x=probabilities,
                    nbins=30,
                    title="Distribution of Food Security Risk Scores",
                    labels={'x': 'Risk Score', 'y': 'Count'}
                )
                st.plotly_chart(fig_risk, use_container_width=True)
                
                # Confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, predictions)
                
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title="Confusion Matrix",
                    labels={'x': 'Predicted', 'y': 'Actual'},
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_cm, use_container_width=True)
        
        with tab4:
            st.subheader("Data Explorer")
            
            # Data table
            st.write("Sample of the dataset:")
            st.dataframe(features_df.head(20))
            
            # Download button
            csv = features_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="food_security_data.csv",
                mime="text/csv"
            )
            
            # Statistics
            st.subheader("Dataset Statistics")
            st.write(features_df.describe())
    
    else:
        st.info("👈 Please generate sample data using the sidebar to begin exploration.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p><strong>Food Security Monitoring System</strong></p>
    <p>Author: <a href="https://github.com/kryptologyst" target="_blank">kryptologyst</a></p>
    <p>This is a research demonstration tool for educational purposes.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
