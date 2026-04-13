"""Visualization tools for food security monitoring."""

import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium import plugins

logger = logging.getLogger(__name__)


class FoodSecurityVisualizer:
    """Visualization tools for food security monitoring."""
    
    def __init__(self, config: Dict):
        """Initialize visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.geo_config = config['geographic']
        self.viz_config = config['geographic']['visualization']
        
    def create_feature_distribution_plots(self, data: pd.DataFrame, 
                                        save_path: Optional[str] = None) -> None:
        """Create feature distribution plots.
        
        Args:
            data: Feature DataFrame
            save_path: Path to save the plot
        """
        feature_cols = [
            'crop_yield', 'rainfall', 'market_access_score', 
            'poverty_rate', 'food_price_index', 'population_density',
            'conflict_index', 'infrastructure_score'
        ]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(feature_cols):
            if feature in data.columns:
                # Create histogram
                axes[i].hist(data[feature], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{feature.replace("_", " ").title()}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_correlation_heatmap(self, data: pd.DataFrame, 
                                 save_path: Optional[str] = None) -> None:
        """Create correlation heatmap.
        
        Args:
            data: Feature DataFrame
            save_path: Path to save the plot
        """
        feature_cols = [
            'crop_yield', 'rainfall', 'market_access_score', 
            'poverty_rate', 'food_price_index', 'population_density',
            'conflict_index', 'infrastructure_score', 'food_insecure'
        ]
        
        # Calculate correlation matrix
        corr_matrix = data[feature_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_food_security_map(self, geo_data: gpd.GeoDataFrame, 
                               predictions: Optional[np.ndarray] = None,
                               probabilities: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None) -> folium.Map:
        """Create interactive food security map.
        
        Args:
            geo_data: Geographic data
            predictions: Model predictions
            probabilities: Prediction probabilities
            save_path: Path to save the map
            
        Returns:
            Folium map object
        """
        # Create base map
        center_lat = self.geo_config['map']['center_lat']
        center_lon = self.geo_config['map']['center_lon']
        zoom_level = self.geo_config['map']['zoom_level']
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_level,
            tiles=self.geo_config['map']['tile_layer']
        )
        
        # Add markers
        for idx, row in geo_data.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            
            # Determine marker color based on predictions
            if predictions is not None and idx < len(predictions):
                if predictions[idx] == 1:  # Food insecure
                    color = self.viz_config['food_insecure_color']
                    status = 'Food Insecure'
                else:  # Food secure
                    color = self.viz_config['food_secure_color']
                    status = 'Food Secure'
            else:
                color = 'blue'
                status = 'Unknown'
            
            # Create popup text
            popup_text = f"""
            <b>Region {idx}</b><br>
            Status: {status}<br>
            Latitude: {lat:.4f}<br>
            Longitude: {lon:.4f}
            """
            
            if probabilities is not None and idx < len(probabilities):
                prob = probabilities[idx]
                popup_text += f"<br>Risk Score: {prob:.3f}"
            
            # Add marker
            folium.CircleMarker(
                location=[lat, lon],
                radius=self.viz_config['marker_size'],
                popup=popup_text,
                color='black',
                weight=1,
                fillColor=color,
                fillOpacity=self.viz_config['marker_opacity']
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Food Security Status</b></p>
        <p><i class="fa fa-circle" style="color:''' + self.viz_config['food_secure_color'] + '''"></i> Food Secure</p>
        <p><i class="fa fa-circle" style="color:''' + self.viz_config['food_insecure_color'] + '''"></i> Food Insecure</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        if save_path:
            m.save(save_path)
        
        return m
    
    def create_risk_heatmap(self, geo_data: gpd.GeoDataFrame, 
                          probabilities: np.ndarray,
                          save_path: Optional[str] = None) -> folium.Map:
        """Create risk heatmap using probability values.
        
        Args:
            geo_data: Geographic data
            probabilities: Risk probabilities
            save_path: Path to save the map
            
        Returns:
            Folium map with heatmap layer
        """
        # Create base map
        center_lat = self.geo_config['map']['center_lat']
        center_lon = self.geo_config['map']['center_lon']
        zoom_level = self.geo_config['map']['zoom_level']
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_level,
            tiles=self.geo_config['map']['tile_layer']
        )
        
        # Prepare data for heatmap
        heat_data = []
        for idx, row in geo_data.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            risk = probabilities[idx] if idx < len(probabilities) else 0.0
            heat_data.append([lat, lon, risk])
        
        # Add heatmap layer
        plugins.HeatMap(
            heat_data,
            name='Food Security Risk',
            min_opacity=0.2,
            max_zoom=18,
            radius=25,
            blur=15,
            gradient={0.0: 'green', 0.5: 'yellow', 1.0: 'red'}
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        if save_path:
            m.save(save_path)
        
        return m
    
    def create_time_series_plot(self, data: pd.DataFrame, 
                              feature: str = 'food_insecure',
                              save_path: Optional[str] = None) -> None:
        """Create time series plot for food security trends.
        
        Args:
            data: Time series data
            feature: Feature to plot
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        if 'time' in data.columns:
            plt.plot(data['time'], data[feature], linewidth=2)
            plt.xlabel('Time')
        else:
            plt.plot(data[feature], linewidth=2)
            plt.xlabel('Index')
        
        plt.ylabel(feature.replace('_', ' ').title())
        plt.title(f'Food Security Trend - {feature.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_feature_importance_plot(self, feature_importance: Dict[str, float],
                                     model_name: str = 'Model',
                                     save_path: Optional[str] = None) -> None:
        """Create feature importance plot.
        
        Args:
            feature_importance: Dictionary of feature importance scores
            model_name: Name of the model
            save_path: Path to save the plot
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        features, importances = zip(*sorted_features)
        
        # Create horizontal bar plot
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(features))
        plt.barh(y_pos, importances, align='center')
        plt.yticks(y_pos, [f.replace('_', ' ').title() for f in features])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {model_name}')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Invert y-axis to show most important at top
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, data: pd.DataFrame, 
                                  geo_data: gpd.GeoDataFrame,
                                  predictions: Optional[np.ndarray] = None,
                                  probabilities: Optional[np.ndarray] = None) -> go.Figure:
        """Create interactive dashboard.
        
        Args:
            data: Feature data
            geo_data: Geographic data
            predictions: Model predictions
            probabilities: Prediction probabilities
            
        Returns:
            Plotly dashboard figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Food Security Distribution', 'Risk Score Distribution',
                          'Geographic Distribution', 'Feature Correlation'),
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "scattermapbox"}, {"type": "heatmap"}]]
        )
        
        # Food security distribution
        if predictions is not None:
            secure_count = np.sum(predictions == 0)
            insecure_count = np.sum(predictions == 1)
            
            fig.add_trace(
                go.Pie(
                    labels=['Food Secure', 'Food Insecure'],
                    values=[secure_count, insecure_count],
                    name="Food Security Status"
                ),
                row=1, col=1
            )
        
        # Risk score distribution
        if probabilities is not None:
            fig.add_trace(
                go.Histogram(
                    x=probabilities,
                    name="Risk Score Distribution",
                    nbinsx=30
                ),
                row=1, col=2
            )
        
        # Geographic distribution
        fig.add_trace(
            go.Scattermapbox(
                lat=geo_data['latitude'],
                lon=geo_data['longitude'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=probabilities if probabilities is not None else 'blue',
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Risk Score")
                ),
                name="Geographic Distribution"
            ),
            row=2, col=1
        )
        
        # Feature correlation
        feature_cols = [
            'crop_yield', 'rainfall', 'market_access_score', 
            'poverty_rate', 'food_price_index'
        ]
        corr_matrix = data[feature_cols].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Food Security Monitoring Dashboard",
            height=800,
            showlegend=False
        )
        
        # Update map layout
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(
                    lat=self.geo_config['map']['center_lat'],
                    lon=self.geo_config['map']['center_lon']
                ),
                zoom=self.geo_config['map']['zoom_level']
            )
        )
        
        return fig
    
    def save_all_visualizations(self, data: pd.DataFrame, 
                               geo_data: gpd.GeoDataFrame,
                               predictions: Optional[np.ndarray] = None,
                               probabilities: Optional[np.ndarray] = None,
                               output_dir: str = 'assets') -> None:
        """Save all visualizations to output directory.
        
        Args:
            data: Feature data
            geo_data: Geographic data
            predictions: Model predictions
            probabilities: Prediction probabilities
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Feature distributions
        self.create_feature_distribution_plots(
            data, str(output_path / 'feature_distributions.png')
        )
        
        # Correlation heatmap
        self.create_correlation_heatmap(
            data, str(output_path / 'correlation_heatmap.png')
        )
        
        # Food security map
        if predictions is not None:
            self.create_food_security_map(
                geo_data, predictions, probabilities,
                str(output_path / 'food_security_map.html')
            )
        
        # Risk heatmap
        if probabilities is not None:
            self.create_risk_heatmap(
                geo_data, probabilities,
                str(output_path / 'risk_heatmap.html')
            )
        
        # Interactive dashboard
        dashboard = self.create_interactive_dashboard(
            data, geo_data, predictions, probabilities
        )
        dashboard.write_html(str(output_path / 'interactive_dashboard.html'))
        
        logger.info(f"All visualizations saved to {output_path}")


def main():
    """Main function for visualization."""
    # This would be called from the main visualization script
    pass


if __name__ == "__main__":
    main()
