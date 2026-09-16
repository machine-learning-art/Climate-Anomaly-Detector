"""
Plot trends module for visualizing climate anomaly detection results.

This module provides functions to plot actual temperature data, model predictions,
and identified anomalies from the model_application_results table.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from typing import Optional, Dict, Any, Tuple
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClimateTrendPlotter:
    """
    Class for plotting climate data trends and anomalies.
    
    Provides visualization of actual temperature data, model predictions,
    and identified anomalies from the anomaly detection pipeline.
    
    IMPROVED: Data is loaded once during initialization and reused for all plots.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plotter with configuration.
        
        Args:
            config: Configuration dictionary containing database path and other parameters
        """
        self.config = config or {}
        self.db_path = self.config.get('db_path', 'climate.db')
        self.data_cache: Optional[pd.DataFrame] = None
        logger.info(f"Initialized ClimateTrendPlotter with db_path: {self.db_path}")

    def load_data_from_database(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Load climate data and model results from database.
        
        Args:
            start_date: Start date for filtering (YYYY-MM-DD format)
            end_date: End date for filtering (YYYY-MM-DD format)
            force_reload: If True, reload data even if cached
            
        Returns:
            DataFrame containing climate data with anomaly information
        """
        # Return cached data if available and not forcing reload
        if self.data_cache is not None and not force_reload:
            logger.info("Using cached data")
            return self.data_cache.copy()
        
        try:
            conn = sqlite3.connect(self.db_path)

            # Base query to get all relevant columns
            query = """
                SELECT date, temp_max, temp_min, avg_temp,
                       avg_reconstruction, is_anomaly, reconstruction_error
                FROM model_application_results
            """

            params = []

            # Add date filtering if specified
            if start_date or end_date:
                query += " WHERE date BETWEEN ? AND ?"
                params.extend([start_date or '1900-01-01', end_date or '2100-12-31'])

            query += " ORDER BY date"

            df = pd.read_sql_query(query, conn, params=params)

            # Convert date column to datetime
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

                # Scale temperature values from tenths of degrees Celsius to actual degrees
                temp_columns = ['temp_max', 'temp_min', 'avg_temp', 'avg_reconstruction']
                for col in temp_columns:
                    if col in df.columns:
                        df[col] = df[col] / 10.0
            
            # Cache the data for subsequent use
            self.data_cache = df.copy()
            logger.info(f"Loaded {len(df)} records from database and cached for future use")
            return df

        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            raise

    def plot_temperature_trends(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        figsize: tuple = (14, 8),
        title: str = "Climate Temperature Trends with Anomaly Detection"
    ) -> plt.Figure:
        """
        Plot temperature trends showing actual data, predictions, and anomalies.

        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            figsize: Figure size (width, height)
            title: Title for the plot

        Returns:
            Matplotlib Figure object
        """
        # Load data from database (will use cache if available)
        df = self.load_data_from_database(start_date, end_date)

        if df.empty:
            logger.warning("No data available for plotting")
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title(title)
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return fig

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Plot actual temperature ranges
        if 'temp_max' in df.columns and 'temp_min' in df.columns:
            ax.fill_between(
                df.index,
                df['temp_max'],
                df['temp_min'],
                alpha=0.3,
                color='blue',
                label='Actual Temperature Range'
            )

        # Plot average temperature
        if 'avg_temp' in df.columns:
            ax.plot(
                df.index,
                df['avg_temp'],
                color='blue',
                linewidth=2,
                label='Actual Average Temperature'
            )

        # Plot model predictions
        if 'avg_reconstruction' in df.columns:
            ax.plot(
                df.index,
                df['avg_reconstruction'],
                color='green',
                linestyle='--',
                linewidth=2,
                label='Model Prediction'
            )

        # Highlight anomalies
        if 'is_anomaly' in df.columns:
            anomalies = df[df['is_anomaly'] == 1]

            if not anomalies.empty:
                ax.scatter(
                    anomalies.index,
                    anomalies['avg_temp'],
                    color='red',
                    s=100,
                    marker='X',
                    label='Anomalies'
                )

        # Add title and labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()

        logger.info("Created temperature trends plot")
        return fig

    def plot_anomaly_errors(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        figsize: tuple = (14, 6),
        title: str = "Reconstruction Errors and Anomalies"
    ) -> plt.Figure:
        """
        Plot reconstruction errors with anomaly markers.

        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            figsize: Figure size (width, height)
            title: Title for the plot

        Returns:
            Matplotlib Figure object
        """
        # Load data from database (will use cache if available)
        df = self.load_data_from_database(start_date, end_date)

        if df.empty:
            logger.warning("No data available for plotting")
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title(title)
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return fig

        # Create figure and axis
        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot reconstruction errors
        if 'reconstruction_error' in df.columns:
            color = 'tab:blue'
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Reconstruction Error', color=color, fontsize=12)

            # Plot all errors
            ax1.plot(df.index, df['reconstruction_error'], color=color, alpha=0.7, label='Error')

            # Highlight anomaly errors
            anomalies = df[df['is_anomaly'] == 1]
            if not anomalies.empty:
                ax1.scatter(
                    anomalies.index,
                    anomalies['reconstruction_error'],
                    color='red',
                    s=100,
                    marker='X',
                    label='Anomaly Error'
                )

            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)

        # Create second y-axis for anomaly flag
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Anomaly Flag', color=color, fontsize=12)

        if 'is_anomaly' in df.columns:
            # Plot anomaly flags as step function
            ax2.plot(
                df.index,
                df['is_anomaly'] * 0.5,  # Scale for visibility
                color=color,
                drawstyle='steps-post',
                alpha=0.3,
                label='Anomaly Flag'
            )

        # Add title and legend
        ax1.set_title(title, fontsize=14, fontweight='bold')
        fig.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
        plt.tight_layout()

        logger.info("Created anomaly error plot")
        return fig

    def plot_error_distribution(self, figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Plot distribution of reconstruction errors with anomaly threshold.

        Args:
            figsize: Figure size (width, height)

        Returns:
            Matplotlib Figure object
        """
        # Load data from database (will use cache if available)
        df = self.load_data_from_database()

        if df.empty or 'reconstruction_error' not in df.columns:
            logger.warning("No error data available for distribution plot")
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title('Error Distribution')
            ax.text(0.5, 0.5, 'No error data available', ha='center', va='center')
            return fig

        # Create figure and axis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Histogram of all errors
        sns.histplot(df['reconstruction_error'], kde=True, ax=ax1, color='blue', alpha=0.7)
        ax1.set_title('Error Distribution')
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Frequency')

        # Histogram of anomaly errors
        anomalies = df[df['is_anomaly'] == 1]
        if not anomalies.empty:
            sns.histplot(anomalies['reconstruction_error'], kde=True, ax=ax2, color='red', alpha=0.7)
            ax2.set_title('Anomaly Error Distribution')
        else:
            ax2.text(0.5, 0.5, 'No anomalies found', ha='center', va='center')
            ax2.set_title('Anomaly Error Distribution')

        ax2.set_xlabel('Reconstruction Error')
        ax2.set_ylabel('Frequency')

        plt.tight_layout()
        logger.info("Created error distribution plot")
        return fig

    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300) -> None:
        """
        Save a matplotlib figure to file.

        Args:
            fig: Matplotlib Figure object
            filename: Output filename (with extension)
            dpi: Resolution in dots per inch
        """
        try:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {filename}")
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            raise

    def create_comprehensive_report(self, output_dir: str = 'reports') -> Dict[str, Any]:
        """
        Create comprehensive visualization report with multiple plots.
        
        IMPROVED: Data is loaded only once at the beginning and reused for all plots.

        Args:
            output_dir: Directory to save plots

        Returns:
            Dictionary containing information about created plots
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'plots_created': [],
            'timestamp': datetime.now().isoformat(),
            'output_directory': output_dir,
            'data_records_loaded': 0
        }

        try:
            # Load data once at the beginning (will be cached for all subsequent plot calls)
            df = self.load_data_from_database()
            results['data_records_loaded'] = len(df)

            # Generate and save temperature trends plot
            temp_fig = self.plot_temperature_trends()
            temp_filename = os.path.join(output_dir, 'plot_temperature_trends.png')
            self.save_plot(temp_fig, temp_filename)
            results['plots_created'].append('temperature_trends.png')
            plt.close(temp_fig)

            # Generate and save anomaly errors plot
            error_fig = self.plot_anomaly_errors()
            error_filename = os.path.join(output_dir, 'plot_anomaly_errors.png')
            self.save_plot(error_fig, error_filename)
            results['plots_created'].append('anomaly_errors.png')
            plt.close(error_fig)

            # Generate and save error distribution plot
            dist_fig = self.plot_error_distribution()
            dist_filename = os.path.join(output_dir, 'plot_error_distribution.png')
            self.save_plot(dist_fig, dist_filename)
            results['plots_created'].append('error_distribution.png')
            plt.close(dist_fig)

            logger.info(f"Created comprehensive report with {len(results['plots_created'])} plots in {output_dir}")
            return results

        except Exception as e:
            logger.error(f"Error creating comprehensive report: {e}")
            raise

    def clear_cache(self) -> None:
        """
        Clear the cached data.
        
        Useful when you want to reload data with different parameters.
        """
        self.data_cache = None
        logger.info("Cleared data cache")

if __name__ == "__main__":
    # Example usage
    config = {
        'db_path': 'climate.db'
    }

    plotter = ClimateTrendPlotter(config)

    ## Create and display temperature trends plot
    # print("Creating temperature trends plot...")
    # temp_fig = plotter.plot_temperature_trends()
    # plt.show()

    ## Create and display error distribution plot
    # print("Creating error distribution plot...")
    # dist_fig = plotter.plot_error_distribution()
    # plt.show()

    ## Create and display anomaly errors plot
    # print("Creating anomaly errors plot...")
    # error_fig = plotter.plot_anomaly_errors()
    # plt.show()

    # Create comprehensive report
    print("Creating comprehensive report...")
    report_results = plotter.create_comprehensive_report('demo')
    print(f"Report created successfully. Plots saved to: {report_results['output_directory']}")
    print(f"Data records loaded: {report_results['data_records_loaded']}")