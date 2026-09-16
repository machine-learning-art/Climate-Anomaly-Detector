# climate_anomaly_detector/src/preprocess.py

import pandas as pd
import numpy as np
import sqlite3
from scipy.signal import detrend
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClimateDataPreprocessor:
    def __init__(self, db_path: str = "climate.db"):
        """
        Initialize the preprocessor with database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

    def load_data_from_db(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load climate data from SQLite database with optional date filtering.

        Args:
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)

        Returns:
            DataFrame containing the loaded data
        """
        try:
            query = "SELECT * FROM daily_temps"
            params = []

            if start_date or end_date:
                query += " WHERE date BETWEEN ? AND ?"
                params.extend([start_date or '1900-01-01', end_date or datetime.now().strftime('%Y-%m-%d')])

            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(query, conn, params=params)
                logger.info(f"Loaded {len(df)} records from database")
                return df

        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw climate data by handling missing values and outliers.

        Args:
            df: Raw DataFrame containing climate data

        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()

        # Convert date column to datetime if not already
        if 'date' in cleaned_df.columns:
            cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])

        # Handle missing values - forward fill then backward fill
        for col in ['temp_max', 'temp_min']:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].ffill().bfill()

        # Remove extreme outliers using IQR method
        for temp_col in ['temp_max', 'temp_min']:
            if temp_col in cleaned_df.columns:
                q1 = cleaned_df[temp_col].quantile(0.25)
                q3 = cleaned_df[temp_col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # Cap outliers instead of removing them
                cleaned_df[temp_col] = np.where(
                    cleaned_df[temp_col] < lower_bound,
                    lower_bound,
                    np.where(
                        cleaned_df[temp_col] > upper_bound,
                        upper_bound,
                        cleaned_df[temp_col]
                    )
                )

        logger.info("Data cleaning completed")
        return cleaned_df

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional features from raw temperature data.

        Args:
            df: Cleaned DataFrame containing climate data

        Returns:
            DataFrame with additional calculated features
        """
        # Make a copy to avoid modifying original
        feature_df = df.copy()

        # Calculate average temperature
        if all(col in feature_df.columns for col in ['temp_max', 'temp_min']):
            feature_df['avg_temp'] = (feature_df['temp_max'] + feature_df['temp_min']) / 2

        # Calculate daily temperature range
        if all(col in feature_df.columns for col in ['temp_max', 'temp_min']):
            feature_df['temp_range'] = feature_df['temp_max'] - feature_df['temp_min']

        # Calculate moving averages (7-day and 30-day)
        if 'avg_temp' in feature_df.columns:
            feature_df['ma_7day'] = feature_df['avg_temp'].rolling(window=7, min_periods=1).mean()
            feature_df['ma_30day'] = feature_df['avg_temp'].rolling(window=30, min_periods=1).mean()

        # Calculate temperature anomalies (difference from 30-day moving average)
        if all(col in feature_df.columns for col in ['avg_temp', 'ma_30day']):
            feature_df['temp_anomaly'] = feature_df['avg_temp'] - feature_df['ma_30day']

        logger.info("Feature calculation completed")
        return feature_df

    def detrend_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply detrending to temperature data to remove long-term trends.

        Args:
            df: DataFrame containing climate data with temperature columns

        Returns:
            DataFrame with detrended temperature values
        """
        # Make a copy to avoid modifying original
        detrended_df = df.copy()

        for temp_col in ['temp_max', 'temp_min', 'avg_temp']:
            if temp_col in detrended_df.columns:
                detrended_values = detrend(detrended_df[temp_col].values)
                detrended_df[f'detrended_{temp_col}'] = detrended_values

        logger.info("Detrending completed")
        return detrended_df

    def normalize_data(
        self,
        df: pd.DataFrame,
        method: str = 'standard',
        columns: Optional[list] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normalize temperature data using specified method.

        Args:
            df: DataFrame containing climate data
            method: Normalization method ('standard' or 'minmax')
            columns: List of columns to normalize (default: all numeric columns)

        Returns:
            Tuple of (normalized DataFrame, scaler parameters)
        """
        # Make a copy to avoid modifying original
        normalized_df = df.copy()

        if columns is None:
            # Default to temperature-related columns
            columns = [col for col in df.columns
                      if any(temp in col.lower() for temp in ['temp', 'avg', 'range'])]
            columns = [col for col in columns if df[col].dtype in ['float64', 'int64']]

        scaler_params = {}

        if method == 'standard':
            scaler = StandardScaler()
            for col in columns:
                if col in normalized_df.columns:
                    scaled_values = scaler.fit_transform(normalized_df[[col]])
                    normalized_df[f'scaled_{col}'] = scaled_values
                    # Store scaler parameters for inverse transform
                    scaler_params[col] = {
                        'mean': scaler.mean_[0],
                        'scale': scaler.scale_[0]
                    }
        elif method == 'minmax':
            scaler = MinMaxScaler()
            for col in columns:
                if col in normalized_df.columns:
                    scaled_values = scaler.fit_transform(normalized_df[[col]])
                    normalized_df[f'scaled_{col}'] = scaled_values
                    # Store scaler parameters for inverse transform
                    scaler_params[col] = {
                        'min': scaler.data_min_[0],
                        'scale': scaler.data_range_[0]
                    }
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        logger.info(f"{method.capitalize()} normalization completed")
        return normalized_df, scaler_params

    def prepare_for_model(
        self,
        df: pd.DataFrame,
        sequence_length: int = 7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for time series modeling by creating sequences.

        Args:
            df: DataFrame containing processed climate data
            sequence_length: Number of days to include in each sequence

        Returns:
            Tuple of (X, y) arrays ready for model training
        """
        # Select relevant columns for modeling
        feature_cols = [col for col in df.columns if 'scaled_' in col]
        if not feature_cols:
            raise ValueError("No scaled features found. Run normalize_data first.")

        X = []
        y = []

        # Convert to numpy array
        values = df[feature_cols].values

        for i in range(len(values) - sequence_length):
            X.append(values[i:i+sequence_length])
            y.append(values[i+sequence_length])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Prepared {len(X)} sequences of length {sequence_length}")
        return X, y

    def save_processed_data(self, df: pd.DataFrame) -> None:
        """
        Save processed data back to SQLite database in a separate table.

        Args:
            df: Processed DataFrame containing climate data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create processed data table if it doesn't exist
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS processed_data (
                        date TEXT PRIMARY KEY,
                        temp_max REAL,
                        temp_min REAL,
                        avg_temp REAL,
                        temp_range REAL,
                        ma_7day REAL,
                        ma_30day REAL,
                        temp_anomaly REAL,
                        detrended_temp_max REAL,
                        detrended_temp_min REAL,
                        detrended_avg_temp REAL,
                        scaled_avg_temp REAL
                    )
                """)

                # Save only the columns that exist in both DataFrames
                existing_cols = [col for col in df.columns if col in [
                    'date', 'temp_max', 'temp_min', 'avg_temp',
                    'temp_range', 'ma_7day', 'ma_30day',
                    'temp_anomaly', 'detrended_temp_max',
                    'detrended_temp_min', 'detrended_avg_temp',
                    'scaled_avg_temp'
                ]]

                df[existing_cols].to_sql("processed_data", conn, if_exists="replace", index=False)
                logger.info(f"Saved {len(df)} processed records to database")

        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            raise

    def run_full_pipeline(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline from database to processed data.

        Args:
            start_date: Start date for data loading (optional)
            end_date: End date for data loading (optional)

        Returns:
            Processed DataFrame ready for anomaly detection
        """
        # Step 1: Load data from database
        df = self.load_data_from_db(start_date, end_date)

        # Step 2: Clean the data
        cleaned_df = self.clean_data(df)

        # Step 3: Calculate features
        feature_df = self.calculate_features(cleaned_df)

        # Step 4: Detrend the data
        detrended_df = self.detrend_data(feature_df)

        # Step 5: Normalize the data
        normalized_df, scaler_params = self.normalize_data(detrended_df)

        # Step 6: Save processed data back to database
        self.save_processed_data(normalized_df)

        return normalized_df, scaler_params

if __name__ == "__main__":
    # Example usage
    preprocessor = ClimateDataPreprocessor()

    # Run full preprocessing pipeline for the last year of data
    processed_data = preprocessor.run_full_pipeline(
        start_date="2025-01-01",
        end_date="2025-12-31"
    )

    print(f"Processed {len(processed_data)} records")