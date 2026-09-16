# climate_anomaly_detector/src/pipeline_predict.py

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime
import logging
from typing import Dict, Any, Optional, Tuple
import json

# Import existing components
from pipeline_train import AnomalyDetectionPipeline
from preprocess import ClimateDataPreprocessor
from detect import AnomalyDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPredictor:
    """
    Class for applying trained anomaly detection models to full datasets
    and storing results in a structured database table.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model applier.

        Args:
            config: Configuration dictionary containing pipeline parameters
        """
        self.config = config or {}
        self.pipeline = AnomalyDetectionPipeline(config)
        self.preprocessor = ClimateDataPreprocessor(self.config.get('db_path', 'climate.db'))
        self.detector = AnomalyDetector(
            model_path=self.config.get('model_path', 'anomaly_model.h5'),
            config_path=self.config.get('config_path', 'detector_config.json')
        )
        self._load_trained_model()
        self.logger = logging.getLogger(__name__)

    def _load_trained_model(self) -> None:
        """
        Load trained model from file, handling both old and new formats.
        """
        # Try multiple possible paths
        possible_paths = [
            self.config.get('model_path', 'anomaly_model.h5'),
            self.config.get('model_path', 'anomaly_model.h5').replace('.h5', '.keras')
        ]

        loaded = False
        for model_path in possible_paths:
            if os.path.exists(model_path):
                try:
                    self.detector.load_model(model_path)
                    logger.info(f"Loaded trained model from {model_path}")

                    # Load configuration to get threshold
                    config = self.detector.load_config()
                    if 'threshold' in config:
                        self.detector.threshold = config['threshold']
                        logger.info(f"Set anomaly threshold to {self.detector.threshold}")
                    loaded = True
                    break

                except Exception as e:
                    logger.warning(f"Failed to load {model_path}: {e}")
                    continue

        if not loaded:
            raise FileNotFoundError(
                f"No valid trained model found at any of: {possible_paths}. "
                "Please train the model first using pipeline_train.py"
            )

    def _initialize_results_table(self) -> None:
        """
        Initialize the SQLite database table for storing model application results.
        Creates table with columns: date, temp_max, temp_min, avg_reconstruction, is_anomaly
        plus additional metadata columns.
        """
        db_path = self.config.get('db_path', 'climate.db')

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_application_results (
                    application_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_timestamp TEXT NOT NULL,
                    date DATE NOT NULL,
                    temp_max REAL,
                    temp_min REAL,
                    avg_temp REAL,
                    reconstruction_error REAL,
                    avg_reconstruction REAL,
                    is_anomaly INTEGER NOT NULL,
                    error_percentile REAL,
                    distance_to_threshold REAL,
                    model_version TEXT,
                    threshold REAL,
                    config_json TEXT,
                    affected_rows TEXT
                )
            """)
            logger.info("Model application results table initialized.")

    def _create_date_sequence_mapping(self, df: pd.DataFrame, sequence_length: int = 7) -> Dict[int, Tuple[str, list]]:
        """
        Create mapping from sequence indices to original dates and affected rows.
        Ensures datetime context is maintained throughout the preprocessing pipeline.
    
        Args:
            df: DataFrame with date index (may be integer or datetime)
            sequence_length: Number of days in each sequence
    
        Returns:
            Dictionary mapping sequence index to (representative_date, row_indices)
        """
        mapping = {}
        dates = df.index
    
        for seq_idx in range(len(df) - sequence_length + 1):
            start_row = seq_idx
            end_row = min(seq_idx + sequence_length, len(df))
    
            # Use the last date of each sequence as representative
            representative_date = dates[end_row - 1]
    
            # Convert to string representation based on type
            if hasattr(representative_date, 'strftime'):
                # Datetime object - format as proper date string
                date_str = representative_date.strftime('%Y-%m-%d')
            elif isinstance(representative_date, (int, float)):
                # Numeric index - try to extract actual dates from DataFrame columns
                if 'date' in df.columns and end_row - 1 < len(df):
                    date_value = df.iloc[end_row - 1].get('date')
                    if hasattr(date_value, 'strftime'):
                        # Date column exists as datetime object
                        date_str = date_value.strftime('%Y-%m-%d')
                    elif isinstance(date_value, (int, float)):
                        # Date column is numeric - try to convert to datetime assuming Unix timestamp
                        try:
                            import pandas as pd
                            date_obj = pd.to_datetime(date_value, unit='s')
                            date_str = date_obj.strftime('%Y-%m-%d')
                        except (ValueError, TypeError):
                            # Fallback if conversion fails
                            date_str = str(int(representative_date))
                    else:
                        # Other types - convert to string
                        date_str = str(date_value)
                elif 'date' in df.columns and end_row - 1 < len(df):
                    # Try to get date from columns if available
                    date_value = df.iloc[end_row - 1].get('date')
                    if hasattr(date_value, 'strftime'):
                        date_str = date_value.strftime('%Y-%m-%d')
                    else:
                        date_str = str(int(representative_date))
                else:
                    # Fallback to numeric representation
                    date_str = str(int(representative_date))
            else:
                # Other types - convert directly to string
                date_str = str(representative_date)
    
            mapping[seq_idx] = (
                date_str,
                list(range(start_row, end_row))
            )
    
        return mapping

    def apply_model_to_full_dataset(self) -> pd.DataFrame:
        """
        Apply trained model to the complete dataset and return results with date mapping.

        Returns:
            DataFrame containing application results for all data points
        """
        # Load and preprocess full dataset
        df = self._load_and_preprocess_data()

        # Prepare sequences for prediction
        X, _ = self.preprocessor.prepare_for_model(df)

        if len(X) == 0:
            raise ValueError("No data available after preprocessing")

        # Apply model to all data
        errors, is_anomaly = self.detector.detect_anomalies(X)
        reconstructions = self.detector.model.predict(X)

        # Create comprehensive results DataFrame
        results_df = self._create_comprehensive_results(
            df=df,
            X=X,
            errors=errors,
            is_anomaly=is_anomaly,
            reconstructions=reconstructions
        )

        return results_df

    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess the full dataset.

        Returns:
            Processed DataFrame ready for model application
        """
        # Determine data source
        if 'csv_path' in self.config:
            df = self.pipeline.fetcher.fetch_and_save(csv_path=self.config['csv_path'])
        elif 'noaa_params' in self.config:
            noaa_params = self.config['noaa_params']
            df = self.pipeline.fetcher.fetch_and_save(**noaa_params)
        else:
            # Load from database if no source specified
            df = self.preprocessor.load_data_from_db(
                start_date=self.config.get('start_date'),
                end_date=self.config.get('end_date')
            )

        # Preprocess the data
        processed_df, scaler_params = self.preprocessor.run_full_pipeline(
            start_date=self.config.get('start_date'),
            end_date=self.config.get('end_date')
        )

        # Store scaler parameters in detector for inverse transformation
        self.detector.scaler_params = scaler_params

        return processed_df

    def _create_comprehensive_results(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        errors: np.ndarray,
        is_anomaly: np.ndarray,
        reconstructions: np.ndarray
    ) -> pd.DataFrame:
        """
        Create detailed results DataFrame mapping original data to model predictions.

        Args:
            df: Original processed DataFrame with date index
            X: Input sequences used for prediction
            errors: Reconstruction errors for each sequence
            is_anomaly: Binary array indicating anomalies
            reconstructions: Model predictions

        Returns:
            DataFrame with comprehensive results including original data and predictions
        """
        # Get sequence length from preprocessing
        sequence_length = X.shape[1] if len(X.shape) > 1 else 7

        # Create date mapping for sequences
        date_mapping = self._create_date_sequence_mapping(df, sequence_length)

        # Apply inverse transformation to get original scale reconstructions
        try:
            inverse_transformed = self.detector.inverse_transform_reconstruction(reconstructions)
            logger.info(f"Inverse transformed keys: {list(inverse_transformed.keys())}")
        except Exception as e:
            logger.warning(f"Failed to apply inverse transform: {e}. Using scaled values instead.")
            # Create empty structure to avoid KeyError
            inverse_transformed = {'avg_temp': []}

        results = []

        for seq_idx in range(len(X)):
            representative_date, row_indices = date_mapping.get(seq_idx, (None, []))

            if not representative_date:
                continue

            # Get original data values for this sequence
            original_data = df.iloc[row_indices]

            # Calculate average reconstruction for the sequence (actual temperature predictions)
            # Use inverse-transformed values if available
            if self.detector.scaler_params and 'avg_temp' in inverse_transformed:
                # Get the avg_temp reconstructions at the last timestep from inverse transform
                avg_reconstruction = np.mean(inverse_transformed['avg_temp'][seq_idx])
            else:
                # No proper inverse transform available - use scaled values and note this in logs
                logger.warning(f"Using scaled reconstruction values for sequence {seq_idx} (no proper inverse transform)")
                avg_reconstruction = np.mean(reconstructions[seq_idx, -1, :])  # Use last timestep

            results.append({
                'date': representative_date,
                'temp_max': float(original_data['temp_max'].mean()) if 'temp_max' in original_data.columns else None,
                'temp_min': float(original_data['temp_min'].mean()) if 'temp_min' in original_data.columns else None,
                'avg_temp': float(original_data['avg_temp'].mean()) if 'avg_temp' in original_data.columns else None,
                'reconstruction_error': float(errors[seq_idx]),
                'avg_reconstruction': float(avg_reconstruction),
                'is_anomaly': bool(is_anomaly[seq_idx]),
                'error_percentile': float(np.percentile(errors, (seq_idx / len(X)) * 100)),
                'distance_to_threshold': float(errors[seq_idx] - self.detector.threshold) if self.detector.threshold else None,
                'sequence_index': seq_idx,
                'affected_rows': row_indices
            })

        return pd.DataFrame(results).set_index('date')

    def save_results_to_database(self, results_df: pd.DataFrame) -> int:
        """
        Save model application results to SQLite database.

        Args:
            results_df: DataFrame containing application results

        Returns:
            The application_id for the saved results
        """
        self._initialize_results_table()

        db_path = self.config.get('db_path', 'climate.db')
        run_timestamp = datetime.now().isoformat()
        model_version = self.config.get('model_version', '1.0')

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            for _, row in results_df.iterrows():
                # Convert affected_rows list to JSON string
                affected_rows_json = json.dumps(row['affected_rows']) if 'affected_rows' in row else None

                cursor.execute("""
                    INSERT INTO model_application_results (
                        run_timestamp, date, temp_max, temp_min,
                        avg_temp, avg_reconstruction, is_anomaly,
                        reconstruction_error, error_percentile,
                        distance_to_threshold, model_version, threshold,
                        config_json, affected_rows
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_timestamp,
                    row.name,
                    float(row['temp_max']) if pd.notna(row['temp_max']) else None,
                    float(row['temp_min']) if pd.notna(row['temp_min']) else None,
                    float(row['avg_temp']) if pd.notna(row['avg_temp']) else None,
                    float(row['avg_reconstruction']),
                    int(row['is_anomaly']),
                    float(row['reconstruction_error']),
                    float(row.get('error_percentile', 0)),
                    float(row.get('distance_to_threshold', 0)) if row.get('distance_to_threshold') else None,
                    model_version,
                    float(self.detector.threshold) if self.detector.threshold else None,
                    json.dumps(self.config),
                    affected_rows_json
                ))

            application_id = cursor.lastrowid
            logger.info(f"Saved {len(results_df)} results to database with application_id: {application_id}")
            return application_id

    def get_anomaly_summary(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve summary of anomalies from the database.

        Args:
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)

        Returns:
            DataFrame containing anomaly summary
        """
        db_path = self.config.get('db_path', 'climate.db')

        with sqlite3.connect(db_path) as conn:
            query = """
                SELECT date, temp_max, temp_min, avg_temp,
                       avg_reconstruction, is_anomaly,
                       reconstruction_error, error_percentile
                FROM model_application_results
                WHERE is_anomaly = 1
            """

            params = []

            if start_date or end_date:
                query += " AND date BETWEEN ? AND ?"
                params.extend([start_date or '1900-01-01', end_date or datetime.now().strftime('%Y-%m-%d')])

            query += " ORDER BY date"

            df = pd.read_sql_query(query, conn, params=params)
            return df

    def get_temporal_analysis(self) -> Dict[str, Any]:
        """
        Perform temporal analysis of anomalies.
        Uses SQLite-compatible date functions instead of PostgreSQL EXTRACT.
        """
        db_path = self.config.get('db_path', 'climate.db')

        with sqlite3.connect(db_path) as conn:
            # Get all anomaly dates with SQLite-compatible date extraction
            query = """
                SELECT date,
                    strftime('%m', date) as month,
                    strftime('%w', date) as day_of_week
                FROM model_application_results
                WHERE is_anomaly = 1
            """

            df = pd.read_sql_query(query, conn)

            if len(df) == 0:
                return {'message': 'No anomalies found'}

            # Convert month from string to integer for analysis
            df['month'] = df['month'].astype(int)
            # day_of_week is already numeric (0-6)

            # Ensure date column is datetime - handle both string and integer formats
            if pd.api.types.is_numeric_dtype(df['date']):
                # If dates are stored as integers, convert to datetime assuming they're Unix timestamps
                df['date'] = pd.to_datetime(df['date'], unit='s')
            else:
                # Convert string dates to datetime
                df['date'] = pd.to_datetime(df['date'])

            # Perform temporal analysis
            results = {
                'total_anomalies': len(df),
                'date_range': {
                    'start': df['date'].min().strftime('%Y-%m-%d'),
                    'end': df['date'].max().strftime('%Y-%m-%d')
                },
                'monthly_distribution': df['month'].value_counts().to_dict(),
                'day_of_week_distribution': df['day_of_week'].value_counts().to_dict(),
                'anomalies_per_month': df.groupby(df['date'].dt.to_period('M')).size().to_dict()
            }

            return results

    def run_full_application(self) -> Dict[str, Any]:
        """
        Run complete model application workflow: apply model, save results, and analyze.

        Returns:
            Dictionary containing results summary
        """
        try:
            # Step 1: Apply model to full dataset
            logger.info("Applying model to full dataset...")
            results_df = self.apply_model_to_full_dataset()

            # Step 2: Save results to database
            logger.info("Saving results to database...")
            application_id = self.save_results_to_database(results_df)

            # Step 3: Perform analysis
            logger.info("Performing temporal analysis...")
            temporal_analysis = self.get_temporal_analysis()

            # Step 4: Get anomaly summary
            anomalies_summary = self.get_anomaly_summary()

            return {
                'application_id': application_id,
                'results_shape': results_df.shape,
                'total_anomalies': len(anomalies_summary),
                'temporal_analysis': temporal_analysis,
                'date_range': {
                    'start': results_df.index.min(),
                    'end': results_df.index.max()
                }
            }

        except Exception as e:
            logger.error(f"Model application failed: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    config = {
        'db_path': 'climate.db',
        'model_path': 'climate_anomaly_detector/src/model/anomaly_model.h5',
        'config_path': 'climate_anomaly_detector/src/configs/detector_config.json',
        'csv_path': 'climate_anomaly_detector/demo/sample_demo.csv',
        'start_date': '2022-01-01',
        'end_date': '2025-12-31',
        'model_version': '1.0'
    }

    # Create applier instance
    applier = ModelPredictor(config)

    # Run full application workflow
    results_summary = applier.run_full_application()

    print("\nModel Application Summary:")
    print(f"Application ID: {results_summary['application_id']}")
    print(f"Results shape: {results_summary['results_shape']}")
    print(f"Total anomalies detected: {results_summary['total_anomalies']}")
    print(f"Date range analyzed: {results_summary['date_range']['start']} to {results_summary['date_range']['end']}")

    if 'temporal_analysis' in results_summary:
        temp_analysis = results_summary['temporal_analysis']
        if isinstance(temp_analysis, dict) and 'total_anomalies' in temp_analysis:
            print(f"\nTemporal Analysis:")
            print(f"Total anomalies: {temp_analysis['total_anomalies']}")
            print("Monthly distribution:")
            for month, count in temp_analysis.get('monthly_distribution', {}).items():
                print(f"  Month {month}: {count} anomalies")