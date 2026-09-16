# climate_anomaly_detector/src/pipeline_train.py

from fetch import ClimateDataFetcher
from preprocess import ClimateDataPreprocessor
from detect import AnomalyDetector
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List

class AnomalyDetectionPipeline:
    """
    End-to-end pipeline for climate anomaly detection.
    Orchestrates data fetching, preprocessing, and anomaly detection.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the anomaly detection pipeline.

        Args:
            config: Configuration dictionary containing pipeline parameters
        """
        self.config = config or {}
        self.fetcher = ClimateDataFetcher(self.config.get('db_path', 'climate.db'))
        self.preprocessor = ClimateDataPreprocessor(self.config.get('db_path', 'climate.db'))
        self.detector = AnomalyDetector(
            model_path=self.config.get('model_path', 'anomaly_model.h5'),
            config_path=self.config.get('config_path', 'detector_config.json')
        )
        self.test_data = None  # Store test data for later evaluation
        self.sequence_to_row_mapping = None  # Initialize sequence mapping
        self.logger = logging.getLogger(__name__)


    def run(self, data_source: str = 'csv') -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the complete anomaly detection pipeline.

        Args:
            data_source: Source of data ('noaa' or 'csv')

        Returns:
            Tuple containing (processed_dataframe, anomalies_indices)
        """
        try:
            # Step 1: Fetch and save data
            if data_source == 'noaa':
                noaa_params = self.config.get('noaa_params', {})
                df = self.fetcher.fetch_from_noaa(**noaa_params)
            elif data_source == 'csv':
                csv_path = self.config.get('csv_path')
                if not csv_path:
                    raise ValueError("CSV path must be provided for CSV data source")
                df = self.fetcher.load_from_csv(csv_path)
            else:
                raise ValueError(f"Unknown data source: {data_source}")

            # Save to database
            self.fetcher.save_to_db(df)
            self.logger.info(f"Saved {len(df)} records to database")

            # Step 2: Preprocess data
            processed_df, scaler_params = self.preprocessor.run_full_pipeline(
                start_date=self.config.get('start_date'),
                end_date=self.config.get('end_date')
            )

            # Step 3: Prepare model input
            X, y = self.preprocessor.prepare_for_model(processed_df)

            if len(X) == 0:
                raise ValueError("No data available for training after preprocessing")

            # Store the mapping from sequences to original rows
            # This would need to be implemented in the preprocessor
            self.sequence_to_row_mapping = self._create_sequence_mapping(len(processed_df), X.shape[0])

            # Step 4: Train detector (on normal data)
            train_size = int(0.8 * len(X))
            self.detector.train(
                X_train=X[:train_size],
                epochs=self.config.get('epochs', 20),
                batch_size=self.config.get('batch_size', 32),
                early_stopping_patience=self.config.get('patience', 5)
            )

            # Step 5: Set threshold using normal data
            self.detector.set_threshold(X[:train_size])

            # Store test data for later evaluation
            self.test_data = X[train_size:]  # Store the test set

            # Step 6: Detect anomalies in test data
            errors, is_anomaly = self.detector.detect_anomalies(X[train_size:])
            anomalies = np.where(is_anomaly == 1)[0]

            return processed_df, anomalies

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

    def run_with_cross_validation(
        self,
        data_source: str = 'csv',
        n_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Run pipeline with cross-validation for robust model evaluation.

        Args:
            data_source: Source of data ('noaa' or 'csv')
            n_folds: Number of folds for time-series CV

        Returns:
            Dictionary containing cross-validation results and analysis
        """
        try:
            # Step 1: Fetch and preprocess data (same as before)
            if data_source == 'noaa':
                noaa_params = self.config.get('noaa_params', {})
                df = self.fetcher.fetch_from_noaa(**noaa_params)
            elif data_source == 'csv':
                csv_path = self.config.get('csv_path')
                if not csv_path:
                    raise ValueError("CSV path must be provided for CSV data source")
                df = self.fetcher.load_from_csv(csv_path)

            # Save to database
            self.fetcher.save_to_db(df)
            # self.logger.info(f"Saved {len(df)} records to database")

            # Step 2: Preprocess data
            processed_df, _ = self.preprocessor.run_full_pipeline(
                start_date=self.config.get('start_date'),
                end_date=self.config.get('end_date')
            )

            # Step 3: Prepare model input
            X, y = self.preprocessor.prepare_for_model(processed_df)

            if len(X) == 0:
                raise ValueError("No data available for training after preprocessing")

            # Step 4: Build model (but don't train yet)
            self.detector.build_model(input_shape=X.shape[1:])

            # Step 5: Run cross-validation
            cv_results = self.detector.time_series_cross_validation(
                X,
                n_splits=n_folds,
                epochs=self.config.get('epochs', 20),
                batch_size=self.config.get('batch_size', 32)
            )

            # Step 6: Analyze cross-validation results
            analysis = self._analyze_cv_results(cv_results, processed_df)

            return {
                'cv_results': cv_results,
                'analysis': analysis,
                'processed_data_shape': processed_df.shape
            }

        except Exception as e:
            self.logger.error(f"Cross-validation pipeline failed: {e}")
            raise

    def _analyze_cv_results(
        self,
        cv_results: Dict[str, Any],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze cross-validation results to provide insights.

        Args:
            cv_results: Results from time-series CV
            df: Original DataFrame for context

        Returns:
            Dictionary containing analysis and recommendations
        """
        analysis = {
            'threshold_stability': cv_results['threshold_stability'],
            'error_distribution': {
                'mean': cv_results['overall_mean_error'],
                'std': cv_results['overall_std_error']
            },
            'recommendations': []
        }

        # Calculate min/max from concatenated errors instead of individual arrays
        all_errors = np.concatenate(cv_results['fold_errors'])
        analysis['error_distribution'].update({
            'min': np.min(all_errors),
            'max': np.max(all_errors)
        })

        # Check threshold stability
        if not analysis['threshold_stability']:
            analysis['recommendations'].append(
                "Threshold varies significantly across folds. Consider: "
                "- More training data\n"
                "- Different model architecture\n"
                "- Manual threshold adjustment"
            )

        # Check for potential seasonality issues
        if 'seasonal_patterns' in str(df.columns):
            analysis['recommendations'].append(
                "Consider using seasonal cross-validation to ensure model works "
                "across different seasons and years."
            )

        return analysis

    def run_seasonal_cv(
        self,
        data_source: str = 'csv'
    ) -> Dict[str, Any]:
        """
        Run pipeline with seasonal cross-validation.

        Args:
            data_source: Source of data ('noaa' or 'csv')

        Returns:
            Dictionary containing seasonal CV results
        """
        try:
            # Step 1-3: Same as before (fetch, save, preprocess)
            if data_source == 'noaa':
                noaa_params = self.config.get('noaa_params', {})
                df = self.fetcher.fetch_from_noaa(**noaa_params)
            elif data_source == 'csv':
                csv_path = self.config.get('csv_path')
                df = self.fetcher.load_from_csv(csv_path)

            self.fetcher.save_to_db(df)
            processed_df, _ = self.preprocessor.run_full_pipeline(
                start_date=self.config.get('start_date'),
                end_date=self.config.get('end_date')
            )

            # Step 4: Prepare data with dates
            X, y = self.preprocessor.prepare_for_model(processed_df)

            if len(X) == 0:
                raise ValueError("No data available for training after preprocessing")

            # Get corresponding dates (need to modify prepare_for_model or track dates)
            dates = processed_df.index  # Assuming date is index

            # Step 5: Build model
            self.detector.build_model(input_shape=X.shape[1:])

            # Step 6: Run seasonal CV
            seasonal_results = self.detector.seasonal_split_cross_validation(
                X,
                dates=dates
            )

            return {
                'seasonal_results': seasonal_results,
                'processed_data_shape': processed_df.shape
            }

        except Exception as e:
            self.logger.error(f"Seasonal CV pipeline failed: {e}")
            raise

    def evaluate(self, test_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate the anomaly detector performance.
        """
        if self.detector.model is None:
            raise ValueError("Detector not trained. Run pipeline first.")

        # Use stored test data or provided test data
        evaluation_data = test_data if test_data is not None else self.test_data

        if evaluation_data is None:
            raise ValueError("No test data available. Either run pipeline first or provide test_data.")

        return self.detector.evaluate(evaluation_data)

    def get_anomaly_report(self, anomalies: np.ndarray) -> pd.DataFrame:
        """
        Generate a detailed report of detected anomalies.

        Args:
            anomalies: Array of anomaly indices (indices into test set)

        Returns:
            DataFrame containing anomaly details with original data context
        """
        if self.test_data is None:
            raise ValueError("No test data available. Run pipeline first.")

        # Extract the actual anomalous sequences from test data
        anomalous_sequences = self.test_data[anomalies]

        # Get scores for these specific sequences
        anomaly_scores_df = self.detector.get_anomaly_scores(anomalous_sequences)

        # For now, return scores with sequence information
        # A more complete solution would map back to original timestamps
        report = anomaly_scores_df.copy()
        report['sequence_index'] = anomalies

        self.logger.info(f"Generated anomaly report for {len(report)} anomalous sequences")
        return report

    def _create_sequence_mapping(self, n_rows: int, n_sequences: int) -> Dict[int, List[int]]:
        """
        Create mapping from sequence indices to original data row indices.

        Args:
            n_rows: Number of rows in original DataFrame
            n_sequences: Number of sequences created

        Returns:
            Dictionary mapping sequence index to list of original row indices
        """
        # This is a simplified version - actual implementation depends on how sequences are created
        mapping = {}
        window_size = 7  # Should match the timesteps used in preprocessing

        for seq_idx in range(n_sequences):
            start_row = seq_idx
            end_row = min(seq_idx + window_size, n_rows)
            mapping[seq_idx] = list(range(start_row, end_row))

        return mapping

if __name__ == "__main__":
    # Example configuration
    config = {
        'db_path': 'climate.db',
        'model_path': 'src/model/anomaly_model.h5',
        'config_path': 'src/configs/detector_config.json',
        'csv_path': 'demo/sample_demo.csv',
        'start_date': '2022-01-01',
        'end_date': '2025-12-31',
        'epochs': 25,
        'batch_size': 64,
        'patience': 7
    }

    # Run different types of validation
    pipeline = AnomalyDetectionPipeline(config)

    print("\nRunning time-series cross-validation...")
    cv_results = pipeline.run_with_cross_validation(data_source='csv', n_folds=5)
    print(f"Threshold stability: {cv_results['analysis']['threshold_stability']}")

    # Print error distribution metrics
    error_dist = cv_results['analysis']['error_distribution']
    print("\nError Distribution Metrics:")
    print(f"  Mean reconstruction error: {error_dist['mean']:.4f}")
    print(f"  Standard deviation: {error_dist['std']:.4f}")
    print(f"  Minimum error: {error_dist['min']:.4f}")
    print(f"  Maximum error: {error_dist['max']:.4f}")

    # Print recommendations if any
    if cv_results['analysis']['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(cv_results['analysis']['recommendations'], 1):
            print(f"{i}. {rec}")
    else:
        print("\nNo specific recommendations - model appears stable.")

    # print("\nRunning seasonal cross-validation...")
    # seasonal_results = pipeline.run_seasonal_cv(data_source='csv')
    # print("Seasonal analysis complete")

    # Traditional run (for actual anomaly detection)
    print("\nRunning traditional pipeline for anomaly detection...")
    processed_data, anomalies = pipeline.run(data_source='csv')
    print(f"Found {len(anomalies)} anomalies in the dataset")

    # Evaluate model
    if len(anomalies) > 0:
        # Step 1: Evaluate model performance on test data
        print("\nEvaluating model performance...")
        evaluation_results = pipeline.evaluate()
        print("Evaluation Metrics:")
        for metric, value in evaluation_results.items():
            print(f"  {metric}: {value:.4f}")

        # Step 2: Generate detailed anomaly report
        print("\nGenerating anomaly report...")
        anomaly_report = pipeline.get_anomaly_report(anomalies)
        print(f"\nAnomaly Report Summary:")
        print(f"Total anomalies detected: {len(anomaly_report)}")
        if len(anomaly_report) > 0:
            print(f"Date range of anomalies: {anomaly_report.index[0]} to {anomaly_report.index[-1]}")
            print(f"Average reconstruction error: {anomaly_report['reconstruction_error'].mean():.4f}")
            print(f"Reconstruction errors: {anomaly_report['reconstruction_error'].values}")
            print(f"Error percentiles: {anomaly_report['error_percentile'].values}")
            print(f"Distances to threshold: {anomaly_report['distance_to_threshold'].values}")
    else:
        print("\nNo anomalies detected - no evaluation needed.")