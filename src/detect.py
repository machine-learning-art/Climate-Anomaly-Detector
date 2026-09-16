# climate_anomaly_detector/src/detect.py

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import logging
import os
import json
from datetime import datetime

os.environ["KERAS_BACKEND"] = "torch"

# Keras/TensorFlow imports
import keras
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import plot_model
#import tensorflow as tf

# set random seed 
keras.utils.set_random_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    LSTM Autoencoder-based anomaly detector for climate data.
    Detects anomalies in time series temperature data using reconstruction error.
    """

    def __init__(self, model_path: str = "anomaly_model.h5", config_path: str = "detector_config.json"):
        """
        Initialize the anomaly detector.

        Args:
            model_path: Path to save/load trained model
            config_path: Path to save/load detector configuration
        """
        self.model = None
        self.model_path = model_path
        self.config_path = config_path
        self.threshold = None
        self.scaler_params = None
        self.input_shape = None

    def build_model(self, input_shape: Tuple[int, int], encoding_dim: int = 32) -> Model:
        """
        Build LSTM autoencoder model for anomaly detection.

        Args:
            input_shape: Shape of input data (timesteps, features)
            encoding_dim: Size of the encoded representation

        Returns:
            Compiled Keras model
        """
        self.input_shape = input_shape

        # Input layer
        inputs = Input(shape=input_shape)

        # Encoder
        encoder = LSTM(encoding_dim, activation='relu')(inputs)
        encoder = Dense(encoding_dim)(encoder)

        # Decoder
        decoder = RepeatVector(input_shape[0])(encoder)
        decoder = LSTM(input_shape[1], return_sequences=True)(decoder)

        # Output layer
        outputs = TimeDistributed(Dense(input_shape[1]))(decoder)

        # Create and compile model
        self.model = Model(inputs=inputs, outputs=outputs)

        optimizer = Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        logger.info(f"Built LSTM autoencoder with input shape {input_shape}")
        return self.model

    def train(
        self,
        X_train: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        validation_split: float = 0.1,
        early_stopping_patience: int = 5
    ) -> Dict[str, Any]:
        """
        Train the autoencoder on normal (non-anomalous) data.

        Args:
            X_train: Training data (sequences of climate features)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            early_stopping_patience: Patience for early stopping

        Returns:
            Dictionary containing training history and metrics
        """
        if self.model is None:
            self.build_model(X_train.shape[1:])

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=self.model_path.replace('.h5', '.keras'),
                save_best_only=True,
                monitor='val_loss'
            )
        ]

        # Train model
        history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        # Save training configuration
        self._save_config({
            'input_shape': list(self.input_shape),
            'encoding_dim': self.model.layers[2].units,  # Encoder LSTM units
            'threshold': self.threshold if self.threshold is not None else None,
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'early_stopping_patience': early_stopping_patience,
            'training_date': datetime.now().isoformat()
        })

        logger.info("Model training completed")
        return history.history

    def time_series_cross_validation(
        self,
        X: np.ndarray,
        n_splits: int = 5,
        epochs: int = 20,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Perform time-series cross-validation for robust model evaluation.

        Args:
            X: Complete dataset (sequences of climate features)
            n_splits: Number of folds for cross-validation
            epochs: Training epochs per fold
            batch_size: Batch size for training

        Returns:
            Dictionary containing cross-validation results and metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        cv_results = {
            'fold_errors': [],
            'fold_thresholds': [],
            'fold_metrics': []
        }

        for i in range(n_splits):
            # Calculate fold boundaries
            val_start = int(i * len(X) / n_splits)
            val_end = int((i + 1) * len(X) / n_splits)

            # Create train/validation splits (preserving temporal order)
            train_data = np.concatenate([X[:val_start], X[val_end:]])
            val_data = X[val_start:val_end]

            if len(train_data) == 0 or len(val_data) == 0:
                logger.warning(f"Fold {i+1}: Empty split, skipping")
                continue

            # Train on this fold's training data
            history = self.model.fit(
                train_data, train_data,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=0  # Reduce output for CV
            )

            # Calculate reconstruction errors on validation set
            reconstructions = self.model.predict(val_data)
            errors = np.mean(np.square(val_data - reconstructions), axis=(1, 2))

            # Store results for this fold
            cv_results['fold_errors'].append(errors)

            # Calculate threshold and metrics for this fold
            threshold = np.percentile(errors, 95.0)
            cv_results['fold_thresholds'].append(threshold)

            # Basic metrics
            metrics = {
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'min_error': np.min(errors),
                'max_error': np.max(errors),
                'threshold': threshold,
                'final_loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1]
            }
            cv_results['fold_metrics'].append(metrics)

            logger.info(f"Fold {i+1}: Mean error={metrics['mean_error']:.4f}, Threshold={threshold:.4f}")

        # Calculate overall statistics
        all_errors = np.concatenate(cv_results['fold_errors'])
        cv_results.update({
            'overall_mean_error': np.mean(all_errors),
            'overall_std_error': np.std(all_errors),
            'overall_threshold_range': (
                min(cv_results['fold_thresholds']),
                max(cv_results['fold_thresholds'])
            ),
            'threshold_stability': np.std(cv_results['fold_thresholds']) < 0.1
        })

        logger.info(f"Cross-validation completed: {n_splits} folds")
        return cv_results

    def seasonal_split_cross_validation(
        self,
        X: np.ndarray,
        dates: pd.DatetimeIndex,
        n_years: int = 3
    ) -> Dict[str, Any]:
        """
        Perform cross-validation using seasonal splits to ensure model works across seasons.

        Args:
            X: Complete dataset (sequences of climate features)
            dates: Datetime index corresponding to the data sequences
            n_years: Number of years for training (validation uses remaining years)

        Returns:
            Dictionary containing seasonal CV results
        """
        # Group by year and create seasonal splits
        unique_years = dates.year.unique()
        if len(unique_years) < 2:
            raise ValueError("Need at least 2 years of data for seasonal cross-validation")

        cv_results = {
            'yearly_errors': {},
            'yearly_thresholds': {}
        }

        for year in unique_years:
            # Training: all years except current
            train_mask = dates.year != year
            val_mask = dates.year == year

            train_data = X[train_mask]
            val_data = X[val_mask]

            if len(train_data) == 0 or len(val_data) == 0:
                logger.warning(f"Year {year}: Empty split, skipping")
                continue

            # Train model
            self.model.fit(
                train_data, train_data,
                epochs=20,
                batch_size=32,
                validation_split=0.1,
                verbose=0
            )

            # Evaluate on this year's data
            reconstructions = self.model.predict(val_data)
            errors = np.mean(np.square(val_data - reconstructions), axis=(1, 2))

            threshold = np.percentile(errors, 95.0)

            cv_results['yearly_errors'][year] = errors
            cv_results['yearly_thresholds'][year] = threshold

            logger.info(f"Year {year}: Mean error={np.mean(errors):.4f}, Threshold={threshold:.4f}")

        return cv_results

    def detect_anomalies(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in test data using reconstruction error.

        Args:
            X_test: Test data (sequences of climate features)

        Returns:
            Tuple of (reconstruction_errors, is_anomaly)
            - reconstruction_errors: MSE for each sequence
            - is_anomaly: Binary array indicating anomalies (1=anomaly, 0=normal)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Predict reconstructions
        reconstructions = self.model.predict(X_test)

        # Calculate reconstruction error (MSE)
        errors = np.mean(np.square(X_test - reconstructions), axis=(1, 2))

        # Determine anomalies based on threshold
        is_anomaly = np.where(errors > self.threshold, 1, 0) if self.threshold else np.zeros_like(errors)

        logger.info(f"Detected {np.sum(is_anomaly)} anomalies in {len(X_test)} samples")
        return errors, is_anomaly

    def set_threshold(
        self,
        X_normal: np.ndarray,
        threshold_percentile: float = 95.0
    ) -> float:
        """
        Set anomaly threshold based on reconstruction error distribution.
        Automatically saves the configuration with the new threshold.

        Args:
            X_normal: Data known to be normal (non-anomalous)
            threshold_percentile: Percentile for threshold calculation

        Returns:
            The calculated threshold value
        """
        errors = self.detect_anomalies(X_normal)[0]
        self.threshold = np.percentile(errors, threshold_percentile)

        logger.info(f"Set anomaly threshold at {self.threshold:.4f} ({threshold_percentile}th percentile")

        # Save configuration with the new threshold
        config_to_save = {
            'input_shape': list(self.input_shape) if self.input_shape else None,
            'encoding_dim': self.model.layers[2].units if self.model and len(self.model.layers) > 2 else None,
            'threshold': self.threshold,
            'training_date': datetime.now().isoformat()
        }

        with open(self.config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        logger.info(f"Configuration saved to {self.config_path}")

        return self.threshold

    def evaluate(
        self,
        X_test: np.ndarray,
        y_true: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate detector performance.

        Args:
            X_test: Test data (sequences of climate features)
            y_true: Ground truth labels (optional)

        Returns:
            Dictionary containing evaluation metrics
        """
        errors, predictions = self.detect_anomalies(X_test)

        metrics = {
            'mean_reconstruction_error': np.mean(errors),
            'std_reconstruction_error': np.std(errors),
            'max_reconstruction_error': np.max(errors),
            'min_reconstruction_error': np.min(errors)
        }

        if y_true is not None:
            from sklearn.metrics import (
                precision_score,
                recall_score,
                f1_score,
                roc_auc_score,
                confusion_matrix
            )

            # Calculate metrics
            metrics.update({
                'precision': precision_score(y_true, predictions),
                'recall': recall_score(y_true, predictions),
                'f1_score': f1_score(y_true, predictions),
                'roc_auc': roc_auc_score(y_true, errors) if len(np.unique(y_true)) > 1 else 0,
                'confusion_matrix': confusion_matrix(y_true, predictions).tolist()
            })

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def save_model(self, path: Optional[str] = None) -> None:
        """
        Save the trained model to file using modern Keras format.
        """
        save_path = path or self.model_path

        # Use new Keras format (.keras extension)
        if not save_path.endswith('.keras'):
            save_path = save_path.replace('.h5', '.keras')

        keras.saving.save_model(self.model, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, path: Optional[str] = None) -> None:
        """
        Load a trained model from file with custom object handling for compatibility.
        """
        load_path = path or self.model_path

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found at {load_path}")

        try:
            # Handle both old .h5 and new .keras formats
            if load_path.endswith('.h5'):
                custom_objects = {
                    'mse': keras.losses.MeanSquaredError(),
                    'mean_squared_error': keras.losses.MeanSquaredError()
                }
                self.model = keras.models.load_model(
                    load_path,
                    custom_objects=custom_objects
                )
            else:
                # New format doesn't need custom objects
                self.model = keras.saving.load_model(load_path)

            logger.info(f"Model loaded from {load_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ValueError(
                f"Could not load model from {load_path}. Error: {str(e)}"
            ) from e

    # def _save_config(self, config: Dict[str, Any]) -> None:
    #     """
    #     Save detector configuration to JSON file.

    #     Args:
    #         config: Configuration dictionary
    #     """
    #     with open(self.config_path, 'w') as f:
    #         json.dump(config, f, indent=2)
    #     logger.info(f"Configuration saved to {self.config_path}")

    def _save_config(self, config: Dict[str, Any]) -> None:
        """
        Save detector configuration to JSON file.
        Includes the current threshold value if set.
        """
        # Create base config with all parameters
        config_to_save = {
            'input_shape': list(self.input_shape) if self.input_shape else None,
            'encoding_dim': self.model.layers[2].units if self.model and len(self.model.layers) > 2 else None,
            'threshold': self.threshold,  # Save current threshold value
            'training_date': datetime.now().isoformat()
        }

        with open(self.config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        logger.info(f"Configuration saved to {self.config_path}")

    def load_config(self) -> Dict[str, Any]:
        """
        Load detector configuration from JSON file.

        Returns:
            Configuration dictionary
        """
        if not os.path.exists(self.config_path):
            return {}

        with open(self.config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {self.config_path}")
        return config

    def plot_model_architecture(self, file_path: str = "model_architecture.png") -> None:
        """
        Generate and save a visualization of the model architecture.

        Args:
            file_path: Path to save the plot image
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        plot_model(self.model, to_file=file_path, show_shapes=True)
        logger.info(f"Model architecture saved to {file_path}")

    def get_anomaly_scores(
        self,
        X_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Get anomaly scores with detailed information for each sample.

        Args:
            X_test: Test data (sequences of climate features)

        Returns:
            DataFrame containing anomaly scores and metadata
        """
        errors, is_anomaly = self.detect_anomalies(X_test)

        # Create results DataFrame
        results = pd.DataFrame({
            'reconstruction_error': errors,
            'is_anomaly': is_anomaly,
            'error_percentile': np.percentile(errors, np.linspace(0, 100, len(errors)))
        })

        if self.threshold:
            results['distance_to_threshold'] = results['reconstruction_error'] - self.threshold

        logger.info(f"Generated anomaly scores for {len(results)} samples")
        return results

    def inverse_transform_reconstruction(self, reconstructions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply inverse transformation to model reconstructions using stored scaler parameters.

        Args:
            reconstructions: Model output (reconstructed data in scaled space)
                        Shape: (n_sequences, sequence_length, n_features)

        Returns:
            Dictionary containing inverse-transformed values for each feature
            Keys are feature names, values are numpy arrays of original scale values
        """
        if self.scaler_params is None:
            raise ValueError("No scaler parameters available. Cannot perform inverse transform.")

        # Initialize result dictionary to store inverse-transformed values per feature
        inverse_transformed = {}

        # Get the feature columns from scaler params (they should match the model's input features)
        for i, (feature_name, params) in enumerate(self.scaler_params.items()):
            # Extract the feature's reconstructions: shape (n_sequences, sequence_length)
            feature_reconstructions = reconstructions[..., i]

            if 'mean' in params and 'scale' in params:
                # StandardScaler inverse transform
                mean_val = float(params['mean'])
                scale_val = float(params['scale'])

                # Inverse transform: (x * scale) + mean
                original_values = feature_reconstructions * scale_val + mean_val

            elif 'min' in params and 'scale' in params:
                # MinMaxScaler inverse transform
                min_val = float(params['min'])
                scale_val = float(params['scale'])

                # Inverse transform: (x * scale) + min
                original_values = feature_reconstructions * scale_val + min_val

            else:
                raise ValueError(f"Unknown scaler parameters format for {feature_name}")

            inverse_transformed[feature_name] = original_values

        return inverse_transformed