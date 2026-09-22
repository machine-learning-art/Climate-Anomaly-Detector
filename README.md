# Climate Anomaly Detector

A machine learning-based system for detecting anomalies in climate temperature data using LSTM autoencoders.

## Overview

Climate Anomaly Detector is an end-to-end pipeline that:
- Fetches climate data from NOAA API *(under construction)* or CSV files
- Preprocesses and normalizes the data
- Trains LSTM autoencoder models to learn normal temperature patterns
- Detects anomalies based on reconstruction errors
- Visualizes results with comprehensive plots

## Features

### Core Capabilities
- **Data Fetching**: Retrieve climate data from NOAA API *(under construction)* or local CSV files
- **Advanced Preprocessing**: Clean, normalize, and feature-engineer temperature data
- **Anomaly Detection**: LSTM autoencoder-based detection using reconstruction errors
- **Cross-Validation**: Time-series and seasonal *(under construction)* cross-validation for robust evaluation
- **Visualization**: Interactive plots showing temperature trends and anomalies

### Technical Highlights
- LSTM Autoencoder architecture for anomaly detection
- Comprehensive data preprocessing pipeline
- Time-series cross-validation support
- Seasonal pattern analysis *(under construction)*
- SQLite database integration
- Keras/PyTorch backend with modern .keras format support

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install numpy pandas requests matplotlib seaborn scipy scikit-learn keras torch
```

## Project Structure

```
climate_anomaly_detector/
├── src/                      # Main source code
│   ├── fetch.py              # Data fetching from NOAA API and CSV
│   ├── preprocess.py         # Data cleaning and feature engineering
│   ├── detect.py             # LSTM autoencoder anomaly detection
│   ├── pipeline_train.py     # Training pipeline with cross-validation
│   ├── pipeline_predict.py   # Prediction pipeline for full datasets
│   └── plot_trends.py        # Visualization of results
├── model/                    # Trained models and model artifacts
│   └── anomaly_model.keras   # LSTM autoencoder model weights
├── demo/                     # Demo data and examples
├── configs/                  # Configuration files
└── climate.db                # SQLite database for storing data
```

## API Configuration Setup

### Creating api_config.yaml File

To utilize the NOAA API data fetching functionality, create an `api_config.yaml` configuration file with API credentials.

#### 1. Create the Configuration File

Create a new file named `configs/api_config.yaml` in the project directory:

```yaml
# configs/api_config.yaml
noaa:
  api_token: "YOUR_NOAA_API_TOKEN_HERE"
```

Replace `"YOUR_NOAA_API_TOKEN_HERE"` with the actual NOAA API token.

#### 2. Obtain a NOAA API Token

1. Visit the [NOAA Climate Data Online (CDO) website](https://www.ncdc.noaa.gov/cdo-web/token)
2. Follow the directions to generate a token

#### 3. File Location

The system looks for the configuration file at:
- `./configs/api_config.yaml` (relative to project root)
- Optionally specify a custom path when creating the ClimateDataFetcher instance

```python
fetcher = ClimateDataFetcher(config_path="/custom/path/to/api_config.yaml")
```

#### 4. Configuration Format Details

The YAML file should follow this structure:

```yaml
noaa:
  api_token: "your_api_token_string"
```

- `api_token`: Your NOAA CDO API token (required)
- The token should be a string containing alphanumeric characters and possibly special characters

#### 5. Security Best Practices

1. **Never commit API token** to version control
2. Add `configs/api_config.yaml` to your `.gitignore` file:
   ```
   configs/api_config.yaml
   ```

3. Use environment variables for production deployments instead of hardcoding tokens

#### 6. Verifying Your Configuration

After creating the file, verify it is working by running:

```python
from fetch import ClimateDataFetcher

# Test loading the configuration
fetcher = ClimateDataFetcher()
print(f"API token loaded: {fetcher.api_token is not None}")
```

The output should show `True` if the configuration file is properly set up.

## Usage

### 1. Data Fetching

```python
from fetch import ClimateDataFetcher

# Option 1: Load from CSV
fetcher = ClimateDataFetcher()
fetcher.fetch_and_save(csv_path="demo/sample_demo.csv")

# Option 2: Fetch from NOAA API (requires API token)
fetcher.fetch_and_save(station_id="GHCND:USW00023183",
                        start_date='2026-09-01',
                        end_date='2026-10-01')
```

### 2. Training Pipeline

```python
from pipeline_train import AnomalyDetectionPipeline

config = {
    'db_path': 'climate.db',
    'model_path': 'src/model/anomaly_model.h5',
    'csv_path': 'demo/sample_demo.csv',
    'start_date': '2022-01-01',
    'end_date': '2025-12-31',
    'epochs': 25,
    'batch_size': 64
}

pipeline = AnomalyDetectionPipeline(config)

# Run with cross-validation
cv_results = pipeline.run_with_cross_validation(data_source='csv', n_folds=5)

# Traditional training and detection
processed_data, anomalies = pipeline.run(data_source='csv')
```

### 3. Prediction Pipeline

```python
from pipeline_predict import ModelPredictor

config = {
    'db_path': 'climate.db',
    'model_path': 'src/model/anomaly_model.h5',
    'config_path': 'configs/detector_config.json',
    'csv_path': 'demo/sample_demo.csv'
}

predictor = ModelPredictor(config)
results_summary = predictor.run_full_application()
```

### 4. Visualization

```python
from plot_trends import ClimateTrendPlotter

plotter = ClimateTrendPlotter({'db_path': 'climate.db'})

# Create comprehensive report with all plots
report_results = plotter.create_comprehensive_report('demo')
```

## Data Flow

```
Data Source → Fetch → Preprocess → Train Model → Detect Anomalies → Visualize Results
```

1. **Fetch**: Load data from NOAA API *(under construction)* or CSV files
2. **Preprocess**: Clean, normalize, and feature-engineer the data
3. **Train**: Build LSTM autoencoder on normal (non-anomalous) data
4. **Detect**: Identify anomalies based on reconstruction errors
5. **Visualize**: Create plots showing temperature trends and detected anomalies

## Key Components

### 1. Data Preprocessing (`preprocess.py`)
- Handles missing values with forward/backward filling
- Calculates features: average temperature, temperature range, moving averages
- Applies detrending to remove long-term trends
- Normalizes data using StandardScaler or MinMaxScaler
- Creates time-series sequences for LSTM input

### 2. Anomaly Detection (`detect.py`)
- **LSTM Autoencoder**: Learns normal patterns in temperature data
- **Reconstruction Error**: Identifies anomalies based on prediction errors
- **Thresholding**: Automatically sets threshold using percentile method
- **Cross-Validation**: Time-series and seasonal *(under construction)* cross-validation support

### 3. Training Pipeline (`pipeline_train.py`)
- End-to-end training workflow
- Cross-validation for robust model evaluation
- Seasonal pattern analysis *(under construction)*
- Comprehensive anomaly reporting

### 4. Prediction Pipeline (`pipeline_predict.py`)
- Apply trained models to full datasets
- Store results in structured database table
- Temporal analysis of anomalies
- Summary statistics and reports

## Visualization Examples

The system generates several types of plots:

1. **Temperature Trends**: Shows actual temperatures, model predictions, and anomalies
2. **Reconstruction Errors**: Displays error distribution with anomaly markers
3. **Error Distribution**: Histograms showing normal vs anomalous errors

## Configuration

Example configuration for the training pipeline:

```python
config = {
    'db_path': 'climate.db',
    'model_path': 'src/model/anomaly_model.h5',
    'config_path': 'configs/detector_config.json',
    'csv_path': 'demo/sample_demo.csv',
    'start_date': '2022-01-01',
    'end_date': '2025-12-31',
    'epochs': 25,
    'batch_size': 64,
    'patience': 7,  # Early stopping patience
    'model_version': '1.0'
}
```

## Documentation

### Data Format

The system expects climate data in the following format:

| Column   | Type     | Description                     |
|----------|----------|---------------------------------|
| date     | datetime | Date of observation              |
| temp_max | float    | Maximum temperature (°C × 10)   |
| temp_min | float    | Minimum temperature (°C × 10)   |

Note: Temperatures are stored as tenths of degrees Celsius (e.g., 25.5°C = 255)

### Database Schema

**daily_temps table:**
- `date` TEXT PRIMARY KEY
- `temp_max` REAL
- `temp_min` REAL

**processed_data table (optional):**
- `date` TEXT PRIMARY KEY
- `temp_max` REAL
- `temp_min` REAL
- `avg_temp` REAL
- `temp_range` REAL
- `ma_7day` REAL
- `ma_30day` REAL
- `temp_anomaly` REAL
- `detrended_temp_max` REAL
- `detrended_temp_min` REAL
- `detrended_avg_temp` REAL
- `scaled_avg_temp` REAL

**model_application_results table:**
- `application_id` INTEGER PRIMARY KEY AUTOINCREMENT
- `run_timestamp` TEXT NOT NULL
- `date` DATE NOT NULL
- `temp_max` REAL
- `temp_min` REAL
- `avg_temp` REAL
- `reconstruction_error` REAL
- `avg_reconstruction` REAL
- `is_anomaly` INTEGER NOT NULL

## Troubleshooting

### Common Issues and Solutions

1. **AUTOINCREMENT Syntax Error**
   - Fix: Remove AUTOINCREMENT from the table definition or ensure it's used with INTEGER PRIMARY KEY only

2. **Model Loading Errors**
   - Ensure model files exist at specified paths
   - Try both .h5 and .keras formats for compatibility

3. **Data Validation Failures**
   - Check that CSV files contain required columns: date, temp_max, temp_min
   - Handle missing values appropriately in the data

4. **Cross-Validation Warnings**
   - Ensure sufficient data (at least 2 years recommended)
   - Adjust fold sizes if getting empty splits
