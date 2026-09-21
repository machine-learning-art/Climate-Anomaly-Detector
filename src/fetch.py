# climate_anomaly_detector/src/fetch.py

import os
import requests
import pandas as pd
import sqlite3
import yaml
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClimateDataFetcher:
    def __init__(self, db_path: str = "climate.db", config_path: str = "./configs/api_config.yaml"):
        self.db_path = db_path
        self._initialize_db()
        self.config_path = config_path
        self.api_token = self._load_api_token()

    def _initialize_db(self) -> None:
        """Initialize the SQLite database and create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_temps (
                    date TEXT,
                    station_id TEXT,
                    temp_max REAL,
                    temp_min REAL,
                    PRIMARY KEY (date, station_id)
                )
            """)
            logger.info("Database initialized.")

    def _load_api_token(self) -> Optional[str]:
        """Load API token from configuration file."""
        if not Path(self.config_path).exists():
            logger.warning(f"Configuration file {self.config_path} not found")
            return None

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            if config and 'noaa' in config:
                token = config['noaa'].get('api_token')
                if token:
                    logger.info("API token loaded from configuration file")
                    return token
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

        logger.warning("No valid API token found in configuration")
        return None

    def fetch_from_noaa(
        self,
        dataset_id: str = "GHCND",
        station_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch climate data from the NOAA API.

        Args:
            dataset_id: Dataset identifier (e.g., "GHCND").
            station_id: Station ID for filtering (optional).
            start_date: Start date in YYYY-MM-DD format (optional).
            end_date: End date in YYYY-MM-DD format (optional).
            limit: Maximum number of records to fetch.

        Returns:
            DataFrame containing the fetched data.
        """
        if not self.api_token:
            raise ValueError(
                "NOAA API token not configured. "
                "Read the API Configuration Setup notes in the README file to create an api_config.yaml file."
            )
        
        base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
        headers = {
            'token': self.api_token
        }
        params = {
            "datasetid": dataset_id,
            "limit": limit,
        }

        if station_id:
            params["stationid"] = station_id
        if start_date:
            params["startdate"] = start_date
        if end_date:
            params["enddate"] = end_date

        try:
            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            # Convert JSON to DataFrame (adjust based on actual API response structure)
            df = pd.DataFrame(data["results"])
            logger.info(f"Fetched {len(df)} records from NOAA.")

            # Transform the data to expected format
            temp_data = df[df['datatype'].isin(['TMAX', 'TMIN'])]
            if len(temp_data) == 0:
                raise ValueError("No temperature data found in API response")

            # Extract station ID from attributes column or use provided station_id
            if 'station' in temp_data.columns:
                temp_data['station_id'] = temp_data['station'].str.extract(r':([^:]+)$')[0]
            elif station_id:
                temp_data['station_id'] = station_id.split(':')[-1]  # Extract last part after colon

            temp_data['date'] = pd.to_datetime(temp_data['date']).dt.date
            pivoted_df = temp_data.pivot(
                index=['date', 'station_id'],
                columns='datatype',
                values='value'
            ).reset_index()
            pivoted_df.columns.name = None
            pivoted_df = pivoted_df.rename(columns={
                'TMAX': 'temp_max',
                'TMIN': 'temp_min'
            })
            pivoted_df['temp_max'] = pd.to_numeric(pivoted_df['temp_max'])
            pivoted_df['temp_min'] = pd.to_numeric(pivoted_df['temp_min'])

            logger.info(f"Transformed data to {len(pivoted_df)} temperature records.")
            return pivoted_df
    
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data from NOAA: {e}")
            raise

    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load climate data from a local CSV file.

        Args:
            file_path: Path to the CSV file.

        Returns:
            DataFrame containing the loaded data.
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data from {file_path}.")
            # Rename columns to match expected format
            column_mapping = {
                "DATE": "date",
                "STATION": "station_id",
                "TMAX": "temp_max",
                "TMIN": "temp_min"
            }
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            df = df[["date", "station_id", "temp_max", "temp_min"]]  # Keep only these columns
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise

    def save_to_db(self, df: pd.DataFrame) -> None:
        """
        Save DataFrame to SQLite database.

        Args:
            df: DataFrame containing climate data.
        """
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql("daily_temps", conn, if_exists="append", index=False)
            logger.info(f"Saved {len(df)} records to database.")

    def fetch_and_save(
        self,
        csv_path: Optional[str] = None,
        **noaa_kwargs
    ) -> None:
        """
        Fetch data from NOAA or CSV and save to database.

        Args:
            csv_path: Path to CSV file (if loading from local file).
            noaa_kwargs: Additional arguments for fetch_from_noaa.
        """
        if csv_path:
            df = self.load_from_csv(csv_path)
        else:
            df = self.fetch_from_noaa(**noaa_kwargs)

        # Validate data before saving
        if not self._validate_data(df):
            logger.error("Data validation failed.")
            return

        self.save_to_db(df)

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the structure and content of the DataFrame."""
        required_columns = {"date", "station_id", "temp_max", "temp_min"}
        if not required_columns.issubset(df.columns):
            logger.error(f"Missing columns: {required_columns - set(df.columns)}")
            return False

        # Check for missing values
        if df.isnull().any().any():
            logger.warning("Data contains missing values.")

        return True

if __name__ == "__main__":
    # Example usage
    fetcher = ClimateDataFetcher()

    # # Option 1: Fetch from NOAA API (requires API token)
    # fetcher.fetch_and_save(station_id="GHCND:USW00023183",
    #                         start_date='2026-09-01',
    #                         end_date='2026-10-01')

    # # Option 2: Load from CSV
    fetcher.fetch_and_save(csv_path="demo/sample_demo.csv")