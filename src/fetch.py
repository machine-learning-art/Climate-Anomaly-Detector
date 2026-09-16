# climate_anomaly_detector/src/fetch.py

import os
import requests
import pandas as pd
import sqlite3
from datetime import datetime
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClimateDataFetcher:
    def __init__(self, db_path: str = "climate.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize the SQLite database and create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_temps (
                    date TEXT PRIMARY KEY,
                    temp_max REAL,
                    temp_min REAL
                )
            """)
            logger.info("Database initialized.")

    def fetch_from_noaa(
        self,
        api_token: str,
        dataset_id: str = "GHCN_DAILY",
        station_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch climate data from the NOAA API.

        Args:
            api_token: NOAA API token.
            dataset_id: Dataset identifier (e.g., "GHCN_DAILY").
            station_id: Station ID for filtering (optional).
            start_date: Start date in YYYY-MM-DD format (optional).
            end_date: End date in YYYY-MM-DD format (optional).
            limit: Maximum number of records to fetch.

        Returns:
            DataFrame containing the fetched data.
        """
        base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
        params = {
            "datasetid": dataset_id,
            "limit": limit,
            "token": api_token
        }

        if station_id:
            params["stationid"] = station_id
        if start_date:
            params["startdate"] = start_date
        if end_date:
            params["enddate"] = end_date

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Convert JSON to DataFrame (adjust based on actual API response structure)
            df = pd.DataFrame(data["results"])
            logger.info(f"Fetched {len(df)} records from NOAA.")
            return df

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
                "TMAX": "temp_max",
                "TMIN": "temp_min"
            }
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            df = df[["date", "temp_max", "temp_min"]]  # Keep only these columns
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
            df.to_sql("daily_temps", conn, if_exists="replace", index=False)
            logger.info(f"Saved {len(df)} records to database.")

    def fetch_and_save(
        self,
        api_token: Optional[str] = None,
        csv_path: Optional[str] = None,
        **noaa_kwargs
    ) -> None:
        """
        Fetch data from NOAA or CSV and save to database.

        Args:
            api_token: NOAA API token (if fetching from NOAA).
            csv_path: Path to CSV file (if loading from local file).
            noaa_kwargs: Additional arguments for fetch_from_noaa.
        """
        if api_token:
            df = self.fetch_from_noaa(api_token, **noaa_kwargs)
        elif csv_path:
            df = self.load_from_csv(csv_path)
        else:
            raise ValueError("Either api_token or csv_path must be provided.")

        # Validate data before saving
        if not self._validate_data(df):
            logger.error("Data validation failed.")
            return

        self.save_to_db(df)

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the structure and content of the DataFrame."""
        required_columns = {"date", "temp_max", "temp_min"}
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

    # Option 1: Fetch from NOAA API (requires API token)
    # fetcher.fetch_and_save(api_token="YOUR_NOAA_API_TOKEN", station_id="USW00014735")

    # Option 2: Load from CSV
    fetcher.fetch_and_save(csv_path="climate_anomaly_detector/demo/sample_demo.csv")