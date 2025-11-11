"""
MCP Server for Anomaly Detection using Statistical Methods
Uses FastMCP framework to detect anomalies in time series data.
"""

from fastmcp import FastMCP
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
import json
import psycopg2


# Initialize FastMCP server
mcp = FastMCP("Anomaly Detection Server")

conn = psycopg2.connect(
    host="aws-1-ap-southeast-2.pooler.supabase.com",
    user="postgres.vwffpsdqynpogykytlvg",
    password="1Aashutosh$cronlabs",
    database="postgres",
    port="5432"  # default PostgreSQL port
)


def load_tempt():
    return pd.read_sql('SELECT * FROM tempt;', conn)


def detect_anomalies_moving_average(
    df: pd.DataFrame,
    value_column: str,
    time_column: str,
    window: int = 7,
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    Detect anomalies using moving average method.
    
    Args:
        df: DataFrame with time series data
        value_column: Column name containing values to analyze
        time_column: Column name containing time/datetime
        window: Window size for moving average
        threshold: Number of standard deviations from moving average to consider anomaly
    
    Returns:
        DataFrame with anomaly flags and scores
    """
    df = df.copy()
    df = df.sort_values(time_column)
    
    # Calculate moving average and standard deviation
    df['ma'] = df[value_column].rolling(window=window, min_periods=1, center=True).mean()
    df['ma_std'] = df[value_column].rolling(window=window, min_periods=1, center=True).std()
    
    # Handle NaN values in std (when window is smaller than data points)
    df['ma_std'] = df['ma_std'].fillna(df[value_column].std())
    
    # Calculate z-score from moving average
    df['z_score'] = (df[value_column] - df['ma']) / (df['ma_std'] + 1e-8)
    
    # Mark anomalies
    df['is_anomaly_ma'] = abs(df['z_score']) > threshold
    df['anomaly_score_ma'] = abs(df['z_score'])
    
    return df


def detect_anomalies_standard_deviation(
    df: pd.DataFrame,
    value_column: str,
    time_column: str,
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect anomalies using global standard deviation method.
    
    Args:
        df: DataFrame with time series data
        value_column: Column name containing values to analyze
        time_column: Column name containing time/datetime
        threshold: Number of standard deviations from mean to consider anomaly
    
    Returns:
        DataFrame with anomaly flags and scores
    """
    df = df.copy()
    df = df.sort_values(time_column)
    
    # Calculate global statistics
    mean_val = df[value_column].mean()
    std_val = df[value_column].std()
    
    # Calculate z-score
    df['z_score_std'] = (df[value_column] - mean_val) / (std_val + 1e-8)
    
    # Mark anomalies
    df['is_anomaly_std'] = abs(df['z_score_std']) > threshold
    df['anomaly_score_std'] = abs(df['z_score_std'])
    
    return df


def detect_anomalies_iqr(
    df: pd.DataFrame,
    value_column: str,
    time_column: str,
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Detect anomalies using Interquartile Range (IQR) method.
    
    Args:
        df: DataFrame with time series data
        value_column: Column name containing values to analyze
        time_column: Column name containing time/datetime
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with anomaly flags and scores
    """
    df = df.copy()
    df = df.sort_values(time_column)
    
    # Calculate quartiles
    Q1 = df[value_column].quantile(0.25)
    Q3 = df[value_column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Mark anomalies
    df['is_anomaly_iqr'] = (df[value_column] < lower_bound) | (df[value_column] > upper_bound)
    df['anomaly_score_iqr'] = np.where(
        df[value_column] < lower_bound,
        (lower_bound - df[value_column]) / (IQR + 1e-8),
        np.where(
            df[value_column] > upper_bound,
            (df[value_column] - upper_bound) / (IQR + 1e-8),
            0
        )
    )
    
    return df


# ... (keep your other functions: detect_anomalies_moving_average, etc.) ...

def detect_anomalies_core(
    data: str,
    time_column: str,
    value_column: str,
    aggregation_level: Optional[str] = None,
    methods: Optional[List[str]] = None,
    window: int = 7,
    threshold: float = 2.0,
    iqr_multiplier: float = 1.5
) -> Dict[str, Any]:
    """
    Core function to detect anomalies in time series data.
    """
    try:
        if methods is None:
            methods = ["moving_average", "standard_deviation", "iqr"]

        data_list = json.loads(data)
        df = pd.DataFrame(data_list)

        if time_column not in df.columns:
            return {"error": f"Time column '{time_column}' not found in data."}

        if value_column not in df.columns:
            return {"error": f"Value column '{value_column}' not found in data."}

        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        anomaly_df = df.copy()
        results = {}

        if "moving_average" in methods:
            anomaly_df = detect_anomalies_moving_average(anomaly_df, value_column, time_column, window, threshold)
            results["moving_average"] = {
                "anomalies": anomaly_df[anomaly_df['is_anomaly_ma']].to_dict('records')
            }

        if "standard_deviation" in methods:
            anomaly_df = detect_anomalies_standard_deviation(anomaly_df, value_column, time_column, threshold)
            results["standard_deviation"] = {
                "anomalies": anomaly_df[anomaly_df['is_anomaly_std']].to_dict('records')
            }

        if "iqr" in methods:
            anomaly_df = detect_anomalies_iqr(anomaly_df, value_column, time_column, iqr_multiplier)
            results["iqr"] = {
                "anomalies": anomaly_df[anomaly_df['is_anomaly_iqr']].to_dict('records')
            }

        return {
            "total_records": len(anomaly_df),
            "time_column": time_column,
            "value_column": value_column,
            "results": results
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def detect_anomalies(
    time_column: str,
    value_column: str,
    aggregation_level: Optional[str] = None,
    methods: str = '["moving_average","standard_deviation","iqr"]'
):
    """
    Detect anomalies from the 'tempt' table.

    Args:
        time_column: Name of the timestamp/datetime column in tempt.
        value_column: Name of the numeric column to analyze.
        aggregation_level: Optional column to group by before detection.
        methods: JSON list of detection methods to apply.
    """
    try:
        df = load_tempt()

        if df.empty:
            return {"error": "'tempt' table is empty. Refresh it before running anomaly detection."}

        data_string = df.to_json(orient="records")
        methods_list = json.loads(methods)

        return detect_anomalies_core(
            data=data_string,
            time_column=time_column,
            value_column=value_column,
            aggregation_level=aggregation_level,
            methods=methods_list
        )

    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="http", host="0.0.0.0", port=8000)
