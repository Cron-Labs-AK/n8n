"""
MCP Server for Anomaly Detection (PostgreSQL Only Version)
"""

from fastmcp import FastMCP
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
import json
import psycopg2
import psycopg2.extras
import traceback

mcp = FastMCP("Anomaly Detection Server")

# -----------------------------------------------------
# PostgreSQL Fetch Helper (ALWAYS used, data param removed)
# -----------------------------------------------------
conn = psycopg2.connect(
    host="aws-1-ap-southeast-2.pooler.supabase.com",
    user="postgres.vwffpsdqynpogykytlvg",
    password="1Aashutosh$cronlabs",
    database="postgres",
    port="5432"  # default PostgreSQL port
)

#Fetch table names
df_tables = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';", conn)

#Example: specify the tables you want
tables = ['Tempt']

for table in tables:
    df = pd.read_sql(f'SELECT * FROM "{table}";', conn)



# -----------------------------------------------------
# Anomaly detection methods (unchanged logic)
# -----------------------------------------------------
def detect_anomalies_moving_average(df, value_column, time_column, window=7, threshold=2.0):
    df = df.copy()
    df["ma"] = df[value_column].rolling(window=window, center=True).mean()
    df["ma_std"] = df[value_column].rolling(window=window, center=True).std()
    df["ma_std"] = df["ma_std"].fillna(df[value_column].std())
    df["z_ma"] = (df[value_column] - df["ma"]) / (df["ma_std"] + 1e-8)
    df["is_anomaly_ma"] = abs(df["z_ma"]) > threshold
    return df


def detect_anomalies_standard_deviation(df, value_column, time_column, threshold=3.0):
    df = df.copy()
    mean_val = df[value_column].mean()
    std_val = df[value_column].std()
    df["z_std"] = (df[value_column] - mean_val) / (std_val + 1e-8)
    df["is_anomaly_std"] = abs(df["z_std"]) > threshold
    return df


def detect_anomalies_iqr(df, value_column, time_column, multiplier=1.5):
    df = df.copy()
    Q1 = df[value_column].quantile(0.25)
    Q3 = df[value_column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    df["is_anomaly_iqr"] = (df[value_column] < lower) | (df[value_column] > upper)
    return df


# -----------------------------------------------------
# FINAL CORE: Always reads from table (no `data`)
# -----------------------------------------------------
def detect_anomalies_core(
    db_connection: str,
    table: str,
    time_column: str,
    value_column: str,
    methods: List[str]
) -> Dict[str, Any]:

    df = fetch_table_from_postgres(db_connection, table, time_column, value_column)

    results = {}
    combined_flag = pd.Series(False, index=df.index)

    if "moving_average" in methods:
        df = detect_anomalies_moving_average(df, value_column, time_column)
        combined_flag |= df["is_anomaly_ma"]
        results["moving_average"] = df[df["is_anomaly_ma"]].to_dict("records")

    if "standard_deviation" in methods:
        df = detect_anomalies_standard_deviation(df, value_column, time_column)
        combined_flag |= df["is_anomaly_std"]
        results["standard_deviation"] = df[df["is_anomaly_std"]].to_dict("records")

    if "iqr" in methods:
        df = detect_anomalies_iqr(df, value_column, time_column)
        combined_flag |= df["is_anomaly_iqr"]
        results["iqr"] = df[df["is_anomaly_iqr"]].to_dict("records")

    return {
        "table": table,
        "time_column": time_column,
        "value_column": value_column,
        "methods_used": methods,
        "total_records": len(df),
        "total_anomalies": int(combined_flag.sum()),
        "anomalies": df[combined_flag].to_dict("records")
    }


# -----------------------------------------------------
# MCP Tool (no `data`, table-only input)
# -----------------------------------------------------
SUPABASE_DB_CONNECTION = "postgresql://postgres.vwffpsdqynpogykytlvg:[YOUR-PASSWORD]@aws-1-ap-southeast-2.pooler.supabase.com:6543/postgres"

@mcp.tool()
def detect_anomalies(
    time_column: str,
    value_column: str,
    methods: str = '["moving_average","standard_deviation","iqr"]'
):
    try:
        methods_list = json.loads(methods)
        return detect_anomalies_core(
            db_connection=SUPABASE_DB_CONNECTION,
            time_column=time_column,
            value_column=value_column,
            methods=methods_list
        )
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
