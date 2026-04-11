"""
MCP Stats Server — a lightweight, standalone Model Context Protocol server
that exposes a `profile_data_statistics` tool for computing descriptive
statistics on a JSON-serialized pandas DataFrame.

Run standalone:   python mcp_stats_server.py
The server uses stdio transport so it can be launched as a subprocess
by any MCP client (e.g. the BusinessInsightsAgent).
"""

import json
import logging

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, format="%(asctime)s | MCP-Stats | %(message)s")
logger = logging.getLogger("mcp_stats_server")

# ── Create the MCP server ────────────────────────────────────
mcp = FastMCP("AgenticBI-Stats")


@mcp.tool()
def profile_data_statistics(data_json: str) -> str:
    """
    Accepts a JSON string (records-orient) representing a DataFrame and
    returns a JSON object with descriptive statistics for every column.

    Numeric columns: count, mean, median, std, min, max, q25, q75, outlier_count
    Categorical columns: count, unique, top_values (top 5 by frequency)
    """
    try:
        records = json.loads(data_json)
        df = pd.DataFrame(records)
    except Exception as e:
        return json.dumps({"error": f"Failed to parse data: {e}"})

    profile: dict = {"row_count": len(df), "col_count": len(df.columns), "columns": {}}

    for col in df.columns:
        series = df[col]
        col_info: dict = {"dtype": str(series.dtype), "null_count": int(series.isna().sum())}

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) == 0:
                col_info.update({"count": 0, "mean": None, "median": None})
            else:
                q25 = float(clean.quantile(0.25))
                q75 = float(clean.quantile(0.75))
                iqr = q75 - q25
                outlier_mask = (clean < (q25 - 1.5 * iqr)) | (clean > (q75 + 1.5 * iqr))
                col_info.update({
                    "count": int(len(clean)),
                    "mean": round(float(clean.mean()), 4),
                    "median": round(float(clean.median()), 4),
                    "std": round(float(clean.std()), 4) if len(clean) > 1 else 0.0,
                    "min": round(float(clean.min()), 4),
                    "max": round(float(clean.max()), 4),
                    "q25": round(q25, 4),
                    "q75": round(q75, 4),
                    "outlier_count": int(outlier_mask.sum()),
                })
        else:
            clean = series.dropna().astype(str)
            top_vals = clean.value_counts().head(5)
            col_info.update({
                "count": int(len(clean)),
                "unique": int(clean.nunique()),
                "top_values": {str(k): int(v) for k, v in top_vals.items()},
            })

        profile["columns"][col] = col_info

    return json.dumps(profile)


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting AgenticBI MCP Stats Server (stdio transport)…")
    mcp.run(transport="stdio")
