"""
Centralized configuration for all Agentic BI agents.
Database credentials, LLM model, paths, and logging setup.
"""

import os
import logging

# ---------------------------------------------------------
# DATABASE CONFIGURATION
# ---------------------------------------------------------
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": "5432",
}

# ---------------------------------------------------------
# LLM CONFIGURATION
# ---------------------------------------------------------
LLM_MODEL = "llama3"

# ---------------------------------------------------------
# VECTOR DB CONFIGURATION
# ---------------------------------------------------------
# Resolve relative to THIS file's directory so it works regardless of CWD
_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(_AGENTS_DIR, "chroma_db_data")

# ---------------------------------------------------------
# EMBEDDING MODEL
# ---------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a named logger with consistent formatting.
    Usage:  logger = get_logger("agent.nl2sql")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
