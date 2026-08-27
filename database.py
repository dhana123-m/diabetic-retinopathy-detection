"""
Database module for Diabetic Retinopathy Detection.
Handles SQLite database initialization and operations.
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATABASE_PATH, DATABASE_DIR
from ml.utils import logger


def get_connection() -> sqlite3.Connection:
    """Get a database connection."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Initialize the database and create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            predicted_class INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            original_image_path TEXT,
            gradcam_image_path TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")


def insert_prediction(
    filename: str,
    predicted_class: int,
    class_name: str,
    confidence: float,
    risk_level: str,
    original_image_path: str = "",
    gradcam_image_path: str = "",
) -> int:
    """Insert a new prediction record and return the record id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions
        (filename, predicted_class, class_name, confidence, risk_level,
         timestamp, original_image_path, gradcam_image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            predicted_class,
            class_name,
            confidence,
            risk_level,
            datetime.now().isoformat(),
            original_image_path,
            gradcam_image_path,
        ),
    )
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    logger.info(f"Prediction recorded with id={record_id}")
    return record_id


def get_all_predictions(limit: int = 100) -> list[dict]:
    """Retrieve all prediction records, most recent first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_prediction_by_id(record_id: int) -> dict | None:
    """Retrieve a single prediction record by id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_statistics() -> dict:
    """Get summary statistics for the dashboard."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total FROM predictions")
    total = cursor.fetchone()["total"]

    cursor.execute(
        "SELECT class_name, COUNT(*) as count FROM predictions GROUP BY predicted_class ORDER BY predicted_class"
    )
    class_counts = {row["class_name"]: row["count"] for row in cursor.fetchall()}

    cursor.execute(
        "SELECT risk_level, COUNT(*) as count FROM predictions GROUP BY risk_level"
    )
    risk_counts = {row["risk_level"]: row["count"] for row in cursor.fetchall()}

    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 10")
    recent = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return {
        "total": total,
        "class_counts": class_counts,
        "risk_counts": risk_counts,
        "recent": recent,
    }


def clear_history() -> None:
    """Delete all prediction records."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    logger.info("Prediction history cleared.")
