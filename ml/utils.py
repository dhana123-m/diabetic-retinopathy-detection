"""
Utility functions for the Diabetic Retinopathy Detection system.
"""

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_NAMES, CLASS_SHORT_NAMES, RISK_LEVELS, NUM_CLASSES


def setup_logging(name: str = "dr_detection", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logging()


def get_class_name(class_index: int) -> str:
    """Return full class name for a given class index."""
    return CLASS_NAMES.get(class_index, "Unknown")


def get_class_short_name(class_index: int) -> str:
    """Return short class name for a given class index."""
    return CLASS_SHORT_NAMES.get(class_index, "Unknown")


def get_risk_level(class_index: int) -> str:
    """Return risk level for a given class index."""
    return RISK_LEVELS.get(class_index, "Unknown")


def probabilities_to_class(probs: np.ndarray) -> int:
    """Convert probability array to predicted class index."""
    return int(np.argmax(probs))


def format_confidence(confidence: float) -> str:
    """Format confidence as a percentage string."""
    return f"{confidence * 100:.1f}%"


def ensure_directories(*dirs: Path) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
