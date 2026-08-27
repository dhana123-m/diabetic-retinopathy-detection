"""
Basic test suite for Diabetic Retinopathy Detection System.
Run with: python -m pytest tests/test_basic.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CLASS_NAMES, CLASS_SHORT_NAMES, RISK_LEVELS, NUM_CLASSES,
    IMAGE_SIZE, SUPPORTED_EXTENSIONS,
)
from ml.utils import (
    get_class_name, get_class_short_name, get_risk_level,
    format_confidence, probabilities_to_class,
)


class TestClassMapping:
    def test_class_names_count(self):
        assert len(CLASS_NAMES) == NUM_CLASSES

    def test_get_class_name(self):
        assert get_class_name(0) == "No Diabetic Retinopathy"
        assert get_class_name(4) == "Proliferative Diabetic Retinopathy"
        assert get_class_name(99) == "Unknown"

    def test_get_class_short_name(self):
        assert get_class_short_name(0) == "No DR"
        assert get_class_short_name(2) == "Moderate DR"

    def test_get_risk_level(self):
        assert get_risk_level(0) == "Low"
        assert get_risk_level(3) == "High"
        assert get_risk_level(4) == "Very High"


class TestRiskMapping:
    def test_risk_levels_count(self):
        assert len(RISK_LEVELS) == NUM_CLASSES

    def test_risk_increases_with_class(self):
        risk_order = ["Low", "Low-Moderate", "Moderate-High", "High", "Very High"]
        for i, expected in enumerate(risk_order):
            assert RISK_LEVELS[i] == expected


class TestUtils:
    def test_format_confidence(self):
        assert format_confidence(0.914) == "91.4%"
        assert format_confidence(0.0) == "0.0%"
        assert format_confidence(1.0) == "100.0%"

    def test_probabilities_to_class(self):
        probs = np.array([0.1, 0.2, 0.5, 0.15, 0.05])
        assert probabilities_to_class(probs) == 2

    def test_probabilities_to_class_all_equal(self):
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        assert probabilities_to_class(probs) == 0  # argmax returns first


class TestImageValidation:
    def test_supported_extensions(self):
        assert ".png" in SUPPORTED_EXTENSIONS
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".jpeg" in SUPPORTED_EXTENSIONS
        assert ".bmp" not in SUPPORTED_EXTENSIONS

    def test_config_paths_are_relative(self):
        from config import BASE_DIR, DATA_DIR, MODELS_DIR
        assert DATA_DIR.name == "data"
        assert MODELS_DIR.name == "models"


class TestDatabase:
    def test_database_operations(self):
        from database import (
            get_connection, init_database, insert_prediction,
            get_all_predictions, clear_history,
        )
        import tempfile

        # Use a temp database
        from config import DATABASE_DIR
        test_db = DATABASE_DIR / "test_predictions.db"

        import config
        original_db = config.DATABASE_PATH
        config.DATABASE_PATH = test_db

        try:
            init_database()
            record_id = insert_prediction(
                filename="test.png",
                predicted_class=2,
                class_name="Moderate Diabetic Retinopathy",
                confidence=0.914,
                risk_level="Moderate-High",
                original_image_path="test.png",
                gradcam_image_path="test_gradcam.png",
            )
            assert record_id > 0

            predictions = get_all_predictions()
            assert len(predictions) >= 1

            clear_history()
            predictions = get_all_predictions()
            assert len(predictions) == 0
        finally:
            config.DATABASE_PATH = original_db
            if test_db.exists():
                test_db.unlink()


class TestFlaskApp:
    def test_app_creation(self):
        from app import create_app
        application = create_app()
        assert application is not None

    def test_app_routes(self):
        from app import create_app
        application = create_app()
        client = application.test_client()

        # Test main routes
        response = client.get("/")
        assert response.status_code == 200

        response = client.get("/analysis")
        assert response.status_code == 200

        response = client.get("/history")
        assert response.status_code == 200

        response = client.get("/about")
        assert response.status_code == 200

        response = client.get("/dashboard")
        assert response.status_code == 200

    def test_predict_no_image(self):
        from app import create_app
        application = create_app()
        client = application.test_client()

        response = client.post("/predict")
        assert response.status_code == 400

    def test_predict_invalid_format(self):
        from app import create_app
        application = create_app()
        client = application.test_client()

        data = {"image": (b"fake data", "test.bmp")}
        response = client.post("/predict", data=data, content_type="multipart/form-data")
        assert response.status_code == 400

    def test_stats_api(self):
        from app import create_app
        application = create_app()
        client = application.test_client()

        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.get_json()
        assert "total" in data


class TestPreprocessing:
    def test_image_size_config(self):
        assert IMAGE_SIZE == 224

    def test_numpy_preprocessing(self):
        img = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        processed = img.astype(np.float32) / 255.0
        assert processed.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0


class TestDemoMode:
    def test_demo_prediction(self):
        import config
        import ml.predict as predict_mod
        original_config = config.DEMO_MODE
        original_predict = predict_mod.DEMO_MODE
        config.DEMO_MODE = True
        predict_mod.DEMO_MODE = True

        try:
            from ml.predict import DRPredictor
            predictor = DRPredictor()
            result = predictor.predict("nonexistent.png")
            assert "predicted_class" in result
            assert "confidence" in result
            assert "risk_level" in result
            assert 0 <= result["predicted_class"] < NUM_CLASSES
        finally:
            config.DEMO_MODE = original_config
            predict_mod.DEMO_MODE = original_predict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
