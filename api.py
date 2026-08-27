"""
Flask ML API for Diabetic Retinopathy Detection.

Deployable API backend serving the static frontend (Vercel).
Exposes JSON endpoints for prediction, history, statistics and media files.

Run locally:
    python api.py

Run in production (gunicorn):
    gunicorn api:app --workers 1 --threads 2 --timeout 300
"""

import os
import sys
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    UPLOADS_DIR, MAX_CONTENT_LENGTH, SUPPORTED_EXTENSIONS, MAX_UPLOAD_SIZE_MB,
    DEMO_MODE, FLASK_HOST, FLASK_PORT, MODELS_DIR, MODEL_PATH,
)
from database import init_database, insert_prediction, get_all_predictions, get_prediction_by_id, get_statistics, clear_history
from ml.predict import get_predictor
from ml.gradcam import generate_gradcam
from ml.utils import logger

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

_model_loaded = False
_model_load_lock = threading.Lock()


def _load_model_on_startup() -> None:
    """Load the trained model once at startup (non-blocking, in background)."""
    global _model_loaded
    with _model_load_lock:
        if _model_loaded:
            return
        predictor = get_predictor()
        if DEMO_MODE:
            logger.info("API started in DEMO MODE (simulated predictions).")
            _model_loaded = True
            return
        if not MODEL_PATH.exists():
            logger.error(f"Model not found: {MODEL_PATH}")
            return
        ok = predictor.load_model()
        _model_loaded = ok
        logger.info(f"Model load finished: loaded={ok}")


def _validate_image(file) -> tuple[bool, str]:
    """Validate an uploaded image file."""
    if file is None or file.filename == "":
        return False, "No file selected."
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported format. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    return True, ""


def _safe_filename(original: str) -> str:
    """Generate a safe unique filename."""
    ext = Path(original).suffix.lower()
    return f"{uuid.uuid4().hex[:12]}{ext}"


@app.route("/api/health")
def health():
    """Health check used by the host platform."""
    predictor = get_predictor()
    status = "ok" if (_model_loaded or DEMO_MODE) else "starting"
    return jsonify({"status": status, "demo_mode": DEMO_MODE, "model": "loaded" if _model_loaded else "loading"}), (200 if status == "ok" else 503)


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    """Handle image upload and return prediction result as JSON."""
    if request.method == "OPTIONS":
        return ("", 204)

    file = request.files.get("image")
    valid, msg = _validate_image(file)
    if not valid:
        return jsonify({"error": msg}), 400

    try:
        safe_name = _safe_filename(file.filename)
        save_path = UPLOADS_DIR / safe_name
        file.save(str(save_path))

        predictor = get_predictor()
        if not predictor.model and not DEMO_MODE:
            if not predictor.load_model():
                return jsonify({"error": "AI model is still loading. Please retry in a few seconds."}), 503

        prediction = predictor.predict(save_path)

        try:
            gradcam_result = generate_gradcam(save_path, None, prediction["predicted_class"])
        except Exception as e:
            logger.warning(f"Grad-CAM generation failed: {e}")
            gradcam_result = {}

        record_id = insert_prediction(
            filename=file.filename,
            predicted_class=prediction["predicted_class"],
            class_name=prediction["class_name"],
            confidence=prediction["confidence"],
            risk_level=prediction["risk_level"],
            original_image_path=safe_name,
            gradcam_image_path=gradcam_result.get("overlay_filename", ""),
        )

        return jsonify({
            "success": True,
            "record_id": record_id,
            "prediction": prediction,
            "gradcam": {
                "original": safe_name,
                "overlay": gradcam_result.get("overlay_filename", ""),
                "heatmap": gradcam_result.get("heatmap_filename", ""),
                "comparison": gradcam_result.get("comparison_filename", ""),
            },
        })

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/api/history", methods=["GET"])
def history_list():
    """List prediction history."""
    limit = request.args.get("limit", 200, type=int)
    return jsonify({"predictions": get_all_predictions(limit=limit)})


@app.route("/api/history/<int:record_id>", methods=["GET"])
def history_detail(record_id):
    """Fetch a single prediction record."""
    record = get_prediction_by_id(record_id)
    if record is None:
        return jsonify({"error": "Record not found."}), 404
    return jsonify({"prediction": record})


@app.route("/api/history", methods=["DELETE"])
def history_clear():
    """Clear all prediction history."""
    try:
        clear_history()
        return jsonify({"success": True, "message": "History cleared."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    """Dashboard statistics."""
    return jsonify(get_statistics())


@app.route("/api/media/<path:filename>")
def media(filename):
    """Serve uploaded/processed images (original, heatmap, overlay)."""
    return send_from_directory(str(UPLOADS_DIR), filename)


if __name__ == "__main__":
    init_database()
    threading.Thread(target=_load_model_on_startup, daemon=True).start()
    print(f"\n  RetinaAI ML API running at http://{FLASK_HOST}:{FLASK_PORT}/api/health")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)