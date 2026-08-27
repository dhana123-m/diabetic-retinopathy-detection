"""
Flask application for Diabetic Retinopathy Detection System.
Main entry point for the web application.
"""

import os
import sys
import uuid
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    SECRET_KEY, UPLOADS_DIR, STATIC_DIR, TEMPLATES_DIR,
    MAX_CONTENT_LENGTH, SUPPORTED_EXTENSIONS, MAX_UPLOAD_SIZE_MB,
    DEMO_MODE, FLASK_HOST, FLASK_PORT, DEBUG_MODE, MODEL_PATH,
    CLASS_SHORT_NAMES, CLASS_NAMES, RISK_LEVELS, NUM_CLASSES,
    RESULTS_DIR,
)
from database import init_database, insert_prediction, get_all_predictions, get_statistics, clear_history
from ml.predict import get_predictor, predict_image
from ml.gradcam import generate_gradcam
from ml.utils import logger

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATES_DIR),
)
app.config["SECRET_KEY"] = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _validate_image(file) -> tuple[bool, str]:
    """Validate an uploaded image file."""
    if file.filename == "":
        return False, "No file selected."
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported format. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    if file.content_length and file.content_length > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        return False, f"File too large. Maximum size: {MAX_UPLOAD_SIZE_MB}MB."
    return True, ""


def _safe_filename(original: str) -> str:
    """Generate a safe unique filename."""
    ext = Path(original).suffix.lower()
    unique_name = f"{uuid.uuid4().hex[:12]}{ext}"
    return unique_name


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    """Home/dashboard page."""
    stats = get_statistics()
    return render_template("index.html", stats=stats, demo_mode=DEMO_MODE)


@app.route("/dashboard")
def dashboard():
    """Dashboard page with analytics."""
    stats = get_statistics()
    return render_template("dashboard.html", stats=stats, demo_mode=DEMO_MODE)


@app.route("/analysis")
def analysis():
    """Image analysis/upload page."""
    return render_template("analysis.html", demo_mode=DEMO_MODE)


@app.route("/history")
def history():
    """Prediction history page."""
    predictions = get_all_predictions(limit=200)
    return render_template("history.html", predictions=predictions, demo_mode=DEMO_MODE)


@app.route("/about")
def about():
    """About/model information page."""
    return render_template("about.html", demo_mode=DEMO_MODE)


@app.route("/result/<int:record_id>")
def result_detail(record_id):
    """View a specific prediction result."""
    from database import get_prediction_by_id
    record = get_prediction_by_id(record_id)
    if record is None:
        return redirect(url_for("history"))
    return render_template("result.html", record=record, demo_mode=DEMO_MODE)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and prediction."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    valid, msg = _validate_image(file)
    if not valid:
        return jsonify({"error": msg}), 400

    try:
        safe_name = _safe_filename(file.filename)
        save_path = UPLOADS_DIR / safe_name
        file.save(str(save_path))

        # Run prediction
        prediction = predict_image(save_path)

        # Generate Grad-CAM
        try:
            gradcam_result = generate_gradcam(
                save_path,
                None,  # Will be preprocessed internally
                prediction["predicted_class"],
            )
        except Exception as e:
            logger.warning(f"Grad-CAM generation failed: {e}")
            gradcam_result = {
                "overlay_filename": "",
                "comparison_filename": "",
                "heatmap_filename": "",
            }

        # Store in database
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
                "comparison": gradcam_result.get("comparison_filename", ""),
                "heatmap": gradcam_result.get("heatmap_filename", ""),
            },
        })

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/clear-history", methods=["POST"])
def clear_history_route():
    """Clear all prediction history."""
    try:
        clear_history()
        return jsonify({"success": True, "message": "History cleared."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    """Serve uploaded/processed images."""
    return send_from_directory(str(UPLOADS_DIR), filename)


@app.route("/results/<path:filename>")
def serve_result(filename):
    """Serve evaluation result files."""
    return send_from_directory(str(RESULTS_DIR), filename)


@app.route("/api/stats")
def api_stats():
    """API endpoint for dashboard statistics."""
    return jsonify(get_statistics())


# ──────────────────────────────────────────────
# Error Handlers
# ──────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"File too large. Maximum size: {MAX_UPLOAD_SIZE_MB}MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return render_template("base.html", error="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("base.html", error="Internal server error"), 500


# ──────────────────────────────────────────────
# Application Startup
# ──────────────────────────────────────────────

def create_app():
    """Application factory."""
    init_database()
    predictor = get_predictor()
    if not DEMO_MODE:
        loaded = predictor.load_model()
        if not loaded:
            logger.warning(
                "No trained model found. Run ml/train.py to train a model, "
                "or enable DEMO_MODE in config.py for UI testing."
            )
    else:
        logger.info("Application started in DEMO MODE.")
    return app


if __name__ == "__main__":
    application = create_app()
    print("\n" + "=" * 60)
    print("  Diabetic Retinopathy Detection System")
    print("=" * 60)
    if DEMO_MODE:
        print("  ** DEMO MODE ACTIVE **")
        print("  Predictions are simulated for interface testing.")
        print("  Real AI model is NOT being used.")
    print(f"\n  Running at: http://{FLASK_HOST}:{FLASK_PORT}")
    print("  Press Ctrl+C to stop.\n")
    application.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG_MODE)
