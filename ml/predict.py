"""
Prediction module for Diabetic Retinopathy Detection.
Handles single-image inference with confidence and risk assessment.
Auto-detects non-fundus images and converts them.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as app_config
from config import (
    MODEL_PATH, IMAGE_SIZE, CLASS_NAMES, CLASS_SHORT_NAMES,
    RISK_LEVELS, NUM_CLASSES, DEMO_MODE, UPLOADS_DIR,
)
from ml.utils import logger
from ml.focal_loss import FocalLoss


def validate_fundus_image(image_path: str | Path) -> dict:
    """Check if an image is a retinal fundus photograph.

    Returns dict with:
        is_fundus: bool
        confidence: float (0-1, how likely it is a fundus image)
        reason: str explanation
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {"is_fundus": False, "confidence": 0.0, "reason": "Could not read image"}

    h, w = img.shape[:2]
    scores = []

    # 1. Aspect ratio check - fundus images are roughly 1:1
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect < 1.5:
        scores.append(("aspect_ratio", 0.3))
    elif aspect < 2.5:
        scores.append(("aspect_ratio", 0.1))
    else:
        scores.append(("aspect_ratio", 0.0))

    # 2. Dark corners - fundus images have a circular mask
    corner_size = max(10, min(h, w) // 10)
    corners = [
        img[:corner_size, :corner_size],
        img[:corner_size, -corner_size:],
        img[-corner_size:, :corner_size],
        img[-corner_size:, -corner_size:],
    ]
    corner_brightness = np.mean([c.mean() for c in corners])
    center = img[h // 4:3 * h // 4, w // 4:3 * w // 4]
    center_brightness = center.mean()
    if corner_brightness < 40 and center_brightness > 80:
        scores.append(("dark_corners", 0.3))
    elif corner_brightness < 80:
        scores.append(("dark_corners", 0.15))
    else:
        scores.append(("dark_corners", 0.0))

    # 3. Color distribution - fundus images are orange-red dominant
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0].mean()
    s_channel = hsv[:, :, 1].mean()
    if 5 < h_channel < 25 and s_channel > 50:
        scores.append(("fundus_color", 0.3))
    elif 0 < h_channel < 30 and s_channel > 30:
        scores.append(("fundus_color", 0.15))
    else:
        scores.append(("fundus_color", 0.0))

    # 4. Resolution check - fundus images are typically high-res
    if min(h, w) >= 400:
        scores.append(("resolution", 0.1))
    elif min(h, w) >= 200:
        scores.append(("resolution", 0.05))
    else:
        scores.append(("resolution", 0.0))

    total = sum(s for _, s in scores)
    reasons = [f"{name}: {'pass' if s > 0 else 'fail'}" for name, s in scores if s > 0]
    fail_reasons = [f"{name}" for name, s in scores if s == 0]

    is_fundus = total >= 0.5
    reason = f"score={total:.2f}"
    if is_fundus:
        reason += f" ({', '.join(reasons)} passed)"
    else:
        reason += f" ({', '.join(fail_reasons)} failed)"

    logger.info(f"Fundus validation: is_fundus={is_fundus}, {reason}")
    return {"is_fundus": is_fundus, "confidence": min(total, 1.0), "reason": reason}


class DRPredictor:
    """Singleton-like predictor that loads model once and reuses it."""

    def __init__(self):
        self.model = None
        self._model_loaded = False

    def load_model(self) -> bool:
        """Load the trained model."""
        if self._model_loaded:
            return True
        if app_config.DEMO_MODE:
            logger.info("Demo mode: skipping real model load.")
            self._model_loaded = True
            return True
        if not MODEL_PATH.exists():
            logger.error(f"Model not found at {MODEL_PATH}")
            return False
        try:
            self.model = tf.keras.models.load_model(str(MODEL_PATH), custom_objects={"FocalLoss": FocalLoss})
            self._model_loaded = True
            logger.info("Model loaded successfully for prediction.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def preprocess_image(self, image_path: str | Path) -> np.ndarray:
        """Load and preprocess a single image for inference."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32)
        # EfficientNet preprocessing
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, image_path: str | Path) -> dict:
        """Run prediction on a single image and return full result dict.

        Auto-detects non-fundus images and converts them before prediction.
        """
        if DEMO_MODE:
            return self._demo_predict()

        if not self._model_loaded:
            if not self.load_model():
                raise RuntimeError("Model is not loaded. Cannot predict.")

        image_path = Path(image_path)

        # Validate input image
        validation = validate_fundus_image(image_path)
        image_type = "fundus"
        converted_path = None

        if not validation["is_fundus"]:
            logger.info(f"Non-fundus image detected ({validation['reason']}). Converting...")
            from ml.fundus_converter import convert_to_fundus

            try:
                conv_result = convert_to_fundus(image_path)
                converted_path = UPLOADS_DIR / f"converted_{image_path.stem}.png"

                fundus_rgb = cv2.cvtColor(conv_result["fundus_image"], cv2.COLOR_BGR2RGB)
                from PIL import Image as PILImage
                PILImage.fromarray(fundus_rgb).save(str(converted_path))

                image_path = converted_path
                image_type = "converted"
                logger.info(f"Converted image saved to {converted_path}")
            except Exception as e:
                logger.warning(f"Fundus conversion failed: {e}. Using original image.")

        processed = self.preprocess_image(image_path)
        predictions = self.model(processed, training=False)
        probs = predictions[0].numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        # Entropy-based confidence warning
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(NUM_CLASSES)
        normalized_entropy = entropy / max_entropy
        low_confidence_warning = None
        if confidence < 0.4 or normalized_entropy > 0.85:
            low_confidence_warning = (
                f"Low prediction confidence ({confidence:.1%}). "
                "The model is uncertain about this image."
            )

        result = {
            "predicted_class": pred_class,
            "class_name": CLASS_NAMES[pred_class],
            "class_short_name": CLASS_SHORT_NAMES[pred_class],
            "confidence": confidence,
            "confidence_str": f"{confidence * 100:.1f}%",
            "risk_level": RISK_LEVELS[pred_class],
            "probabilities": {
                CLASS_SHORT_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)
            },
            "all_class_names": {
                str(i): CLASS_NAMES[i] for i in range(NUM_CLASSES)
            },
            "image_type": image_type,
            "fundus_validation": validation,
            "low_confidence_warning": low_confidence_warning,
            "converted_path": str(converted_path) if converted_path else None,
        }
        logger.info(
            f"Prediction: {result['class_name']} "
            f"(class {pred_class}, confidence {result['confidence_str']}, "
            f"image_type={image_type})"
        )
        return result

    def _demo_predict(self) -> dict:
        """Generate a simulated prediction for demo mode."""
        import random
        random.seed()
        pred_class = random.randint(0, NUM_CLASSES - 1)
        probs = np.random.dirichlet(np.ones(NUM_CLASSES))
        # Boost the chosen class
        probs[pred_class] = max(probs) + 0.5
        probs = probs / probs.sum()
        confidence = float(probs[pred_class])

        result = {
            "predicted_class": pred_class,
            "class_name": CLASS_NAMES[pred_class],
            "class_short_name": CLASS_SHORT_NAMES[pred_class],
            "confidence": confidence,
            "confidence_str": f"{confidence * 100:.1f}%",
            "risk_level": RISK_LEVELS[pred_class],
            "probabilities": {
                CLASS_SHORT_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)
            },
            "all_class_names": {
                str(i): CLASS_NAMES[i] for i in range(NUM_CLASSES)
            },
            "image_type": "demo",
            "fundus_validation": {"is_fundus": False, "confidence": 0.0, "reason": "Demo mode"},
            "low_confidence_warning": None,
            "converted_path": None,
        }
        return result

    def _demo_predict(self) -> dict:
        """Generate a simulated prediction for demo mode."""
        import random
        random.seed()
        pred_class = random.randint(0, NUM_CLASSES - 1)
        probs = np.random.dirichlet(np.ones(NUM_CLASSES))
        # Boost the chosen class
        probs[pred_class] = max(probs) + 0.5
        probs = probs / probs.sum()
        confidence = float(probs[pred_class])

        result = {
            "predicted_class": pred_class,
            "class_name": CLASS_NAMES[pred_class],
            "class_short_name": CLASS_SHORT_NAMES[pred_class],
            "confidence": confidence,
            "confidence_str": f"{confidence * 100:.1f}%",
            "risk_level": RISK_LEVELS[pred_class],
            "probabilities": {
                CLASS_SHORT_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)
            },
            "all_class_names": {
                i: CLASS_NAMES[i] for i in range(NUM_CLASSES)
            },
        }
        return result


# Global predictor instance
_predictor = None


def get_predictor() -> DRPredictor:
    """Get the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = DRPredictor()
    return _predictor


def predict_image(image_path: str | Path) -> dict:
    """Convenience function to predict a single image."""
    predictor = get_predictor()
    return predictor.predict(image_path)


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage: python -m ml.predict <image_path>")
        sys.exit(1)
    result = predict_image(_sys.argv[1])
    print(f"\nPrediction: {result['class_name']}")
    print(f"Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence_str']}")
    print(f"Risk: {result['risk_level']}")
