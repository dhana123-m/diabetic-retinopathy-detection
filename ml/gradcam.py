"""
Grad-CAM Explainable AI module for Diabetic Retinopathy Detection.
Generates heatmaps showing which regions of a retinal image influenced the prediction.
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    IMAGE_SIZE, GRADCAM_HEATMAP_ALPHA, GRADCAM_COLORMAP, DEMO_MODE,
    UPLOADS_DIR, MODEL_PATH,
)
from ml.utils import logger


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Automatically find the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if hasattr(layer, "output"):
            output = layer.output
            if hasattr(output, "shape") and len(output.shape) == 4:
                logger.info(f"Selected Grad-CAM layer: {layer.name} (output shape: {output.shape})")
                return layer.name
    for layer in reversed(model.layers):
        if hasattr(layer, "output"):
            output = layer.output
            if hasattr(output, "shape") and len(output.shape) == 4:
                logger.info(f"Selected Grad-CAM layer (fallback): {layer.name}")
                return layer.name
    raise ValueError("Could not find a suitable convolutional layer for Grad-CAM.")


class GradCAMGenerator:
    """Generates Grad-CAM heatmaps for model predictions."""

    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.conv_layer_name = find_last_conv_layer(model)
        self.grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[
                model.get_layer(self.conv_layer_name).output,
                model.output,
            ],
        )

    def generate(self, image: np.ndarray, predicted_class: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the given image and predicted class.

        Args:
            image: Preprocessed image array of shape (1, H, W, 3)
            predicted_class: Index of the predicted class

        Returns:
            heatmap: 2D numpy array (H, W) with values in [0, 1]
        """
        image_tensor = tf.cast(image, tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image_tensor)
            loss = predictions[:, predicted_class]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads

        # Weight the channels by gradient importance
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.numpy()

    def overlay_heatmap(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = GRADCAM_HEATMAP_ALPHA,
    ) -> np.ndarray:
        """Overlay heatmap on original image."""
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Ensure original is uint8
        if original_image.dtype != np.uint8:
            orig = (original_image * 255).astype(np.uint8)
        else:
            orig = original_image.copy()

        overlay = cv2.addWeighted(orig, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay

    def generate_and_save(
        self,
        original_image_path: str | Path,
        processed_image: np.ndarray,
        predicted_class: int,
        save_dir: str | Path | None = None,
    ) -> dict:
        """
        Full Grad-CAM pipeline: generate heatmap, overlay, save results.

        Returns dict with paths to saved images.
        """
        if save_dir is None:
            save_dir = UPLOADS_DIR
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate heatmap
        heatmap = self.generate(processed_image, predicted_class)

        # Load original image for overlay
        original = cv2.imread(str(original_image_path))
        if original is None:
            raise ValueError(f"Could not read original image: {original_image_path}")
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Create overlay
        overlay = self.overlay_heatmap(original_rgb, heatmap)

        # Generate filenames
        stem = Path(original_image_path).stem
        gradcam_filename = f"{stem}_gradcam.png"
        heatmap_filename = f"{stem}_heatmap.png"
        overlay_filename = f"{stem}_overlay.png"

        gradcam_path = save_dir / gradcam_filename
        heatmap_path = save_dir / heatmap_filename
        overlay_path = save_dir / overlay_filename

        # Save heatmap only
        plt.figure(figsize=(8, 8))
        plt.imshow(heatmap, cmap=GRADCAM_COLORMAP)
        plt.title("Grad-CAM Heatmap")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(str(heatmap_path), dpi=150, bbox_inches="tight")
        plt.close()

        # Save overlay
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # Save side-by-side comparison
        comparison_filename = f"{stem}_comparison.png"
        comparison_path = save_dir / comparison_filename

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        axes[0].imshow(original_rgb)
        axes[0].set_title("Original Fundus Image", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap=GRADCAM_COLORMAP)
        axes[1].set_title("Grad-CAM Heatmap", fontsize=14)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=14)
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(str(comparison_path), dpi=150, bbox_inches="tight")
        plt.close()

        result = {
            "gradcam_path": str(gradcam_path),
            "heatmap_path": str(heatmap_path),
            "overlay_path": str(overlay_path),
            "comparison_path": str(comparison_path),
            "gradcam_filename": gradcam_filename,
            "heatmap_filename": heatmap_filename,
            "overlay_filename": overlay_filename,
            "comparison_filename": comparison_filename,
        }

        logger.info(f"Grad-CAM visualizations saved for {original_image_path}")
        return result


def _preprocess_image(image_path: str | Path) -> np.ndarray:
    """Load and preprocess an image for the model."""
    img = tf.keras.utils.load_img(str(image_path), target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img = tf.keras.utils.img_to_array(img)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return np.expand_dims(img, axis=0)


def generate_gradcam(
    original_image_path: str | Path,
    processed_image: np.ndarray | None,
    predicted_class: int,
) -> dict:
    """Convenience function to generate Grad-CAM for a single image."""
    if DEMO_MODE:
        return _demo_gradcam(original_image_path)

    if processed_image is None:
        processed_image = _preprocess_image(original_image_path)

    model = tf.keras.models.load_model(str(MODEL_PATH))
    generator = GradCAMGenerator(model)
    return generator.generate_and_save(original_image_path, processed_image, predicted_class)


def _demo_gradcam(original_image_path: str | Path) -> dict:
    """Generate placeholder Grad-CAM for demo mode."""
    stem = Path(original_image_path).stem
    save_dir = UPLOADS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    original = cv2.imread(str(original_image_path))
    if original is not None:
        h, w = original.shape[:2]
    else:
        h, w = IMAGE_SIZE, IMAGE_SIZE

    # Create a random heatmap for demo
    heatmap = np.random.rand(h, w).astype(np.float32)
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    heatmap_filename = f"{stem}_heatmap.png"
    heatmap_path = save_dir / heatmap_filename
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, cmap=GRADCAM_COLORMAP)
    plt.title("Grad-CAM Heatmap (Demo)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(heatmap_path), dpi=150, bbox_inches="tight")
    plt.close()

    overlay_filename = f"{stem}_overlay.png"
    overlay_path = save_dir / overlay_filename
    if original is not None:
        orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        overlay = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(orig_rgb, 0.6, overlay, 0.4, 0)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(overlay_path), np.zeros((h, w, 3), dtype=np.uint8))

    comparison_filename = f"{stem}_comparison.png"
    comparison_path = save_dir / comparison_filename
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    if original is not None:
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(np.zeros((h, w, 3)))
    axes[0].set_title("Original Fundus Image (Demo)", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(heatmap, cmap=GRADCAM_COLORMAP)
    axes[1].set_title("Grad-CAM Heatmap (Demo)", fontsize=14)
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(str(comparison_path), dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "gradcam_path": str(heatmap_path),
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
        "comparison_path": str(comparison_path),
        "gradcam_filename": heatmap_filename,
        "heatmap_filename": heatmap_filename,
        "overlay_filename": overlay_filename,
        "comparison_filename": comparison_filename,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ml.gradcam <image_path>")
        sys.exit(1)
    result = generate_gradcam(sys.argv[1], None, 0)
    print(f"Grad-CAM saved: {result}")
