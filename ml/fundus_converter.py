"""
Fundus Image Converter - Transforms regular eye photos into fundus-like images.

This module applies computer vision techniques to convert front-facing eye
photographs into images that resemble retinal fundus photographs, enabling
the diabetic retinopathy model to process them.

DISCLAIMER: This is an experimental preprocessing pipeline. Converting a
front-facing eye photo to a fundus image is an approximation. The resulting
image will NOT contain actual retinal features visible through a fundus camera.
Predictions on converted images should be treated as experimental estimates,
not clinical diagnoses.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import IMAGE_SIZE
from ml.utils import logger


class FundusConverter:
    """Converts regular eye photos into fundus-like images."""

    FUNDUS_COLORS = {
        "retina_base": np.array([40, 60, 140]),
        "retina_bright": np.array([30, 100, 200]),
        "optic_disc": np.array([50, 180, 220]),
        "vessel_dark": np.array([20, 30, 80]),
    }

    def convert(self, image_path: str | Path, output_size: int = IMAGE_SIZE) -> dict:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        logger.info(f"Input image shape: {img.shape}")

        eye_region = self._detect_and_crop_eye(img)
        fundus = self._apply_fundus_color_transform(eye_region)
        fundus = self._apply_circular_mask(fundus)
        fundus = self._add_retinal_texture(fundus)
        fundus = self._enhance_fundus(fundus)
        fundus = self._apply_vignette(fundus)
        fundus = cv2.resize(fundus, (output_size, output_size))

        return {
            "fundus_image": fundus,
            "original_shape": img.shape,
            "eye_detected": eye_region is not None,
            "conversion_quality": self._estimate_quality(fundus),
        }

    def _detect_and_crop_eye(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        eye_mask = self._detect_iris_region(img, gray, hsv)

        if eye_mask is not None:
            coords = cv2.boundingRect(eye_mask.astype(np.uint8))
            x, y, rw, rh = coords
            cx, cy = x + rw // 2, y + rh // 2
            radius = max(rw, rh) // 2
            radius = int(radius * 1.3)

            x1 = max(0, cx - radius)
            y1 = max(0, cy - radius)
            x2 = min(w, cx + radius)
            y2 = min(h, cy + radius)

            cropped = img[y1:y2, x1:x2].copy()
            logger.info(f"Eye detected and cropped to {cropped.shape}")
            return cropped

        logger.info("No specific eye region detected, using center crop")
        side = min(h, w)
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        return img[y1:y1 + side, x1:x1 + side].copy()

    def _detect_iris_region(self, img: np.ndarray, gray: np.ndarray, hsv: np.ndarray):
        h, w = img.shape[:2]

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w) // 4,
            param1=100, param2=40,
            minRadius=min(h, w) // 8, maxRadius=min(h, w) // 2,
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            best = circles[0, 0]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (best[0], best[1]), best[2], 255, -1)
            logger.info(f"Iris detected via HoughCircles at ({best[0]},{best[1]}) r={best[2]}")
            return mask

        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([25, 150, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 60])
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

        eye_region = cv2.bitwise_and(dark_mask, cv2.bitwise_not(skin_mask))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        eye_region = cv2.morphologyEx(eye_region, cv2.MORPH_CLOSE, kernel)
        eye_region = cv2.morphologyEx(eye_region, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(eye_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > (h * w * 0.005):
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [largest], -1, 255, -1)
                logger.info("Eye region detected via color segmentation")
                return mask

        return None

    def _apply_fundus_color_transform(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

        h_new = np.clip(h * 0.3 + 5, 0, 179)
        s_new = np.clip(s * 1.4 + 30, 0, 255)
        v_new = np.clip(v * 0.65 + 20, 0, 255)

        hsv[:,:,0] = h_new
        hsv[:,:,1] = s_new
        hsv[:,:,2] = v_new

        fundus_hsv = hsv.astype(np.uint8)
        fundus_bgr = cv2.cvtColor(fundus_hsv, cv2.COLOR_HSV2BGR)

        b, g, r = cv2.split(fundus_bgr)
        b = np.clip(b * 1.3 + 15, 0, 255).astype(np.uint8)
        g = np.clip(g * 0.9 + 5, 0, 255).astype(np.uint8)
        r = np.clip(r * 0.7, 0, 255).astype(np.uint8)

        fundus_bgr = cv2.merge([b, g, r])

        lab = cv2.cvtColor(fundus_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        fundus_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        logger.info("Fundus color transform applied")
        return fundus_bgr

    def _apply_circular_mask(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = min(h, w) // 2 - 2
        cv2.circle(mask, center, radius, 255, -1)

        mask_blur = cv2.GaussianBlur(mask, (21, 21), 10)
        mask_3ch = cv2.merge([mask_blur, mask_blur, mask_blur]).astype(np.float32) / 255.0

        black = np.zeros_like(img, dtype=np.float32)
        result = (img.astype(np.float32) * mask_3ch + black * (1 - mask_3ch)).astype(np.uint8)

        logger.info("Circular fundus mask applied")
        return result

    def _add_retinal_texture(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        overlay = img.copy()

        np.random.seed(42)

        vessel_layer = np.zeros((h, w), dtype=np.float32)
        for _ in range(5):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            x2 = np.random.randint(w // 4, 3 * w // 4)
            y2 = np.random.randint(h // 4, 3 * h // 4)
            thickness = np.random.randint(1, 3)
            cv2.line(vessel_layer, (x1, y1), (x2, y2), 0.3, thickness)

        vessel_layer = cv2.GaussianBlur(vessel_layer, (15, 15), 5)
        overlay = cv2.addWeighted(overlay, 0.92, 
                                   cv2.merge([vessel_layer * 80, vessel_layer * 40, vessel_layer * 20]).astype(np.uint8), 
                                   0.08, 0)

        center = (w // 2, h // 2)
        radius = min(h, w) // 8
        optic_disc = np.zeros((h, w, 3), dtype=np.float32)
        cv2.circle(optic_disc, (int(center[0] + w * 0.1), center[1]), radius, (80, 160, 200), -1)
        optic_disc = cv2.GaussianBlur(optic_disc, (31, 31), 15)
        overlay = cv2.addWeighted(overlay, 0.88, optic_disc.astype(np.uint8), 0.12, 0)

        logger.info("Retinal texture overlay applied")
        return overlay

    def _enhance_fundus(self, img: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.3)

        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(1.2)

        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.1)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _apply_vignette(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        X = cv2.getGaussianKernel(w, w * 0.6)
        Y = cv2.getGaussianKernel(h, h * 0.6)
        vignette = Y * X.T
        vignette = vignette / vignette.max()
        vignette = np.power(vignette, 0.8)

        vignette_3ch = cv2.merge([vignette, vignette, vignette])
        result = (img.astype(np.float32) * vignette_3ch).astype(np.uint8)

        logger.info("Vignette effect applied")
        return result

    def _estimate_quality(self, fundus: np.ndarray) -> str:
        hsv = cv2.cvtColor(fundus, cv2.COLOR_BGR2HSV)
        h_mean = hsv[:,:,0].mean()
        s_mean = hsv[:,:,1].mean()

        h, w = fundus.shape[:2]
        center = fundus[h//4:3*h//4, w//4:3*w//4]
        edge_top = fundus[:h//8, :]
        center_bright = center.mean()
        edge_bright = edge_top.mean()

        if s_mean > 60 and 5 < h_mean < 25 and center_bright > edge_bright * 1.2:
            return "good"
        elif s_mean > 30 and center_bright > edge_bright:
            return "moderate"
        else:
            return "low"


def convert_to_fundus(image_path: str | Path, output_size: int = IMAGE_SIZE) -> dict:
    """Convenience function to convert an image to fundus-like appearance."""
    converter = FundusConverter()
    result = converter.convert(image_path, output_size)
    return result


def save_converted_fundus(image_path: str | Path, output_path: str | Path | None = None) -> str:
    """Convert and save fundus-like image. Returns the output path."""
    result = convert_to_fundus(image_path)

    if output_path is None:
        stem = Path(image_path).stem
        output_path = Path(image_path).parent / f"{stem}_fundus_converted.png"

    fundus_img = result["fundus_image"]
    fundus_rgb = cv2.cvtColor(fundus_img, cv2.COLOR_BGR2RGB)
    Image.fromarray(fundus_rgb).save(str(output_path))

    logger.info(f"Fundus-converted image saved to {output_path}")
    logger.info(f"  Quality: {result['conversion_quality']}")
    logger.info(f"  Eye detected: {result['eye_detected']}")

    return str(output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ml.fundus_converter <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    result_path = save_converted_fundus(image_path, output_path)
    print(f"Converted fundus image saved: {result_path}")
