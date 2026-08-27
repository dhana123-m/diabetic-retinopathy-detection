"""
Realistic Synthetic Retinal Fundus Image Generator.
Generates fundus-like images with class-specific pathological features.
"""

import sys
from pathlib import Path
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import IMAGE_SIZE


class FundusImageGenerator:
    """Generates synthetic retinal fundus images with realistic features."""

    BASE_RETINA_COLOR = (30, 60, 140)
    RETINA_COLORS = [
        (25, 50, 130), (30, 65, 150), (35, 55, 135),
        (28, 58, 145), (32, 62, 142), (20, 45, 125),
    ]

    def __init__(self, size: int = IMAGE_SIZE, seed: int = None):
        self.size = size
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(self, dr_grade: int) -> np.ndarray:
        """Generate a synthetic fundus image for a given DR grade (0-4)."""
        img = self._create_retina_background()
        img = self._add_optic_disc(img)
        img = self._add_blood_vessels(img, dr_grade)
        img = self._add_pathology(img, dr_grade)
        img = self._apply_circular_mask(img)
        img = self._apply_vignette(img)
        img = self._add_noise(img)
        return img

    def _create_retina_background(self) -> np.ndarray:
        h, w = self.size, self.size
        base = np.zeros((h, w, 3), dtype=np.uint8)
        center_x, center_y = w // 2, h // 2

        base_color = random.choice(self.RETINA_COLORS)

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        norm_dist = np.clip(dist / max_dist, 0, 1)

        for c in range(3):
            channel_val = base_color[c]
            gradient = channel_val * (1 - norm_dist * 0.6)
            noise = np.random.normal(0, 3, (h, w))
            base[:, :, c] = np.clip(gradient + noise, 0, 255).astype(np.uint8)

        # Add subtle color variation
        noise_layer = np.random.normal(0, 5, (h, w, 3))
        base = np.clip(base.astype(np.float32) + noise_layer, 0, 255).astype(np.uint8)

        return base

    def _add_optic_disc(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        result = img.copy()

        disc_x = int(w * random.uniform(0.55, 0.68))
        disc_y = int(h * random.uniform(0.42, 0.58))
        disc_radius = int(min(h, w) * random.uniform(0.06, 0.09))

        # Yellowish-white optic disc
        overlay = np.zeros_like(result, dtype=np.float32)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - disc_x) ** 2 + (Y - disc_y) ** 2)
        mask = np.exp(-(dist ** 2) / (2 * (disc_radius * 0.7) ** 2))

        disc_color = np.array([80, 180, 220], dtype=np.float32)
        for c in range(3):
            overlay[:, :, c] = mask * disc_color[c]

        result = cv2.addWeighted(result, 0.85, overlay.astype(np.uint8), 0.15, 0)
        return result

    def _add_blood_vessels(self, img: np.ndarray, dr_grade: int) -> np.ndarray:
        h, w = img.shape[:2]
        result = img.copy()
        vessel_layer = np.zeros((h, w, 3), dtype=np.float32)

        disc_x = int(w * 0.62)
        disc_y = int(h * 0.50)

        num_vessels = random.randint(8, 15)
        vessel_count_factor = 1.0 + dr_grade * 0.1

        for _ in range(num_vessels):
            angle = random.uniform(0, 2 * np.pi)
            length = random.randint(int(min(h, w) * 0.15), int(min(h, w) * 0.4))
            thickness = random.uniform(1.0, 3.0)

            x, y = float(disc_x), float(disc_y)
            points = [(int(x), int(y))]

            steps = random.randint(15, 30)
            for s in range(steps):
                angle += random.gauss(0, 0.3)
                step_len = length / steps
                x += step_len * np.cos(angle)
                y += step_len * np.sin(angle)
                points.append((int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))))

            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            current_thickness = max(1, int(thickness * (1 - 0.3)))
            cv2.polylines(vessel_layer, [pts], False,
                          (15 * vessel_count_factor, 25 * vessel_count_factor, 60 * vessel_count_factor),
                          thickness=current_thickness)

        vessel_layer = cv2.GaussianBlur(vessel_layer, (3, 3), 1)
        result = cv2.addWeighted(result, 0.92, vessel_layer.astype(np.uint8), 0.08, 0)
        return result

    def _add_pathology(self, img: np.ndarray, dr_grade: int) -> np.ndarray:
        if dr_grade == 0:
            return img

        h, w = img.shape[:2]
        result = img.copy()

        if dr_grade >= 1:
            result = self._add_microaneurysms(result, count=random.randint(3, 8))

        if dr_grade >= 2:
            result = self._add_hemorrhages(result, count=random.randint(2, 6))
            result = self._add_hard_exudates(result, count=random.randint(1, 4))

        if dr_grade >= 3:
            result = self._add_microaneurysms(result, count=random.randint(5, 12))
            result = self._add_hemorrhages(result, count=random.randint(4, 10))
            result = self._add_cotton_wool_spots(result, count=random.randint(2, 5))
            result = self._add_venous_beading(result, count=random.randint(1, 3))

        if dr_grade >= 4:
            result = self._add_neovascularization(result, count=random.randint(1, 3))
            result = self._add_vitreous_hemorrhage(result, count=random.randint(1, 2))
            result = self._add_hard_exudates(result, count=random.randint(3, 8))
            result = self._add_cotton_wool_spots(result, count=random.randint(3, 7))

        return result

    def _add_microaneurysms(self, img: np.ndarray, count: int) -> np.ndarray:
        h, w = img.shape[:2]
        result = img.copy()

        for _ in range(count):
            cx = random.randint(int(w * 0.2), int(w * 0.85))
            cy = random.randint(int(h * 0.2), int(h * 0.85))
            radius = random.randint(1, 3)

            # Red dot
            cv2.circle(result, (cx, cy), radius, (20, 30, 120), -1)
            cv2.circle(result, (cx, cy), radius + 1, (15, 25, 100), 1)

        return result

    def _add_hemorrhages(self, img: np.ndarray, count: int) -> np.ndarray:
        h, w = img.shape[:2]
        result = img.copy()

        for _ in range(count):
            cx = random.randint(int(w * 0.15), int(w * 0.85))
            cy = random.randint(int(h * 0.15), int(h * 0.85))
            rw = random.randint(4, 12)
            rh = random.randint(3, 10)
            angle = random.randint(0, 360)

            overlay = result.copy()
            cv2.ellipse(overlay, (cx, cy), (rw, rh), angle, 0, 360,
                        (15, 20, 100), -1)
            cv2.ellipse(overlay, (cx, cy), (rw, rh), angle, 0, 360,
                        (10, 15, 80), 1)

            alpha = random.uniform(0.3, 0.6)
            mask_3ch = np.zeros_like(result, dtype=np.float32)
            cv2.ellipse(mask_3ch, (cx, cy), (rw, rh), angle, 0, 360, (1, 1, 1), -1)
            mask_3ch = cv2.GaussianBlur(mask_3ch, (5, 5), 2)

            result = (result.astype(np.float32) * (1 - mask_3ch * alpha) +
                      overlay.astype(np.float32) * mask_3ch * alpha).astype(np.uint8)

        return result

    def _add_hard_exudates(self, img: np.ndarray, count: int) -> np.ndarray:
        h, w = img.shape[:2]
        result = img.copy()

        for _ in range(count):
            cx = random.randint(int(w * 0.2), int(w * 0.8))
            cy = random.randint(int(h * 0.2), int(h * 0.8))

            num_dots = random.randint(3, 8)
            for _ in range(num_dots):
                dx = cx + random.randint(-15, 15)
                dy = cy + random.randint(-15, 15)
                radius = random.randint(1, 4)

                overlay = result.copy()
                cv2.circle(overlay, (dx, dy), radius, (60, 170, 210), -1)

                alpha = random.uniform(0.3, 0.6)
                cv2.circle(result, (dx, dy), radius,
                           (int(60 * alpha + result[dy, dx, 0] * (1 - alpha)),
                            int(170 * alpha + result[dy, dx, 1] * (1 - alpha)),
                            int(210 * alpha + result[dy, dx, 2] * (1 - alpha))), -1)

        return result

    def _add_cotton_wool_spots(self, img: np.ndarray, count: int) -> np.ndarray:
        h, w = img.shape[:2]
        result = img.copy()

        for _ in range(count):
            cx = random.randint(int(w * 0.2), int(w * 0.8))
            cy = random.randint(int(h * 0.2), int(h * 0.8))
            radius = random.randint(5, 15)

            overlay = np.zeros_like(result, dtype=np.float32)
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            mask = np.exp(-(dist ** 2) / (2 * (radius * 0.6) ** 2))

            cwool_color = np.array([70, 160, 200], dtype=np.float32)
            for c in range(3):
                overlay[:, :, c] = mask * cwool_color[c]

            alpha = random.uniform(0.15, 0.35)
            result = cv2.addWeighted(result, 1 - alpha, overlay.astype(np.uint8), alpha, 0)

        return result

    def _add_venous_beading(self, img: np.ndarray, count: int) -> np.ndarray:
        h, w = img.shape[:2]
        result = img.copy()

        for _ in range(count):
            x1 = random.randint(int(w * 0.2), int(w * 0.5))
            y1 = random.randint(int(h * 0.2), int(h * 0.8))
            x2 = x1 + random.randint(30, 80)
            y2 = y1 + random.randint(-20, 20)

            thickness = random.randint(2, 4)
            num_beads = random.randint(4, 8)

            for b in range(num_beads):
                t = b / num_beads
                bx = int(x1 + t * (x2 - x1))
                by = int(y1 + t * (y2 - y1))
                bead_r = random.randint(2, 4)
                cv2.circle(result, (bx, by), bead_r, (10, 15, 90), -1)

        return result

    def _add_neovascularization(self, img: np.ndarray, count: int) -> np.ndarray:
        h, w = img.shape[:2]
        result = img.copy()

        for _ in range(count):
            cx = random.randint(int(w * 0.25), int(w * 0.75))
            cy = random.randint(int(h * 0.25), int(h * 0.75))

            num_branches = random.randint(3, 7)
            for _ in range(num_branches):
                angle = random.uniform(0, 2 * np.pi)
                length = random.randint(10, 30)
                thickness = random.randint(1, 2)

                points = [(cx, cy)]
                x, y = float(cx), float(cy)
                for s in range(5):
                    angle += random.gauss(0, 0.5)
                    x += (length / 5) * np.cos(angle)
                    y += (length / 5) * np.sin(angle)
                    points.append((int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))))

                pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(result, [pts], False, (10, 15, 90), thickness=thickness)

            # Small hemorrhage around neo-vessels
            cv2.circle(result, (cx, cy), random.randint(5, 12), (15, 20, 100), -1)

        return result

    def _add_vitreous_hemorrhage(self, img: np.ndarray, count: int) -> np.ndarray:
        h, w = img.shape[:2]
        result = img.copy()

        for _ in range(count):
            cx = random.randint(int(w * 0.2), int(w * 0.8))
            cy = random.randint(int(h * 0.2), int(h * 0.8))
            radius = random.randint(15, 35)

            overlay = np.zeros_like(result, dtype=np.float32)
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            mask = np.exp(-(dist ** 2) / (2 * (radius * 0.5) ** 2))

            for c in range(3):
                overlay[:, :, c] = mask * [15, 20, 100][c]

            alpha = random.uniform(0.2, 0.45)
            result = cv2.addWeighted(result, 1 - alpha, overlay.astype(np.uint8), alpha, 0)

        return result

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
        return result

    def _apply_vignette(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        X = cv2.getGaussianKernel(w, w * 0.6)
        Y = cv2.getGaussianKernel(h, h * 0.6)
        vignette = Y * X.T
        vignette = vignette / vignette.max()
        vignette = np.power(vignette, 0.7)
        vignette_3ch = cv2.merge([vignette, vignette, vignette])
        return (img.astype(np.float32) * vignette_3ch).astype(np.uint8)

    def _add_noise(self, img: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, 2, img.shape)
        result = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return result


def generate_dataset(output_dir: str | Path, images_per_class: int = 100, seed: int = 42):
    """Generate a full synthetic dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = FundusImageGenerator(seed=seed)

    records = []
    class_names = {
        0: "No Diabetic Retinopathy",
        1: "Mild Diabetic Retinopathy",
        2: "Moderate Diabetic Retinopathy",
        3: "Severe Diabetic Retinopathy",
        4: "Proliferative Diabetic Retinopathy",
    }

    for grade in range(5):
        for i in range(images_per_class):
            filename = f"synth_grade{grade}_{i:04d}.png"
            filepath = output_dir / filename

            img = generator.generate(grade)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(str(filepath))

            records.append({"id_code": filename.replace(".png", ""), "diagnosis": grade})

            if (i + 1) % 25 == 0:
                print(f"  Grade {grade}: {i + 1}/{images_per_class} generated")

    print(f"\nTotal: {len(records)} images generated in {output_dir}")

    csv_path = output_dir.parent / "train.csv"
    with open(csv_path, "w") as f:
        f.write("id_code,diagnosis\n")
        for rec in records:
            f.write(f"{rec['id_code']},{rec['diagnosis']}\n")

    print(f"CSV saved to {csv_path}")
    return records


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    print("Generating realistic synthetic fundus images...")
    print("This creates class-specific pathological features for each DR grade.\n")

    images_per_class = 100
    if len(sys.argv) > 1:
        images_per_class = int(sys.argv[1])

    generate_dataset(data_dir / "train_images", images_per_class=images_per_class)
