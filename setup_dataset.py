"""
Dataset Setup Script for Diabetic Retinopathy Detection.
Downloads and prepares the APTOS 2019 Blindness Detection dataset.
"""

import os
import sys
import subprocess
import zipfile
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, TRAIN_IMAGES_DIR, TRAIN_CSV, TEST_IMAGES_DIR


def check_kaggle_cli():
    """Check if kaggle CLI is installed."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def install_kaggle():
    """Install kaggle CLI."""
    print("[*] Installing kaggle CLI...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    print("[+] kaggle installed.")


def check_kaggle_credentials():
    """Check if Kaggle API credentials exist."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    if kaggle_json.exists():
        return True
    return False


def setup_kaggle_credentials():
    """Guide user to set up Kaggle credentials."""
    print()
    print("=" * 60)
    print("  KAGGLE API SETUP REQUIRED")
    print("=" * 60)
    print()
    print("  To download the APTOS 2019 dataset:")
    print()
    print("  1. Go to https://www.kaggle.com/account/login")
    print("  2. Click 'Create New API Token' (Downloads kaggle.json)")
    print("  3. Place kaggle.json in:", Path.home() / ".kaggle")
    print()
    print("  Then run this script again.")
    print("=" * 60)
    print()


def download_dataset():
    """Download APTOS 2019 dataset using Kaggle API."""
    competition = "aptos2019-blindness-detection"

    print(f"[*] Downloading {competition} dataset...")

    # Download train images
    print("[*] Downloading training images...")
    subprocess.check_call([
        "kaggle", "competitions", "download", "-c", competition,
        "-f", "train_images.zip",
        "-p", str(DATA_DIR),
    ])

    # Download train CSV
    print("[*] Downloading training labels...")
    subprocess.check_call([
        "kaggle", "competitions", "download", "-c", competition,
        "-f", "train.csv",
        "-p", str(DATA_DIR),
    ])

    print("[+] Download complete.")


def extract_dataset():
    """Extract downloaded zip files."""
    train_images_zip = DATA_DIR / "train_images.zip"
    if train_images_zip.exists():
        print("[*] Extracting train images...")
        with zipfile.ZipFile(str(train_images_zip), "r") as zip_ref:
            zip_ref.extractall(str(DATA_DIR))
        train_images_zip.unlink()
        print("[+] Train images extracted.")

    # Move images to correct directory if needed
    extracted_dir = DATA_DIR / "train_images"
    if not extracted_dir.exists():
        # Check if images were extracted to a different location
        for item in DATA_DIR.iterdir():
            if item.is_dir() and item.name != "train_images":
                images = list(item.glob("*.png")) + list(item.glob("*.jpg"))
                if images:
                    extracted_dir.mkdir(parents=True, exist_ok=True)
                    for img in images:
                        img.rename(extracted_dir / img.name)
                    item.rmdir()


def verify_dataset():
    """Verify dataset is properly set up."""
    print("[*] Verifying dataset...")

    issues = []

    if not TRAIN_CSV.exists():
        issues.append(f"Missing: {TRAIN_CSV}")
    else:
        df = _quick_csv_check()
        if df is not None:
            print(f"    CSV: {len(df)} records found")
            print(f"    Classes: {sorted(df['diagnosis'].unique())}")
            for cls in sorted(df["diagnosis"].unique()):
                count = len(df[df["diagnosis"] == cls])
                print(f"      Class {cls}: {count} images")

    if not TRAIN_IMAGES_DIR.exists():
        issues.append(f"Missing directory: {TRAIN_IMAGES_DIR}")
    else:
        image_count = len(list(TRAIN_IMAGES_DIR.glob("*.*")))
        print(f"    Images: {image_count} files in train_images/")

    if issues:
        print("\n[!] Issues found:")
        for issue in issues:
            print(f"    - {issue}")
        return False

    print("[+] Dataset verified successfully!")
    return True


def _quick_csv_check():
    """Quick CSV validation."""
    try:
        import pandas as pd
        df = pd.read_csv(str(TRAIN_CSV))
        if "id" not in df.columns or "diagnosis" not in df.columns:
            print("[!] CSV missing required columns 'id' and 'diagnosis'")
            return None
        return df
    except Exception as e:
        print(f"[!] Error reading CSV: {e}")
        return None


def create_sample_dataset():
    """
    Create a small synthetic dataset for testing the pipeline.
    This generates colored retinal-like images with labels.
    """
    import numpy as np
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        from PIL import Image, ImageDraw

    print("[*] Creating sample dataset for pipeline testing...")

    TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    np.random.seed(42)

    samples_per_class = 20
    for class_id in range(5):
        for i in range(samples_per_class):
            img_id = f"sample_{class_id}_{i:03d}"

            # Create retinal-like synthetic image
            img = Image.new("RGB", (512, 512), color=(20, 10, 5))
            draw = ImageDraw.Draw(img)

            # Draw circular retinal background
            draw.ellipse([64, 64, 448, 448], fill=(180, 80, 30))

            # Draw optic disc
            cx, cy = 320, 256
            draw.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], fill=(240, 200, 100))

            # Add class-specific features
            rng = np.random.RandomState(class_id * 1000 + i)

            if class_id >= 1:
                # Microaneurysms (small red dots)
                for _ in range(class_id * 5):
                    x = rng.randint(100, 412)
                    y = rng.randint(100, 412)
                    r = rng.randint(2, 5)
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=(200, 20, 20))

            if class_id >= 2:
                # Hard exudates (yellow patches)
                for _ in range(class_id * 3):
                    x = rng.randint(120, 392)
                    y = rng.randint(120, 392)
                    r = rng.randint(4, 10)
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=(240, 220, 80))

            if class_id >= 3:
                # Hemorrhages (larger dark red)
                for _ in range((class_id - 2) * 4):
                    x = rng.randint(100, 412)
                    y = rng.randint(100, 412)
                    r = rng.randint(5, 12)
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=(150, 30, 30))

            if class_id >= 4:
                # Neovascularization (new vessels)
                for _ in range(6):
                    x1 = rng.randint(100, 412)
                    y1 = rng.randint(100, 412)
                    x2 = x1 + rng.randint(-60, 60)
                    y2 = y1 + rng.randint(-60, 60)
                    draw.line([x1, y1, x2, y2], fill=(200, 50, 50), width=2)

            # Add noise
            img_array = np.array(img)
            noise = rng.normal(0, 8, img_array.shape).astype(np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

            img.save(str(TRAIN_IMAGES_DIR / f"{img_id}.png"))
            records.append({"id": img_id, "diagnosis": class_id})

    # Write CSV
    with open(str(TRAIN_CSV), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "diagnosis"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[+] Sample dataset created: {len(records)} images across 5 classes")
    print(f"    Images: {TRAIN_IMAGES_DIR}")
    print(f"    CSV: {TRAIN_CSV}")


def main():
    print("=" * 60)
    print("  Diabetic Retinopathy Dataset Setup")
    print("=" * 60)
    print()

    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Check if dataset already exists
    if verify_dataset():
        print("\n[+] Dataset is ready for training!")
        return

    # Try Kaggle download
    print()
    print("Choose dataset option:")
    print("  [1] Download APTOS 2019 from Kaggle (requires API key)")
    print("  [2] Create sample dataset for testing pipeline")
    print()

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        if not check_kaggle_cli():
            install_kaggle()
        if not check_kaggle_credentials():
            setup_kaggle_credentials()
            return
        download_dataset()
        extract_dataset()
        verify_dataset()
    elif choice == "2":
        create_sample_dataset()
    else:
        print("Invalid choice.")
        return

    print("\n[+] Setup complete! Run 'python ml/train.py' to start training.")


if __name__ == "__main__":
    main()
