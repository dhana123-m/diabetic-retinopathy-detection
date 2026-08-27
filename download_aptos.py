"""Download APTOS 2019 dataset using Bearer token auth."""
import requests
import os
import sys
import zipfile
import io
import pandas as pd

sys.path.insert(0, str(os.path.dirname(__file__)))
from config import DATA_DIR, TRAIN_IMAGES_DIR, TRAIN_CSV

TOKEN = os.environ.get("KAGGLE_API_TOKEN")
if not TOKEN:
    print("ERROR: KAGGLE_API_TOKEN environment variable is not set.")
    print("Set it to your Kaggle API token before running this script.")
    sys.exit(1)
HEADERS = {"Authorization": "Bearer " + TOKEN}
COMP = "aptos2019-blindness-detection"


def download_file(filename, save_path):
    url = f"https://www.kaggle.com/api/v1/competitions/{COMP}/data/download/{filename}"
    print(f"Downloading {filename}...")
    r = requests.get(url, headers=HEADERS, allow_redirects=True, stream=True)
    if r.status_code == 200:
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1024 / 1024:.1f}MB / {total / 1024 / 1024:.1f}MB ({pct:.0f}%)", end="", flush=True)
        print(f"\n  Saved: {save_path} ({downloaded} bytes)")
        return True
    else:
        print(f"  ERROR {r.status_code}: {r.text[:200]}")
        return False


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Download train.csv
    csv_path = DATA_DIR / "aptos_train.csv"
    if not csv_path.exists():
        ok = download_file("train.csv", str(csv_path))
        if not ok:
            print("Failed to download train.csv")
            return
    else:
        print(f"train.csv already exists: {csv_path}")

    # Copy to standard name if needed
    standard_csv = DATA_DIR / "train.csv"
    if not standard_csv.exists() or standard_csv.resolve() != csv_path.resolve():
        df = pd.read_csv(str(csv_path))
        # Standardize column name
        if "id_code" in df.columns:
            df = df.rename(columns={"id_code": "id"})
        df.to_csv(str(standard_csv), index=False)
        print(f"Saved standardized CSV: {standard_csv}")

    df = pd.read_csv(str(standard_csv))
    print(f"Records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    for cls in sorted(df["diagnosis"].unique()):
        count = (df["diagnosis"] == cls).sum()
        print(f"  Class {cls}: {count} images")

    # Download train images zip
    zip_path = DATA_DIR / "train_images.zip"
    img_count = len(list(TRAIN_IMAGES_DIR.glob("*.png")))
    if img_count < 3000:
        ok = download_file("train_images.zip", str(zip_path))
        if not ok:
            print("Failed to download train_images.zip")
            return

        print("Extracting train images...")
        with zipfile.ZipFile(str(zip_path), "r") as z:
            z.extractall(str(DATA_DIR))
        zip_path.unlink(missing_ok=True)
        print("Extraction complete.")
    else:
        print(f"Images already exist: {img_count} files")

    img_count = len(list(TRAIN_IMAGES_DIR.glob("*.png")))
    print(f"\nFinal image count: {img_count}")
    print("Dataset ready for training!")


if __name__ == "__main__":
    main()
