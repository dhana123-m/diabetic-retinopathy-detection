"""Extract APTOS 2019 dataset and organize into train_images/."""
import os
import sys
import zipfile
import shutil
import pandas as pd

sys.path.insert(0, str(os.path.dirname(__file__)))
from config import DATA_DIR, TRAIN_IMAGES_DIR

def main():
    zip_path = DATA_DIR / "aptos2019-dataset.zip"
    if not zip_path.exists():
        print(f"ZIP not found: {zip_path}")
        return

    # Clear old synthetic images
    if TRAIN_IMAGES_DIR.exists():
        old_count = len(list(TRAIN_IMAGES_DIR.glob("*.png")))
        print(f"Clearing {old_count} old synthetic images...")
        for f in TRAIN_IMAGES_DIR.glob("*.png"):
            f.unlink()

    # Extract
    print("Extracting ZIP (this may take a minute)...")
    with zipfile.ZipFile(str(zip_path), "r") as z:
        names = z.namelist()
        print(f"  ZIP contains {len(names)} files")

        # Find the image folder
        img_prefix = None
        for name in names:
            if "/" in name and name.endswith(".png"):
                img_prefix = name.split("/")[0]
                break

        if not img_prefix:
            print("  No image folder found in ZIP")
            return

        print(f"  Image folder: {img_prefix}/")

        # Extract only the images
        extracted = 0
        for name in names:
            if name.startswith(img_prefix + "/") and name.endswith(".png"):
                fname = name.split("/")[-1]
                target = TRAIN_IMAGES_DIR / fname
                with z.open(name) as src, open(str(target), "wb") as dst:
                    dst.write(src.read())
                extracted += 1
                if extracted % 500 == 0:
                    print(f"    Extracted {extracted} images...")

        print(f"  Extracted {extracted} images total")

    # Update CSV - ensure it has 'id' column
    csv_path = DATA_DIR / "train.csv"
    if csv_path.exists():
        df = pd.read_csv(str(csv_path))
        if "id_code" in df.columns:
            df = df.rename(columns={"id_code": "id"})
        # Remove .png extension from id if present
        df["id"] = df["id"].str.replace(".png", "", regex=False)
        df.to_csv(str(csv_path), index=False)
        print(f"CSV updated: {len(df)} records")
        for cls in sorted(df["diagnosis"].unique()):
            count = (df["diagnosis"] == cls).sum()
            print(f"  Class {cls}: {count} images")

    # Count actual images
    img_count = len(list(TRAIN_IMAGES_DIR.glob("*.png")))
    print(f"\nFinal image count in train_images/: {img_count}")

    # Cleanup zip
    if img_count >= 3000:
        print("Removing ZIP to save space...")
        zip_path.unlink()
        print("Done!")

if __name__ == "__main__":
    main()
