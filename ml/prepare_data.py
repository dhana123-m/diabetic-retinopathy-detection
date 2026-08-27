"""
Data preparation module for Diabetic Retinopathy Detection.
Handles dataset loading, tf.data pipelines, and advanced augmentation.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_DIR, TRAIN_CSV, TRAIN_IMAGES_DIR, IMAGE_SIZE,
    BATCH_SIZE, NUM_CLASSES, VALIDATION_SPLIT, RANDOM_SEED,
    SUPPORTED_EXTENSIONS, AUGMENTATION_CONFIG,
)
from ml.utils import logger, ensure_directories


def load_csv() -> pd.DataFrame:
    """Load and validate the training CSV."""
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Training CSV not found at {TRAIN_CSV}. "
            "Run 'python setup_dataset.py' to download the dataset."
        )
    df = pd.read_csv(str(TRAIN_CSV))
    if "id" not in df.columns or "diagnosis" not in df.columns:
        raise ValueError("CSV must contain 'id' and 'diagnosis' columns.")
    logger.info(f"Loaded CSV with {len(df)} records.")
    dist = df["diagnosis"].value_counts().sort_index()
    for cls, count in dist.items():
        logger.info(f"  Class {cls}: {count} images ({count/len(df)*100:.1f}%)")
    return df


def validate_images(df: pd.DataFrame) -> pd.DataFrame:
    """Remove entries with missing or corrupted image files."""
    valid_rows = []
    for _, row in df.iterrows():
        image_path = _find_image_path(row["id"])
        if image_path is None:
            continue
        if not _is_valid_image(image_path):
            continue
        valid_rows.append(row)
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    removed = len(df) - len(valid_df)
    if removed > 0:
        logger.warning(f"Removed {removed} invalid/missing images.")
    logger.info(f"Valid images: {len(valid_df)}")
    return valid_df


def _find_image_path(image_id) -> Path | None:
    """Find image file given an id, checking multiple extensions."""
    for ext in SUPPORTED_EXTENSIONS:
        path = TRAIN_IMAGES_DIR / f"{image_id}{ext}"
        if path.exists():
            return path
    return None


def _is_valid_image(path: Path) -> bool:
    """Check if an image file can be read by OpenCV."""
    try:
        img = cv2.imread(str(path))
        return img is not None and img.size > 0
    except Exception:
        return False


def compute_class_weights(df: pd.DataFrame) -> dict:
    """Compute balanced class weights for imbalanced datasets."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.array(sorted(df["diagnosis"].unique()))
    weights = compute_class_weight("balanced", classes=classes, y=df["diagnosis"].values)
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Class weights: {weight_dict}")
    return weight_dict


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train-validation split."""
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_SPLIT,
        stratify=df["diagnosis"],
        random_state=RANDOM_SEED,
    )
    logger.info(f"Train: {len(train_df)}, Validation: {len(val_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _load_single_image(image_path, label):
    """Load and preprocess a single image."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32)
    return img, label


def _augment_image(image, label):
    """Apply random augmentations to an image."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.image.random_saturation(image, 0.85, 1.15)
    image = tf.image.random_hue(image, 0.05)

    if tf.random.uniform([]) > 0.7:
        crop_size = int(IMAGE_SIZE * 0.85)
        padded = tf.image.resize_with_crop_or_pad(
            image,
            IMAGE_SIZE + crop_size // 4,
            IMAGE_SIZE + crop_size // 4,
        )
        image = tf.image.random_crop(padded, [IMAGE_SIZE, IMAGE_SIZE, 3])

    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def _preprocess_efficientnet(image, label):
    """Apply EfficientNet preprocessing."""
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label


def create_tf_dataset(
    df: pd.DataFrame,
    augment: bool = False,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """Create an optimized tf.data pipeline."""
    image_paths = []
    labels = []
    for _, row in df.iterrows():
        path = _find_image_path(row["id"])
        if path is not None:
            image_paths.append(str(path))
            labels.append(int(row["diagnosis"]))

    labels_onehot = tf.keras.utils.to_categorical(labels, NUM_CLASSES)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels_onehot))
    dataset = dataset.map(
        _load_single_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=RANDOM_SEED)

    if augment:
        dataset = dataset.map(
            _augment_image,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.map(
        _preprocess_efficientnet,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_data_generator(df: pd.DataFrame, augment: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load all images and labels into numpy arrays (fallback for non-tf.data usage)."""
    from tensorflow.keras.applications.efficientnet import preprocess_input

    images = []
    labels = []
    for _, row in df.iterrows():
        image_path = _find_image_path(row["id"])
        if image_path is None:
            continue
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        images.append(img)
        labels.append(int(row["diagnosis"]))

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return images, labels


if __name__ == "__main__":
    ensure_directories(DATA_DIR)
    df = load_csv()
    df = validate_images(df)
    train_df, val_df = split_data(df)
    train_ds = create_tf_dataset(train_df, augment=True, shuffle=True)
    val_ds = create_tf_dataset(val_df, augment=False, shuffle=False)
    print(f"\nDataset ready: {len(train_df)} train, {len(val_df)} validation.")
    print(f"Train batches: {len(train_ds)}, Val batches: {len(val_ds)}")
