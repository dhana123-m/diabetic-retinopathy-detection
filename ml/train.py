"""
Advanced Training module for Diabetic Retinopathy Detection.
Implements EfficientNet-B0 transfer learning with:
  - Cosine annealing with warmup
  - Label smoothing
  - Focal loss for class imbalance
  - Two-stage fine-tuning
  - Mixed precision (when GPU available)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras.saving
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard, LearningRateScheduler,
)
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, Input,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    IMAGE_SIZE, BATCH_SIZE, EPOCHS, FINE_TUNE_EPOCHS,
    INITIAL_LEARNING_RATE, FINE_TUNE_LEARNING_RATE,
    DROPOUT_RATE, RANDOM_SEED, NUM_CLASSES, MODEL_PATH,
    MODEL_HISTORY_PATH, MODELS_DIR,
)
from ml.prepare_data import (
    load_csv, validate_images, split_data, create_tf_dataset,
    compute_class_weights,
)
from ml.utils import logger, ensure_directories

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Enable mixed precision for GPU speedup
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision enabled for GPU training.")
    except RuntimeError:
        pass


@keras.saving.register_keras_serializable(package="ml.train")
class FocalLoss(tf.keras.losses.Loss):
    """Focal loss for handling class imbalance."""

    def __init__(self, gamma=2.0, alpha=None, **kwargs):
        kwargs.setdefault("reduction", "sum_over_batch_size")
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self._cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction="none")

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = self._cce(y_true, y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - pt, self.gamma)
        focal_loss = focal_weight * ce
        if self.alpha is not None:
            class_weight = tf.reduce_sum(y_true * self.alpha, axis=-1)
            focal_loss = focal_loss * class_weight
        return focal_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha.tolist() if self.alpha is not None else None,
        })
        return config


class CosineWarmupScheduler:
    """Cosine annealing learning rate scheduler with linear warmup."""

    def __init__(self, warmup_epochs, total_epochs, base_lr, min_lr=1e-7):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def __call__(self, epoch, lr):
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))


def build_model() -> tuple:
    """Build EfficientNet-B0 with improved classification head. Returns (model, base_model)."""
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling=None,
    )

    base_model.trainable = False

    out = base_model.output
    out = GlobalAveragePooling2D()(out)
    out = BatchNormalization()(out)
    out = Dense(512, activation="relu")(out)
    out = BatchNormalization()(out)
    out = Dropout(DROPOUT_RATE)(out)
    out = Dense(256, activation="relu")(out)
    out = BatchNormalization()(out)
    out = Dropout(DROPOUT_RATE * 0.7)(out)
    outputs = Dense(NUM_CLASSES, activation="softmax", dtype="float32")(out)

    model = Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_DR")
    return model, base_model


def get_class_weight_array(class_weights: dict) -> np.ndarray:
    """Convert class weights dict to array for focal loss."""
    arr = np.ones(NUM_CLASSES, dtype=np.float32)
    for k, v in class_weights.items():
        arr[k] = v
    return arr


def train_model() -> dict:
    """Complete training pipeline."""
    ensure_directories(MODELS_DIR, MODELS_DIR.parent / "results")

    total_start = time.time()

    # ── Stage 1: Data ──
    logger.info("=" * 60)
    logger.info("STAGE 1: Loading and preparing data")
    logger.info("=" * 60)

    df = load_csv()
    df = validate_images(df)
    if len(df) < 10:
        raise ValueError("Not enough valid images for training.")

    train_df, val_df = split_data(df)
    class_weights = compute_class_weights(train_df)

    train_ds = create_tf_dataset(train_df, augment=True, shuffle=True)
    val_ds = create_tf_dataset(val_df, augment=False, shuffle=False)

    steps_per_epoch = len(train_df) // BATCH_SIZE
    val_steps = len(val_df) // BATCH_SIZE

    # ── Stage 2: Build Model ──
    logger.info("=" * 60)
    logger.info("STAGE 2: Building EfficientNet-B0 model")
    logger.info("=" * 60)

    model, base_model = build_model()

    focal_alpha = get_class_weight_array(class_weights)
    loss_fn = FocalLoss(gamma=2.0, alpha=focal_alpha)

    lr_schedule = CosineDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        decay_steps=steps_per_epoch * EPOCHS,
        alpha=1e-6,
    )
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )

    trainable_count = sum(p.numpy().size for p in model.trainable_weights)
    total_count = sum(p.numpy().size for p in model.weights)
    logger.info(f"Trainable params: {trainable_count:,} / {total_count:,}")

    # ── Stage 3: Train head ──
    logger.info("=" * 60)
    logger.info("STAGE 3: Training classification head (frozen backbone)")
    logger.info("=" * 60)

    head_callbacks = [
        EarlyStopping(
            monitor="val_accuracy", patience=8,
            restore_best_weights=True, verbose=1,
        ),
        ModelCheckpoint(
            str(MODEL_PATH), monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        LearningRateScheduler(
            CosineWarmupScheduler(
                warmup_epochs=3, total_epochs=EPOCHS,
                base_lr=INITIAL_LEARNING_RATE,
            )
        ),
    ]

    history_head = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=head_callbacks,
        verbose=1,
    )

    head_acc = max(history_head.history["val_accuracy"])
    head_auc = max(history_head.history.get("val_auc", [0]))
    logger.info(f"Head training best - Val Acc: {head_acc:.4f}, Val AUC: {head_auc:.4f}")

    # ── Stage 4: Fine-tune ──
    logger.info("=" * 60)
    logger.info("STAGE 4: Fine-tuning upper layers of backbone")
    logger.info("=" * 60)

    base_model.trainable = True

    unfreeze_from = len(base_model.layers) - 40
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    logger.info(f"Unfrozen backbone layers: {trainable_count}/{len(base_model.layers)}")

    lr_schedule_ft = CosineDecay(
        initial_learning_rate=FINE_TUNE_LEARNING_RATE,
        decay_steps=steps_per_epoch * FINE_TUNE_EPOCHS,
        alpha=1e-8,
    )
    optimizer_ft = Adam(learning_rate=lr_schedule_ft)

    model.compile(
        optimizer=optimizer_ft,
        loss=loss_fn,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )

    finetune_callbacks = [
        EarlyStopping(
            monitor="val_accuracy", patience=10,
            restore_best_weights=True, verbose=1,
        ),
        ModelCheckpoint(
            str(MODEL_PATH), monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        CosineWarmupScheduler(
            warmup_epochs=2, total_epochs=FINE_TUNE_EPOCHS,
            base_lr=FINE_TUNE_LEARNING_RATE,
        ),
    ]

    history_finetune = model.fit(
        train_ds,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_ds,
        callbacks=finetune_callbacks,
        verbose=1,
    )

    ft_acc = max(history_finetune.history["val_accuracy"])
    ft_auc = max(history_finetune.history.get("val_auc", [0]))
    logger.info(f"Fine-tune best - Val Acc: {ft_acc:.4f}, Val AUC: {ft_auc:.4f}")

    # ── Save final model ──
    model.save(str(MODEL_PATH))
    logger.info(f"Model saved to {MODEL_PATH}")

    # ── Save training history ──
    combined_history = {}
    for key in ["accuracy", "val_accuracy", "loss", "val_loss", "auc", "val_auc"]:
        head_vals = [float(x) for x in history_head.history.get(key, [])]
        ft_vals = [float(x) for x in history_finetune.history.get(key, [])]
        combined_history[f"head_{key}"] = head_vals
        combined_history[f"finetune_{key}"] = ft_vals

    with open(MODEL_HISTORY_PATH, "w") as f:
        json.dump(combined_history, f, indent=2)

    total_time = time.time() - total_start
    logger.info(f"Training complete in {total_time/60:.1f} minutes.")
    logger.info(f"Best head accuracy: {head_acc:.4f}")
    logger.info(f"Best fine-tune accuracy: {ft_acc:.4f}")

    return combined_history


if __name__ == "__main__":
    history = train_model()
    logger.info("Training finished successfully.")
