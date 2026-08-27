"""Train on real APTOS 2019 data with reduced epochs for CPU."""
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
)
from ml.train import FocalLoss, CosineWarmupScheduler, build_model, get_class_weight_array
from ml.prepare_data import (
    load_csv, validate_images, split_data, create_tf_dataset, compute_class_weights,
)
from ml.utils import logger, ensure_directories
from config import (
    IMAGE_SIZE, BATCH_SIZE, MODEL_PATH, MODEL_HISTORY_PATH, MODELS_DIR,
)

tf.random.set_seed(42)
np.random.seed(42)

import json

# Reduced epochs for CPU training on real data
HEAD_EPOCHS = 15
FINETUNE_EPOCHS = 10
HEAD_LR = 1e-3
FINETUNE_LR = 5e-6


def train():
    ensure_directories(MODELS_DIR, MODELS_DIR.parent / "results")
    total_start = time.time()

    # Load data
    logger.info("=" * 60)
    logger.info("Loading APTOS 2019 data...")
    logger.info("=" * 60)
    df = load_csv()
    df = validate_images(df)
    logger.info(f"Valid images: {len(df)}")
    for cls in sorted(df["diagnosis"].unique()):
        count = (df["diagnosis"] == cls).sum()
        logger.info(f"  Class {cls}: {count} ({count/len(df)*100:.1f}%)")

    train_df, val_df = split_data(df)
    class_weights = compute_class_weights(train_df)

    train_ds = create_tf_dataset(train_df, augment=True, shuffle=True)
    val_ds = create_tf_dataset(val_df, augment=False, shuffle=False)

    steps_per_epoch = len(train_df) // BATCH_SIZE
    val_steps = len(val_df) // BATCH_SIZE
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Steps/epoch: {steps_per_epoch}")

    # Build model
    logger.info("=" * 60)
    logger.info("Building EfficientNet-B0 model...")
    logger.info("=" * 60)
    model, base_model = build_model()

    focal_alpha = get_class_weight_array(class_weights)
    loss_fn = FocalLoss(gamma=2.0, alpha=focal_alpha)

    # Stage 1: Train head (frozen backbone)
    logger.info("=" * 60)
    logger.info(f"STAGE 1: Training head ({HEAD_EPOCHS} epochs, frozen backbone)")
    logger.info("=" * 60)

    optimizer = tf.keras.optimizers.Adam(learning_rate=HEAD_LR)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )

    head_callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.LearningRateScheduler(
            CosineWarmupScheduler(warmup_epochs=3, total_epochs=HEAD_EPOCHS, base_lr=HEAD_LR)
        ),
    ]

    history_head = model.fit(
        train_ds,
        epochs=HEAD_EPOCHS,
        validation_data=val_ds,
        callbacks=head_callbacks,
        verbose=1,
    )

    head_acc = max(history_head.history["val_accuracy"])
    logger.info(f"Head training best Val Acc: {head_acc:.4f}")

    # Stage 2: Fine-tune backbone
    logger.info("=" * 60)
    logger.info(f"STAGE 2: Fine-tuning backbone ({FINETUNE_EPOCHS} epochs)")
    logger.info("=" * 60)

    base_model.trainable = True
    unfreeze_from = len(base_model.layers) - 40
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    trainable = sum(1 for l in base_model.layers if l.trainable)
    logger.info(f"Unfrozen: {trainable}/{len(base_model.layers)} backbone layers")

    lr_schedule_ft = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=FINETUNE_LR,
        decay_steps=steps_per_epoch * FINETUNE_EPOCHS,
        alpha=1e-8,
    )
    optimizer_ft = tf.keras.optimizers.Adam(learning_rate=lr_schedule_ft)

    model.compile(
        optimizer=optimizer_ft,
        loss=loss_fn,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )

    ft_callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    history_ft = model.fit(
        train_ds,
        epochs=FINETUNE_EPOCHS,
        validation_data=val_ds,
        callbacks=ft_callbacks,
        verbose=1,
    )

    ft_acc = max(history_ft.history["val_accuracy"])
    logger.info(f"Fine-tune best Val Acc: {ft_acc:.4f}")

    # Save
    model.save(str(MODEL_PATH))
    logger.info(f"Model saved: {MODEL_PATH}")

    combined = {}
    for key in ["accuracy", "val_accuracy", "loss", "val_loss", "auc", "val_auc"]:
        head_vals = [float(x) for x in history_head.history.get(key, [])]
        ft_vals = [float(x) for x in history_ft.history.get(key, [])]
        combined[f"head_{key}"] = head_vals
        combined[f"finetune_{key}"] = ft_vals

    with open(MODEL_HISTORY_PATH, "w") as f:
        json.dump(combined, f, indent=2)

    total_time = time.time() - total_start
    logger.info(f"Training complete in {total_time/60:.1f} minutes.")
    logger.info(f"Best head acc: {head_acc:.4f}, Best fine-tune acc: {ft_acc:.4f}")


if __name__ == "__main__":
    train()
