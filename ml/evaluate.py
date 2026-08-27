"""
Comprehensive Evaluation module for Diabetic Retinopathy Detection.
Generates full metrics, ROC curves, confusion matrix, and per-class analysis.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MODEL_PATH, MODEL_HISTORY_PATH, RESULTS_DIR, NUM_CLASSES,
    CLASS_SHORT_NAMES, BATCH_SIZE,
)
from ml.prepare_data import load_csv, validate_images, split_data, create_data_generator
from ml.utils import logger, ensure_directories
from ml.focal_loss import FocalLoss


def load_trained_model():
    """Load the trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run ml/train.py first."
        )
    model = tf.keras.models.load_model(str(MODEL_PATH), custom_objects={"FocalLoss": FocalLoss})
    logger.info(f"Model loaded from {MODEL_PATH}")
    return model


def evaluate() -> dict:
    """Run full evaluation pipeline."""
    ensure_directories(RESULTS_DIR)

    model = load_trained_model()

    logger.info("Loading validation data...")
    df = load_csv()
    df = validate_images(df)
    _, val_df = split_data(df)

    val_images, val_labels = create_data_generator(val_df, augment=False)
    from tensorflow.keras.applications.efficientnet import preprocess_input
    val_images = preprocess_input(val_images.copy())

    logger.info("Running predictions on validation set...")
    predictions = model.predict(val_images, batch_size=BATCH_SIZE, verbose=1)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = val_labels

    # Compute core metrics
    accuracy = float(accuracy_score(true_labels, pred_labels))
    precision_macro = float(precision_score(true_labels, pred_labels, average="macro", zero_division=0))
    recall_macro = float(recall_score(true_labels, pred_labels, average="macro", zero_division=0))
    f1_macro = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
    f1_weighted = float(f1_score(true_labels, pred_labels, average="weighted", zero_division=0))

    # One-hot encode for ROC/multiclass
    true_onehot = tf.keras.utils.to_categorical(true_labels, NUM_CLASSES)

    report_str = classification_report(
        true_labels, pred_labels,
        target_names=[CLASS_SHORT_NAMES[i] for i in range(NUM_CLASSES)],
        zero_division=0,
    )
    report_dict = classification_report(
        true_labels, pred_labels,
        target_names=[CLASS_SHORT_NAMES[i] for i in range(NUM_CLASSES)],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "total_samples": len(true_labels),
        "classification_report": report_dict,
    }

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision (macro): {precision_macro:.4f}")
    logger.info(f"Recall (macro): {recall_macro:.4f}")
    logger.info(f"F1 (macro): {f1_macro:.4f}")
    logger.info(f"F1 (weighted): {f1_weighted:.4f}")

    # Save metrics
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(report_str)
    logger.info(f"Classification report saved.")

    # Generate all plots
    _plot_confusion_matrix(true_labels, pred_labels)
    _plot_normalized_confusion_matrix(true_labels, pred_labels)
    _plot_roc_curves(true_onehot, predictions)
    _plot_precision_recall_curves(true_onehot, predictions)
    _plot_per_class_performance(report_dict)
    _plot_accuracy_loss()
    _plot_training_summary()

    logger.info(f"All evaluation results saved to {RESULTS_DIR}")
    return metrics


def _plot_confusion_matrix(true_labels, pred_labels):
    """Plot raw confusion matrix."""
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[CLASS_SHORT_NAMES[i] for i in range(NUM_CLASSES)],
        yticklabels=[CLASS_SHORT_NAMES[i] for i in range(NUM_CLASSES)],
    )
    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    logger.info("Confusion matrix saved.")


def _plot_normalized_confusion_matrix(true_labels, pred_labels):
    """Plot normalized confusion matrix (percentages)."""
    cm = confusion_matrix(true_labels, pred_labels, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="RdYlBu_r",
        xticklabels=[CLASS_SHORT_NAMES[i] for i in range(NUM_CLASSES)],
        yticklabels=[CLASS_SHORT_NAMES[i] for i in range(NUM_CLASSES)],
        vmin=0, vmax=1,
    )
    plt.title("Normalized Confusion Matrix", fontsize=16)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix_normalized.png", dpi=150)
    plt.close()
    logger.info("Normalized confusion matrix saved.")


def _plot_roc_curves(true_onehot, predictions):
    """Plot per-class ROC curves and macro-average."""
    plt.figure(figsize=(10, 8))
    colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336", "#9C27B0"]

    macro_auc_sum = 0
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(true_onehot[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)
        macro_auc_sum += roc_auc
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f"{CLASS_SHORT_NAMES[i]} (AUC = {roc_auc:.3f})")

    macro_auc = macro_auc_sum / NUM_CLASSES
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curves (Macro AUC = {macro_auc:.3f})", fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close()
    logger.info(f"ROC curves saved (Macro AUC = {macro_auc:.4f})")


def _plot_precision_recall_curves(true_onehot, predictions):
    """Plot per-class precision-recall curves."""
    plt.figure(figsize=(10, 8))
    colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336", "#9C27B0"]

    for i in range(NUM_CLASSES):
        precision, recall, _ = precision_recall_curve(true_onehot[:, i], predictions[:, i])
        ap = average_precision_score(true_onehot[:, i], predictions[:, i])
        plt.plot(recall, precision, color=colors[i], lw=2,
                 label=f"{CLASS_SHORT_NAMES[i]} (AP = {ap:.3f})")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves", fontsize=16)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "precision_recall_curves.png", dpi=150)
    plt.close()
    logger.info("Precision-recall curves saved.")


def _plot_per_class_performance(report_dict: dict):
    """Plot per-class precision, recall, f1."""
    classes = [CLASS_SHORT_NAMES[i] for i in range(NUM_CLASSES)]
    precisions = [report_dict[c]["precision"] for c in classes if c in report_dict]
    recalls = [report_dict[c]["recall"] for c in classes if c in report_dict]
    f1s = [report_dict[c]["f1-score"] for c in classes if c in report_dict]
    supports = [report_dict[c]["support"] for c in classes if c in report_dict]
    valid_classes = [c for c in classes if c in report_dict]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    x = np.arange(len(valid_classes))
    width = 0.25

    ax1.bar(x - width, precisions, width, label="Precision", color="#2196F3")
    ax1.bar(x, recalls, width, label="Recall", color="#4CAF50")
    ax1.bar(x + width, f1s, width, label="F1-Score", color="#FF9800")
    ax1.set_ylabel("Score")
    ax1.set_title("Per-Class Metrics", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_classes, rotation=15)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(valid_classes, supports, color="#2196F3", alpha=0.7)
    ax2.set_ylabel("Support (Samples)")
    ax2.set_title("Class Distribution in Validation Set", fontsize=14)
    ax2.set_xticklabels(valid_classes, rotation=15)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "per_class_performance.png", dpi=150)
    plt.close()
    logger.info("Per-class performance plot saved.")


def _plot_accuracy_loss():
    """Plot training accuracy and loss curves."""
    if not MODEL_HISTORY_PATH.exists():
        logger.warning("Training history not found, skipping accuracy/loss plot.")
        return

    with open(MODEL_HISTORY_PATH) as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for stage, color_prefix in [("head", "blue"), ("finetune", "red")]:
        acc_key = f"{stage}_accuracy"
        val_acc_key = f"{stage}_val_accuracy"
        loss_key = f"{stage}_loss"
        val_loss_key = f"{stage}_val_loss"

        if acc_key not in history:
            continue

        acc = history[acc_key]
        val_acc = history[val_acc_key]
        loss = history[loss_key]
        val_loss = history[val_loss_key]

        offset = 0 if stage == "head" else len(history.get("head_accuracy", []))
        epochs_range = range(offset + 1, offset + len(acc) + 1)

        axes[0, 0].plot(epochs_range, acc, "b-" if stage == "head" else "r-",
                        label=f"{'Head' if stage == 'head' else 'Fine-tune'} Train")
        axes[0, 1].plot(epochs_range, val_acc, "b-" if stage == "head" else "r-",
                        label=f"{'Head' if stage == 'head' else 'Fine-tune'} Val")
        axes[1, 0].plot(epochs_range, loss, "b-" if stage == "head" else "r-",
                        label=f"{'Head' if stage == 'head' else 'Fine-tune'} Train")
        axes[1, 1].plot(epochs_range, val_loss, "b-" if stage == "head" else "r-",
                        label=f"{'Head' if stage == 'head' else 'Fine-tune'} Val")

    axes[0, 0].set_title("Training Accuracy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Validation Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Training Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Validation Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_loss.png", dpi=150)
    plt.close()
    logger.info("Accuracy/loss plot saved.")


def _plot_training_summary():
    """Create a training summary dashboard."""
    if not MODEL_HISTORY_PATH.exists():
        return

    with open(MODEL_HISTORY_PATH) as f:
        history = json.load(f)

    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy comparison
    labels = ["Head Best", "Fine-tune Best", "Final"]
    head_best = max(history.get("head_val_accuracy", [0]))
    ft_best = max(history.get("finetune_val_accuracy", [0]))
    values = [head_best, ft_best, metrics.get("accuracy", 0)]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = axes[0].bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy Progression")
    axes[0].set_ylim(0, 1.1)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Key metrics
    metric_names = ["Accuracy", "Precision", "Recall", "F1 (macro)", "F1 (weighted)"]
    metric_values = [
        metrics.get("accuracy", 0),
        metrics.get("precision_macro", 0),
        metrics.get("recall_macro", 0),
        metrics.get("f1_macro", 0),
        metrics.get("f1_weighted", 0),
    ]
    colors2 = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
    bars2 = axes[1].barh(metric_names, metric_values, color=colors2, edgecolor="white")
    axes[1].set_xlim(0, 1.1)
    axes[1].set_title("Final Metrics")
    for bar, val in zip(bars2, metric_values):
        axes[1].text(val + 0.02, bar.get_y() + bar.get_height()/2.,
                     f"{val:.3f}", va="center", fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="x")

    # Training info
    head_epochs = len(history.get("head_accuracy", []))
    ft_epochs = len(history.get("finetune_accuracy", []))
    info_text = (
        f"Total Epochs: {head_epochs + ft_epochs}\n"
        f"  Head: {head_epochs} epochs\n"
        f"  Fine-tune: {ft_epochs} epochs\n\n"
        f"Dataset Samples: {metrics.get('total_samples', 'N/A')}\n"
        f"Model: EfficientNet-B0\n"
        f"Loss: Focal Loss (gamma=2.0)\n"
        f"Optimizer: Adam + Cosine Decay"
    )
    axes[2].text(0.1, 0.5, info_text, transform=axes[2].transAxes,
                 fontsize=12, verticalalignment="center",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    axes[2].axis("off")
    axes[2].set_title("Training Summary")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_summary.png", dpi=150)
    plt.close()
    logger.info("Training summary saved.")


if __name__ == "__main__":
    metrics = evaluate()
    print(f"\nEvaluation complete. Accuracy: {metrics['accuracy']:.2%}")
    print(f"F1 Score (macro): {metrics['f1_macro']:.2%}")
