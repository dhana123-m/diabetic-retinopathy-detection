"""
Focal loss for class imbalance.

Defined in its own module (instead of ml.train) so lightweight consumers
such as the prediction API can register the custom object without pulling
in the full training stack (pandas/sklearn).
"""

import keras.saving
import tensorflow as tf


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