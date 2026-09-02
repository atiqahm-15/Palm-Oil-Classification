"""CORAL ordinal classification head, loss, metric, and label conversion.

Ripeness is ordinal (unripe < underripe < ripe < overripe — CLASS_NAMES in dataset.py is
already in this order), but the baseline MobileNetV2 run treats it as 4 unrelated
categories via plain softmax + categorical cross-entropy, which doesn't encode "underripe
predicted as ripe" as a smaller mistake than "unripe predicted as overripe". This is an
experiment to see whether an ordinal-aware head reduces the underripe<->ripe confusion
found in the baseline's error analysis (see PROJECT_PLAN.md decisions log).

CORAL (Cao, Mirjalili & Raschka, 2019, "Rank Consistent Ordinal Regression for Neural
Networks with Application to Age Estimation") reframes K-class ordinal classification as
K-1 binary "is the true rank greater than k?" tasks sharing all weights except a per-task
bias — the shared-weights-differing-bias construction is what gives it its rank-consistency
guarantee. Implemented here as a single Dense(1, use_bias=False) plus a learned per-task
bias vector.

The saved inference model outputs a plain (batch, 4) per-class probability vector — same
shape/semantics as the baseline's softmax output — via CoralLogitsToProbs, computed from
consecutive differences of the cumulative "P(rank > k)" probabilities. This lets
evaluate.py work completely unchanged for both the categorical baseline and this ordinal
model: it never needs to know which one it's looking at.
"""
import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="ordinal")
class CoralLayer(tf.keras.layers.Layer):
    """Pooled features -> (batch, num_classes - 1) rank-consistent logits."""

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.dense = tf.keras.layers.Dense(1, use_bias=False, name="coral_shared_weight")

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="coral_bias", shape=(self.num_classes - 1,), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        return self.dense(inputs) + self.bias  # (batch, 1) + (K-1,) -> (batch, K-1)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


@keras.saving.register_keras_serializable(package="ordinal")
class CoralLogitsToProbs(tf.keras.layers.Layer):
    """(batch, K-1) rank logits -> (batch, K) per-class probabilities.

    P(class=k) = P(rank>k-1) - P(rank>k), with P(rank>-1) := 1 and P(rank>K-1) := 0.
    Small negative values from imperfect rank-monotonicity are clipped and renormalized.
    """

    def call(self, logits):
        cum_probs = tf.sigmoid(logits)  # (batch, K-1)
        ones = tf.ones_like(cum_probs[:, :1])
        zeros = tf.zeros_like(cum_probs[:, :1])
        full = tf.concat([ones, cum_probs, zeros], axis=1)  # (batch, K+1)
        probs = full[:, :-1] - full[:, 1:]  # (batch, K)
        probs = tf.nn.relu(probs)
        return probs / (tf.reduce_sum(probs, axis=1, keepdims=True) + 1e-8)


@keras.saving.register_keras_serializable(package="ordinal")
def coral_loss(y_true_cumulative, logits):
    """Per-sample summed binary cross-entropy across the K-1 threshold tasks.

    Returns one value per sample (not reduced to a scalar) so Keras can apply
    sample_weight and batch-averaging itself, same contract as built-in losses.
    """
    per_task = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_cumulative, logits=logits)
    return tf.reduce_sum(per_task, axis=1)


@keras.saving.register_keras_serializable(package="ordinal")
class OrdinalAccuracy(tf.keras.metrics.Metric):
    """Accuracy computed by decoding both true and predicted rank from cumulative labels/logits."""

    def __init__(self, name="ordinal_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_rank = tf.reduce_sum(tf.cast(y_true > 0.5, tf.float32), axis=1)
        pred_rank = tf.reduce_sum(tf.cast(tf.sigmoid(y_pred) > 0.5, tf.float32), axis=1)
        matches = tf.cast(tf.equal(true_rank, pred_rank), tf.float32)
        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.size(matches), tf.float32))

    def result(self):
        return self.correct / self.total

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


def onehot_to_cumulative(label_onehot: tf.Tensor) -> tf.Tensor:
    """(batch, K) one-hot -> (batch, K-1) cumulative binary: [rank>0, rank>1, ..., rank>K-2]."""
    num_classes = label_onehot.shape[-1]
    rank = tf.argmax(label_onehot, axis=-1)
    thresholds = tf.range(num_classes - 1, dtype=rank.dtype)
    return tf.cast(rank[:, None] > thresholds[None, :], tf.float32)
