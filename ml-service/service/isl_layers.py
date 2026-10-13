"""Keras custom layers used by the committed ISL classifier.

These definitions mirror the classes in the training notebook.  They live in a
normal Python module so the ``.keras`` artifact can be loaded by the service.

INTEGRATION_WARNING for Person A (B1):
    This file contains model architecture only.  It must not contain dataset
    parsing or preprocessing logic.  If the training notebook changes its
    custom layers, update this module and the model manifest together, then
    rerun the model-load and numerical-parity tests.
"""

from __future__ import annotations

import keras
from keras import layers
import tensorflow as tf


@keras.saving.register_keras_serializable(package="SignBridge")
class LandmarkAugmentation(layers.Layer):
    """Training-only jitter and complete-frame masking from the notebook."""

    def __init__(self, jitter_std: float = 0.005, mask_prob: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.jitter_std = float(jitter_std)
        self.mask_prob = float(mask_prob)

    def call(self, inputs, training=None):
        if not training:
            return inputs
        noise = tf.random.normal(
            tf.shape(inputs), mean=0.0, stddev=self.jitter_std
        )
        augmented = inputs + noise
        time_steps = tf.shape(inputs)[1]
        time_mask = (
            tf.random.uniform((tf.shape(inputs)[0], time_steps, 1))
            > self.mask_prob
        )
        return augmented * tf.cast(time_mask, inputs.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"jitter_std": self.jitter_std, "mask_prob": self.mask_prob})
        return config


@keras.saving.register_keras_serializable(package="SignBridge")
class Conv1DTokenBlock(layers.Layer):
    """Depthwise-separable temporal tokenizer from the training notebook."""

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        dilation_rate: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.dilation_rate = int(dilation_rate)

    def build(self, input_shape):
        self.dw_conv = layers.DepthwiseConv1D(
            kernel_size=self.kernel_size,
            padding="same",
            dilation_rate=self.dilation_rate,
            use_bias=False,
        )
        self.pw_conv = layers.Conv1D(
            self.filters, kernel_size=1, padding="same", use_bias=False
        )
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation("swish")
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.dw_conv(inputs)
        x = self.pw_conv(x)
        x = self.bn(x, training=training)
        return self.act(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "dilation_rate": self.dilation_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="SignBridge")
class TransformerEncoderBlock(layers.Layer):
    """Transformer encoder block from the training notebook."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.dropout_rate = float(dropout)

    def build(self, input_shape):
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(self.ff_dim, activation="swish"),
                layers.Dropout(self.dropout_rate),
                layers.Dense(self.d_model),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs, training=None):
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout_rate,
            }
        )
        return config


CUSTOM_OBJECTS = {
    "LandmarkAugmentation": LandmarkAugmentation,
    "Conv1DTokenBlock": Conv1DTokenBlock,
    "TransformerEncoderBlock": TransformerEncoderBlock,
    "SignBridge>LandmarkAugmentation": LandmarkAugmentation,
    "SignBridge>Conv1DTokenBlock": Conv1DTokenBlock,
    "SignBridge>TransformerEncoderBlock": TransformerEncoderBlock,
}
