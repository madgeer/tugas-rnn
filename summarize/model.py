"""
Arsitektur Seq2Seq Vanilla RNN (SimpleRNN) dengan TensorFlow/Keras.

Encoder : Embedding -> SimpleRNN -> hidden state
Decoder : Embedding -> SimpleRNN (initial_state = encoder state) -> Dense
Seq2Seq : menggabungkan Encoder dan Decoder; training memakai teacher forcing.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow import keras

from config import ExperimentConfig
from preprocessing import PAD_ID


class Encoder(keras.layers.Layer):
    """Encoder vanilla RNN: Embedding + SimpleRNN."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout)

        self.embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,  # PENTING: RNN mengabaikan posisi <PAD>
            name="encoder_embedding",
        )
        self.dropout = (
            keras.layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        )
        # SimpleRNN murni: satu hidden state, recurrence h_t = tanh(Wx + Uh + b).
        self.rnn = keras.layers.SimpleRNN(
            units=self.hidden_dim,
            activation="tanh",
            return_sequences=False,
            return_state=True,
            name="encoder_rnn",
        )

    def call(self, inputs, training: Optional[bool] = None):
        x = self.embedding(inputs)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        _, state = self.rnn(x, training=training)
        return state

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update(
            {
                "vocab_size": self.vocab_size,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout_rate,
            }
        )
        return cfg


class Decoder(keras.layers.Layer):
    """Decoder vanilla RNN: Embedding + SimpleRNN + Linear(Dense) proyeksi vocab."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout)

        self.embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=False,
            name="decoder_embedding",
        )
        self.dropout = (
            keras.layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        )
        self.rnn = keras.layers.SimpleRNN(
            units=self.hidden_dim,
            activation="tanh",
            return_sequences=True,
            return_state=True,
            name="decoder_rnn",
        )
        self.dense = keras.layers.Dense(
            units=self.vocab_size,
            activation=None,  # logits; loss memakai from_logits=True
            name="decoder_output",
        )

    def call(
        self,
        inputs,
        initial_state,
        training: Optional[bool] = None,
        return_state: bool = False,
    ):
        x = self.embedding(inputs)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        outputs, state = self.rnn(
            x,
            initial_state=initial_state,
            training=training,
        )
        logits = self.dense(outputs)
        if return_state:
            return logits, state
        return logits

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update(
            {
                "vocab_size": self.vocab_size,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout_rate,
            }
        )
        return cfg


class Seq2Seq(keras.Model):
    """Model Seq2Seq yang menggabungkan Encoder dan Decoder."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        pad_id: int = PAD_ID,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = int(pad_id)

    def call(self, inputs, training: Optional[bool] = None):
        encoder_inputs, decoder_inputs = inputs
        state = self.encoder(encoder_inputs, training=training)
        logits = self.decoder(
            decoder_inputs,
            state,
            training=training,
            return_state=False,
        )
        return logits

    def build_with_dummy(self, max_article_len: int, max_dec_len: int) -> None:
        """Bangun bobot model dengan satu batch dummy (berguna sebelum load weights)."""
        dummy_encoder = tf.zeros((1, int(max_article_len)), dtype=tf.int32)
        dummy_decoder = tf.zeros((1, int(max_dec_len)), dtype=tf.int32)
        self((dummy_encoder, dummy_decoder), training=False)


class MaskedSparseCategoricalCrossentropy(keras.losses.Loss):
    """
    Cross-entropy per token, mengabaikan posisi <PAD>.

    Output model berupa logits (batch, time, vocab), target (batch, time).
    """

    def __init__(self, pad_id: int = PAD_ID, name: str = "masked_sparse_ce", **kwargs):
        super().__init__(name=name, reduction="sum", **kwargs)
        self.pad_id = int(pad_id)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        token_loss = keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        mask = tf.cast(tf.not_equal(y_true, self.pad_id), token_loss.dtype)
        denom = tf.maximum(tf.reduce_sum(mask), tf.cast(1.0, token_loss.dtype))
        return tf.reduce_sum(token_loss * mask) / denom

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg["pad_id"] = self.pad_id
        return cfg


class MaskedTokenAccuracy(keras.metrics.Metric):
    """Akurasi prediksi token, hanya pada token non-<PAD>."""

    def __init__(self, pad_id: int = PAD_ID, name: str = "masked_token_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_id = int(pad_id)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        mask = tf.cast(tf.not_equal(y_true, self.pad_id), tf.float32)
        match = tf.cast(tf.equal(y_true, pred), tf.float32)
        self.correct.assign_add(tf.reduce_sum(match * mask))
        self.total.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.correct / tf.maximum(self.total, tf.cast(1.0, self.total.dtype))

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


def build_model(
    vocab_size: int,
    exp_cfg: ExperimentConfig,
    max_article_len: int,
    max_dec_len: int,
) -> Seq2Seq:
    """Buat model Seq2Seq baru sesuai konfigurasi eksperimen."""
    encoder = Encoder(
        vocab_size=vocab_size,
        embedding_dim=exp_cfg.embedding_dim,
        hidden_dim=exp_cfg.hidden_dim,
        dropout=exp_cfg.dropout,
        name="encoder",
    )
    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=exp_cfg.embedding_dim,
        hidden_dim=exp_cfg.hidden_dim,
        dropout=exp_cfg.dropout,
        name="decoder",
    )
    model = Seq2Seq(encoder, decoder, pad_id=PAD_ID, name="seq2seq_vanilla_rnn")
    model.build_with_dummy(max_article_len, max_dec_len)
    return model


def make_optimizer(exp_cfg: ExperimentConfig) -> keras.optimizers.Optimizer:
    """Buat optimizer berdasarkan nama pada konfigurasi."""
    name = exp_cfg.optimizer.lower()
    kwargs = {
        "learning_rate": exp_cfg.learning_rate,
        "clipnorm": exp_cfg.clipnorm,
    }
    if name == "adam":
        return keras.optimizers.Adam(**kwargs)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(**kwargs)
    if name == "sgd":
        return keras.optimizers.SGD(momentum=0.9, nesterov=True, **kwargs)
    raise ValueError(
        f"Optimizer '{exp_cfg.optimizer}' tidak dikenal. Gunakan adam/rmsprop/sgd."
    )


def compile_model(model: Seq2Seq, exp_cfg: ExperimentConfig) -> None:
    model.compile(
        optimizer=make_optimizer(exp_cfg),
        loss=MaskedSparseCategoricalCrossentropy(pad_id=PAD_ID),
        metrics=[MaskedTokenAccuracy(pad_id=PAD_ID)],
    )
