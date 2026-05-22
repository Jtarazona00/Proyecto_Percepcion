"""Arquitectura: BiLSTM 128 -> Dropout -> BiLSTM 64 -> Dropout -> Dense 64 -> Dense N.

Usa Bidirectional para que cada LSTM lea la secuencia hacia adelante y hacia
atras. Util en secuencias cortas (30 frames) donde el final de la sena puede
ser tan informativo como el inicio.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input

import config


def construir_modelo(dropout: float = 0.4, num_classes: int | None = None,
                     frames: int | None = None, features: int | None = None):
    """BiLSTM. Lee config en tiempo de llamada (no en import) para soportar el
    dataset PUCP-305, cuyas clases se setean dinamicamente con config.set_classes.
    """
    nc = num_classes if num_classes is not None else config.NUM_CLASSES
    fr = frames if frames is not None else config.FRAMES
    ft = features if features is not None else config.FEATURES_PER_FRAME
    if nc < 2:
        raise ValueError(
            f"NUM_CLASSES={nc}. Para PUCP-305 llama config.set_classes(...) antes "
            "de construir el modelo, o pasa num_classes explicito."
        )
    model = Sequential([
        Input(shape=(fr, ft)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(dropout),
        Bidirectional(LSTM(64)),
        Dropout(dropout),
        Dense(64, activation='relu'),
        Dense(nc, activation='softmax'),
    ])
    return model
