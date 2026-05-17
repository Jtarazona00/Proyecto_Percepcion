"""Arquitectura: BiLSTM 128 -> Dropout -> BiLSTM 64 -> Dropout -> Dense 64 -> Dense N.

Usa Bidirectional para que cada LSTM lea la secuencia hacia adelante y hacia
atras. Util en secuencias cortas (30 frames) donde el final de la sena puede
ser tan informativo como el inicio.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input

from config import FRAMES, FEATURES_PER_FRAME, NUM_CLASSES


def construir_modelo(dropout: float = 0.4):
    model = Sequential([
        Input(shape=(FRAMES, FEATURES_PER_FRAME)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(dropout),
        Bidirectional(LSTM(64)),
        Dropout(dropout),
        Dense(64, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax'),
    ])
    return model
