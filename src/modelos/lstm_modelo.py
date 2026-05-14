"""Arquitectura: LSTM 128 -> Dropout -> LSTM 64 -> Dropout -> Dense 64 -> Dense 21."""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

from config import FRAMES, FEATURES_PER_FRAME, NUM_CLASSES


def construir_modelo(dropout=0.3):
    model = Sequential([
        Input(shape=(FRAMES, FEATURES_PER_FRAME)),
        LSTM(128, return_sequences=True),
        Dropout(dropout),
        LSTM(64),
        Dropout(dropout),
        Dense(64, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax'),
    ])
    return model
