"""Compilacion y entrenamiento del modelo LSTM."""
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import LEARNING_RATE, EPOCHS, BATCH_SIZE, MODEL_PATH
from src.modelos.lstm_modelo import construir_modelo


def entrenar(X_train, y_train, X_val, y_val):
    model = construir_modelo()

    # Compilacion del modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ModelCheckpoint('modelo_lsp_final.keras', monitor='val_accuracy',
                       save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]

    # Entrenamiento
    historial = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return model, historial
