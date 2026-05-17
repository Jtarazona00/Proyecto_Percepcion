"""Data augmentation sobre secuencias de keypoints (30, 258).

Layout de features (definido por src.preprocesamiento.extraccion_keypoints):
    [  0:132]  pose:     33 landmarks x (x, y, z, visibility)
    [132:195]  mano_izq: 21 landmarks x (x, y, z)
    [195:258]  mano_der: 21 landmarks x (x, y, z)
"""
from __future__ import annotations

import numpy as np


# Indices x de cada bloque (para reflejo horizontal y traslacion)
_POSE_X = list(range(0, 132, 4))
_POSE_Y = list(range(1, 132, 4))
_LH_X = list(range(132, 195, 3))
_LH_Y = list(range(133, 195, 3))
_RH_X = list(range(195, 258, 3))
_RH_Y = list(range(196, 258, 3))


def mirror_horizontal(seq: np.ndarray) -> np.ndarray:
    """Refleja la secuencia sobre el eje vertical e intercambia mano izq/der.

    Util para que el modelo aprenda independiente de si la persona es zurda o
    diestra. Aproximadamente duplica la variabilidad del dataset.
    """
    seq = seq.copy()
    # 1. Reflejar coordenadas x: x' = 1 - x (MediaPipe normaliza a [0, 1])
    for i in _POSE_X + _LH_X + _RH_X:
        seq[:, i] = 1.0 - seq[:, i]
    # 2. Intercambiar bloques de mano izquierda y derecha
    lh = seq[:, 132:195].copy()
    seq[:, 132:195] = seq[:, 195:258]
    seq[:, 195:258] = lh
    return seq


def agregar_ruido(seq: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Suma ruido gaussiano pequeño a todas las coordenadas."""
    return seq + np.random.normal(0, sigma, seq.shape).astype(np.float32)


def escalar(seq: np.ndarray, factor: float | None = None) -> np.ndarray:
    """Escala uniformemente todas las coordenadas (simula zoom)."""
    if factor is None:
        factor = float(np.random.uniform(0.95, 1.05))
    return seq * factor


def trasladar(seq: np.ndarray, max_shift: float = 0.05) -> np.ndarray:
    """Traslada la secuencia en X e Y (simula que la persona se movio un poco)."""
    seq = seq.copy()
    dx = float(np.random.uniform(-max_shift, max_shift))
    dy = float(np.random.uniform(-max_shift, max_shift))
    for i in _POSE_X + _LH_X + _RH_X:
        seq[:, i] += dx
    for i in _POSE_Y + _LH_Y + _RH_Y:
        seq[:, i] += dy
    return seq


def frame_dropout(seq: np.ndarray, max_drop: int = 3) -> np.ndarray:
    """Pone a 0 algunos frames al azar (simula oclusion temporal)."""
    seq = seq.copy()
    n_drop = np.random.randint(1, max_drop + 1)
    idx = np.random.choice(seq.shape[0], n_drop, replace=False)
    seq[idx] = 0.0
    return seq


def augmentar_secuencia(seq: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Aplica una composicion aleatoria de augmentations a una secuencia."""
    seq = seq.copy()
    if np.random.rand() < p:
        seq = mirror_horizontal(seq)
    if np.random.rand() < p:
        seq = agregar_ruido(seq)
    if np.random.rand() < p:
        seq = escalar(seq)
    if np.random.rand() < p:
        seq = trasladar(seq)
    if np.random.rand() < p:
        seq = frame_dropout(seq)
    return seq.astype(np.float32)


def expandir_train_set(X_train: np.ndarray, y_train: np.ndarray,
                       factor: int = 4, semilla: int | None = 42):
    """Expande el train set replicandolo `factor` veces con augmentations.

    El primer bloque queda sin augmentar (originales).
    """
    if semilla is not None:
        np.random.seed(semilla)
    X_out = [X_train]
    y_out = [y_train]
    for _ in range(factor - 1):
        X_out.append(np.array([augmentar_secuencia(s) for s in X_train]))
        y_out.append(y_train)
    return np.concatenate(X_out), np.concatenate(y_out)
