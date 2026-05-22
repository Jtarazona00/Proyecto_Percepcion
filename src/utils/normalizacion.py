"""Normalizacion de secuencias de keypoints (T, 258) para invarianza a la
posicion y el tamano de la persona en la camara.

Por cada frame:
- centro  = punto medio de los hombros (pose landmarks 11 y 12)
- escala  = distancia entre hombros
- a las coords (x, y) se les resta el centro y se dividen por la escala;
  z tambien se divide por la escala; visibility queda intacta.

Reglas de robustez:
- Si los hombros no estan (pose ausente en ese frame), el frame se deja igual.
- Los bloques de mano ausentes (todo ceros) se dejan en cero (no se desplazan).

Se aplica DESPUES de la data augmentation, porque el `mirror_horizontal` de
augmentacion.py asume coordenadas MediaPipe en [0, 1] (x' = 1 - x).

Layout de 258 (ver extraccion_keypoints / augmentacion):
    [  0:132] pose     33 x (x, y, z, visibility)
    [132:195] mano izq 21 x (x, y, z)
    [195:258] mano der 21 x (x, y, z)
"""
from __future__ import annotations

import numpy as np

_HOMBRO_IZQ = 11
_HOMBRO_DER = 12


def normalizar_secuencia(seq: np.ndarray) -> np.ndarray:
    """(T, 258) -> (T, 258) normalizada (centrada en hombros + escalada)."""
    out = seq.astype(np.float32).copy()
    for t in range(out.shape[0]):
        pose = out[t, 0:132].reshape(33, 4)
        lh = out[t, 132:195].reshape(21, 3)
        rh = out[t, 195:258].reshape(21, 3)

        ls = pose[_HOMBRO_IZQ, :2]
        rs = pose[_HOMBRO_DER, :2]
        # Sin hombros (pose ausente) -> no tocar este frame
        if not (np.any(ls) or np.any(rs)):
            continue

        centro = (ls + rs) / 2.0
        escala = float(np.linalg.norm(ls - rs))
        if escala < 1e-6:
            escala = 1.0

        # Pose presente -> normalizar las 33 (visibility intacta)
        pose[:, 0] = (pose[:, 0] - centro[0]) / escala
        pose[:, 1] = (pose[:, 1] - centro[1]) / escala
        pose[:, 2] = pose[:, 2] / escala

        # Manos: solo si el bloque tiene datos (no todo ceros)
        for mano in (lh, rh):
            if np.any(mano):
                mano[:, 0] = (mano[:, 0] - centro[0]) / escala
                mano[:, 1] = (mano[:, 1] - centro[1]) / escala
                mano[:, 2] = mano[:, 2] / escala

        out[t, 0:132] = pose.reshape(-1)
        out[t, 132:195] = lh.reshape(-1)
        out[t, 195:258] = rh.reshape(-1)
    return out


def normalizar_batch(X: np.ndarray) -> np.ndarray:
    """(N, T, 258) -> (N, T, 258) normalizada."""
    return np.stack([normalizar_secuencia(s) for s in X]).astype(np.float32)
