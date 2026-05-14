"""Carga y particionado de VideoLSP10: estructura esperada data/processed/<clase>/<seq>.npy."""
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from config import PROCESSED_DIR, CLASSES, SPLIT_TEST, SPLIT_VAL


def cargar_dataset(processed_dir: Path = PROCESSED_DIR):
    X, y = [], []
    for idx, clase in enumerate(CLASSES):
        carpeta = Path(processed_dir) / clase
        if not carpeta.exists():
            continue
        for archivo in sorted(carpeta.glob("*.npy")):
            X.append(np.load(archivo))
            y.append(idx)
    return np.array(X), np.array(y)


def particionar(X, y, semilla=42):
    """Split estratificado 80/10/10."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=SPLIT_TEST, stratify=y, random_state=semilla,
    )
    val_size = SPLIT_VAL / (1 - SPLIT_TEST)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size, stratify=y_tmp, random_state=semilla,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
