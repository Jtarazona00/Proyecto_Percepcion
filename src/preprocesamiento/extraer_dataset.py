"""Procesa carpetas con frames .jpg del dataset VideoLSP10 -> arrays (30, 258).

Estructura esperada de entrada: <ruta>/<clase>_r.<id>/<frame>.jpg
Estructura de salida: <salida>/<clase>/<id>.npy
"""
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from config import FRAMES, FEATURES_PER_FRAME, CLASSES, PROCESSED_DIR
from .extraccion_keypoints import crear_holistic, extraer_keypoints


def parse_nombre_carpeta(nombre: str):
    """Carpeta '<clase>_r.<id>' -> (clase, id_int) o (None, None) si no matchea."""
    partes = nombre.rsplit("_r.", 1)
    if len(partes) != 2:
        return None, None
    try:
        return partes[0], int(partes[1])
    except ValueError:
        return None, None


def procesar_carpeta_jpg(carpeta: Path, holistic) -> Optional[np.ndarray]:
    """Lee imagenes .jpg de la carpeta, muestrea uniformemente a FRAMES,
    pasa cada frame por MediaPipe y devuelve array (FRAMES, FEATURES_PER_FRAME).
    """
    archivos = sorted(
        carpeta.glob("*.jpg"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )
    n = len(archivos)
    if n == 0:
        return None
    indices = np.linspace(0, n - 1, FRAMES).astype(int)
    secuencia = []
    for i in indices:
        img = cv2.imread(str(archivos[i]))
        if img is None:
            secuencia.append(np.zeros(FEATURES_PER_FRAME))
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        resultados = holistic.process(rgb)
        secuencia.append(extraer_keypoints(resultados))
    return np.array(secuencia, dtype=np.float32)


def procesar_dataset(carpeta_rgb, salida=PROCESSED_DIR, verbose: bool = True):
    """Procesa todas las subcarpetas <clase>_r.<id>/ -> .npy en salida/<clase>/<id>.npy.

    Es resumible: salta las muestras cuyo .npy ya existe.
    """
    carpeta_rgb = Path(carpeta_rgb)
    salida = Path(salida)
    for clase in CLASSES:
        (salida / clase).mkdir(parents=True, exist_ok=True)

    tareas = []
    for c in sorted(carpeta_rgb.iterdir()):
        if not c.is_dir():
            continue
        clase, idx = parse_nombre_carpeta(c.name)
        if clase not in CLASSES:
            continue
        destino = salida / clase / f"{idx}.npy"
        if not destino.exists():
            tareas.append((c, destino))

    if verbose:
        print(f"Pendientes: {len(tareas)} secuencias")

    try:
        from tqdm import tqdm
        iterator = tqdm(tareas, desc="MediaPipe")
    except ImportError:
        iterator = tareas

    with crear_holistic() as holistic:
        for carpeta, destino in iterator:
            arr = procesar_carpeta_jpg(carpeta, holistic)
            if arr is not None and arr.shape == (FRAMES, FEATURES_PER_FRAME):
                np.save(destino, arr)

    if verbose:
        print("Extraccion completa.")
