"""Pipeline para el dataset PUCP-305 (videos LSP + anotaciones ELAN).

Flujo:
1. Explorar el .zip extraido para encontrar pares (video, .eaf).
2. Parsear cada .eaf para obtener glosas con intervalos de tiempo.
3. Segmentar cada video por intervalo con OpenCV.
4. Pasar cada segmento por MediaPipe Holistic -> (30, 258).
5. Filtrar glosas con menos de MIN_MUESTRAS_POR_CLASE para evitar clases inutiles.
6. Guardar como <salida>/<glosa>/<idx>.npy.

Es resumible: salta los .npy ya existentes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import cv2

from config import FRAMES, FEATURES_PER_FRAME, MIN_MUESTRAS_POR_CLASE
from .extraccion_keypoints import crear_holistic, extraer_keypoints
from .parse_elan import parsear_eaf, encontrar_video_asociado


def encontrar_pares_video_eaf(raiz: Path):
    """Recorre la carpeta extraida y devuelve lista de (video_path, eaf_path)."""
    raiz = Path(raiz)
    pares = []
    eafs = list(raiz.rglob("*.eaf"))
    for eaf in eafs:
        video = encontrar_video_asociado(eaf, carpeta_base=raiz)
        if video is not None:
            pares.append((video, eaf))
    return pares


def descubrir_glosas(pares, tiers: Iterable[str] | None = None,
                     min_muestras: int = MIN_MUESTRAS_POR_CLASE):
    """Hace una pasada sobre todos los .eaf y devuelve:

    - conteo: Counter glosa -> # apariciones totales
    - glosas_validas: lista de glosas con >= min_muestras

    No segmenta video todavia. Util para decidir vocabulario antes de procesar.
    """
    conteo: Counter = Counter()
    for _, eaf in pares:
        for anot in parsear_eaf(eaf, tiers_interes=tiers):
            conteo[anot["glosa"]] += 1
    glosas_validas = sorted(
        g for g, n in conteo.items() if n >= min_muestras
    )
    return conteo, glosas_validas


def _muestrear_segmento(video_cap, inicio_ms: int, fin_ms: int,
                        frames_objetivo: int = FRAMES):
    """Lee un segmento de video y devuelve `frames_objetivo` frames muestreados."""
    duracion = max(fin_ms - inicio_ms, 1)
    times = np.linspace(inicio_ms, fin_ms, frames_objetivo).astype(int)
    frames = []
    for t in times:
        video_cap.set(cv2.CAP_PROP_POS_MSEC, float(t))
        ok, frame = video_cap.read()
        if not ok or frame is None:
            return None
        frames.append(frame)
    return frames


def procesar_dataset_pucp305(raiz_extraida: Path, salida: Path,
                              tiers: Iterable[str] | None = None,
                              glosas_permitidas: Sequence[str] | None = None,
                              max_muestras_por_clase: int | None = None,
                              verbose: bool = True):
    """Procesa todo el dataset PUCP-305 -> .npy por glosa.

    - raiz_extraida: carpeta donde se descomprimio el zip
    - salida: donde guardar los .npy (por ejemplo, Drive/PUCP305_processed)
    - tiers: filtra a estos tiers ELAN (None = todos)
    - glosas_permitidas: filtra a estas glosas (None = todas las validas)
    - max_muestras_por_clase: tope opcional para balancear el dataset

    Devuelve dict glosa -> n_muestras_guardadas.
    """
    raiz_extraida = Path(raiz_extraida)
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)

    pares = encontrar_pares_video_eaf(raiz_extraida)
    if verbose:
        print(f"Encontrados {len(pares)} pares (video, .eaf).")

    if not pares:
        return {}

    # Pre-conteo para reportar y validar
    conteo, glosas_validas = descubrir_glosas(
        pares, tiers=tiers, min_muestras=MIN_MUESTRAS_POR_CLASE
    )
    if verbose:
        print(f"Total glosas unicas: {len(conteo)}")
        print(f"Glosas con >= {MIN_MUESTRAS_POR_CLASE} muestras: {len(glosas_validas)}")

    if glosas_permitidas:
        permitidas = {g.upper() for g in glosas_permitidas}
    else:
        permitidas = set(glosas_validas)

    # Importacion local de tqdm para no forzar dep en src/
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):  # noqa: E306
            return x

    # Crear subcarpetas
    for g in permitidas:
        (salida / g).mkdir(parents=True, exist_ok=True)

    guardados: Counter = Counter()

    with crear_holistic() as holistic:
        for video_path, eaf_path in tqdm(pares, desc="Videos"):
            anotaciones = [a for a in parsear_eaf(eaf_path, tiers_interes=tiers)
                            if a["glosa"] in permitidas]
            if not anotaciones:
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue

            try:
                for anot in anotaciones:
                    glosa = anot["glosa"]

                    # Tope por clase
                    if (max_muestras_por_clase is not None
                            and guardados[glosa] >= max_muestras_por_clase):
                        continue

                    # Idempotencia: nombre = stem_video__inicio_ms.npy
                    nombre = f"{video_path.stem}__{anot['inicio_ms']}.npy"
                    destino = salida / glosa / nombre
                    if destino.exists():
                        guardados[glosa] += 1
                        continue

                    frames = _muestrear_segmento(
                        cap, anot["inicio_ms"], anot["fin_ms"]
                    )
                    if frames is None:
                        continue

                    secuencia = []
                    for frame in frames:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb.flags.writeable = False
                        resultados = holistic.process(rgb)
                        secuencia.append(extraer_keypoints(resultados))
                    arr = np.array(secuencia, dtype=np.float32)

                    if arr.shape == (FRAMES, FEATURES_PER_FRAME):
                        np.save(destino, arr)
                        guardados[glosa] += 1
            finally:
                cap.release()

    if verbose:
        print(f"\nClases guardadas: {len(guardados)}")
        for g, n in guardados.most_common(10):
            print(f"  {g}: {n}")

    return dict(guardados)
