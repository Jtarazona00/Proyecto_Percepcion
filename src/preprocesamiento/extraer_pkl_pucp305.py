"""Extractor para el paquete PUCP-305 con MediaPipe YA precalculado (.pkl).

A diferencia de `extraer_dataset_pucp305.py` (que asume videos crudos + ELAN y
corre MediaPipe), este modulo consume el paquete `13691887/`:

- ANNOTATIONS.zip   -> train/val/test (FILENAME,CLASS_ID, SIN header)
- MEDIAPIPE/<FILENAME>.pkl -> lista de frames; cada frame es un dict
      {'pose': PoseLandmarkerResult,         # MediaPipe Tasks
       'hands': HandLandmarkerResult,        # MediaPipe Tasks
       'holistic_legacy': {                  # MediaPipe Holistic clasico (protobuf)
           'pose_landmarks': NormalizedLandmarkList | None,   # 33 (x,y,z,visibility)
           'left_hand_landmarks': NormalizedLandmarkList | None,  # 21 (x,y,z)
           'right_hand_landmarks': NormalizedLandmarkList | None, # 21 (x,y,z)
           ... }}

Produce el MISMO vector de 258 features por frame que
`extraccion_keypoints.extraer_keypoints`, para reutilizar el modelo y la
augmentation existentes:
    [  0:132] pose 33 x (x,y,z,visibility)
    [132:195] mano izq 21 x (x,y,z)
    [195:258] mano der 21 x (x,y,z)

IMPORTANTE: des-picklear los .pkl requiere `mediapipe` instalado (los objetos
son clases de mediapipe). Por eso NO se importa mediapipe aqui a nivel de
modulo (asi el archivo es importable sin mediapipe), pero `pickle.load` SI lo
necesitara en tiempo de ejecucion (Colab).
"""
from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

POSE_N = 33
HAND_N = 21
FEATURES = POSE_N * 4 + HAND_N * 3 + HAND_N * 3  # 258
ID_NA_DEFECTO = 222


# ----------------------------------------------------------------------------
# 1) Limpieza de anotaciones: quita la clase #N/A y remapea IDs a 0..N-1
# ----------------------------------------------------------------------------
def _leer_split_zip(ann_zip: Path, nombre: str) -> pd.DataFrame:
    with zipfile.ZipFile(ann_zip) as z:
        data = z.read(f"ANNOTATIONS/{nombre}_labels.csv")
    return pd.read_csv(io.BytesIO(data), header=None, names=["FILENAME", "CLASS_ID"])


def limpiar_anotaciones(ann_zip: Path, ref_csv: Path, id_na: int = ID_NA_DEFECTO):
    """Devuelve (splits, labels_map).

    - splits: dict {'train','val','test'} -> DataFrame con FILENAME,CLASS_ID remapeado a 0..N-1
    - labels_map: DataFrame NEW_ID,OLD_ID,LABEL (ordenado por NEW_ID), sin la clase #N/A

    `ref_csv` = videos_ref_annotations.csv (FILENAME,CLASS_ID,LABEL).
    """
    ann_zip = Path(ann_zip)
    ref = pd.read_csv(ref_csv)

    ids_validos = sorted(i for i in ref.CLASS_ID.unique() if i != id_na)
    old2new = {old: new for new, old in enumerate(ids_validos)}

    labels_map = ref[ref.CLASS_ID != id_na].copy()
    labels_map["NEW_ID"] = labels_map.CLASS_ID.map(old2new)
    labels_map = (labels_map.rename(columns={"CLASS_ID": "OLD_ID"})
                  [["NEW_ID", "OLD_ID", "LABEL"]]
                  .sort_values("NEW_ID")
                  .reset_index(drop=True))

    splits = {}
    for s in ["train", "val", "test"]:
        df = _leer_split_zip(ann_zip, s)
        df = df[df.CLASS_ID != id_na].copy()
        df["CLASS_ID"] = df.CLASS_ID.map(old2new)
        splits[s] = df.reset_index(drop=True)

    return splits, labels_map


# ----------------------------------------------------------------------------
# 2) Extraccion de features desde un frame del pkl
# ----------------------------------------------------------------------------
def _flatten_landmarks(landmark_list, n: int, dims: int) -> np.ndarray:
    """landmark_list = protobuf con .landmark, o None -> vector plano de n*dims."""
    if landmark_list is None:
        return np.zeros(n * dims, dtype=np.float32)
    lms = landmark_list.landmark
    if dims == 4:
        arr = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in lms],
                       dtype=np.float32)
    else:
        arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
    if arr.shape[0] != n:  # defensivo: rellena/recorta a n landmarks
        out = np.zeros((n, dims), dtype=np.float32)
        out[:min(n, arr.shape[0])] = arr[:n]
        arr = out
    return arr.flatten()


def _mano_desde_tasks(hands_result, lado: str) -> np.ndarray | None:
    """Recupera una mano (21x3) desde HandLandmarkerResult de Tasks.

    `lado` = 'Left' o 'Right' (convencion MediaPipe, en perspectiva de imagen).
    Devuelve None si no hay esa mano.
    """
    try:
        for cats, lms in zip(hands_result.handedness, hands_result.hand_landmarks):
            if cats and cats[0].category_name == lado:
                arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                return arr.flatten()
    except Exception:
        pass
    return None


def features_frame(frame: dict, usar_fallback_manos: bool = True) -> np.ndarray:
    """Un frame del pkl -> vector de 258 features (mismo layout que extraer_keypoints)."""
    hl = frame.get("holistic_legacy", {}) or {}
    pose = _flatten_landmarks(hl.get("pose_landmarks"), POSE_N, 4)
    lh = _flatten_landmarks(hl.get("left_hand_landmarks"), HAND_N, 3)
    rh = _flatten_landmarks(hl.get("right_hand_landmarks"), HAND_N, 3)

    # Si la Holistic legacy no detecto una mano, intenta con la API Tasks (mejor
    # detector de manos). Las manos son la senal mas importante en LSP.
    if usar_fallback_manos and "hands" in frame:
        if not lh.any():
            alt = _mano_desde_tasks(frame["hands"], "Left")
            if alt is not None:
                lh = alt
        if not rh.any():
            alt = _mano_desde_tasks(frame["hands"], "Right")
            if alt is not None:
                rh = alt

    return np.concatenate([pose, lh, rh]).astype(np.float32)


def secuencia_features(frames: list, frames_objetivo: int = 30,
                       usar_fallback_manos: bool = True) -> np.ndarray | None:
    """Lista de frames (longitud variable) -> tensor (frames_objetivo, 258).

    Remuestrea uniformemente por indices (linspace) para fijar la longitud.
    """
    n = len(frames)
    if n == 0:
        return None
    idx = np.linspace(0, n - 1, frames_objetivo).round().astype(int)
    seq = np.stack([features_frame(frames[i], usar_fallback_manos) for i in idx])
    return seq.astype(np.float32)


def cargar_pkl(ruta: Path):
    with open(ruta, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------
# 3) Construir arrays (X, y) de un split
# ----------------------------------------------------------------------------
def construir_arrays(df: pd.DataFrame, mediapipe_dir: Path,
                     frames_objetivo: int = 30, usar_fallback_manos: bool = True,
                     verbose: bool = True):
    """Recorre el split y devuelve (X, y, faltantes).

    X: (N, frames_objetivo, 258) float32 ; y: (N,) int64.
    `mediapipe_dir`: carpeta con los <FILENAME>.pkl.
    """
    mediapipe_dir = Path(mediapipe_dir)
    X, y, faltantes = [], [], []

    try:
        from tqdm.auto import tqdm
        it = tqdm(df.itertuples(index=False), total=len(df), desc="pkl")
    except ImportError:
        it = df.itertuples(index=False)

    for fila in it:
        ruta = mediapipe_dir / f"{fila.FILENAME}.pkl"
        if not ruta.exists():
            faltantes.append(fila.FILENAME)
            continue
        seq = secuencia_features(cargar_pkl(ruta), frames_objetivo,
                                 usar_fallback_manos)
        if seq is None:
            faltantes.append(fila.FILENAME)
            continue
        X.append(seq)
        y.append(int(fila.CLASS_ID))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    if verbose:
        print(f"  {len(X)} muestras  |  faltantes/invalidos: {len(faltantes)}")
    return X, y, faltantes


# ----------------------------------------------------------------------------
# 4) Diagnostico: longitud de secuencias y tasa de deteccion de manos
# ----------------------------------------------------------------------------
def diagnostico(df: pd.DataFrame, mediapipe_dir: Path, n_muestra: int = 200,
                semilla: int = 42):
    """Muestra estadisticas utiles antes de extraer todo el dataset."""
    mediapipe_dir = Path(mediapipe_dir)
    rng = np.random.default_rng(semilla)
    fns = df.FILENAME.tolist()
    sel = rng.choice(len(fns), size=min(n_muestra, len(fns)), replace=False)

    largos, manos_legacy, manos_con_fallback, vacios = [], 0, 0, 0
    total_frames = 0
    for i in sel:
        ruta = mediapipe_dir / f"{fns[i]}.pkl"
        if not ruta.exists():
            continue
        frames = cargar_pkl(ruta)
        largos.append(len(frames))
        for fr in frames:
            total_frames += 1
            hl = fr.get("holistic_legacy", {}) or {}
            tiene_legacy = (hl.get("left_hand_landmarks") is not None
                            or hl.get("right_hand_landmarks") is not None)
            if tiene_legacy:
                manos_legacy += 1
            feat = features_frame(fr, usar_fallback_manos=True)
            tiene_fallback = feat[132:195].any() or feat[195:258].any()
            if tiene_fallback:
                manos_con_fallback += 1
            if not feat.any():
                vacios += 1

    largos = np.array(largos)
    print(f"Muestras inspeccionadas: {len(largos)}")
    print(f"Frames por video -> min={largos.min()} max={largos.max()} "
          f"mediana={int(np.median(largos))} media={largos.mean():.1f}")
    print(f"Frames con mano (solo Holistic legacy): {manos_legacy/total_frames:.1%}")
    print(f"Frames con mano (legacy + fallback Tasks): {manos_con_fallback/total_frames:.1%}")
    print(f"Frames totalmente vacios (sin pose ni manos): {vacios/total_frames:.1%}")
    return largos
