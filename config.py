"""Configuracion global del proyecto LSP.

Soporta dos datasets:
- VideoLSP10 (default): 10 frases en LSP, ~60 muestras/clase
- PUCP-305: 305 glosas LSP de PUCP, samples variables/clase

Cambia DATASET para usar uno u otro.
"""
from pathlib import Path
import os

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# === Selector de dataset ===
# Override desde el entorno: os.environ['DATASET'] = 'pucp305'
DATASET = os.environ.get("DATASET", "videolsp10").lower()


# Tensor de entrada: (FRAMES, FEATURES_PER_FRAME)
# 33*4 (pose) + 21*3 (mano izq) + 21*3 (mano der) = 258
FRAMES = 30
FEATURES_PER_FRAME = 258


if DATASET == "videolsp10":
    # Dataset VideoLSP10 — 10 clases reales segun label.txt
    CLASSES = [
        "ayudame",
        "por_favor",
        "disculpame",
        "cual_es_tu_nombre",
        "donde_vives_tu",
        "no_entiendo",
        "que_haces_tu",
        "hola_como_estas_tu",
        "gracias",
        "hasta_manana",
    ]
elif DATASET == "pucp305":
    # PUCP-305 — las clases se cargan dinamicamente desde el dataset
    # (se rellenan en runtime por extraer_dataset_pucp305.descubrir_clases)
    CLASSES = []
else:
    raise ValueError(f"DATASET desconocido: {DATASET}. Usa 'videolsp10' o 'pucp305'.")


NUM_CLASSES = len(CLASSES)


# Hiperparametros entrenamiento
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 100
SPLIT_TRAIN = 0.8
SPLIT_VAL = 0.1
SPLIT_TEST = 0.1

# Filtro de calidad para PUCP-305: minimo de muestras por clase para incluirla
MIN_MUESTRAS_POR_CLASE = 10


MODEL_PATH = MODELS_DIR / f"modelo_{DATASET}_final.keras"
SCALER_PATH = MODELS_DIR / f"scaler_svm_{DATASET}.joblib"
SVM_PATH = MODELS_DIR / f"svm_baseline_{DATASET}.joblib"


def set_classes(nuevas_clases):
    """Setter dinamico de CLASSES (usado por pipeline PUCP-305)."""
    global CLASSES, NUM_CLASSES
    CLASSES = list(nuevas_clases)
    NUM_CLASSES = len(CLASSES)
