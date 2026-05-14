"""Configuracion global del proyecto LSP."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Dataset VideoLSP10
CLASSES = [
    "hola", "gracias", "porfavor", "si", "no",
    "bueno", "malo", "ayuda", "dolor", "agua",
    "comida", "bano", "doctor", "enfermera", "hospital",
    "familia", "casa", "frio", "calor", "perdon",
    "buenos_dias",
]
NUM_CLASSES = len(CLASSES)

# Tensor de entrada: (FRAMES, FEATURES_PER_FRAME)
# 33*4 (pose) + 21*3 (mano izq) + 21*3 (mano der) = 258
FRAMES = 30
FEATURES_PER_FRAME = 258

# Hiperparametros entrenamiento
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 100
SPLIT_TRAIN = 0.8
SPLIT_VAL = 0.1
SPLIT_TEST = 0.1

MODEL_PATH = MODELS_DIR / "modelo_lsp_final.keras"
SCALER_PATH = MODELS_DIR / "scaler_svm.joblib"
SVM_PATH = MODELS_DIR / "svm_baseline.joblib"
