# Proyecto Percepción — Reconocimiento de Lengua de Señas Peruana (LSP)

Sistema de reconocimiento de Lengua de Señas Peruana en tiempo real con síntesis de voz, orientado al **Hospital Regional Docente de Trujillo**. Proyecto del curso **Percepción Computacional** (UPAO, 2026-1).

## Objetivo

Permitir que personas sordas o con discapacidad auditiva se comuniquen con personal médico mediante una cámara web: el sistema reconoce señas en video, las clasifica y las pronuncia en voz alta.

## Estado actual

- **Sección 3** — Adquisición y preprocesamiento: completa.
- **Sección 4.1** — Baseline SVM con features estadísticas: **67.3 %** de precisión.
- **Sección 4.2** — Arquitectura del modelo LSTM: definida.
- **Sección 4.3** — Entrenamiento: convergencia en época 73, **val_accuracy 93.5 %**.

## Arquitectura

```
Webcam → MediaPipe Holistic → tensor (30, 258)
       → LSTM 128 → Dropout
       → LSTM 64  → Dropout
       → Dense 64 (ReLU)
       → Dense 21 (Softmax) → palabra reconocida → TTS
```

- **Dataset**: VideoLSP10 — 21 clases, 1,701 secuencias de 30 frames.
- **Features por frame (258)**: pose (33 × 4) + mano izquierda (21 × 3) + mano derecha (21 × 3).
- **Entrenamiento**: Adam lr=0.0005, sparse_categorical_crossentropy, batch=32, 100 épocas, split 80/10/10, callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau).

## Estructura del proyecto

```
percepcion/
├── config.py                       # rutas, clases, hiperparámetros
├── main.py                         # entrypoint
├── requirements.txt
└── src/
    ├── preprocesamiento/
    │   ├── extraccion_keypoints.py # MediaPipe → vector de 258
    │   └── captura.py              # webcam/video → (30, 258)
    ├── modelos/
    │   ├── baseline_svm.py         # SVM con features estadísticas (774)
    │   └── lstm_modelo.py          # arquitectura LSTM
    ├── entrenamiento/
    │   └── entrenar.py             # compile + callbacks + fit
    ├── evaluacion/
    │   └── evaluar.py              # métricas + matriz de confusión
    ├── inferencia/
    │   ├── tiempo_real.py          # webcam + predicción + TTS
    │   └── tts.py                  # síntesis de voz
    └── utils/
        └── datos.py                # carga y split del dataset
```

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Nota**: MediaPipe requiere Python 3.8–3.11. TensorFlow en Windows entrena en CPU (para GPU usa Google Colab).

## Dependencias y por qué cada una

| Paquete | Por qué se necesita |
|---|---|
| `numpy` | Manejo de los tensores (30, 258) y operaciones vectoriales del pipeline. |
| `opencv-python` | Captura de video desde la webcam y manipulación de frames. |
| `mediapipe` | Extracción de landmarks (pose + manos) — la base del feature vector. |
| `tensorflow` | Construir, entrenar y cargar el modelo LSTM. |
| `scikit-learn` | Baseline SVM, StandardScaler, métricas (accuracy, classification report, confusion matrix) y split estratificado. |
| `matplotlib` | Graficar curvas de loss/accuracy y matriz de confusión. |
| `seaborn` | Mejora visual de las gráficas de evaluación. |
| `joblib` | Persistir el SVM y el scaler entrenados (`.joblib`). |
| `pyttsx3` | Síntesis de voz **offline** — la palabra reconocida se pronuncia sin internet. |
| `gTTS` | Síntesis de voz **online** de Google — alternativa con voz más natural. |

## Uso

```powershell
python main.py baseline       # entrena SVM y reporta precisión
python main.py entrenar       # entrena el modelo LSTM
python main.py evaluar        # carga modelo y evalúa sobre test
python main.py tiempo_real    # inferencia desde webcam + TTS
```

El dataset debe estar en `data/processed/<clase>/<secuencia>.npy`, donde cada archivo es un array de forma `(30, 258)`.

## Stack por fase

| Fase | Herramientas |
|---|---|
| Preprocesamiento | OpenCV, MediaPipe, NumPy |
| Modelo | TensorFlow/Keras, scikit-learn, Google Colab |
| Evaluación | matplotlib, scikit-learn, TensorBoard |
| TTS | pyttsx3 / gTTS |
| MLOps (futuro) | Docker, Flask/FastAPI, MLflow |
