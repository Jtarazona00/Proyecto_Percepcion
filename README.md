# Proyecto Percepción — Reconocimiento de Lengua de Señas Peruana (LSP)

Sistema de reconocimiento de Lengua de Señas Peruana en tiempo real con síntesis de voz, orientado al **Hospital Regional Docente de Trujillo**. Proyecto del curso **Percepción Computacional** (UPAO, 2026-1).


## Estado actual

- **Sección 3** — Adquisición y preprocesamiento: completa (600 secuencias × 30 frames extraídas con MediaPipe).
- **Sección 4.1** — Baseline SVM con features estadísticas: **85.00 %** de precisión en test.
- **Sección 4.2** — Arquitectura BiLSTM definida (capa final `Dense N` ajustable).
- **Sección 4.3** — Entrenamiento completado: **val_accuracy mejor 98.36 %**, train_accuracy 99.16 %.
- **Sección 5** — Evaluación completada: **test_accuracy 83.33 %**, macro F1 = 0.83.
- **Sección 6** — Despliegue / producción: pendiente.
- **Ampliación de vocabulario**: pipeline para PUCP-305 (~305 glosas LSP) listo en `notebooks/02_pucp305_pipeline_colab.ipynb`. Soporte multi-dataset vía `DATASET` env var (`videolsp10` | `pucp305`).

## Resultados finales

| Modelo | Test Accuracy | Notas |
|---|---|---|
| Baseline SVM (RBF) | **85.00 %** | Features estadísticas: media + std + rango → 774 features |
| LSTM (30, 258) | **83.33 %** | val_accuracy mejor: 98.36 %, F1 macro = 0.83 |

- **Clases con F1 = 1.00**: `no_entiendo`, `que_haces_tu`.
- **Confusiones recurrentes**: `ayudame` ↔ `disculpame`, `donde_vives_tu` ↔ `hasta_manana` (movimientos visualmente similares).
- **Gap train-test** (~16 %) indica overfitting moderado, esperable con 60 muestras por clase; mitigado con Dropout 0.3, EarlyStopping y ReduceLROnPlateau.

> El modelo entrenado (`modelo_lsp_final.keras`), las curvas de aprendizaje y la matriz de confusión se guardan en Google Drive (`/MyDrive/VideoLSP10_models/`) tras ejecutar el notebook.

## Arquitectura

```
Webcam → MediaPipe Holistic → tensor (30, 258)
       → LSTM 128 → Dropout
       → LSTM 64  → Dropout
       → Dense 64 (ReLU)
       → Dense 10 (Softmax) → palabra reconocida → TTS
```

- **Dataset principal**: [VideoLSP10](https://github.com/videoLSP/VideoLSP10) — 10 clases (ayudame, por_favor, disculpame, cual_es_tu_nombre, donde_vives_tu, no_entiendo, que_haces_tu, hola_como_estas_tu, gracias, hasta_manana), capturadas con Kinect (RGB + depth + skeleton).
- **Dataset ampliado (opcional)**: [PUCP-305 glosas](https://datos.pucp.edu.pe/dataset.xhtml?persistentId=hdl%3A20.500.12534%2FJU4OLG) — ~305 glosas de LSP en videos continuos con anotaciones ELAN (Pontificia Universidad Católica del Perú). Pipeline en `notebooks/02_pucp305_pipeline_colab.ipynb`.
- **Features por frame (258)**: pose (33 × 4) + mano izquierda (21 × 3) + mano derecha (21 × 3).
- **Entrenamiento**: Adam lr=0.0005, sparse_categorical_crossentropy, batch=32, 100 épocas, split 80/10/10, callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau).

### ¿Qué hace cada paso?

1. **Webcam → MediaPipe Holistic**: en lugar de pasarle la imagen cruda al modelo, MediaPipe extrae solo los puntos clave del cuerpo y manos (un "esqueleto de palitos"). Esto descarta ruido como fondo, ropa o iluminación.
2. **Tensor (30, 258)**: una seña es movimiento, no una pose fija. Se acumulan 30 frames seguidos (~1 segundo) en una matriz donde cada fila es un instante en el tiempo.
3. **LSTM 128 → Dropout**: la LSTM es una red diseñada para entender secuencias. "Lee" los 30 frames en orden y detecta patrones de movimiento. El Dropout apaga neuronas al azar durante el entrenamiento para evitar que el modelo se memorice los datos.
4. **LSTM 64 → Dropout**: una segunda LSTM más chica que refina lo aprendido y resume toda la secuencia en un solo vector.
5. **Dense 64 (ReLU)**: capa clásica que transforma ese resumen en 64 características de alto nivel. ReLU le da no-linealidad para captar patrones complejos.
6. **Dense 10 (Softmax)**: la capa final tiene una neurona por clase (las 10 señas del dataset). Softmax convierte las salidas en probabilidades que suman 1 → se elige la palabra con mayor probabilidad.
7. **TTS**: la palabra ganadora se pasa a un sintetizador de voz (`pyttsx3` o `gTTS`) y se reproduce por los parlantes.

| Etapa | Analogía |
|---|---|
| MediaPipe | Convierte la imagen en un esqueleto de palitos |
| Tensor (30, 258) | Graba 1 segundo de ese esqueleto moviéndose |
| LSTM 128 + 64 | Lee el movimiento y lo resume en una idea |
| Dense 64 | Compara esa idea con patrones aprendidos |
| Dense 10 (Softmax) | Decide qué palabra es, con un % de confianza |
| TTS | La pronuncia en voz alta |

## Estructura del proyecto

```
percepcion/
├── config.py                       # rutas, clases, hiperparámetros
├── main.py                         # entrypoint
├── requirements.txt
├── notebooks/
│   ├── 01_pipeline_completo_colab.ipynb    # pipeline VideoLSP10 (10 clases)
│   └── 02_pucp305_pipeline_colab.ipynb     # pipeline PUCP-305 (~305 glosas LSP)
└── src/
    ├── preprocesamiento/
    │   ├── extraccion_keypoints.py        # MediaPipe → vector de 258
    │   ├── captura.py                      # webcam/video → (30, 258)
    │   ├── extraer_dataset.py              # VideoLSP10: JPGs → .npy
    │   ├── parse_elan.py                   # parser de .eaf (PUCP-305)
    │   └── extraer_dataset_pucp305.py      # PUCP-305: video+ELAN → .npy
    ├── modelos/
    │   ├── baseline_svm.py                 # SVM con features estadísticas (774)
    │   └── lstm_modelo.py                  # arquitectura BiLSTM
    ├── entrenamiento/
    │   └── entrenar.py                     # compile + callbacks + fit + class weights
    ├── evaluacion/
    │   └── evaluar.py                      # métricas + matriz de confusión
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
| `scikit-learn` | Baseline SVM, StandardScaler, métricas (accuracy, classification report, confusion matrix) y split estrategico. |
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
| MLOps | Docker, Flask/FastAPI, MLflow |
