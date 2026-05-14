"""Inferencia en tiempo real desde webcam con ventana deslizante de 30 frames."""
from collections import deque

import numpy as np
import cv2
import tensorflow as tf

from config import FRAMES, CLASSES, MODEL_PATH
from src.preprocesamiento.extraccion_keypoints import (
    crear_holistic, procesar_frame, extraer_keypoints, dibujar_landmarks,
)
from src.inferencia.tts import hablar


def inferir_tiempo_real(camara=0, umbral=0.7):
    model = tf.keras.models.load_model(MODEL_PATH)
    cap = cv2.VideoCapture(camara)
    buffer = deque(maxlen=FRAMES)
    ultima_palabra = None

    with crear_holistic() as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            resultados = procesar_frame(frame, holistic)
            buffer.append(extraer_keypoints(resultados))

            etiqueta = ""
            if len(buffer) == FRAMES:
                entrada = np.expand_dims(np.array(buffer), axis=0)
                probs = model.predict(entrada, verbose=0)[0]
                idx = int(np.argmax(probs))
                if probs[idx] >= umbral:
                    etiqueta = f"{CLASSES[idx]} ({probs[idx]:.2f})"
                    if CLASSES[idx] != ultima_palabra:
                        ultima_palabra = CLASSES[idx]
                        hablar(CLASSES[idx])

            frame = dibujar_landmarks(frame, resultados)
            cv2.putText(frame, etiqueta, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("LSP en tiempo real", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
