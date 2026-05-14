"""Captura de secuencias de 30 frames desde webcam o video."""
import numpy as np
import cv2

from config import FRAMES
from .extraccion_keypoints import crear_holistic, procesar_frame, extraer_keypoints, dibujar_landmarks


def capturar_secuencia_webcam(camara=0, mostrar=True):
    """Devuelve un array (FRAMES, 258)."""
    cap = cv2.VideoCapture(camara)
    secuencia = []
    with crear_holistic() as holistic:
        while len(secuencia) < FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            resultados = procesar_frame(frame, holistic)
            secuencia.append(extraer_keypoints(resultados))
            if mostrar:
                frame = dibujar_landmarks(frame, resultados)
                cv2.imshow("Captura LSP", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    cap.release()
    cv2.destroyAllWindows()
    return np.array(secuencia)


def secuencia_desde_video(ruta_video):
    cap = cv2.VideoCapture(str(ruta_video))
    secuencia = []
    with crear_holistic() as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            resultados = procesar_frame(frame, holistic)
            secuencia.append(extraer_keypoints(resultados))
    cap.release()

    # Normalizar a FRAMES por muestreo uniforme.
    secuencia = np.array(secuencia)
    if len(secuencia) >= FRAMES:
        idx = np.linspace(0, len(secuencia) - 1, FRAMES).astype(int)
        return secuencia[idx]
    padding = np.zeros((FRAMES - len(secuencia), secuencia.shape[1]))
    return np.vstack([secuencia, padding])
