"""Fase tiempo real (2/2) — Webcam en vivo en el PC.

Webcam -> MediaPipe Holistic -> ventana deslizante de 30 frames -> reconocedor
con 'locking in' (DetectorContinuo) -> al finalizar, el LLM arma la frase y se
habla en voz alta (pyttsx3, offline).

IMPORTANTE: requiere un Python 3.11 o 3.12 (MediaPipe Holistic NO existe en 3.13).
Ver notebooks/README_tiempo_real.md para el setup. Ejecutar desde la raiz del repo:

    set GEMINI_API_KEY=tu_key        (Windows;  export en Linux/Mac)
    python -m src.inferencia.realtime_webcam --modelo models/modelo_pucp305_final.keras --clases models/classes_pucp305.json

Controles en la ventana:
    f / ESPACIO : finalizar -> arma la frase con el LLM y la habla
    c           : limpiar la secuencia de glosas
    q           : salir
"""
from __future__ import annotations

import argparse
import json
import os
from collections import deque
from pathlib import Path


def _texto(frame, txt, org, escala=0.7, color=(0, 255, 0), grosor=2):
    import cv2
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 2)
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor)


def main():
    ap = argparse.ArgumentParser(description="LSP en tiempo real (webcam).")
    ap.add_argument("--modelo", default="models/modelo_pucp305_final.keras")
    ap.add_argument("--clases", default="models/classes_pucp305.json")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--umbral", type=float, default=0.5, help="confianza top-1 minima")
    ap.add_argument("--estabilidad", type=int, default=6, help="frames seguidos para fijar")
    ap.add_argument("--cada", type=int, default=3, help="predecir cada N frames (fluidez)")
    ap.add_argument("--gemini-key", default=os.environ.get("GEMINI_API_KEY"))
    args = ap.parse_args()

    import numpy as np
    import cv2
    import tensorflow as tf
    import config
    from src.preprocesamiento.extraccion_keypoints import (
        crear_holistic, procesar_frame, extraer_keypoints, dibujar_landmarks,
    )
    from src.inferencia.inferencia_continua import DetectorContinuo

    if not Path(args.modelo).exists():
        raise SystemExit(f"No encuentro el modelo: {args.modelo}\n"
                         "Descargalo de Drive (PUCP305_models/modelo_pucp305_final.keras).")
    with open(args.clases, encoding="utf-8") as f:
        clases = json.load(f)
    config.set_classes(clases)
    model = tf.keras.models.load_model(args.modelo, compile=False)
    print(f"Modelo cargado: {len(clases)} clases, FRAMES={config.FRAMES}.")
    if not args.gemini_key:
        print("AVISO: sin GEMINI_API_KEY -> al finalizar solo se muestran/hablan las "
              "glosas concatenadas (sin frase del LLM).")

    detector = DetectorContinuo(clases, umbral=args.umbral, estabilidad=args.estabilidad)
    buffer = deque(maxlen=config.FRAMES)
    cap = cv2.VideoCapture(args.camara)
    n = 0
    probs = None

    with crear_holistic() as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # espejo, mas natural para el usuario
            resultados = procesar_frame(frame, holistic)
            buffer.append(extraer_keypoints(resultados))

            n += 1
            if len(buffer) == config.FRAMES and n % args.cada == 0:
                probs = model.predict(np.array(buffer)[None], verbose=0)[0]
                fijada = detector.actualizar(probs)
                if fijada:
                    print(f"[fijada] {fijada}   secuencia={[s[0] for s in detector.secuencia]}")

            frame = dibujar_landmarks(frame, resultados)
            # overlay: top-1 actual + barra de confianza
            if detector.ultima_top1:
                _texto(frame, f"{detector.ultima_top1}  {detector.ultima_conf:.0%}", (20, 40))
                ancho = int(300 * detector.ultima_conf)
                cv2.rectangle(frame, (20, 50), (20 + ancho, 65), (0, 255, 0), -1)
                cv2.rectangle(frame, (20, 50), (320, 65), (255, 255, 255), 1)
            # glosas fijadas
            glosas = " ".join(s[0] for s in detector.secuencia)
            _texto(frame, f"Frase: {glosas}", (20, 100), escala=0.6, color=(0, 255, 255))
            _texto(frame, "f=hablar  c=limpiar  q=salir", (20, frame.shape[0] - 20),
                   escala=0.5, color=(200, 200, 200))

            cv2.imshow("LSP en tiempo real", frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            elif tecla == ord("c"):
                detector.reiniciar()
                print("secuencia limpiada.")
            elif tecla in (ord("f"), ord(" ")):
                if not detector.secuencia:
                    print("(sin glosas todavia)")
                    continue
                frase = _armar(detector.secuencia, args.gemini_key)
                print(f"\n>>> FRASE: {frase}\n")
                _hablar(frase)

    cap.release()
    cv2.destroyAllWindows()


def _armar(secuencia, gemini_key):
    """Con key -> LLM; sin key -> glosas concatenadas (fallback)."""
    if gemini_key:
        try:
            from src.inferencia.llm_frase import armar_frase
            return armar_frase(secuencia, api_key=gemini_key, contexto="hospitalario")
        except Exception as e:
            print(f"(LLM fallo: {e}; uso glosas concatenadas)")
    return " ".join(slot[0] for slot in secuencia)


def _hablar(frase):
    try:
        from src.inferencia.tts import hablar
        hablar(frase)
    except Exception as e:
        print(f"(TTS no disponible: {e})")


if __name__ == "__main__":
    main()
