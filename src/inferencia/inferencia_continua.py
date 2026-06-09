"""Inferencia continua: de video(s) o stream a secuencia de glosas (top-k).

Núcleo compartido entre el demo sobre video (Colab) y la webcam en vivo (PC).
Cada seña reconocida se entrega como su lista de candidatos top-k, que es lo que
consume el LLM (`src.inferencia.llm_frase.armar_frase`) para armar la frase.

Los imports de tensorflow / mediapipe / cv2 se hacen DENTRO de las funciones para
que este modulo sea importable sin esas dependencias.
"""
from __future__ import annotations


def predecir_secuencia(seq, model, clases, top_k: int = 5) -> dict:
    """Una secuencia (FRAMES, 258) -> dict con top-k labels y confianzas."""
    import numpy as np
    p = model.predict(seq[None].astype("float32"), verbose=0)[0]
    idx = np.argsort(p)[-top_k:][::-1]
    return {
        "top_labels": [clases[int(i)] for i in idx],
        "top": [(clases[int(i)], float(p[int(i)])) for i in idx],
        "top1": clases[int(idx[0])],
        "conf": float(p[int(idx[0])]),
    }


def reconocer_video(ruta_video, model, clases, top_k: int = 5) -> dict:
    """Procesa un video (una seña) -> dict top-k. Usa MediaPipe Holistic via
    `secuencia_desde_video` (remuestrea a config.FRAMES = igual que el entrenamiento)."""
    from src.preprocesamiento.captura import secuencia_desde_video
    seq = secuencia_desde_video(ruta_video)
    return predecir_secuencia(seq, model, clases, top_k)


def construir_secuencia_glosas(items, model, clases, top_k: int = 5,
                               verbose: bool = True):
    """`items`: lista de (etiqueta_esperada_o_None, ruta_video), en el orden en que
    se "senan". Devuelve (secuencia, reporte):

    - secuencia: lista de listas top-k (lo que recibe el LLM).
    - reporte:   por seña, dict con esperado/top1/conf/top5/estado (OK|top5|FALLA),
                 imitando el "locking in" del demo (la glosa se fija con su confianza).
    """
    secuencia, reporte = [], []
    for esperado, ruta in items:
        r = reconocer_video(ruta, model, clases, top_k)
        secuencia.append(r["top_labels"])
        if esperado is None:
            estado = "-"
        elif esperado == r["top1"]:
            estado = "OK"
        elif esperado in r["top_labels"]:
            estado = "top5"
        else:
            estado = "FALLA"
        info = {"esperado": esperado, "top1": r["top1"], "conf": r["conf"],
                "top5": r["top_labels"], "estado": estado}
        reporte.append(info)
        if verbose:
            obj = esperado if esperado is not None else "(libre)"
            print(f"seña: {obj:22} -> {r['top1']:22} {r['conf']:.0%}  "
                  f"locking in...  [{estado}]")
    return secuencia, reporte
