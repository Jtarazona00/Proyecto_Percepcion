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


class DetectorContinuo:
    """Mecanismo de 'locking in' para un stream (webcam).

    En cada frame recibe las probabilidades del modelo sobre la ventana actual y
    decide cuando FIJAR una glosa: cuando la misma clase top-1 supera `umbral`
    durante `estabilidad` frames seguidos. Tras fijar entra en `cooldown` frames
    para no fijar la misma seña repetida. Las glosas fijadas se acumulan como
    listas top-k (lo que luego recibe el LLM).

    No depende de tensorflow/mediapipe: se le pasan las probabilidades ya calculadas.
    """

    def __init__(self, clases, umbral: float = 0.5, estabilidad: int = 6,
                 cooldown: int = 12, top_k: int = 5):
        self.clases = clases
        self.umbral = umbral
        self.estabilidad = estabilidad
        self.cooldown = cooldown
        self.top_k = top_k
        self.secuencia: list[list[str]] = []   # top-k por glosa fijada
        self._cand = None        # clase candidata en evaluacion
        self._conteo = 0         # frames seguidos sosteniendo el candidato
        self._cooldown = 0       # frames restantes de enfriamiento
        self.ultima_conf = 0.0
        self.ultima_top1 = None

    def actualizar(self, probs):
        """probs: vector de probabilidades (len = nº clases). Devuelve la glosa
        recien FIJADA en este frame, o None."""
        import numpy as np
        idx = np.argsort(probs)[-self.top_k:][::-1]
        top1 = self.clases[int(idx[0])]
        conf = float(probs[int(idx[0])])
        self.ultima_top1, self.ultima_conf = top1, conf

        if self._cooldown > 0:
            self._cooldown -= 1

        # acumular estabilidad solo si supera umbral y se mantiene la misma clase
        if conf >= self.umbral and top1 == self._cand:
            self._conteo += 1
        elif conf >= self.umbral:
            self._cand = top1
            self._conteo = 1
        else:
            self._cand = None
            self._conteo = 0

        ya_fijada = self.secuencia and self.secuencia[-1][0] == top1
        if (self._conteo >= self.estabilidad and self._cooldown == 0
                and not ya_fijada):
            fijada_topk = [self.clases[int(i)] for i in idx]
            self.secuencia.append(fijada_topk)
            self._cooldown = self.cooldown
            self._conteo = 0
            self._cand = None
            return top1
        return None

    def reiniciar(self):
        self.secuencia.clear()
        self._cand = None
        self._conteo = 0
        self._cooldown = 0
