"""Sintesis de voz.

- `hablar`: pyttsx3, OFFLINE y en tiempo real (PC local). NO funciona en Colab
  (no hay dispositivo de audio).
- `sintetizar_archivo`: gTTS, genera un .mp3. Es lo que se usa en Colab; el mp3
  se reproduce con `IPython.display.Audio(ruta)`.
"""

_engine = None


def _motor():
    global _engine
    if _engine is None:
        import pyttsx3
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 160)
    return _engine


def hablar(texto):
    """Habla en voz alta en el PC local (pyttsx3, offline, tiempo real)."""
    motor = _motor()
    motor.say(texto)
    motor.runAndWait()


def sintetizar_archivo(texto, ruta="frase.mp3", lang="es", tld="com.mx"):
    """Genera un .mp3 con gTTS (sirve en Colab). `tld` ajusta el acento del
    espanol (ej. 'com.mx', 'es', 'com'). Devuelve la ruta del archivo."""
    from gtts import gTTS
    gTTS(text=texto, lang=lang, tld=tld).save(ruta)
    return ruta
