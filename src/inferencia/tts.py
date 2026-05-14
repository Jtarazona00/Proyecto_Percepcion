"""Sintesis de voz para la palabra reconocida."""
import pyttsx3

_engine = None


def _motor():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 160)
    return _engine


def hablar(texto):
    motor = _motor()
    motor.say(texto)
    motor.runAndWait()
