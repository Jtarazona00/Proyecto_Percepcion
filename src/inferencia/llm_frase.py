"""Fase 2 — Interpretacion con LLM: convierte una secuencia de glosas LSP en una
frase natural en espanol, usando la API de Gemini (Google).

La LSP no usa articulos ni conjugaciones y su orden difiere del espanol, por eso
hace falta un LLM que reordene e interprete. Como el reconocedor entrega top-5
candidatos por sena, cada "slot" de la secuencia puede ser una lista de
candidatos; el LLM elige el mas coherente con el contexto (esto es lo que hace
que un top-5 ~81% sea util aunque el top-1 sea ~50%).

Requiere `pip install google-generativeai` y una API key gratis de Google AI
Studio (https://aistudio.google.com/apikey). NO se importa el SDK a nivel de
modulo para que el archivo sea importable sin la dependencia.
"""
from __future__ import annotations

import os

MODELO_DEFECTO = "gemini-1.5-flash"  # gratis y rapido; si tu key no lo tiene, usa listar_modelos()

_PROMPT_SISTEMA = (
    "Eres un interprete de Lengua de Senas Peruana (LSP) en un contexto hospitalario. "
    "Recibes una secuencia de glosas (palabras-sena) en el ORDEN en que se senaron. "
    "La LSP no usa articulos ni conjugaciones y su orden difiere del espanol. "
    "Cuando un paso ofrece varios candidatos, elige el que de la frase mas coherente. "
    "Devuelve UNA sola frase en espanol natural y gramatical que exprese lo que el "
    "paciente probablemente quiso decir. Se conciso. No expliques ni agregues comillas; "
    "responde solo la frase."
)


def _formatear_glosas(secuencia) -> str:
    """`secuencia`: lista donde cada elemento es un str (una glosa) o una lista
    de str (candidatos top-k para esa sena)."""
    lineas = []
    for i, slot in enumerate(secuencia, 1):
        if isinstance(slot, (list, tuple)):
            lineas.append(f"{i}. uno de: {', '.join(map(str, slot))}")
        else:
            lineas.append(f"{i}. {slot}")
    return "\n".join(lineas)


def construir_prompt(secuencia, contexto: str = "hospitalario") -> str:
    return (
        f"{_PROMPT_SISTEMA}\n\n"
        f"Contexto: {contexto}.\n\n"
        f"Glosas senadas (en orden):\n{_formatear_glosas(secuencia)}\n\n"
        f"Frase en espanol:"
    )


def armar_frase(secuencia, api_key: str | None = None,
                modelo: str = MODELO_DEFECTO, contexto: str = "hospitalario") -> str:
    """Secuencia de glosas (o de candidatos top-k) -> frase en espanol via Gemini."""
    import google.generativeai as genai

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Falta la API key de Gemini. Pasa api_key=... o define la variable de "
            "entorno GEMINI_API_KEY (gratis en https://aistudio.google.com/apikey)."
        )
    genai.configure(api_key=api_key)
    respuesta = genai.GenerativeModel(modelo).generate_content(construir_prompt(secuencia, contexto))
    return respuesta.text.strip()


def listar_modelos(api_key: str | None = None):
    """Lista los modelos Gemini disponibles para tu key (utiles para generar texto).
    Util si `MODELO_DEFECTO` da error de modelo no encontrado."""
    import google.generativeai as genai
    genai.configure(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
    return [m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods]
