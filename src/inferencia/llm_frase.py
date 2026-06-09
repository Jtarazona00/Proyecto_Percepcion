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

# None = autoseleccionar un modelo valido de la cuenta (recomendado: los nombres
# concretos como 'gemini-1.5-flash' los va deprecando Google). Puedes forzar uno
# pasando modelo='...' a armar_frase.
MODELO_DEFECTO = None

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


def _ordenar_por_preferencia(nombres):
    """Ordena modelos por preferencia: 'flash' (rapido/gratis) antes que 'pro';
    version mas alta antes (orden alfab. descendente aproxima 2.5 > 2.0 > 1.5);
    se evitan variantes 'vision'/'embedding'/'aqa' que no sirven para chat."""
    def puntaje(n):
        bajo = n.lower()
        s = 0
        if "flash" in bajo:
            s += 100
        elif "pro" in bajo:
            s += 50
        if any(x in bajo for x in ("vision", "embedding", "aqa")):
            s -= 1000
        return (s, bajo)  # a igual puntaje, alfabetico
    return sorted(nombres, key=puntaje, reverse=True)


def _resolver_modelo(genai, preferido: str | None):
    """Devuelve un nombre de modelo valido. Si `preferido` esta disponible lo usa;
    si no, autoselecciona el mejor de los que soportan generateContent."""
    disponibles = [m.name for m in genai.list_models()
                   if "generateContent" in m.supported_generation_methods]
    if not disponibles:
        raise RuntimeError("Tu key de Gemini no tiene modelos con generateContent.")
    if preferido:
        # acepta 'gemini-x' o 'models/gemini-x'
        for n in disponibles:
            if n == preferido or n == f"models/{preferido}" or n.endswith("/" + preferido):
                return n
    return _ordenar_por_preferencia(disponibles)[0]


def armar_frase(secuencia, api_key: str | None = None,
                modelo: str | None = MODELO_DEFECTO, contexto: str = "hospitalario") -> str:
    """Secuencia de glosas (o de candidatos top-k) -> frase en espanol via Gemini.

    `modelo=None` autoselecciona uno valido de tu cuenta. Si pasas uno y no existe,
    cae automaticamente al autoseleccionado (los nombres de Gemini cambian con el
    tiempo)."""
    import google.generativeai as genai

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Falta la API key de Gemini. Pasa api_key=... o define la variable de "
            "entorno GEMINI_API_KEY (gratis en https://aistudio.google.com/apikey)."
        )
    genai.configure(api_key=api_key)
    nombre = _resolver_modelo(genai, modelo)
    prompt = construir_prompt(secuencia, contexto)
    try:
        respuesta = genai.GenerativeModel(nombre).generate_content(prompt)
    except Exception:
        # fallback final: reintenta con el mejor autoseleccionado
        alt = _resolver_modelo(genai, None)
        respuesta = genai.GenerativeModel(alt).generate_content(prompt)
    return respuesta.text.strip()


def listar_modelos(api_key: str | None = None):
    """Lista los modelos Gemini que soportan generateContent, ya ordenados por
    preferencia (el primero es el que usaria armar_frase con modelo=None)."""
    import google.generativeai as genai
    genai.configure(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
    nombres = [m.name for m in genai.list_models()
               if "generateContent" in m.supported_generation_methods]
    return _ordenar_por_preferencia(nombres)
