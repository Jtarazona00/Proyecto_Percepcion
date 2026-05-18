"""Parser de archivos ELAN (.eaf) para extraer glosas LSP con intervalos de tiempo.

Formato ELAN (XML):
- TIME_ORDER contiene TIME_SLOT con ID y TIME_VALUE en ms
- TIER puede tener varios LINGUISTIC_TYPE
- ANNOTATION dentro de TIER tiene ALIGNABLE_ANNOTATION con TIME_SLOT_REF1/2
- ANNOTATION_VALUE contiene la glosa (label de la sena)

Para PUCP-305 tipicamente nos interesan los tiers tipo "GLOSA" o equivalente.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


def _normalizar_glosa(texto: str) -> str:
    """Limpia espacios y unifica capitalizacion para evitar duplicados."""
    return texto.strip().upper().replace("  ", " ")


def parsear_eaf(ruta_eaf: Path, tiers_interes: Iterable[str] | None = None):
    """Lee un .eaf y devuelve lista de dicts:
        {"glosa": str, "inicio_ms": int, "fin_ms": int, "tier": str}

    Si `tiers_interes` es None, devuelve anotaciones de TODOS los tiers.
    """
    ruta_eaf = Path(ruta_eaf)
    if not ruta_eaf.is_file():
        return []

    try:
        tree = ET.parse(ruta_eaf)
    except ET.ParseError:
        return []
    root = tree.getroot()

    # Indexar time slots: ID -> ms
    time_slots: dict[str, int] = {}
    time_order = root.find("TIME_ORDER")
    if time_order is not None:
        for ts in time_order.findall("TIME_SLOT"):
            ts_id = ts.get("TIME_SLOT_ID")
            ts_val = ts.get("TIME_VALUE")
            if ts_id and ts_val is not None:
                try:
                    time_slots[ts_id] = int(ts_val)
                except ValueError:
                    pass

    filtro = None
    if tiers_interes:
        filtro = {t.upper() for t in tiers_interes}

    resultados = []
    for tier in root.findall("TIER"):
        tier_id = tier.get("TIER_ID", "")
        if filtro and tier_id.upper() not in filtro:
            continue
        for annot in tier.findall("ANNOTATION"):
            alignable = annot.find("ALIGNABLE_ANNOTATION")
            if alignable is None:
                continue
            ref1 = alignable.get("TIME_SLOT_REF1")
            ref2 = alignable.get("TIME_SLOT_REF2")
            valor_elem = alignable.find("ANNOTATION_VALUE")
            if ref1 is None or ref2 is None or valor_elem is None:
                continue
            t1 = time_slots.get(ref1)
            t2 = time_slots.get(ref2)
            if t1 is None or t2 is None:
                continue
            glosa = (valor_elem.text or "").strip()
            if not glosa:
                continue
            resultados.append({
                "glosa": _normalizar_glosa(glosa),
                "inicio_ms": min(t1, t2),
                "fin_ms": max(t1, t2),
                "tier": tier_id,
            })

    return resultados


def listar_tiers(ruta_eaf: Path):
    """Devuelve los nombres de los tiers en un .eaf (util para descubrir estructura)."""
    ruta_eaf = Path(ruta_eaf)
    try:
        tree = ET.parse(ruta_eaf)
    except (ET.ParseError, FileNotFoundError):
        return []
    return [tier.get("TIER_ID", "") for tier in tree.getroot().findall("TIER")]


def encontrar_video_asociado(ruta_eaf: Path, carpeta_base: Path | None = None):
    """Intenta resolver el video asociado a un .eaf.

    ELAN guarda referencias a media en <MEDIA_DESCRIPTOR MEDIA_URL="file:..." />.
    Si la ruta del XML no existe, busca un .mp4 con el mismo stem en carpeta_base.
    """
    ruta_eaf = Path(ruta_eaf)
    try:
        tree = ET.parse(ruta_eaf)
    except (ET.ParseError, FileNotFoundError):
        return None
    root = tree.getroot()

    for md in root.findall(".//MEDIA_DESCRIPTOR"):
        url = md.get("MEDIA_URL", "")
        if url.startswith("file:"):
            url = url.replace("file://", "").replace("file:", "")
        candidato = Path(url)
        if candidato.is_file():
            return candidato

    # Fallback: buscar por stem en la carpeta del .eaf o carpeta_base
    base = carpeta_base or ruta_eaf.parent
    stem = ruta_eaf.stem
    for ext in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
        cand = base / f"{stem}{ext}"
        if cand.is_file():
            return cand
    # Busqueda recursiva como ultimo recurso
    for v in base.rglob(f"{stem}.*"):
        if v.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            return v
    return None
