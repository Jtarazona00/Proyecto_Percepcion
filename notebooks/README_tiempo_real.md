# Fase tiempo real (2/2) — Webcam en vivo en tu PC

El reconocimiento con webcam en directo corre en **tu PC** (Colab no da cámara
fluida). MediaPipe Holistic **no existe en Python 3.13**, así que necesitas un
**Python 3.11 o 3.12** en un entorno aparte. No tienes que desinstalar tu 3.13.

## 1. Instalar Python 3.11 o 3.12
Descarga e instala, por ejemplo, Python 3.12 desde python.org (marca "Add to PATH"
o usa el lanzador `py -3.12`).

## 2. Clonar el repo y crear el entorno
```powershell
git clone https://github.com/Jtarazona00/Proyecto_Percepcion.git
cd Proyecto_Percepcion

py -3.12 -m venv .venv-rt
.\.venv-rt\Scripts\Activate.ps1        # PowerShell
python --version                        # debe decir 3.12.x
```

## 3. Instalar dependencias
```powershell
pip install --upgrade pip
pip install tensorflow==2.18.0 mediapipe==0.10.14 opencv-python google-generativeai pyttsx3 numpy
```

## 4. Descargar el modelo entrenado desde Drive
Crea la carpeta `models/` en el repo y copia ahí (desde `Drive/MyDrive/PUCP305_models/`):
- `modelo_pucp305_final.keras`
- `classes_pucp305.json`

```
Proyecto_Percepcion/
  models/
    modelo_pucp305_final.keras
    classes_pucp305.json
```

## 5. (Opcional) API key de Gemini para que arme la frase
```powershell
$env:GEMINI_API_KEY = "tu_key"
```
Sin key, al finalizar solo se concatenan/hablan las glosas (sin frase del LLM).

## 6. Ejecutar
```powershell
python -m src.inferencia.realtime_webcam
# o con rutas/ajustes explicitos:
python -m src.inferencia.realtime_webcam --modelo models/modelo_pucp305_final.keras --clases models/classes_pucp305.json --umbral 0.5
```

### Controles en la ventana
| Tecla | Acción |
|-------|--------|
| `f` o `ESPACIO` | finalizar → el LLM arma la frase y se habla |
| `c` | limpiar la secuencia de glosas |
| `q` | salir |

## Notas honestas
- El top-1 del modelo es ~50%, así que con `--umbral 0.5` algunas señas costará
  fijarlas. Baja el umbral (ej. `0.35`) si no fija casi nada, o súbelo si fija de más.
- El modelo se entrenó con señas **aisladas** remuestreadas a 30 frames; en vivo la
  ventana deslizante de 30 frames puede no cubrir señas largas. Es el límite
  conocido aislado→continuo; el demo sobre video (notebook 05) es más fiel.
- Ajusta `--estabilidad` (frames seguidos para fijar) y `--cada` (cada cuántos
  frames predice) según la fluidez de tu equipo.
