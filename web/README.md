# Demo web — Señas a voz en el navegador

Réplica del concepto del video objetivo, **100% en el navegador**: webcam →
MediaPipe Holistic (JS) → tu modelo en TensorFlow.js → glosas con "locking in" →
frase → voz. **No necesita Python 3.13 ni instalar MediaPipe**: solo un servidor
HTTP estático (que cualquier Python sirve).

## 1. Exportar el modelo a TensorFlow.js (una vez)
Corre `notebooks/06_exportar_tfjs_colab.ipynb` en Colab. Descarga `web_model.zip` y
descomprímelo **dentro de esta carpeta** `web/`, de modo que quede:

```
web/
  lsp_demo.html
  web_model/
    model.json
    group1-shard1of1.bin   (y demás .bin)
    classes_pucp305.json
```

## 2. Servir por HTTP (no abrir con doble clic)
El navegador bloquea `getUserMedia` y `fetch` con `file://`. Desde la carpeta `web/`:

```powershell
# tu Python 3.13 sirve esto sin problema (http.server no usa mediapipe/tf)
python -m http.server 8000
```

Abre **http://localhost:8000/lsp_demo.html** (usa `localhost`, no `127.0.0.1`).

## 3. Usar
1. Clic en **▶ Iniciar cámara** (acepta el permiso).
2. Haz una seña; cuando el modelo se estabiliza, la glosa se fija como **chip** arriba.
3. (Opcional) pega tu **API key de Gemini** en el campo para que arme una frase natural;
   sin key, concatena las glosas.
4. **🗣️ Armar frase y hablar** → la voz del navegador (Web Speech) la pronuncia en español.

## Notas honestas
- **Riesgo de conversión:** el modelo es un BiLSTM; TF.js a veces falla al cargar capas
  Bidirectional. Si la consola del navegador muestra un error al cargar `model.json`,
  avísame y pasamos al plan B (mini-backend Flask que sirve el modelo sin convertir).
- **Precisión:** top-1 ~50%, top-5 ~81%. El candidato "más probable" puede fallar; por eso
  se envían los **top-5** al LLM, que desambigua por contexto.
- **Aislado→continuo:** el modelo se entrenó con señas aisladas (30 frames); en vivo la
  ventana deslizante puede no cubrir señas largas. Ajusta `UMBRAL`/`ESTABILIDAD` en el
  `<script>` si fija de más o de menos.
- **Seguridad de la key:** la key de Gemini se usa solo en tu navegador. No publiques este
  HTML con la key escrita; el campo la mantiene fuera del archivo.
