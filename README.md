# M&E QC Inspector

Prototipo experimental de QC asistido, local y de solo lectura sobre el audio original. Compara printmaster, M&E y stems en **5.1 y 2.0**. Su objetivo es preparar una prueba de viabilidad profesional, no certificar entregas.

## Estado comprobado y límites

- Implementados: lectura PCM WAV/BWAV/RF64 discreta/interleaved, identidad SHA-256, cobertura y timecode, reconstrucción del original, diferencias M&E, controles de formato/true peak, candidatos de revisión, informes y evaluación.
- Whisper small multilingüe y Silero VAD instalados localmente con tamaños y hashes publicados verificados. CLI v1.9.5 compilado desde el commit fijado; reconocimiento real comprobado con una frase sintética en CPU y Metal. No se descarga nada al analizar.
- Probados con señales sintéticas: DSP, FFmpeg, informes, cancelación y métricas. Estas pruebas no demuestran precisión perceptual sobre cine.
- Detección de PFX dentro de DX: cola exploratoria de revisión por actividad. **No separa voz y PFX, ni demuestra por sí sola que falte un sonido.**
- Ejecutado un ensayo local con un programa real 2.0 autorizado y revisión auditiva parcial de candidatos. Esa revisión confirmó diferencias de cancelación, incluidas diferencias aceptables de baja audibilidad. No hay porcentajes de precisión/recall verificados ni validación real 5.1 completa; ver [validación](docs/VALIDATION.md).
- Un cómputo `complete` no significa una entrega aprobada. El null test no certifica un M&E: puede cancelar un M&E incompleto o dejar residual en un reemplazo correcto.

## Instalación local y demo sintética

Equipo objetivo declarado: Mac M2 / 16 GB. Se comprobó Python 3.14.8 y FFmpeg 9.0.2. Las versiones exactas usadas están en `requirements-dev.lock`; FFmpeg/ffprobe son herramientas externas ya disponibles. No cambiar Python ni paquetes globales.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements-dev.lock
.venv/bin/python scripts/make_fixtures.py --output runs/example-inputs --layout 2.0 --defect
PYTHONPATH=src .venv/bin/python -m mne_qc inspect examples/manifest.synthetic.json
PYTHONPATH=src .venv/bin/python -m mne_qc analyze examples/manifest.synthetic.json --output runs/example-report --previews
```

Abrir `runs/example-report/report.html` localmente. Escuchar las muestras sintéticas con nivel de monitorización prudente. Los directorios de salida deben ser nuevos: no se sobrescriben runs anteriores. El ejemplo necesita la generación previa de archivos; no hay audio de producción en el repositorio.

Para comprobar ambos formatos, casos correctos/defectuosos y cancelación:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke.py --output runs/my-smoke
```

Los tests ASR usan un backend simulado y lo distinguen de una inferencia real. No se descargan ni ejecutan modelos como efecto de los tests.

## Preparar un conjunto real

Usar un manifiesto privado bajo `data/local/`, excluido de Git. El ejemplo muestra la estructura. Declarar **cada canal** por etiqueta e índice base cero. Un archivo interleaved tiene varias entradas con la misma ruta; los discretos tienen una ruta por canal e índice cero.

- Roles obligatorios: `PM`, `ME`, `DX`. Añadir `MX`, `FX` para el modo completo. Optionals: roles `OPT_A`, `OPT_B`, etc.
- Etiquetas: `L,R` o `L,R,C,LFE,Ls,Rs`. La aplicación no adivina el orden del export por número de canales.
- `start_sample` es el inicio del archivo en el timeline común a 48 kHz; `program_span` es `[inicio, fin)` en muestras. No usar un BWF time reference como FPS.
- Para timecode, declarar `fps` exacto, por ejemplo `"24000/1001"`, y `start_frame` como frame count continuo. DF solo se implementa para `30000/1001`. Sin esos datos se reportan muestras/segundos.
- Stems para reconstrucción: misma versión, mapa de canales y procesamiento que el PM. Una guía DX mono permite escucha/ASR, pero no reconstrucción del PM multicanal.
- `synthetic: true` identifica fixtures; sus resultados nunca pueden satisfacer los gates de eficacia real.

Reglas autorizadas del título pueden documentarse en `exceptions`, por ejemplo:

```json
{"kind":"language", "span":[48000,96000], "channels":["L","R"],
 "policy":"allowed", "reason":"Canción autorizada según instrucciones del título"}
```

Una excepción solo se aplica si cubre todo el evento y sus canales. Conserva la detección y la evidencia; no borra la presencia de lenguaje ni demuestra inteligibilidad.

## IA local opcional

En esta máquina se han adquirido, con autorización expresa, whisper.cpp v1.9.5, Whisper `small` **multilingüe** y Silero VAD v6.2.0. Revisiones, hashes y binario están registrados en `models.lock.json`; los archivos grandes y `vendor/` están excluidos de Git. El binario del tag v1.9.5 informa `1.9.5-dev`, y su commit se ha contrastado con el inventario. Ver [dependencias](docs/DEPENDENCIES.md).

Con binario y pesos ya adquiridos y comprobados:

```sh
PYTHONPATH=src .venv/bin/python -m mne_qc analyze data/local/manifest.json \
  --output runs/real-session --speech both \
  --binary vendor/whisper.cpp/build/bin/whisper-cli \
  --model models/ggml-small.bin --vad-model models/ggml-silero-v6.2.0.bin
```

`fixed` examina ventanas de 30 s con 5 s de solapamiento sin selector VAD. `vad` usa el selector; `both` ejecuta ambas variantes secuencialmente. Examina todos los canales de ME y la referencia DX, sin limitarse a las regiones de diálogo de DX. Los derivados ASR son PCM16 mono/16 kHz sin normalización y se eliminan al terminar; el análisis DSP conserva las muestras originales.

Una transcripción es una hipótesis. Un resultado vacío no demuestra ausencia de lenguaje. No hay score calibrado de probabilidad ni separación automática validada.

Metal funcionó en ejecución normal y detectó un dispositivo Apple M2 Pro. Dentro del sandbox, el backend GPU falló al inicializarse; existe la alternativa explícita `--no-gpu` para ejecutar ASR en CPU. No se cambia de backend silenciosamente. La elección queda registrada en la configuración del run.

Las pruebas unitarias siguen usando mocks. Para repetir únicamente la integración real con la frase sintética local ya generada, en un directorio nuevo:

```sh
.venv/bin/python scripts/asr_smoke.py --phrase runs/asr-validation/phrase48.wav \
  --output runs/my-real-asr-smoke --cpu
```

Esta prueba detectó voz en `Rs` de 5.1 y `L` de 2.0, con y sin VAD, e incluyó una frase entre los segundos 27 y 35 cruzando ventanas. En el caso sin VAD hubo hipótesis erróneas y rangos demasiado amplios: no se ha demostrado precisión temporal del ASR. Las etiquetas aisladas conocidas como `[MÚSICA]` se conservan como anotaciones del modelo, sin contarlas como palabras habladas; texto como `[MÚSICA] Hola` sigue generando candidato de lenguaje. No se ocultan las palabras hipotéticas ni se convierten en hechos.

Resultados actuales: `runs/asr-integration/validation.json` e informes dentro de `runs/asr-integration/`. Son controles con una sola voz sintética, no validación cinematográfica. La adquisición y estas pruebas no requieren audio de producción ni servicios de transcripción.

## Resultados y revisión

Cada ejecución crea:

- `run.json`: configuración, identidad de entradas, cobertura, mediciones, versiones y estado de cada etapa.
- `report.html`: informe estático sin recursos de red; permite revisar candidatos e importar/exportar decisiones JSON.
- `review.json`: plantilla de decisiones separada de las detecciones. El botón del informe descarga una revisión nueva: elegir dónde guardarla o importarla después; no modifica las fuentes.
- Previews optativos: hasta 20 conjuntos de fragmentos, máximo 10 s por fragmento, 48 kHz/24-bit y configuración original, sin normalizar. El navegador no sustituye una escucha 5.1 correctamente ruteada. La referencia exacta al original permanece en `run.json`.

Ante cancelación o error se guardan las etapas completadas y la cobertura conocida. Si cambia una fuente, se invalida la identidad final y se bloquean los previews incoherentes. Los estados de cómputo no son PASS/FAIL de entrega.

La clasificación independiente FYI / REVIEW / FIX está propuesta y pendiente de implementación. En el informe actual, confirmar que existe un residual no establece por sí solo que requiera corrección.

## Evaluación

```sh
PYTHONPATH=src .venv/bin/python -m mne_qc evaluate \
  --runs runs/example-report/run.json \
  --annotations examples/annotations.synthetic.json \
  --corpus examples/corpus.synthetic.json --output runs/example-evaluation.json
```

El ejemplo devuelve código **2** y `insufficient_evidence`, deliberadamente. Es suficiente para verificar la aritmética y el formato, no la eficacia real. `analyze` devuelve 0 si terminó el cómputo solicitado y 2 si está parcial/fallido; puede contener incidencias aun devolviendo 0. `evaluate` devuelve 0 solo si todos los gates de continuación limitada se alcanzan, nunca como aprobación de entrega.

Ver el [protocolo de validación](docs/VALIDATION.md), el [diseño aprobado](docs/superpowers/specs/2026-10-26-mne-feasibility-design.md) y el [plan](docs/superpowers/plans/2026-10-26-mne-feasibility-plan.md).
