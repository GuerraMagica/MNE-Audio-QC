# Protocolo de validación y límites

## Evidencia disponible

Los tests automatizados y `scripts/smoke.py` ejecutan DSP/FFmpeg sobre fixtures pequeños. Cubren reconstrucción, los dos contraejemplos del null, canales, metadata, true peak controlado, errores/cancelación, candidatos, reportes y aritmética de métricas. El smoke genera casos correctos y defectuosos en 2.0/5.1 y exige `insufficient_evidence` para las conclusiones de eficacia real.

Los tests unitarios de reconocimiento usan un backend simulado para comprobar contratos. **No son pruebas de capacidad de reconocimiento de Whisper.** Tras la autorización expresa del usuario se adquirieron y verificaron los pesos y se ejecutó también la integración real descrita abajo. Posteriormente se completó un ensayo con material real 2.0 y una revisión auditiva parcial; no constituye un corpus de evaluación independiente anotado por completo.

## Ensayo exploratorio con material real 2.0

Se procesó localmente un programa autorizado de unos 41 minutos con PM, M&E y stems DX/MX/FX, PCM24/48 kHz y orden L/R confirmado. Las etapas DSP, técnica y ASR con/sin VAD terminaron, y se verificó la identidad SHA-256 de los originales. El material se declaró EBU R128, no una entrega Netflix: los avisos frente al perfil de pico del prototipo no se contabilizan como defectos de ese conjunto.

El usuario revisó una selección de 12 intervalos y confirmó diferencias de cancelación. También describió un residual que necesitó amplificar +32 dB para percibir y que consideró aceptable. La amplificación de escucha no determina su nivel absoluto en dBFS. Las marcas anteriores de «confirmada» pueden significar existencia de una diferencia, no necesariamente una corrección requerida; no deben convertirse automáticamente en defectos accionables.

El tono inicial de 1 kHz se había impreso independientemente en cada fuente. Se preservó el análisis bruto y se derivaron mediciones excluyendo exclusivamente ese tono, sin normalizar ni alterar los originales. Esa exclusión fue una delimitación documentada de este ensayo; el prototipo no incorpora todavía una exclusión automática general de tonos. Se generó una copia del informe con TC relativos a 24 fps a petición del usuario, sin inferir los FPS originales del material.

Las hipótesis de lenguaje, los controles de escucha y las diferencias aceptables necesitan interpretación independiente. Este único programa y su selección parcial no permiten estimar precisión/recall, demostrar ausencia de omisiones ni validar PFX bajo diálogo. La evaluación real 5.1 sigue pendiente. Los audios, rutas de producción, transcripciones, informes y revisiones individuales permanecen locales y se excluyen del repositorio.

## Integración real local con una voz sintética

Se descargaron los dos archivos de `models.lock.json` desde sus revisiones fijadas y se comprobaron sus tamaños y SHA-256 publicados. Se compiló `whisper-cli` del commit autorizado v1.9.5 (el binario informa `1.9.5-dev`). En ejecución normal, el backend identificó **Apple M2 Pro** y completó la inferencia Metal; CPU también completó la misma frase. Dentro del sandbox, la inicialización GPU abortó; el producto dispone de `--no-gpu` explícito, probado con regresiones. No se modifica el backend silenciosamente.

La frase se sintetizó localmente con la voz macOS `Eddy (Spanish (Spain))`: «Esta es una prueba de control de calidad. Hay una voz en la pista de música y efectos. La puerta se cierra al final de la escena.» El primer intento dentro del sandbox produjo un AIFF de cero muestras pese a devolver código cero; se detectó al comprobar sus frames y se repitió con acceso normal al sintetizador local. Se usó el archivo válido de 8,5905 s, convertido a PCM24/48 kHz para las fixtures y a PCM16/16 kHz para las pruebas directas. No se transmitió ese audio a servicios externos.

Prueba completa: `.venv/bin/python scripts/asr_smoke.py --phrase runs/asr-validation/phrase48.wav --output runs/asr-integration --cpu`. Procesó PM/ME/DX/MX/FX sintéticos con voz contaminante solo en `Rs` (5.1) y `L` (2.0), referencias DX y controles de canales silenciosos. Ejecutó ambas variantes ASR y comprobó identidad final de fuentes y backend. En 2.0 la voz ocupa 27,000–35,591 s y cruza el límite de 30 s; las evidencias de ventanas se mantienen en el timeline original.

| Caso | Sin VAD | Con VAD | Observación |
| --- | --- | --- | --- |
| 5.1, voz en Rs | 2 candidatos, ~34,54 s de etapa | 3 candidatos, ~5,55 s de etapa | Voz localizada en Rs; sin hipótesis léxicas en otros canales de ME |
| 2.0, voz desde 27 s | 2 candidatos, ~20,41 s de etapa | 3 candidatos, ~6,33 s de etapa | VAD acotó 26,98–35,21 s; sin VAD hubo un candidato excesivamente amplio y texto inventado |

Estos tiempos incluyen carga/hash y procesamiento de varios canales y DX, no solo la duración de una frase; no son benchmarks de largometraje. Los máximos RSS del proceso y de sus hijos se registran por separado en `run.json`, son acumulativos y no incluyen memoria GPU ni permiten sumar un pico simultáneo.

El control de silencio directo devolvió `[MÚSICA]` sin VAD. Se separan únicamente etiquetas aisladas conocidas de este tipo como anotaciones no verbales del modelo, conservando texto, rol, canal y offsets en `non_speech_annotations`; no se dan por hechos acústicos. Si hay palabras junto a la etiqueta siguen siendo candidatos. No se filtran alucinaciones léxicas: por ejemplo, el caso 2.0 sin VAD conservó «a los que no se puede ver» como hipótesis adicional y un rango 0–32,8 s. Esto prueba que la integración funciona, pero también que sus rangos/textos necesitan revisión; no demuestra precisión temporal, discernibilidad ni recall de producción.

Evidencia: `runs/asr-integration/validation.json`, los dos `run.json` e informes por formato. La evaluación sigue devolviendo `insufficient_evidence` para eficacia cinematográfica. Es una sola voz sintética en condiciones controladas; no se han cubierto susurros, dialectos, voces enmascaradas ni escenas reales.

No se declara detección fiable de PFX bajo voz. La cola `dx_content_review` señala actividad, incluso durante habla; puede producir muchos avisos. La evaluación de carga y omisiones debe decidir si esto aporta utilidad o requiere un experimento distinto.

## Preparación del corpus real

1. Recibir rutas y rango expresamente autorizados. No buscar audio en otras carpetas. Usar copias de trabajo y guardar manifiestos/transcripciones/resultados bajo directorios privados excluidos de Git.
2. Identificar versiones de PM, ME, DX, MX y FX; mapas de canales, inicio, FPS y región común. Registrar optionals y las reglas específicas del título. Confirmar que se trata de mezclas originales 5.1 y estéreo; un downmix no valida 2.0 independiente.
3. Preparar 30 min de calibración y ≥60 min de evaluación reservada, ≥3 obras/fuentes independientes. Ninguna obra, variante del mismo fragmento o hablante reutilizado deliberadamente para ajustar parámetros debe cruzar particiones. Contabilizar tiempo de programa único, no multiplicarlo por canales/variantes.
4. Reunir en evaluación ≥30 eventos de lenguaje y ≥30 omisiones conocidas por formato; identificar natural/inyectado, pérdidas bajo diálogo, controles negativos y reemplazos válidos. Si no existen omisiones naturales suficientes, declarar esa limitación y no generalizar desde inyecciones.
5. Un profesional anota fragmentos completos, no solo alertas, en muestras `[start,end)`. Etiquetas: `positive`, `negative`, `ambiguous`; funciones: `language`, `omission`. Anotar por separado `policy_issue`: letra audible permitida sigue siendo lenguaje, aunque no sea incumplimiento.

En annotations JSON, cada objeto contiene los campos del ejemplo y un id globalmente único. Marcar los ids de omisiones bajo diálogo en `under_dialogue_ids` del corpus. Un segundo revisor debería resolver ambiguos y revisar una muestra independiente cuando sea posible; si no participa, declarar que la referencia depende de un solo profesional.

## Congelación y ejecución

Calibrar solo con su partición. Fijar ventanas, umbrales, versión de código, dependencias, modelos/pesos, configuración de ASR, reglas del título y matching antes de abrir la evaluación reservada. Si se cambia una decisión tras ver sus resultados, ese material pasa a desarrollo y debe obtenerse otro holdout.

El JSON de corpus contiene `entries`, cada uno con `project_id`, `source_id` (obra original), `original_fragment_id`, `layout`, `partition` (`calibration` o `evaluation`), `span`, `source_span`, `annotations_complete` y `source_hashes` (mapa de rutas a SHA-256 usado por el run). `span` usa el timeline del manifiesto; `source_span` localiza el fragmento en la obra original, también a 48 kHz, con la misma duración. El tiempo de una misma obra se calcula uniendo sus rangos originales: distintas versiones, formatos o project ids del mismo fragmento no suman minutos nuevos. Los rangos y hashes deben corresponder a los archivos anotados, no a otra versión. Un run de pocos segundos no puede justificar minutos declarados en el corpus: se contrastan sus rangos y su cobertura efectiva, y las anotaciones deben quedar dentro del corpus evaluado.

`frozen_protocol` registra `implementation_sha256` y lista de `analysis_config_sha256` admitidos, copiados de runs de calibración congelados. Debe registrar además `evaluation_config_sha256` del JSON canónico de configuración de matching (inicialmente `{"matching_slack_ms":250}`), `backend_sha256` con los hashes admitidos de binario y pesos, y `manifest_sha256` como mapa `project_id|layout` a la identidad del manifiesto de cada conjunto de evaluación. Esta última se obtiene con `inspect` antes del ensayo e incluye mapas, inicios, versiones y excepciones del título. Se confrontan los contenidos reales de los modelos y su verificación final, no solo sus rutas. El hash de configuración de evaluación se devuelve como `protocol_config_sha256`; modificar la holgura después invalida la congelación. `technical_controls_passed` solo puede ser true con evidencia registrada de las pruebas técnicas. Para medir beneficio, añadir `unaided_review_minutes` y `assisted_review_minutes` por clave de métricas `función|formato|modo|variante`; describir condiciones y sesgo si el mismo revisor ya conocía el material.

Ejecutar por separado DSP y ASR sin VAD/con VAD sobre el mismo material. Los modelos actuales ya están adquiridos y su integración básica está comprobada; cualquier cambio de binario/pesos requiere nueva identificación y comprobación. Guardar toda cobertura no evaluable, fallos y tiempos; no eliminar los difíciles de la evaluación. Ninguna etapa IA ausente equivale a un negativo.

## Matching y métricas

- Agrupar avisos solapados del mismo tipo/vista entre canales; no mutar los resultados originales. No absorber una omisión localizada dentro de la cola de actividad DX. Conservar todos los textos/evidencias contribuyentes aunque las hipótesis de transcripción difieran entre ventanas. Matching temporal uno a uno por función y programa: holgura inicial de 250 ms a ambos lados y al menos 50% del intervalo de la alerta dentro del rango expandido. No usar texto idéntico como requisito de detección. La traza de matching conserva los ids y evidencias que contribuyen a cada coincidencia.
- Declarar TP/FP/FN, precisión, recall, falsas alarmas por hora de programa, errores de inicio/fin y cobertura. Mostrar denominadores e intervalos Wilson; no interpretarlos como garantía sobre todas las obras o acentos.
- Ambiguos van aparte, no son negativos. Los positivos de lenguaje y las alertas de QC accionables tienen contadores separados. Separar 5.1/2.0, completo/reducido, variantes ASR, natural/inyectado y pérdidas bajo diálogo.
- Objetivos del experimento: lenguaje ≥95% recall y ≤6 falsas detecciones/h; omisiones completas ≥90% y ≤6 falsas alarmas/h. La función bajo diálogo debe funcionar por separado; el modo reducido es exploratorio.
- Sin suficiente corpus, cobertura, referencia o protocolo congelado, devolver `insufficient_evidence`. Si cumple los objetivos, solo se justifica continuar con validación limitada, no aprobar una entrega ni certificar Netflix.

## Escucha y comprobación manual

Comprobar en Pro Tools u otra estación apropiada el ruteo de L/R/C/LFE/Ls/Rs y el nivel de monitorización. Escuchar M&E original, PM y referencias en contexto; los previews y canales aislados son ayudas, no la referencia definitiva de discernibilidad. Si se dispone de imagen autorizada, verificar sincronía y función dramática de los reemplazos; sin imagen no afirmar esas comprobaciones.

En casos señalados: confirmar si el sonido falta, fue sustituido correctamente, cambió de nivel/canal o está en un optional según la regla del título. Conservar la decisión y motivo en revisión independiente de las detecciones. No corregir ni remezclar automáticamente.

La automatización del navegador del entorno bloqueó la apertura `file:` del informe durante la comprobación inicial. No se hizo un workaround ni se levantó un servidor para eludirlo. Posteriormente el usuario abrió el informe y aportó una revisión auditiva parcial. La generación y el escape del HTML están probados; esa revisión no equivale a una aceptación completa de la interfaz ni a pruebas manuales de todos sus flujos.

## Rendimiento

Registrar tiempo total y por función, duración única del programa, hardware declarado, versiones y memoria observada. El procesamiento es secuencial y acotado por bloques/ventanas, pero los resultados agregados crecen con el número de eventos. La lectura y hashes añaden coste que no debe esconderse. No extrapolar la velocidad de dos segundos sintéticos a una película ni asumir que memoria Python incluye consumo de GPU/Metal.

## Decisión final

Conservar la evidencia para decidir entre continuar el producto híbrido, limitarlo a comparador técnico, validar solo lenguaje, investigar PFX con un experimento adicional o declarar evidencia insuficiente. No presentar las funciones pendientes como implementadas y validadas. La certificación completa de entrega, 7.1/Atmos, loudness dialog-gated, separación neuronal y reparación automática no forman parte de este prototipo.

## Revisión independiente del código

El revisor examinó el código y reprodujo cuatro problemas importantes; no encontró problemas Critical. La pasada de corrección añadió regresiones para: duración/cobertura real del corpus e identidad de fragmentos, congelación de matching/manifiestos/binario/pesos, conservación de ventanas y mediciones tras error de lectura y deduplicación por tipo conservando hipótesis textuales distintas. Se verificaron con ciclos RED→GREEN y regresión completa, sin sustituirlos por una segunda opinión estática. Esto no equivale a validación perceptual ni a una revisión independiente adicional del código corregido.

Evidencia de la fase DSP: `.venv/bin/python -m pytest -q` → 128 pruebas pasaron antes de adquirir modelos. Tras añadir la selección CPU y la separación de etiquetas no verbales, la regresión pasó **134 pruebas**; además se ejecutó la integración real optativa descrita arriba. `.venv/bin/python scripts/smoke.py --output runs/final-smoke` produjo casos 2.0/5.1 correctos y defectuosos completos, identidades de originales verificadas, cancelación conservada y evaluación sintética con evidencia insuficiente. `pip check` no encontró requisitos rotos. Estos tiempos son de pruebas pequeñas en este entorno, no un benchmark de largometraje.

Decisiones de alcance mantenidas tras la revisión: recall/PFX/valor dramático pendientes de corpus anotado y evaluación independiente; el ensayo real estéreo no determina el rendimiento de otros programas ni mide memoria GPU; aceptación completa en Pro Tools e interfaz pendiente. La certificación y funciones excluidas siguen fuera de alcance. La adquisición y la inferencia básica que antes estaban pendientes sí se completaron, con integridad cotejada contra hashes publicados e inventario de fuentes/licencias. Los controles han mostrado errores reales de ASR, por lo que el prototipo no debe usarse como aceptación automática de producción.
