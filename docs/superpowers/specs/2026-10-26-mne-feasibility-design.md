# M&E QC Inspector: diseño de la prueba de viabilidad

Fecha: 2026-10-26. Estado: diseño y plan aprobados expresamente por el usuario. Núcleo DSP, informes y evaluación implementados y probados con fixtures; ASR comprobado con mocks y con una frase sintética local real. Dependencias Python en `.venv`; pesos y whisper.cpp adquiridos con autorización posterior expresa, hashes y commit verificados. Material real y validación perceptual aplazados por el usuario.

## 1. Objetivo y acuerdos

Determinar si una herramienta local puede ayudar a un profesional a revisar M&E: comprobar la reconstrucción del printmaster, localizar diferencias y proponer incidencias de lenguaje discernible y sonidos no dialogados ausentes. El resultado de esta fase será evidencia para decidir si construir la aplicación, reducir su alcance o descartar una función.

El usuario ha aceptado el enfoque de prototipo híbrido de viabilidad y que 5.1 y 2.0 pueden cubrir la primera versión. Puede aportar stems originales de Music y Effects además de DX. Confirma disponer de un Mac M2 con 16 GB de RAM y de fragmentos autorizados de obras distintas con printmaster, M&E y stems, incluyendo mezclas 5.1 y estéreo originales, que puede revisar auditivamente. Estos datos proceden del usuario; todavía no se han inspeccionado archivos ni medido rendimiento. Se mantienen dos modalidades: completa (PM, M&E, DX, MX y FX) y reducida (PM, M&E y DX), con conclusiones limitadas en la segunda. 7.1 queda fuera de esta prueba.

Propuesta de alcance por modalidad: los objetivos de detección de omisiones se exigirán al modo completo; en el reducido esa función será exploratoria, con métricas propias y sin prometer paridad. La detección de lenguaje se realizará sobre M&E y DX y será común a ambas modalidades; deberá quedar evaluada sin depender de MX/FX. La reconstrucción del original solo se exigirá cuando estén disponibles los cinco elementos compatibles.

Decisiones propuestas por el diseñador e incluidas en el diseño aprobado: comenzar la evaluación de lenguaje por castellano; aplicación futura para macOS; criterios cuantitativos de la sección 8; tamaño de corpus de la sección 7. No hay evidencia actual de precisión ni velocidad.

## 2. Enfoques considerados

1. Comparador DSP primero y detección con IA después: sencillo de verificar, pero pospone las incertidumbres que podrían hacer inviable el producto.
2. Prototipo híbrido antes de construir la aplicación: compara un baseline DSP con DSP más detección vocal y reconocimiento de voz locales. Prueba pronto las omisiones y falsas alarmas. Es el enfoque recomendado y aceptado por el usuario.

La prueba no requiere una interfaz completa, integración con Pro Tools, plugins, entrenamiento de modelos, separación generativa obligatoria ni un agente que interprete audio. El constructor elegirá herramientas mantenibles tras comprobar versiones, licencias y compatibilidad. VAD y ASR preentrenados son candidatos, no componentes ya validados.

## 3. Alcance y límites normativos

Referencia inicial: documentación pública para contenido Netflix branded, consultada el 2026-10-26. No se presume aplicable a cualquier entrega licenciada ni a excepciones específicas de un título. Antes de evaluar cumplimiento de una entrega se identificará su Delivery Requirements List y el perfil aplicable.

- [Sound Mix Specifications and Best Practices](https://studiopartner.netflix.net/studio/branded-sound-mix-spec-and-best-practices): PCM 48 kHz/24-bit para masters y stems originales/M&E; requisitos de true peak; reconstrucción del original con DX, Music y Effects; representación de sonidos no dialogados en M&E.
- [M&E Creation & Delivery Guidelines](https://studiopartner.netflix.net/studio/m-and-e-creation-and-delivery-guidelines): M&E rellenado, conservación de música, cambios de FX limitados al relleno necesario, optionals y referencias de QC.

El prototipo comprobará formato y true peak como controles técnicos acotados. No implementará una certificación completa de Netflix ni presentará una medición genérica de loudness como equivalente a dialog-gated loudness. Si se añade esta última, exigirá una validación específica; queda fuera del alcance mínimo. Las reglas y fuentes se identificarán por fecha en el informe.

La representación de vocalizaciones, música vocal, walla y archivos en optionals depende de contexto. Las excepciones de un título deberán registrarse. La detección de voz no equivale a lenguaje discernible ni a incumplimiento.

## 4. Entradas y precondiciones

Se aceptarán WAV/BWAV PCM discretos o interleaved para análisis, aunque aceptar un contenedor no significa aprobarlo para entrega. RF64 debe contemplarse al elegir el lector. Formatos iniciales: 2.0 y 5.1, a 48 kHz/24-bit para los casos de cumplimiento. Los casos deliberadamente incorrectos deberán producir diagnóstico, sin conversión silenciosa.

Cada conjunto identificará rol de archivo, versión, configuración, mapa explícito de canales, sample rate, bit depth declarado, cantidad de muestras, inicio temporal y procedencia. Duración e inicio son comprobaciones distintas: líderes y colas pueden diferir legítimamente. La región de programa común y la cobertura fuera de ella se declararán; no se recortará material para ocultar discrepancias.

Para sumas verificables, PM, DX, MX y FX deben ser exports de la misma versión y configuración, con procesamiento, ganancias, paneo y retornos pertinentes. Una guía mono de DX no permite reconstruir el PM 5.1. Si no se cumplen estas precondiciones, el ensayo de reconstrucción será no evaluable y la revisión heurística podrá continuar con la limitación visible.

Optionals y stems de Foley/PFX serán entradas auxiliares si están disponibles; no obligatorias para ejecutar el modo mínimo. Su ausencia limita atribuir pérdidas. Una referencia de imagen autorizada mejorará la revisión profesional; sin ella no se afirmará sincronía audiovisual ni corrección dramática.

## 5. Análisis propuesto

### 5.1 Integridad y reconstrucción del original

Leer los canales sin normalizar, resamplear ni mezclar a mono. Registrar identidad de las entradas y metadatos necesarios. Procesar por bloques para no cargar un programa entero en RAM.

Calcular el residual PM − (DX + MX + FX) sobre las muestras originales y reportar nivel, distribución temporal y canal. Antes de atribuirlo a contenido, diagnosticar offsets, polaridad, canales intercambiados y discrepancias de exportación. Una tolerancia numérica justificada para cuantización o dither no es una tolerancia perceptual general.

Las pruebas sintéticas sin dither tendrán resultados esperados exactos o acotados por su cuantización conocida. En mezclas reales se reportará el residual medido y las condiciones; no se inventará un umbral universal de cancelación.

Una alineación auxiliar podrá estimar un offset global si hay evidencia suficiente. Siempre conservará el resultado original, el offset y el motivo. Se prohíben time stretching, alineación variable y compensaciones automáticas de EQ/dinámica que puedan esconder un defecto.

### 5.2 Comparación del M&E

Calcular PM − (M&E + DX) como diferencia descriptiva. Comparar también MX + FX con M&E para localizar contenido común, añadidos y cambios. Ninguna de estas restas constituye un veredicto de integridad.

Utilizar análisis por canal y ventanas temporales: correlación, energía por bandas, envolventes y transitorios. Las hipótesis de omisión y de cambio de nivel se conservarán como categorías diferentes. Ganancias auxiliares limitadas, si se usan, no sustituirán las mediciones a nivel original.

Caso crítico A: DX contiene voz y un paso; M&E sustituye ese paso por Foley. La suma puede dejar residual aunque el M&E sea correcto.

Caso crítico B: DX contiene voz y un paso; M&E omite el paso. La suma puede cancelar aunque el M&E sea incompleto.

Por ello se examinará también el contenido no verbal de DX, incluso mientras habla un personaje. Correlación y comparación temporal pueden proponer candidatos; un reemplazo con otra onda puede ser válido. Con PM, DX y M&E no siempre puede identificarse de forma única qué fuente falta. Más stems reducen esa ambigüedad, pero no la eliminan.

No se descartarán intervalos por contener diálogo. No se tratará ausencia de correlación como prueba suficiente de omisión. La separación de fuentes solo se considerará como experimento adicional si el baseline no permite evaluar estos casos; requerirá propuesta específica antes de incorporar modelos grandes.

### 5.3 Lenguaje en M&E

Evaluar actividad vocal y ASR local sobre canales individuales y vistas auxiliares documentadas. Escanear M&E completo, no solo los momentos donde DX contiene voz. Comparar con DX para apoyar hipótesis de fuga, sin exigir coincidencia: puede haber radio, walla u otra voz independiente.

Evaluar dos variantes con el mismo corpus: VAD como selector de candidatos y ASR con ventanas regulares independientes del VAD. Esto medirá las omisiones que introduciría el selector y evitará asumir que un confirmador recupera lo que nunca recibe.

La ausencia de texto no demuestra ausencia de lenguaje. Una transcripción no demuestra que las palabras existan o sean discernibles. La escucha de M&E original a nivel documentado y con su configuración de canales será la referencia de discernibilidad. Las vistas aisladas o realzadas serán evidencia auxiliar, identificada como tal.

Referencia de limitaciones del ASR: [Whisper model card](https://github.com/openai/whisper/blob/main/model-card.md), que documenta posibles transcripciones inexistentes y diferencias de rendimiento entre idiomas. La selección del modelo concreto se realizará antes del ensayo, con versión y pesos identificados.

### 5.4 Evidencia y estados

Cada candidato conservará rango de muestras, timecode si FPS e inicio están definidos, canales implicados, categoría, razón y enlaces locales para revisar los originales. Los resultados auxiliares nunca sustituirán al audio fuente. El informe deberá permitir distinguir detección automática, decisión humana y excepción del título.

Estados: error técnico comprobado; posible incidencia; aceptado por revisión; no evaluable. Una transcripción o un score de modelo no se convertirán en probabilidad calibrada. Sin alertas no significa aprobado. Un proceso cancelado, fallido o parcialmente ejecutado deberá declarar su cobertura y las etapas pendientes.

## 6. Seguridad y alcance operativo

No modificar originales. Trabajar con material sintético y copias autorizadas. Audio, transcripciones y resultados permanecerán locales; no se usarán APIs ni servicios externos para procesarlos. Ningún material se usará para entrenar modelos.

Las descargas de pesos, nuevas dependencias y almacenamiento necesario deberán presentarse con tamaño, origen y licencia antes de ejecutarse cuando sean necesarios. No se instalarán integraciones por defecto. No habrá reparación automática, remezcla, exportación de nuevos masters, subida a Content Hub ni publicaciones.

Al comenzar el diseño, el repositorio estaba vacío y sin commits. Se han comprobado Python, FFmpeg/ffprobe, las dependencias y los runtimes con fixtures. El usuario declaró un Mac M2 con 16 GB de RAM; la ejecución Metal identificó después Apple M2 Pro. Hay tiempos de controles pequeños, no de programas reales, y sigue pendiente un benchmark de producción. El procesamiento por bloques y las etapas de IA secuenciales evitan cargar todos los canales y modelos simultáneamente. No se presupone velocidad en tiempo real ni necesidad de comprar hardware. DSP e informes funcionan sin aceleración de IA; ASR tiene selección CPU explícita y una etapa no disponible se marca pendiente, sin reducir cobertura silenciosamente.

## 7. Corpus y protocolo de evaluación

Primero, fixtures pequeños y reproducibles con señales y eventos conocidos, cubriendo 2.0 y 5.1. Después, material cinematográfico local autorizado, con anotación profesional. No se abrirá material de producción sin una ruta y alcance proporcionados por el usuario.

Matriz mínima de escenarios:

| Grupo | Casos obligatorios |
| --- | --- |
| Reconstrucción | Suma correcta; offset de una muestra; inversión de polaridad; canales intercambiados; stem de otra versión; ganancia alterada; guía mono insuficiente |
| Formato y cobertura | Sample rate/bit depth incorrectos; archivo truncado; cola perdida; líderes distintos; mapa de canales desconocido; interrupción de una etapa |
| Diferencias legítimas | Foley alternativo; relleno de PFX; cambio local de FX justificado; contenido válido en optionals |
| Omisiones | Música, impacto, paso y ambiente ausentes; pérdidas mientras hay diálogo; desaparición parcial; sonido trasladado a otro canal |
| Lenguaje | Fuga de DX; voz independiente de DX; palabra breve; susurro; voz distante/reverberante/filtrada; voz bajo música; voz solo en surround |
| Controles difíciles | Música vocal; instrumentos parecidos a voz; respiración; risa; esfuerzo; walla no discernible; mezcla sin habla; efectos con estructura vocal |
| Fallos del supuesto de null | M&E correcto con residual y M&E incompleto con cancelación, según sección 5.2 |

Propuesta de corpus real inicial: 30 minutos para calibración y al menos 60 minutos reservados para evaluación, procedentes de al menos tres obras o fuentes independientes. Las obras usadas para calibrar no se usarán para evaluar. Ninguna variante del mismo fragmento cruzará esa separación. Incluir muestras reales de ambos formatos; un downmix no demuestra rendimiento sobre mezclas estéreo independientes.

Como objetivo de preparación, reunir al menos 30 eventos discernibles y 30 omisiones conocidas por formato en la evaluación, sin contar copias por canal o distintas versiones como eventos independientes. Las omisiones inyectadas se identificarán como tales. Si no se consigue este material, el informe será exploratorio y no permitirá aprobar la función completa. Esta escala no constituye una validación de producción y deberá ampliarse si los resultados justifican continuar.

Las etiquetas distinguirán presente/ausente, reemplazo aceptable, discernible/no discernible y ambiguo. Se anotarán por separado presencia de lenguaje y posible incumplimiento: una canción autorizada puede contener lenguaje sin ser una incidencia del título. Se medirán por separado la detección de lenguaje y las alertas de QC accionables, sin penalizar como alucinación una letra realmente audible ni presentar todo positivo de lenguaje como error. Un profesional revisará los fragmentos completos de evaluación, no solo las alertas, para contabilizar falsos negativos. Un segundo profesional revisará los ambiguos y una muestra de positivos/negativos cuando sea posible; si no, se declarará que las etiquetas dependen de un solo revisor. Lo ambiguo se reportará aparte, sin convertirlo en negativo.

Definir antes de la evaluación el emparejamiento uno a uno entre alertas y anotaciones, las reglas para unir avisos del mismo evento y las tolerancias temporales por tipo. Un evento encontrado en varios canales contará una vez a nivel de programa; se conservará el detalle por canal. No ajustar estas reglas ni los umbrales después de observar la evaluación reservada.

## 8. Métricas y decisiones

Comparar baseline DSP y sistema híbrido sobre los mismos archivos. Reportar por función, formato y escenario: verdaderos positivos, falsos positivos, falsos negativos, precisión, recall, falsas alarmas por hora de programa, error temporal y segmentos no evaluables. Conservar los denominadores; no agrupar resultados para ocultar un escenario fallido.

Medir además tiempo de cómputo frente a duración de programa, memoria máxima cuando se pueda medir, espacio de resultados y minutos de revisión profesional. La carga de revisión debe compararse con una revisión equivalente sin asistencia; cuando el mismo revisor haya oído ya el material se documentará el sesgo. El equipo objetivo es el Mac M2 con 16 GB confirmado por el usuario; no se fija una velocidad objetivo hasta medir el baseline local.

Criterios de continuación incluidos en el diseño aprobado, a fijar en la configuración antes del ensayo:

- Todos los casos técnicos controlados obligatorios deben producir el resultado esperado o un no evaluable justificado. Nunca aprobar por cancelación un M&E incompleto ni rechazar automáticamente un reemplazo legítimo.
- Lenguaje: recall de al menos 95% de eventos confirmados y no más de 6 falsas detecciones por hora en la evaluación reservada. Reportar también las falsas alertas de QC después de aplicar las reglas del título. Estos son objetivos de producto propuestos, no requisitos de Netflix ni resultados obtenidos.
- Omisiones en modo completo: recall de al menos 90% de omisiones conocidas revisadas y no más de 6 falsas alarmas por hora, desglosando las pérdidas bajo diálogo y las inyectadas frente a naturales. El fallo del subgrupo bajo diálogo impedirá presentar esa capacidad como resuelta. En modo reducido se reportarán resultados exploratorios separados y no se exigirá paridad.
- Cada objetivo debe alcanzarse y reportarse por separado para 5.1 y 2.0, con al menos 30 eventos de lenguaje y 30 omisiones conocidas por formato en la evaluación; falta de eventos suficientes en uno de ellos implica evidencia insuficiente para ese formato. Reportar incertidumbre de los porcentajes; cumplir una cifra sobre esta muestra pequeña solo justifica continuar, nunca garantizar sensibilidad en producción.
- Si solo funciona la comparación técnica, proponer un comparador limitado. Si funciona lenguaje pero no omisiones, ofrecer ese alcance explícito. Si la carga de revisión no mejora o faltan datos, no declarar viable la aplicación completa.

## 9. Entregables y siguiente aprobación

La fase producirá un ejecutor de experimentos, fixtures y manifiestos reproducibles, informe local de incidencias y una evaluación con ejemplos de aciertos, omisiones y falsas alarmas. No será todavía una aplicación de entrega ni una herramienta certificada.

El diseño y el plan están aprobados, y se ha ejecutado la parte independiente del corpus. Equipo objetivo declarado: Mac M2 con 16 GB de RAM. El usuario ha decidido aportar el material autorizado más adelante. Quedan por seleccionar los fragmentos, comprobar tamaño/diversidad y acordar la revisión. Antes de procesar material real: rutas autorizadas, configuración de canales, versiones, idioma y reglas del título. Los dos pesos ya están adquiridos con autorización expresa; se cotejaron revisión/tamaño/hash publicados y se ejecutó una inferencia real, incluida integración multicanal con/sin VAD. `models.lock.json` registra la adquisición. La validación perceptual sigue pendiente y los errores de ASR observados permanecen explícitos.

Revisión independiente del diseño: identificó la atribución de objetivos al modo completo/reducido y la disponibilidad pendiente de corpus y hardware. Se explicitó la propuesta para las modalidades. Después de la revisión, el usuario confirmó hardware y disponibilidad general de material; su selección, anotación y suficiencia para los criterios de evaluación siguen pendientes.

Las decisiones ordinarias sobre bibliotecas, estructura interna, tamaño de bloques y formato de informes quedan a cargo del constructor; deberán justificarse sin añadir infraestructura innecesaria. Cambios de arquitectura, servicios externos, entrenamiento o extensión a 7.1/Atmos requieren volver a acordar el alcance.
