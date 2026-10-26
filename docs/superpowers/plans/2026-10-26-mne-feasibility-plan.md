# M&E QC Inspector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task; superpowers:subagent-driven-development is an alternative if explicitly selected by the user. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir y evaluar un prototipo local que mida diferencias entre printmaster, stems y M&E, proponga incidencias y permita comprobar su utilidad con escucha profesional.

**Architecture:** Paquete Python con CLI, lectura de audio por bloques, análisis DSP sin modificar originales y un adaptador de ASR local ejecutado secuencialmente. Manifiestos JSON describen canales y tiempos; resultados JSON y HTML estático separan hechos medidos, candidatos y revisión humana. La evaluación reservada es una etapa independiente de la calibración.

**Tech Stack:** Python 3.14 disponible, NumPy, SciPy, SoundFile/libsndfile, pytest para desarrollo; FFmpeg/ffprobe ya disponibles; whisper.cpp con Whisper small multilingüe y Silero VAD como primera configuración experimental. Entorno `.venv` local, sin modificar Python del sistema. Comprobar wheels compatibles y fijar versiones antes de instalar; si no los hay, proponer una alternativa de Python sin instalarla automáticamente.

**Spec:** [Diseño aprobado](../specs/2026-10-26-mne-feasibility-design.md).

Estado: plan aprobado; ejecución directa autorizada. Núcleo DSP, informes, CLI y evaluación implementados y probados con fixtures; pesos/runtime adquiridos con autorización posterior y ASR comprobado con una frase sintética local, CPU/Metal y ambas variantes. El usuario ha aplazado expresamente el corpus real y la validación perceptual. No realizar commits ni pushes sin autorización; se conserva la rama local `codex/mne-feasibility`.

Resumen de ejecución: tareas 1–7 implementadas y verificadas a nivel técnico/controles pequeños, incluida tarea 5 con inferencia real sobre una voz sintética; tarea 8 completada para smoke/documentación y pendiente para corpus/escucha profesional. Resultado actual: 134 pruebas pasan; smoke DSP y smoke ASR real completados en ambos formatos. Esto no implica eficacia cinematográfica. Los checkboxes siguientes conservan el detalle del plan; evidencia y límites están en `docs/VALIDATION.md` y el registro de ejecución local.

## Global Constraints

- Equipo objetivo: Mac M2 con 16 GB de RAM. Formatos iniciales: 2.0 y 5.1. 7.1/Atmos fuera de alcance.
- WAV/BWAV PCM discretos o interleaved para análisis; 48 kHz/24-bit para los casos de cumplimiento. Aceptar una entrada no significa aprobar su formato de entrega.
- No modificar originales. Audio, transcripciones y resultados permanecerán locales; no se usarán APIs ni servicios externos para procesarlos.
- No normalizar, resamplear ni mezclar a mono la rama DSP. La rama ASR podrá crear derivados temporales explícitos y documentados.
- Se prohíben time stretching, alineación variable y compensaciones automáticas de EQ/dinámica que puedan esconder un defecto.
- No habrá reparación automática, remezcla, exportación de nuevos masters, subida a Content Hub ni publicaciones.
- Modos completo (PM, M&E, DX, MX, FX) y reducido (PM, M&E, DX). La reconstrucción original requiere stems compatibles; las omisiones del modo reducido son exploratorias.
- Sin alertas no significa aprobado. Un proceso parcial o fallido conserva evidencia y declara cobertura incompleta.
- Objetivos por formato: lenguaje recall ≥95%, ≤6 falsas detecciones/h; omisiones en modo completo recall ≥90%, ≤6 falsas alarmas/h. Al menos 30 eventos de lenguaje y 30 omisiones por formato. Son objetivos experimentales, no requisitos de Netflix.
- Corpus: 30 minutos para calibración y ≥60 minutos de evaluación reservada, ≥3 obras/fuentes; ninguna obra o variante cruza particiones. Revisar ambos formatos y declarar insuficiencia cuando no haya datos suficientes.
- No instalar dependencias ni descargar pesos en esta fase de planificación. En ejecución, presentar antes origen, licencia, tamaño y destino de descargas; no introducir modelos grandes adicionales sin acuerdo.

## Review Focus

1. Canales discretos con orden ambiguo o etiquetas repetidas: no inferir un mapa por número de canales; prueba de tarea 1.
2. Cambio de fuente durante un análisis o preview: conservar resultados parciales y bloquear evidencia derivada incoherente; prueba de tarea 6.
3. Silencio/correlación ambigua y offset de una muestra: nunca inventar una alineación fiable; prueba de tarea 2.
4. Frase que cruza ventanas ASR y voz solo en surround: mantener tiempos de origen y evitar pérdidas/duplicados; prueba de tarea 5.
5. Avisos multicanal duplicados, cero positivos y corpus incompleto: no inflar recall ni emitir un gate favorable; prueba de tarea 7.

## Archivos e interfaces compartidas

Crear únicamente lo necesario para cada tarea. No generar carpetas vacías ni una interfaz gráfica completa.

| Archivos | Responsabilidad |
| --- | --- |
| `pyproject.toml`, `.gitignore`, `requirements-dev.lock`, `README.md` | Instalación local reproducible, exclusión de audio/modelos/resultados, instrucciones |
| `src/mne_qc/domain.py`, `manifest.py` | Contratos, roles, validación de manifiestos y estados |
| `src/mne_qc/audio.py`, `timeline.py` | Lectura por bloques, metadata, cobertura, referencias temporales |
| `src/mne_qc/reconstruction.py`, `technical.py` | Residuales, diagnósticos y controles técnicos |
| `src/mne_qc/compare.py` | Candidatos de diferencias/omisiones y ambigüedad DX |
| `src/mne_qc/speech.py` | Preparación auxiliar y ejecución/parser de whisper-cli |
| `src/mne_qc/runner.py`, `report.py`, `cli.py`, `__main__.py` | Orquestación, cancelación, resultados e interfaz de ejecución |
| `src/mne_qc/evaluation.py` | Matching, métricas, suficiencia y gates |
| `tests/conftest.py`, `tests/test_*.py` | Fixtures pequeños y pruebas por responsabilidad |
| `tests/data/` | Solo metadata y salidas de herramientas pequeñas/sintéticas, sin producción |
| `examples/manifest.synthetic.json`, `examples/annotations.synthetic.json` | Ejemplos seguros ejecutables |
| `scripts/make_fixtures.py`, `scripts/smoke.py` | Generación reproducible y smoke de extremo a extremo |
| `docs/VALIDATION.md`, `docs/DEPENDENCIES.md` | Procedimiento de escucha, límites, versiones, licencias y descargas |

Contratos en `domain.py`, dataclasses salvo JSON de entrada/salida:

- `ChannelSource(path: Path, file_channel: int, label: str)`; índices base cero, labels `L,R` o `L,R,C,LFE,Ls,Rs`. El orden de almacenamiento siempre se declara. Mapeo de 5.1(side) a Ls/Rs se registra, nunca se presume 7.1.
- `Asset(role: str, version: str, channels: tuple[ChannelSource, ...], start_sample: int)`; inicio en timeline de programa a 48 kHz; cada archivo conserva su sample rate nativo para diagnosticar incompatibilidades.
- `Manifest(project_id: str, layout: str, assets: dict[str, Asset], program_span: tuple[int,int], fps: Fraction|None, start_frame: int|None, drop_frame: bool, exceptions: list[dict])`.
- `AudioInfo(sample_rate: int, declared_bits: int|None, subtype: str, frames: int, channels: int, bwf_time_reference: int|None)`; profundidad declarada, no bits efectivamente ocupados.
- `Span(start: int, end: int)` usa muestras e intervalo semiabierto; `Block(start: int, samples: ndarray)` usa float64, forma `(frames, canales)`.
- `Finding(id: str, kind: str, state: str, span: Span|None, channels: tuple[str,...], evidence: dict)`; estados del diseño, sin probabilidad inventada.
- `StageResult(stage: str, status: str, coverage: list[Span], findings: list[Finding], measurements: dict, reason: str|None)`; status `complete/partial/failed/not_evaluable/not_run`.
- `RunResult(status: str, stages: list[StageResult], input_identity: dict, environment: dict)`; status `complete/partial/failed`. Complete significa cómputo solicitado terminado, nunca entrega aprobada.
- `SpeechConfig(binary: Path, model: Path, vad_model: Path|None, language: str, mode: str, threads: int, timeout_seconds: float=180.0, use_gpu: bool=True)` y `Detection(span: Span, channels: tuple[str,...], text: str, evidence: dict)`. `mode` es `fixed` o `vad`; la opción CLI `both` ejecuta ambas configuraciones secuencialmente y conserva etapas y métricas separadas. `--no-gpu` expresa el fallback CPU previsto y queda en la identidad de configuración.
- `Annotation(id: str, project_id: str, layout: str, kind: str, span: Span, channels: tuple[str,...], label: str, injected: bool, policy_issue: bool|None)` y `EvaluationResult(metrics: dict, eligibility: dict, gates: dict)`.
- `StopToken.cancelled() -> bool`; consultar entre bloques y gestionar terminación de procesos hijos.

Los campos se serializan a JSON con `schema_version=1`; no se guardan ndarrays en JSON. La configuración de análisis conserva todos los umbrales, ventanas y variantes. Rutas relativas del manifiesto se resuelven respecto a su archivo; no expandir URLs ni ejecutar expresiones. `fingerprint` devuelve `path,size_bytes,mtime_ns,sha256,status,reason`. `reconstruct` devuelve en `measurements` `peak_abs_by_channel,rms_by_channel,raw_residual,alignment_diagnostics`; el resultado alineado auxiliar, si se solicita, ocupa otra clave. La configuración DSP usa `stft_frames,hop_frames,aggregate_seconds,correlation_threshold,energy_delta_db,activity_floor_dbfs`; los tres últimos se fijan con calibración, no con holdout. La evaluación devuelve `metrics` por función/formato/modo/variante y `eligibility` con los motivos de insuficiencia.

## Task 1: Entrada verificable, lectura y fixtures

**Files:** Crear `pyproject.toml`, `.gitignore`, `requirements-dev.lock`, `src/mne_qc/{__init__,domain,manifest,audio}.py`, `tests/conftest.py`, `tests/test_manifest_audio.py`, `scripts/make_fixtures.py`, `examples/manifest.synthetic.json`.

**Interfaces:** Producir `load_manifest(path: Path) -> Manifest`, `inspect_audio(path: Path) -> AudioInfo`, `iter_blocks(asset: Asset, span: Span, block_frames: int=65536) -> Iterator[Block]`, `fingerprint(path: Path, stop: StopToken) -> dict`. El fingerprint completo usa SHA-256 por bloques; si se interrumpe queda nulo con causa, nunca inventado.

- [ ] Escribir pruebas `test_discrete_matches_interleaved`, `test_duplicate_or_unknown_channel_rejected`, `test_wrong_rate_or_bits_diagnosed`, `test_short_discrete_channel_not_zero_padded`, `test_truncated_wav_rejected`, `test_rf64_and_bwf_metadata`, `test_cancelled_fingerprint_is_explicit`. Afirmar igualdad de muestras y etiquetas en 2.0/5.1; RF64 pequeño con `ds64` válido y BWAV con time reference >2^32; sin inferir FPS del BWF. Cubrir `fmt` extensible y profundidad desconocida.
- [ ] Tras aprobación del plan, comprobar wheels binarios de NumPy/SciPy/SoundFile/pytest para Python 3.14 antes de crear `.venv`. Registrar versiones resueltas y licencias en el lock/documentación; no sustituir Python del sistema si hay incompatibilidad. Instalar solo en el entorno local autorizado.
- [ ] Ejecutar `.venv/bin/python -m pytest tests/test_manifest_audio.py -q`; esperar fallos de los contratos aún no implementados, no fallos de dependencias.
- [ ] Implementar contratos, validación y lectura de solo lectura con SoundFile y ffprobe como fuente de metadata complementaria. Comparar cantidad declarada de muestras con decodificación real para detectar truncados; no aceptar discrepancias como silencio. La manifestación explícita de origen prevalece solo si se registra el conflicto con BWF.
- [ ] Generar señales sintéticas deterministas PCM24 a amplitudes sin clipping: música tonal variable, ambientes, transitorios y componente DX. No simular voz humana mediante tonos para medir ASR. Excluir `.venv/`, `models/`, `vendor/`, `runs/`, `data/local/` y audio generado del seguimiento Git.
- [ ] Repetir las pruebas; esperar todas aprobadas y datos originales intactos. Guardar la evidencia de versiones, no hacer commit.

Asserts esenciales: `assert discrete_samples.shape == interleaved_samples.shape`; `assert np.array_equal(discrete_samples, interleaved_samples)`; `assert cancelled_identity["sha256"] is None`; `assert inspect_audio(bwf_path).bwf_time_reference == 2**32 + 48000`.

## Task 2: Timeline y reconstrucción del printmaster

**Files:** Crear `src/mne_qc/timeline.py`, `src/mne_qc/reconstruction.py`, `tests/test_reconstruction.py`, `tests/test_timeline.py`.

**Interfaces:** Consumir contratos de tarea 1. Producir `program_coverage(manifest: Manifest) -> dict`, `format_timecode(sample: int, fps: Fraction, start_frame: int, drop_frame: bool) -> str`, `reconstruct(manifest: Manifest, stop: StopToken) -> StageResult`, `describe_me_residual(manifest: Manifest, stop: StopToken) -> StageResult`.

- [ ] Escribir `test_exact_sum_cancels`, `test_one_sample_offset_is_reported_not_fixed`, `test_polarity_gain_and_swap_are_candidates`, `test_silence_has_no_reliable_alignment`, `test_leader_and_tail_coverage_visible`, `test_mono_dx_5_1_is_not_evaluable`, `test_reduced_mode_has_no_original_reconstruction`. Afirmar residual máximo exactamente cero en fixtures cuantizados sin dither, offsets en muestras, cobertura explícita y conservación del resultado sin alinear.
- [ ] Añadir `test_timecode_24_25_23976`, `test_dropframe_2997_boundaries` y `test_no_fps_reports_samples_only`; usar `Fraction(24000,1001)` y `Fraction(30000,1001)`, distinguir frame count de etiquetas drop frame y rechazar DF con tasas incompatibles.
- [ ] Ejecutar `.venv/bin/python -m pytest tests/test_reconstruction.py tests/test_timeline.py -q`; verificar rojo por comportamiento faltante.
- [ ] Implementar suma/resta float64 por canal sin alterar entradas. Reportar pico, RMS y ventanas del residual; no declarar un umbral perceptual universal. Diagnósticos de offset global limitados a ±1 s sobre ventanas activas, con score y ambigüedad; cualquier compensación se guarda aparte y nunca sustituye el original. Diferencias de versión o cobertura invalidan la conclusión de reconstrucción, no la escucha heurística.
- [ ] Implementar caso A del diseño (reemplazo válido con residual) y caso B (omisión con cancelación) como fixtures obligatorios. Afirmar que ninguno produce aprobación/rechazo de M&E basado en la resta.
- [ ] Repetir ambos módulos de pruebas; esperar aprobación de todos los casos en 2.0 y 5.1.

Asserts esenciales: `assert reconstruct(exact, stop).measurements["peak_abs_by_channel"] == [0.0] * len(exact.assets["PM"].channels)`; `assert reconstruct(reduced, stop).status == "not_evaluable"`. Las fixtures `exact` y `reduced` se crean en `tests/conftest.py` con los contratos de tarea 1.

## Task 3: Controles técnicos acotados

**Files:** Crear `src/mne_qc/technical.py`, `tests/test_technical.py`, salidas sintéticas de ffprobe/FFmpeg en `tests/data/`.

**Interfaces:** `check_technical(manifest: Manifest, stop: StopToken) -> StageResult`; usar lectura/metadatos de tarea 1 y cobertura de tarea 2.

- [ ] Escribir `test_pcm48_24_profile`, `test_unknown_bits_not_pass`, `test_true_peak_above_minus2_reported`, `test_true_peak_boundary_is_measured`, `test_ffmpeg_failure_not_zero_peak`, `test_each_channel_and_lfe_scanned`. No sustituir true peak por sample peak ni sacar el LFE del control de picos.
- [ ] Ejecutar `.venv/bin/python -m pytest tests/test_technical.py -q`; verificar fallo esperado.
- [ ] Implementar FFmpeg `ebur128=peak=true` por canales conservando su escala, con argumentos estructurados y parser de resumen comprobado contra la versión instalada. Comparar límite −2 dBTP del perfil aplicable; indicar precisión limitada de la salida. Conservar valor y margen de casos frontera. No emitir resultados de dialog-gated loudness ni afirmar certificación del medidor.
- [ ] Verificar con señales de pico conocido y una señal intersample comprobada mediante referencia de sobremuestreo independiente de alta resolución. Fijar antes de medir una tolerancia de validación del medidor de 0,2 dB para esos fixtures; si no cumple, marcar true peak no validado. Esto es verificación de implementación, no certificación de norma.
- [ ] Repetir pruebas. El filtro ya está disponible según ayuda local; si falla al ejecutarlo, conservar diagnóstico y marcar etapa incompleta.

Asserts esenciales: `assert abs(measured_dbtp - reference_dbtp) <= 0.2`; `assert check_technical(broken_backend, stop).status != "complete"`. El fallo de backend se inyecta con mock, no dañando el FFmpeg instalado.

## Task 4: Candidatos de omisión y cambios legítimos

**Files:** Crear `src/mne_qc/compare.py`, `tests/test_compare.py`.

**Interfaces:** `compare_me(manifest: Manifest, config: dict, stop: StopToken) -> StageResult`; consumir tareas 1–2. Configuración inicial: STFT de 2048 muestras, hop 512; agregación de evidencia en ventanas de 0,5 s, con solapamiento; umbrales en configuración registrados y calibrados después.

- [ ] Escribir `test_missing_music_impact_ambience`, `test_partial_missing_event`, `test_foley_replacement_not_automatic_error`, `test_dx_step_missing_despite_null`, `test_pfx_during_speech_retained_for_review`, `test_channel_relocation_not_global_omission`, `test_optional_is_not_evidence_of_main_body_presence`. Exigir categoría, rango y evidencia; no una identificación semántica inventada del sonido.
- [ ] Ejecutar `.venv/bin/python -m pytest tests/test_compare.py -q`; confirmar rojo.
- [ ] Implementar comparación MX+FX frente a M&E mediante correlación normalizada, envolventes, transitorios y energía en bandas, conservando todas las ventanas y canales necesarios en forma agregada. Separar `level_change`, `possible_omission`, `additional_or_replacement`, `spatial_change` y `dx_content_review`. No convertir una correlación baja en omisión comprobada.
- [ ] Para DX, proponer rangos de revisión por actividad/transitorios y evidencia no explicada por MX/FX; mantenerlos aun con habla. El baseline puede no separar un paso de una voz: registrar incertidumbre y medir sus fallos, sin añadir separación neuronal como si ya estuviera aprobada. En modo reducido, etiquetar siempre estas conclusiones como exploratorias.
- [ ] Repetir pruebas y comparar número de candidatos de fixtures defectuosos y controles legítimos. Los resultados sintéticos solo verifican el mecanismo, no el recall cinematográfico.

Asserts esenciales: `assert all(f.state == "posible incidencia" for f in result.findings)`; `assert "dx_content_review" in {f.kind for f in missing_step_result.findings}`. La primera afirmación cubre candidatos semánticos, no los errores técnicos de otras etapas.

## Task 5: Lenguaje local, con y sin selector VAD

**Files:** Crear `src/mne_qc/speech.py`, `tests/test_speech.py`, `tests/data/whisper_output.synthetic.json`, `docs/DEPENDENCIES.md`.

**Interfaces:** `scan_speech(manifest: Manifest, config: SpeechConfig, stop: StopToken) -> StageResult`; producir `Detection` internos y findings con muestras de origen, texto hipotético y modo. Las pruebas unitarias usan un backend simulado; las ejecuciones reales con pesos se registran por separado y no se confunden con mocks.

- [ ] Escribir `test_window_boundary_mapping_and_dedup`, `test_surround_channel_never_skipped`, `test_voice_absent_from_dx_still_candidate`, `test_hallucinated_text_not_confirmed_issue`, `test_missing_binary_or_weights_not_evaluable`, `test_timeout_and_cancellation_terminate_child`, `test_vad_and_fixed_have_distinct_coverage`, `test_no_network_or_automatic_download`.
- [ ] Ejecutar `.venv/bin/python -m pytest tests/test_speech.py -q`; verificar rojo antes de implementar adaptador.
- [ ] Implementar derivados temporales por canal a PCM16 mono/16 kHz solo para ASR, sin normalización, con mapa exacto de origen y padding. Usar FFmpeg ya disponible para resamplear cada ventana con contexto; tomar ventanas de 30 s con 5 s de solapamiento, un canal/proceso a la vez, inicialmente 4 threads. No concatenar segmentos VAD sin reconstruir el mapa temporal; el tiempo devuelto debe referir al original.
- [ ] Implementar invocación por `subprocess` sin shell; modo `fixed` sin VAD y modo `vad` con selector Silero. Confirmar opciones y formato JSON del binario instalado, fijarlas en los tests del parser y registrar su versión/SHA. No enviar a ASR todo un programa/canal en memoria ni saltar canales por tener energía baja sin registrar cobertura.
- [ ] Antes de instalar o descargar, presentar inventario exacto: whisper.cpp `v1.9.5`, fuente oficial, compilación local aislada bajo `vendor/whisper.cpp`, con Metal si está disponible y fallback CPU; Whisper `small` multilingüe (~466 MiB según upstream) y Silero `v6.2.0` (~0,85 MiB). Confirmar tamaños reales, licencia de cada peso y revisión remota antes de pedir la autorización concreta prevista en el diseño. Registrar SHA-256 de pesos después de adquirirlos; no afirmar autenticidad verificada solo porque se calculó un hash local.
- [ ] Tras esa autorización, compilar solo `whisper-cli` y dependencias necesarias, sin Core ML adicional ni servidor; hacer una prueba real con una frase autorizada en castellano y un control sin voz. Verificar manualmente tiempos y que VAD se puede desactivar. Si no hay frase aún autorizada por ruta, esta comprobación queda pendiente y la etapa IA no se declara validada.
- [ ] Repetir pruebas unitarias. Medir por separado carga del modelo, tiempo de inferencia y memoria observada del proceso; la tabla upstream no sustituye mediciones en el M2.

Asserts esenciales: `assert scan_speech(manifest, missing_model, stop).status == "not_evaluable"`; `assert any("Rs" in f.channels for f in surround_result.findings)`; `assert all(f.state == "posible incidencia" for f in hallucination_result.findings)`. Un texto generado por el backend simulado sigue siendo hipótesis.

## Task 6: Ejecutor, recuperación y revisión local

**Files:** Crear `src/mne_qc/{runner,report,cli,__main__}.py`, `tests/test_runner_report.py`, `scripts/smoke.py`; modificar `pyproject.toml` para comando `mne-qc`.

**Interfaces:** `run_analysis(manifest: Manifest, output: Path, speech: SpeechConfig|None, stop: StopToken) -> RunResult`, `write_report(result: RunResult, manifest: Manifest, output: Path) -> Path`, `main(argv: list[str]|None=None) -> int`. CLI: `inspect`, `analyze`, `evaluate`; `analyze --speech off|fixed|vad|both`, `--previews`, sin red por defecto.

- [ ] Escribir `test_failure_preserves_completed_stages`, `test_cancel_preserves_coverage`, `test_source_changed_blocks_preview`, `test_output_never_overwrites_input_or_previous_run`, `test_html_escapes_names_and_asr_text`, `test_no_alerts_is_not_pass`, `test_speech_off_is_visibly_not_run`. Cubrir rutas con espacios, acentos y texto parecido a HTML sin ejecutarlo.
- [ ] Ejecutar `.venv/bin/python -m pytest tests/test_runner_report.py -q`; confirmar rojo.
- [ ] Implementar ejecución secuencial con nuevo directorio de run exclusivo, `run.json` actualizado atómicamente tras cada etapa y `report.html` estático sin recursos de red. Registrar versiones, configuración, identidades y cobertura; manejar SIGINT y fallos de procesos hijos. Fallo de una etapa conserva las ya completadas; no crear un marcador de entrega aprobada.
- [ ] Antes de cada preview, comprobar identidad de las fuentes frente al snapshot inicial; si tamaño/mtime difieren o la verificación SHA final no coincide, invalidar conclusiones afectadas y no generar nueva evidencia. Cancelar un hash deja SHA nulo y estado explícito. Nunca sobrescribir ni borrar originales para recuperar espacio.
- [ ] Implementar previews optativos, limitados a 10 s con contexto de 1 s y señalados como auxiliares. Mantener 48 kHz/24-bit y configuración original; crear una versión estéreo de escucha solo si se pide y documentar coeficientes, sin normalización. Si el sistema no reproduce 5.1 correctamente, enlazar el fragmento para escucha en una estación adecuada, sin afirmar que el navegador valida multicanal.
- [ ] Informe: originales referenciados localmente, muestras/timecode, canal, motivo y medición; revisión humana almacenada en `review.json` por id, sin editar detecciones originales. Excepciones del título condicionan la alerta de QC, no borran que hay lenguaje audible. Evidencia sin FPS usa muestras/segundos, nunca timecode inventado.
- [ ] Ejecutar pruebas y smoke sintético: `.venv/bin/python scripts/smoke.py --output runs/smoke`. Esperar reconstrucción exacta en caso correcto, candidatos localizados en defectos, estado IA pendiente si no se ejecutó y recuperación explícita en cancelación provocada.

Asserts esenciales: `assert interrupted.status == "partial"`; `assert interrupted.stages[0].status == "complete"`; `assert original_sha_before == original_sha_after`; `assert "<script>alert" not in report_html`. Comprobar también que el texto peligroso aparece escapado y sigue siendo legible.

## Task 7: Evaluación reproducible y gates

**Files:** Crear `src/mne_qc/evaluation.py`, `tests/test_evaluation.py`, `examples/annotations.synthetic.json`; conectar `evaluate` en `cli.py`.

**Interfaces:** `evaluate_runs(runs: list[RunResult], annotations: list[Annotation], corpus: dict, config: dict) -> EvaluationResult`. Corpus JSON contiene `project_id`, partición, fuente/obra, duración original única, formato y rol de cada fragmento; hashes/versiones vinculan anotaciones y runs.

- [ ] Escribir `test_duplicate_channels_count_once`, `test_one_to_one_matching`, `test_no_positive_labels_is_not_100_recall`, `test_shared_title_leaks_across_split`, `test_incomplete_coverage_ineligible`, `test_each_format_and_mode_has_own_gate`, `test_ambiguous_and_allowed_lyrics_not_negatives`, `test_under_dialogue_subgroup_cannot_be_hidden`.
- [ ] Ejecutar `.venv/bin/python -m pytest tests/test_evaluation.py -q`; verificar rojo.
- [ ] Implementar matching temporal uno a uno por tipo sobre candidatos ya agrupados: permitir holgura de 250 ms a ambos lados de la anotación y exigir intersección con ≥50% del rango de la alerta; seleccionar la mayor intersección primero, desempatar por id. Para lenguaje no exigir texto idéntico; para omisiones no exigir nombre semántico del sonido. Agrupar avisos del mismo tipo y vista que se solapen en ≥50% del intervalo más corto; conservar canales y evidencia. Congelar estas reglas antes de abrir el holdout y revisar que no favorezcan alertas enormes.
- [ ] Reportar TP/FP/FN, precisión/recall con denominadores, intervalo Wilson del 95% para proporciones, falsas detecciones/hora de programa único, error de inicio/fin y cobertura. Cero positivos o etapa no ejecutada implica métrica no disponible, nunca 100%. Separar detección de lenguaje de incumplimientos; ambiguos excluidos de TP/FP/FN pero su cantidad y duración siguen visibles.
- [ ] Implementar elegibilidad previa a gates: separación de obras y variantes, duración, ≥30 eventos de cada función por formato, anotación completa, cobertura completa y versiones congeladas. Con evidencia insuficiente, resultado `insufficient_evidence`; no se compara un porcentaje aislado con los objetivos. Separar natural/inyectado, bajo diálogo, 2.0/5.1 y completo/reducido. Los resultados del modo reducido no aprueban omisiones del completo.
- [ ] Repetir pruebas y smoke de `evaluate` con ejemplos sintéticos. Debe mostrar explícitamente que esos ejemplos no satisfacen el corpus real aprobado, aunque las métricas aritméticas sean correctas.

Asserts esenciales sobre la tabla de la función/formato/modo/variante correspondiente: `assert counts["tp"] == 1` cuando el mismo evento aparezca en dos canales; `assert counts["fn"] == 1` con una segunda anotación sin aviso; `assert empty_metrics["recall"] is None`; `assert incomplete_gate == "insufficient_evidence"`.

## Task 8: Validación real, documentación y decisión

**Files:** Crear `docs/VALIDATION.md`; completar `README.md`, `docs/DEPENDENCIES.md`; resultados privados en `runs/` y manifiestos autorizados bajo `data/local/`, excluidos de Git.

**Interfaces:** Usar CLI y contratos anteriores. No requiere API ni interfaz nueva.

- [ ] Documentar selección de material autorizado: rutas y rango permitido, versiones coincidentes, mapa de canales, FPS/inicio, castellano y reglas del título. Confirmar qué obras corresponden a calibración y evaluación; no inspeccionar otras carpetas personales para localizar audio.
- [ ] Preparar los 30 min de calibración y ≥60 min de evaluación reservada del diseño, con eventos suficientes por formato; si no se alcanza, realizar solo ensayo exploratorio y explicar qué evidencia falta. Revisar mezclas estéreo originales, no reemplazarlas por downmixes para justificar ese formato.
- [ ] Crear anotaciones profesionales sobre fragmentos completos, con condiciones de escucha, nivel, configuración y autor. Anotar también controles negativos, reemplazos válidos, lenguaje permitido y ambigüedad. No asignar etiquetas de discernibilidad a partir del texto ASR.
- [ ] Calibrar únicamente con su partición; congelar configuración, modelos, binario, protocolo de matching y manifiestos antes de ejecutar el holdout. Correr baseline DSP y variantes ASR sobre el mismo corpus; registrar no evaluables, tiempo, memoria y revisión profesional. Si cambian parámetros tras ver el holdout, esa partición pasa a desarrollo y se necesita evaluación nueva.
- [ ] Ejecutar regresión completa `.venv/bin/python -m pytest -q` y smoke final `.venv/bin/python scripts/smoke.py --output runs/final-smoke` después de la última modificación. Las pruebas de integración con pesos se identifican aparte y no se presentan como ejecutadas cuando faltan modelo/audio. No añadir tests por mera simetría con la implementación.
- [ ] Revisar el informe local y reproducir incidencias y controles en condiciones adecuadas. Validación manual en Pro Tools u otra estación multicanal: confirmar rutas de canal, sincronía con referencia si se dispone de ella y escucha del M&E original; declararla pendiente si no se ha hecho.
- [ ] Concluir por capacidad/formato: continuar, limitar a comparador, limitar a lenguaje, investigar omisiones o evidencia insuficiente. Cumplir objetivos con esta muestra permite continuar, no aprobar entregas Netflix ni sustituir escucha. Si DX/PFX bajo diálogo no resulta útil, presentar evidencia y propuesta acotada antes de añadir separación neuronal.
- [ ] Entregar qué se implementó, pruebas ejecutadas, métricas reales con denominadores, fallos, limitaciones y siguiente acción. Conservar originales y resultados; no publicar ni hacer commits/push sin autorización.

## Revisión del plan frente al diseño

- Entradas, discreto/interleaved, BWF/RF64, versión, canales y profundidad declarada: tarea 1.
- Sample accuracy, cobertura, offsets y los dos contraejemplos del null: tarea 2.
- Perfil técnico acotado y true peak validado sin simular dialog gating: tarea 3.
- Diferencias legítimas, DX/PFX bajo voz, optionals y modo reducido: tarea 4.
- Lenguaje independiente de DX, variantes VAD/ASR y modelo local: tarea 5.
- Evidencia, revisión humana, privacidad, cancelación y fuentes cambiantes: tarea 6.
- Matching, holdout, incertidumbre, suficiencia y gates por formato: tarea 7.
- Corpus, escucha profesional, rendimiento M2 y conclusión limitada: tarea 8.
- No hay commits, instalaciones ni descargas ya realizadas por este plan. El trabajo de anotación y la obtención de modelos siguen siendo dependencias explícitas de la evaluación real.

## Fuentes técnicas verificadas durante la planificación

- [SoundFile](https://python-soundfile.readthedocs.io/): arrays NumPy y lectura por bloques; confirmar soporte efectivo de contenedores con pruebas locales.
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp): Apple Silicon/Metal, ejecución local, tamaño orientativo de small y selección VAD. [Release v1.9.5](https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.5) consultada el 2026-10-26; fijar revisión al adquirirla, no usar master móvil.
- [Script oficial de adquisición de VAD](https://github.com/ggml-org/whisper.cpp/blob/master/models/download-vad-model.sh): nombres de modelos y origen. Inspeccionar la versión fijada antes de ejecutarlo.
- [Documentación FFmpeg](https://github.com/FFmpeg/FFmpeg/blob/master/doc/filters.texi): ebur128/true peak. Se comprobó además `ffmpeg -hide_banner -h filter=ebur128` localmente; esto confirma disponibilidad del filtro, no su exactitud medida.
