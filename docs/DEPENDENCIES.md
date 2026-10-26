# Dependencias y adquisición

## Instalado para esta prueba

Dependencias Python solo en `.venv` local, desde PyPI/files.pythonhosted.org mediante wheels arm64 compatibles con Python 3.14. No se instalaron paquetes globales. Después, el usuario autorizó expresamente los dos pesos y el runtime descritos abajo. Versiones Python completas: `requirements-dev.lock`.

| Componente principal | Versión instalada | Licencia declarada / observación |
| --- | --- | --- |
| NumPy | 2.5.4 | BSD-3-Clause y componentes MIT/0BSD/Zlib/CC0 según metadata |
| SciPy | 1.18.1 | BSD; incluye componentes con avisos propios |
| SoundFile | 0.14.0 | BSD-3-Clause; libsndfile tiene LGPL y avisos propios |
| pytest | 9.1.1 | MIT; desarrollo |
| FFmpeg / ffprobe | 9.0.2, herramientas existentes | Build local con GPL habilitado; no se redistribuye con este proyecto |

La distribución de una aplicación empaquetada requiere revisar las licencias y los avisos de todos sus binarios. Usar una dependencia local no resuelve por sí solo las obligaciones de redistribución. No hay instalador ni distribución creada en esta fase.

## Modelos y runtime adquiridos con autorización

| Artefacto | Origen | Tamaño orientativo | Destino |
| --- | --- | --- | --- |
| whisper.cpp v1.9.5 | [Release oficial](https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.5) | Build observado ~36 MiB, aparte del checkout | `vendor/whisper.cpp/` |
| Whisper small multilingüe GGML | [Model card del repositorio de pesos](https://huggingface.co/ggerganov/whisper.cpp) | ~466 MiB según upstream | `models/ggml-small.bin` |
| Silero VAD v6.2.0 GGML | [Model card de pesos](https://huggingface.co/ggml-org/whisper-vad) | ~0,85 MiB según upstream | `models/ggml-silero-v6.2.0.bin` |

Las tres fuentes declaran MIT. Se inspeccionaron el código CLI y la licencia del checkout v1.9.5. Se descargaron los pesos mediante las URLs fijadas, sin ejecutar scripts remotos. Se verificaron tamaño y SHA-256 LFS publicado antes de darles su nombre final. El commit del checkout coincide con `d1be6fde11ac6e0407606b4e42fe72d34add8037`. El CLI informa `1.9.5-dev`, tal como está declarado en ese tag; no se ha sustituido por master. Las opciones `-ojf`, `-of`, `--vad -vm` y `-ng` se comprobaron en el binario y mediante ejecuciones reales.

Revisiones, URLs fijadas, tamaños exactos y SHA-256 publicados quedan en `models.lock.json`. Los dos archivos suman 488.487.065 bytes (~489 MB decimales), aparte del código y build de whisper.cpp. El inventario registra `downloaded_verified_runtime_built`, el hash del binario y hashes de los artefactos locales enlazados. Es un recibo de esta máquina; no implica que los archivos excluidos de Git existan en otro equipo.

Procedimiento para futuras adquisiciones: fijar revisiones, tamaños y licencia; presentar el inventario y obtener la autorización necesaria; comparar con hashes del origen cuando se publiquen. En esta adquisición los hashes publicados sí se cotejaron. Un hash local aislado no habría demostrado autenticidad.

Se compiló solo el target `whisper-cli` y sus dependencias, con tests/servidor desactivados, CPU/Accelerate y Metal. CMake indicó que OpenMP no estaba disponible; no se instaló porque no impidió las pruebas CPU/Metal. No se añadieron servidor, Docker, Core ML ni servicios de transcripción. Metal funcionó fuera del sandbox sobre Apple M2 Pro; CPU funcionó dentro del sandbox con `--no-gpu`. Las pruebas completas multicanal se ejecutaron en CPU. No se deduce rendimiento de largometraje a partir de estos controles.

## Verificación de tecnología

- [SoundFile: lectura por bloques y soporte NumPy](https://python-soundfile.readthedocs.io/).
- [whisper.cpp: funcionamiento, Apple Silicon, tamaños orientativos y VAD](https://github.com/ggml-org/whisper.cpp).
- [Whisper: limitaciones, alucinaciones y variación por idioma](https://github.com/openai/whisper/blob/main/model-card.md).
- True peak: ayuda local `ffmpeg -hide_banner -h filter=ebur128` y ejecución controlada con referencia independiente. No se ha certificado el medidor ni implementado dialog-gated loudness.
