"""Optional offline whisper-cli adapter. Text is always a review hypothesis."""
from dataclasses import replace
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import time

import soundfile as sf
from scipy.signal import resample_poly

from .audio import fingerprint, inspect_audio, iter_blocks
from .domain import CANDIDATE, Detection, Finding, Manifest, Span, SpeechConfig, StageResult, StopToken
from .process import run_local

# Exact standalone model annotations, not a filter for words such as "música".
# These are retained as model output; their acoustic meaning is unverified.
NON_SPEECH_LABELS={"[música]","[musica]","[music]","[blank_audio]","[silence]",
                   "(música)","(musica)","(music)"}


def parse_whisper_json(raw: dict) -> list[dict]:
    if not isinstance(raw.get("transcription"), list):
        raise ValueError("JSON de Whisper no reconocido")
    result = []
    for entry in raw["transcription"]:
        try:
            start, end = entry["offsets"]["from"], entry["offsets"]["to"]
            text = entry["text"]
            if not isinstance(text, str) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, end)):
                raise ValueError("Tipos o tiempos de ASR inválidos")
            if start < 0 or end <= start:
                raise ValueError("Intervalo ASR inválido")
            if text.strip():
                result.append({"start_ms": start, "end_ms": end, "text": text.strip()})
        except (KeyError, TypeError) as e:
            raise ValueError("Segmento ASR incompleto") from e
    return result


def map_segments(raw: list[dict], origin: int, end: int, label: str) -> list[Detection]:
    result = []
    for item in raw:
        start = max(origin, origin + round(item["start_ms"] * 48))
        finish = min(end, origin + round(item["end_ms"] * 48))
        if finish > start:
            result.append(Detection(Span(start, finish), (label,), item["text"], {"origin_sample": origin}))
    return result


def merge_detections(detections: list[Detection]) -> list[Detection]:
    merged = []
    for d in sorted(detections, key=lambda d: (d.span.start, d.span.end)):
        for old in reversed(merged):
            overlap = min(old.span.end, d.span.end) - max(old.span.start, d.span.start)
            shorter = min(old.span.end-old.span.start, d.span.end-d.span.start)
            if overlap >= shorter * 0.5:
                old.span = Span(min(old.span.start, d.span.start), max(old.span.end, d.span.end))
                old.channels = tuple(dict.fromkeys(old.channels + d.channels))
                old.evidence["text_hypotheses"]=list(dict.fromkeys(old.evidence["text_hypotheses"]+
                                                                d.evidence.get("text_hypotheses",[d.text])))
                old.evidence["contributors"].append({"span":[d.span.start,d.span.end],"channels":list(d.channels),
                                                      "text":d.text,"evidence":deepcopy(d.evidence)})
                break
        else:
            evidence=deepcopy(d.evidence)
            evidence["text_hypotheses"]=list(d.evidence.get("text_hypotheses",[d.text]))
            evidence["contributors"]=[{"span":[d.span.start,d.span.end],"channels":list(d.channels),
                                       "text":d.text,"evidence":deepcopy(d.evidence)}]
            merged.append(Detection(d.span,d.channels,d.text,evidence))
    return merged


def _transcribe_window(path: Path, config: SpeechConfig, stop: StopToken) -> list[dict]:
    output = path.with_suffix(".asr")
    args = [str(config.binary.resolve()), "-m", str(config.model.resolve()), "-f", str(path),
            "-l", config.language, "-t", str(config.threads), "-ojf", "-of", str(output), "-np"]
    if config.mode == "vad":
        args += ["--vad", "-vm", str(config.vad_model.resolve())]
    if not config.use_gpu:
        args += ["-ng"]
    run_local(args, stop, config.timeout_seconds)
    target = Path(str(output) + ".json")
    if not target.is_file() or target.stat().st_size > 20_000_000:
        raise ValueError("Whisper no generó un JSON válido y acotado")
    return parse_whisper_json(json.loads(target.read_text(encoding="utf-8")))


def scan_speech(manifest: Manifest, config: SpeechConfig, stop: StopToken) -> StageResult:
    if config.mode not in {"fixed", "vad"} or config.threads < 1:
        raise ValueError("Modo ASR/threads inválido")
    result = StageResult(f"speech_{config.mode}", "not_evaluable", measurements={"mode": config.mode,
        "coverage_semantics": "input_scanned_vad_may_reject_speech" if config.mode == "vad" else "fixed_windows_full_input",
        "coverage_by_input": [], "derived_format": "PCM16 mono 16000 Hz, sin normalización",
        "non_speech_annotations":[],
        "intelligibility_validation": "human_review_required", "window_seconds": 30, "overlap_seconds": 5})
    result.measurements["use_gpu_requested"]=config.use_gpu
    needed = [config.binary, config.model] + ([config.vad_model] if config.mode == "vad" else [])
    if any(p is None or not Path(p).is_file() for p in needed):
        result.reason = "Falta binario/modelo local; no se descarga automáticamente"
        return result
    started = time.monotonic()
    result.measurements["backend_identity"] = {str(p): fingerprint(Path(p), stop) for p in needed}
    if any(v["status"] != "complete" for v in result.measurements["backend_identity"].values()):
        result.reason = "Identidad de backend incompleta"
        return result
    me_detections, dx_detections = [], []
    expected = sum(len(manifest.assets[r].channels) for r in ("ME", "DX"))
    completed = 0
    result.status = "complete"
    try:
        with tempfile.TemporaryDirectory(prefix="mne-asr-") as directory:
            for role in ("ME", "DX"):
                asset = manifest.assets[role]
                for channel in asset.channels:
                    info = inspect_audio(channel.path)
                    if info.sample_rate != 48000:
                        raise ValueError("ASR requiere timeline 48000 Hz; entrada incompatible")
                    start = max(manifest.program_span[0], asset.start_sample)
                    end = min(manifest.program_span[1], asset.start_sample + info.frames)
                    if end <= start:
                        raise ValueError("Sin cobertura de entrada ASR")
                    single = replace(asset, channels=(channel,))
                    last = start
                    for origin in range(start, end, 25*48000):
                        if stop.cancelled():
                            raise InterruptedError("ASR cancelado")
                        finish = min(origin + 30*48000, end)
                        samples = next(iter_blocks(single, Span(origin, finish), 30*48000)).samples[:, 0]
                        derived = resample_poly(samples, 1, 3)
                        path = Path(directory) / f"{role}_{channel.label}_{origin}.wav"
                        sf.write(path, derived, 16000, subtype="PCM_16")
                        raw = _transcribe_window(path, config, stop)
                        lexical=[]
                        for item in raw:
                            if item["text"].strip().casefold() in NON_SPEECH_LABELS:
                                result.measurements["non_speech_annotations"].append({**item,"role":role,
                                    "channel":channel.label,"origin_sample":origin,"input_end_sample":finish,
                                    "interpretation":"model_annotation_not_acoustically_confirmed"})
                            else:
                                lexical.append(item)
                        detections = map_segments(lexical, origin, finish, channel.label)
                        (me_detections if role == "ME" else dx_detections).extend(detections)
                        # File belongs to this temporary window; remove after tool output is read.
                        for generated in path.parent.glob(path.stem + ".*"):
                            generated.unlink()
                        last = finish
                        result.measurements["coverage_by_input"].append({"role": role, "channel": channel.label,
                                                                        "start": origin, "end": finish})
                    if start == manifest.program_span[0] and last == manifest.program_span[1]:
                        completed += 1
    except (OSError, ValueError, RuntimeError, TimeoutError, InterruptedError) as e:
        result.status, result.reason = "partial", str(e)
    result.measurements["elapsed_seconds"] = time.monotonic() - started
    result.measurements["dx_reference_count"] = len(dx_detections)
    for i, d in enumerate(merge_detections(me_detections)):
        match = any(x.text.casefold().strip() in {t.casefold().strip() for t in d.evidence["text_hypotheses"]} and
                    min(d.span.end, x.span.end) > max(d.span.start, x.span.start) for x in dx_detections)
        result.findings.append(Finding(f"{config.mode}-speech-{i}", "language", CANDIDATE, d.span, d.channels,
            {**d.evidence,"text_hypothesis": d.text, "dx_text_match": match, "intelligibility_confirmed": False,
             "mode": config.mode, "reason": "Escuchar el M&E original; ASR puede alucinar u omitir lenguaje"}))
    if completed == expected:
        result.coverage = [Span(*manifest.program_span)]
    else:
        result.status = "partial"
        result.reason = result.reason or "Cobertura incompleta en uno o más canales"
    backend_after={}
    for p in needed:
        try:
            backend_after[str(p)]=fingerprint(Path(p),stop)
        except OSError as e:
            backend_after[str(p)]={"sha256":None,"status":"failed","reason":str(e)}
    verified=all(v.get("status")=="complete" and v.get("sha256")==result.measurements["backend_identity"][name].get("sha256")
                 for name,v in backend_after.items())
    result.measurements.update(backend_identity_final=backend_after,backend_identity_verified=verified)
    if not verified:
        result.status,result.reason="partial","Cambió el backend/pesos o no se pudo verificar su identidad final"
    return result
