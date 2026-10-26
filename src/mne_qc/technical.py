import math
from pathlib import Path
import re

from .audio import inspect_audio
from .domain import CANDIDATE, Finding, Manifest, Span, StageResult, StopToken, TECHNICAL
from .process import run_local


def measure_true_peak(path: Path, channel: int, stop: StopToken) -> float:
    _, err = run_local(["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-loglevel", "info",
                        "-i", str(path), "-map", "0:a:0", "-af",
                        f"pan=mono|c0=c{channel},ebur128=peak=true:framelog=verbose",
                        "-f", "null", "-"], stop, timeout=7200)
    if "True peak:" not in err:
        raise RuntimeError("FFmpeg no devolvió true peak")
    match = re.search(r"Peak:\s*([+-]?(?:\d+(?:\.\d+)?|inf))\s*dBFS", err.rsplit("True peak:", 1)[1])
    if not match:
        raise RuntimeError("Resumen true peak no reconocido")
    value = float(match.group(1))
    if math.isnan(value) or value == math.inf:
        raise RuntimeError("True peak inválido")
    return value


def check_technical(manifest: Manifest, stop: StopToken) -> StageResult:
    result = StageResult("technical", "complete", measurements={"peaks": [], "format": [],
                         "true_peak_resolution_db": 0.1, "dialog_gated_loudness": "not_implemented",
                         "meter_certified": False})
    cache = {}
    for role, asset in manifest.assets.items():
        for channel in asset.channels:
            if stop.cancelled():
                result.status, result.reason = "partial", "Cancelado"
                return result
            try:
                if channel.path not in cache:
                    cache[channel.path] = inspect_audio(channel.path)
                info = cache[channel.path]
                if channel.file_channel >= info.channels:
                    raise ValueError("Índice de canal fuera del archivo")
                result.measurements["format"].append({"role": role, "label": channel.label,
                    "sample_rate": info.sample_rate, "declared_bits": info.declared_bits, "subtype": info.subtype,
                    "frames": info.frames, "bwf_time_reference": info.bwf_time_reference})
                if info.sample_rate != 48000 or info.declared_bits != 24:
                    result.findings.append(Finding(f"format-{role}-{channel.label}", "format", TECHNICAL,
                        None, (channel.label,), {"role": role, "sample_rate": info.sample_rate,
                            "declared_bits": info.declared_bits, "subtype": info.subtype,
                            "reason": "Perfil de análisis: PCM 48000 Hz/24-bit declarado"}))
                peak = measure_true_peak(channel.path, channel.file_channel, stop)
                record = {"role": role, "label": channel.label, "value_dbtp": peak if math.isfinite(peak) else None,
                          "silence": peak == -math.inf, "margin_db": -2 - peak if math.isfinite(peak) else None}
                result.measurements["peaks"].append(record)
                if peak > -2:
                    result.findings.append(Finding(f"peak-{role}-{channel.label}", "true_peak_exceeded", TECHNICAL,
                        None, (channel.label,), record))
                elif math.isfinite(peak) and abs(peak + 2) <= 0.05:
                    result.findings.append(Finding(f"peak-border-{role}-{channel.label}", "true_peak_boundary", CANDIDATE,
                        None, (channel.label,), {**record, "reason": "Caso frontera: resolución de salida 0,1 dB"}))
            except (ValueError, OSError, RuntimeError, TimeoutError, InterruptedError) as e:
                result.status, result.reason = "partial", str(e)
                result.measurements["peaks"].append({"role": role, "label": channel.label,
                                                      "value_dbtp": None, "error": str(e)})
                if stop.cancelled():
                    return result
    if result.status == "complete":
        result.coverage = [Span(*manifest.program_span)]
    return result
