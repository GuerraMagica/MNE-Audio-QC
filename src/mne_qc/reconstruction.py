import numpy as np
from scipy.signal import correlate, correlation_lags

from .audio import iter_blocks, inspect_audio
from .domain import CANDIDATE, Finding, LAYOUTS, Manifest, Span, StageResult, StopToken
from .timeline import program_coverage


def _alignment(source, target):
    channel = int(np.argmax(np.sum(source * source, axis=0)))
    x, y = source[:, channel], target[:, channel]
    energy = np.linalg.norm(x) * np.linalg.norm(y)
    if energy < 1e-14 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return {"offset_samples": None, "reason": "Silencio o señal insuficiente", "polarity": None}
    corr = correlate(y, x, mode="full", method="fft")
    lags = correlation_lags(len(y), len(x))
    mask = np.abs(lags) <= min(48000, len(x) // 2)
    index = np.flatnonzero(mask)[np.argmax(np.abs(corr[mask]))]
    score = float(abs(corr[index]) / energy)
    # A lag estimate is evidence, not a correction or proof of source identity.
    lag = int(lags[index]) if score >= 0.95 else None
    return {"offset_samples": lag, "score": score, "polarity": 1 if corr[index] >= 0 else -1,
            "gain_at_zero_lag": float(np.dot(x, y) / np.dot(x, x)),
            "reason": "Hipótesis de offset global; señales periódicas pueden ser ambiguas",
            "channel_index": channel}


def _difference(manifest: Manifest, roles, name, stop):
    result = StageResult(name, "not_evaluable")
    if not all(role in manifest.assets for role in roles):
        result.reason = "Faltan stems originales para esta comprobación"
        return result
    labels = LAYOUTS[manifest.layout]
    assets = [manifest.assets[r] for r in roles]
    if any(tuple(c.label for c in a.channels) != labels for a in assets):
        result.reason = "Configuración incompatible: una guía mono no reconstruye multicanal"
        return result
    if len({a.version for a in assets}) != 1:
        result.reason = "Versiones declaradas diferentes"
        return result
    coverage = program_coverage(manifest)
    result.measurements["coverage"] = coverage
    if any(inspect_audio(c.path).sample_rate != 48000 for a in assets for c in a.channels):
        result.reason = "Sample rate incompatible"
        return result
    if any(coverage["assets"][r]["unequal_channel_lengths"] for r in roles):
        result.reason = "Canales discretos con duración diferente"
        return result
    start = max(manifest.program_span[0], *(a.start_sample for a in assets))
    end = min(manifest.program_span[1], *(coverage["assets"][r]["available"][1] for r in roles))
    if end <= start:
        result.reason = "No hay programa común"
        return result
    peak = np.zeros(len(labels))
    squares = np.zeros(len(labels))
    count = 0
    windows = []
    prefix = {r: [] for r in roles}
    generators = [iter_blocks(a, Span(start, end), 24000) for a in assets]
    read_error=None
    try:
        for blocks in zip(*generators, strict=True):
            if stop.cancelled():
                break
            residual = blocks[0].samples.copy()
            for b in blocks[1:]:
                residual -= b.samples
            peak = np.maximum(peak, np.max(np.abs(residual), axis=0))
            squares += np.sum(residual * residual, axis=0)
            n = len(residual)
            count += n
            windows.append({"start": blocks[0].start, "end": blocks[0].start + n,
                            "rms": np.sqrt(np.mean(residual * residual, axis=0)).tolist()})
            if count <= 144000:
                for role, b in zip(roles, blocks):
                    prefix[role].append(b.samples)
    except (OSError,ValueError,RuntimeError) as e:
        read_error=str(e)
    finally:
        for g in generators:
            g.close()
    diagnostics = {}
    if count:
        result.coverage = [Span(start, start + count)]
        p = {r: np.concatenate(b) for r, b in prefix.items()}
        for role in roles[1:]:
            target = p[roles[0]].copy()
            for other in roles[1:]:
                if other != role:
                    target -= p[other]
            diagnostics[role] = _alignment(p[role], target)
    result.measurements.update(peak_abs_by_channel=peak.tolist() if count else None,
                               rms_by_channel=np.sqrt(squares / count).tolist() if count else None,
                               raw_residual=True, windows=windows, alignment_diagnostics=diagnostics,
                               me_verdict="requires_review")
    result.status = "complete" if start == manifest.program_span[0] and end == manifest.program_span[1] and count == end - start else "partial"
    if stop.cancelled():
        result.status, result.reason = "partial", "Cancelado; cobertura conservada"
    if read_error is not None:
        result.status,result.reason="partial",f"Lectura interrumpida; hechos previos conservados: {read_error}"
    if name == "reconstruction" and count and np.max(peak) > 0:
        result.findings.append(Finding("reconstruction-difference", "channel_or_processing_difference", CANDIDATE,
                                       Span(start, start + count), labels,
                                       {"reason": "Residual medido; revisar export, routing, offset y cuantización"}))
    return result


def reconstruct(manifest: Manifest, stop: StopToken) -> StageResult:
    return _difference(manifest, ("PM", "DX", "MX", "FX"), "reconstruction", stop)


def describe_me_residual(manifest: Manifest, stop: StopToken) -> StageResult:
    return _difference(manifest, ("PM", "ME", "DX"), "me_residual", stop)
