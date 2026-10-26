"""Explainable candidates, not semantic proof that a sound is missing."""
import numpy as np
from contextlib import closing
from scipy.signal import stft

from .audio import inspect_audio, iter_blocks
from .domain import CANDIDATE, Finding, LAYOUTS, Manifest, Span, StageResult, StopToken

DEFAULTS = {"stft_frames": 2048, "hop_frames": 512, "aggregate_seconds": 0.5,
            "correlation_threshold": 0.7, "energy_delta_db": 6.0, "activity_floor_dbfs": -80.0}


def _corr(x, y):
    x, y = x - np.mean(x), y - np.mean(y)
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / norm) if norm > 1e-16 else 0.0


def _bands(x, config):
    size = min(config["stft_frames"], len(x))
    if size < 2:
        return np.zeros(5)
    hop = min(config["hop_frames"], size)
    f, _, spec = stft(x, fs=48000, nperseg=size, noverlap=size - hop, boundary=None)
    bounds = (0, 120, 500, 2000, 8000, 24001)
    return np.array([float(np.mean(np.sum(np.abs(spec[(f >= a) & (f < b)])**2, axis=0)))
                     for a, b in zip(bounds, bounds[1:])])


def compare_me(manifest: Manifest, config: dict, stop: StopToken) -> StageResult:
    cfg = {**DEFAULTS, **config}
    if cfg["stft_frames"] < 2 or not 1 <= cfg["hop_frames"] <= cfg["stft_frames"] or cfg["aggregate_seconds"] <= 0:
        raise ValueError("Configuración DSP no válida")
    labels = LAYOUTS[manifest.layout]
    complete = {"MX", "FX"} <= manifest.assets.keys()
    roles = ["PM", "ME", "DX"] + (["MX", "FX"] if complete else [])
    result = StageResult("comparison", "complete", measurements={"mode": "complete" if complete else "reduced",
        "omission_claim": "candidate_only" if complete else "exploratory", "dx_separation_validated": False,
        "config": cfg, "windows": []})
    assets = [manifest.assets[r] for r in roles]
    # DX may be a guide with fewer channels; analyze it separately below.
    required = [a for a in assets if a.role != "DX"]
    if any(tuple(c.label for c in a.channels) != labels for a in required):
        result.status, result.reason = "not_evaluable", "Mapa incompatible para comparación"
        return result
    if any(inspect_audio(c.path).sample_rate != 48000 for a in assets for c in a.channels):
        result.status, result.reason = "not_evaluable", "Sample rate incompatible; no se resamplea DSP"
        return result
    start = max(manifest.program_span[0], *(a.start_sample for a in assets))
    end = min(manifest.program_span[1], *(a.start_sample + min(inspect_audio(c.path).frames for c in a.channels) for a in assets))
    if end <= start:
        result.status, result.reason = "not_evaluable", "Sin cobertura común"
        return result
    # Initial 0.5 s grid with 50% overlap; all thresholds remain explicit and calibratable.
    width = max(2, round(cfg["aggregate_seconds"] * 48000))
    stride = max(1, width // 2)
    floor = 10**(cfg["activity_floor_dbfs"] / 20)
    last = start
    for pos in range(start, end, stride):
        if stop.cancelled():
            break
        span = Span(pos, min(pos + width, end))
        try:
            data={}
            for a in assets:
                with closing(iter_blocks(a,span,width)) as blocks:
                    data[a.role]=next(blocks).samples
        except (OSError,ValueError,RuntimeError) as e:
            result.status,result.reason="partial",f"Lectura interrumpida; ventanas previas conservadas: {e}"
            break
        me, pm = data["ME"], data["PM"]
        if complete:
            ref = data["MX"] + data["FX"]
        elif data["DX"].shape[1] == len(labels):
            ref = pm - data["DX"]
        else:
            ref = pm  # Explicitly exploratory: no pretend spatial reconstruction of a mono guide.
        record = {"start": span.start, "end": span.end, "channels": []}
        for i, label in enumerate(labels):
            x, y = ref[:, i], me[:, i]
            xr, yr = np.sqrt(np.mean(x*x)), np.sqrt(np.mean(y*y))
            if max(xr, yr) <= floor:
                continue
            corr = _corr(x, y)
            delta = float(20 * np.log10(max(yr, 1e-15) / max(xr, 1e-15)))
            xb, yb = _bands(x, cfg), _bands(y, cfg)
            important = xb > max(float(np.max(xb)) * 1e-4, floor**2)
            lost = important & (yb < xb * 10**(-cfg["energy_delta_db"] / 10))
            ev = {"reference": "MX+FX" if complete else "exploratory_PM_minus_DX",
                  "correlation": corr, "level_delta_db": delta, "lost_bands": np.flatnonzero(lost).tolist(),
                  "mode": result.measurements["mode"], "automatic_verdict": False}
            record["channels"].append({"label": label, **ev})
            kind = None
            if xr > floor and (delta < -cfg["energy_delta_db"] or np.any(lost)):
                kind = "possible_omission"
                for other in range(len(labels)):
                    if other != i and np.sqrt(np.mean(me[:, other]**2)) >= xr * 0.5 and _corr(x, me[:, other]) > 0.95:
                        kind = "spatial_change"
                        ev["possible_target_channel"] = labels[other]
                        break
            elif xr > floor and yr > floor and corr < cfg["correlation_threshold"]:
                kind = "additional_or_replacement"
            elif abs(delta) > cfg["energy_delta_db"]:
                kind = "level_change"
            if kind:
                result.findings.append(Finding(f"compare-{pos}-{label}", kind, CANDIDATE, span, (label,), ev))
        # Activity in DX is deliberately not called a non-speech event. The difficult
        # content stays on the review queue, including while speech is present.
        for i, c in enumerate(manifest.assets["DX"].channels):
            rms = float(np.sqrt(np.mean(data["DX"][:, i]**2)))
            if rms > floor:
                result.findings.append(Finding(f"dx-{pos}-{c.label}", "dx_content_review", CANDIDATE, span, (c.label,),
                    {"reason": "DX activo: revisar si contiene PFX/ambiente sin equivalente en M&E",
                     "rms": rms, "speech_not_excluded": True, "source_separation": "not_implemented",
                     "mode": result.measurements["mode"]}))
        result.measurements["windows"].append(record)
        last = span.end
    if last > start:
        result.coverage = [Span(start, last)]
    if (stop.cancelled() or start != manifest.program_span[0] or last != manifest.program_span[1]) and result.reason is None:
        result.status, result.reason = "partial", "Cobertura limitada o cancelación"
    if len({a.version for a in assets}) > 1:
        result.status, result.reason = "partial", "Versiones distintas; comparación solo exploratoria"
    return result
