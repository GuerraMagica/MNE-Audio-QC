from fractions import Fraction

from .audio import inspect_audio
from .domain import Manifest


def program_coverage(manifest: Manifest) -> dict:
    lo, hi = manifest.program_span
    assets = {}
    for role, a in manifest.assets.items():
        infos = [inspect_audio(c.path) for c in a.channels]
        end = a.start_sample + min(i.frames for i in infos)
        start = a.start_sample
        missing = []
        if start > lo:
            missing.append([lo, min(start, hi)])
        if end < hi:
            missing.append([max(end, lo), hi])
        outside = []
        if start < lo:
            outside.append([start, min(end, lo)])
        if end > hi:
            outside.append([max(start, hi), end])
        assets[role] = {"available": [start, end], "missing": missing, "outside_program": outside,
                        "unequal_channel_lengths": len({i.frames for i in infos}) > 1,
                        "bwf_origins": [i.bwf_time_reference for i in infos],
                        "declared_timeline_origin": start}
    return {"program": [lo, hi], "assets": assets}


def format_timecode(sample: int, fps: Fraction, start_frame: int, drop_frame: bool) -> str:
    if fps is None or fps <= 0 or start_frame is None:
        raise ValueError("Timecode requiere FPS e inicio explícitos")
    fps = Fraction(fps)
    nominal = round(fps)
    if drop_frame and fps != Fraction(30000, 1001):
        raise ValueError("Drop frame solo implementado para 30000/1001")
    # Half-sample tolerance only: avoid a label one frame early at rounded boundaries.
    frame = int((Fraction(sample, 48000) * fps + fps / 96000).__floor__()) + start_frame
    negative = frame < 0
    frame = abs(frame)
    if drop_frame:
        tens, remainder = divmod(frame, 17982)
        frame += 18 * tens + 2 * max(0, (remainder - 2) // 1798)
    seconds, ff = divmod(frame, nominal)
    minutes, ss = divmod(seconds, 60)
    hours, mm = divmod(minutes, 60)
    return f'{"-" if negative else ""}{hours:02d}:{mm:02d}:{ss:02d}{";" if drop_frame else ":"}{ff:02d}'
