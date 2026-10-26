"""Read-only PCM access; BWF timestamps do not imply a frame rate."""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import struct

import numpy as np
import soundfile as sf

from .domain import Asset, AudioInfo, Block, Span, StopToken


def _wav_metadata(path):
    size = path.stat().st_size
    result = {"time_reference": None, "data_size": None, "block_align": None}
    with path.open("rb") as f:
        head = f.read(12)
        if len(head) != 12 or head[0:4] not in (b"RIFF", b"RF64") or head[8:] != b"WAVE":
            raise ValueError("Se requiere WAV/BWAV RIFF o RF64")
        declared = struct.unpack_from("<I", head, 4)[0]
        if head[:4] == b"RIFF" and declared + 8 > size:
            raise ValueError("WAV truncado: tamaño RIFF mayor que el archivo")
        ds64_data = None
        while f.tell() + 8 <= size:
            kind, length = struct.unpack("<4sI", f.read(8))
            start = f.tell()
            if kind == b"ds64":
                if length < 28:
                    raise ValueError("RF64 ds64 incompleto")
                data = f.read(28)
                if len(data) != 28:
                    raise ValueError("RF64 truncado")
                riff_size, ds64_data, _, _ = struct.unpack("<QQQI", data)
                if riff_size + 8 > size:
                    raise ValueError("RF64 truncado: tamaño ds64")
            if length == 0xFFFFFFFF:
                if kind != b"data" or ds64_data is None:
                    raise ValueError("Tamaño RF64 desconocido")
                length = ds64_data
            if start + length > size:
                raise ValueError("WAV truncado: tamaño de chunk inválido")
            if kind == b"fmt ":
                f.seek(start)
                fmt = f.read(min(length, 40))
                if len(fmt) < 16:
                    raise ValueError("WAV fmt truncado")
                result["block_align"] = struct.unpack_from("<H", fmt, 12)[0]
            elif kind == b"bext" and length >= 346:
                f.seek(start + 338)
                result["time_reference"] = struct.unpack("<Q", f.read(8))[0]
            elif kind == b"data":
                result["data_size"] = length
            f.seek(start + length + (length % 2))
    return result


def inspect_audio(path: Path) -> AudioInfo:
    path = Path(path)
    meta = _wav_metadata(path)
    try:
        with sf.SoundFile(path, "r") as f:
            if f.subtype not in {"PCM_U8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}:
                raise ValueError("WAV comprimido no admitido")
            align, data_size = meta["block_align"], meta["data_size"]
            if not align or data_size is None or data_size % align or data_size // align != f.frames:
                raise ValueError("WAV truncado o cantidad de muestras inconsistente")
            bits = {"PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32}.get(f.subtype)
            return AudioInfo(f.samplerate, bits, f.subtype, f.frames, f.channels, meta["time_reference"])
    except sf.LibsndfileError as e:
        raise ValueError(f"No se puede decodificar WAV: {e}") from e


def iter_blocks(asset: Asset, span: Span, block_frames: int = 65536):
    if block_frames < 1:
        raise ValueError("Bloque inválido")
    local_start, local_end = span.start - asset.start_sample, span.end - asset.start_sample
    with ExitStack() as stack:
        readers = {}
        for c in asset.channels:
            if c.path not in readers:
                info = inspect_audio(c.path)
                if info.sample_rate != 48000:
                    raise ValueError("Sample rate incompatible con timeline de 48000 Hz")
                if local_start < 0 or local_end > info.frames:
                    raise ValueError("Cobertura insuficiente: duración o muestras no coinciden")
                readers[c.path] = stack.enter_context(sf.SoundFile(c.path, "r"))
                readers[c.path].seek(local_start)
            if c.file_channel >= readers[c.path].channels:
                raise ValueError("Índice de canal fuera del archivo")
        for start in range(span.start, span.end, block_frames):
            n = min(block_frames, span.end - start)
            buffers = {p: r.read(n, dtype="float64", always_2d=True) for p, r in readers.items()}
            if any(len(b) != n for b in buffers.values()):
                raise ValueError("Archivo truncado durante la lectura")
            yield Block(start, np.column_stack([buffers[c.path][:, c.file_channel] for c in asset.channels]))


def fingerprint(path: Path, stop: StopToken) -> dict:
    path = Path(path)
    before = path.stat()
    result = {"path": str(path), "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
              "sha256": None, "status": "cancelled", "reason": "Hash interrumpido"}
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while not stop.cancelled():
            block = f.read(1024 * 1024)
            if not block:
                after = path.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                    result.update(status="changed", reason="Fuente modificada durante el hash")
                else:
                    result.update(sha256=digest.hexdigest(), status="complete", reason=None)
                break
            digest.update(block)
    return result
