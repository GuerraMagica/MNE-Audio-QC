import json
import hashlib
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from .domain import Asset, ChannelSource, LAYOUTS, Manifest


def manifest_identity(manifest: Manifest) -> str:
    """Include channel map, origins, versions and title exceptions, not just paths."""
    raw=json.dumps(asdict(manifest),default=str,sort_keys=True,ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _integer(value, name):
    if type(value) is not int:
        raise ValueError(f"{name}: debe ser un entero")
    return value


def load_manifest(path: Path) -> Manifest:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or raw.get("layout") not in LAYOUTS:
        raise ValueError("Versión de manifiesto o configuración no admitida")
    labels = LAYOUTS[raw["layout"]]
    assets = {}
    for role, a in raw.get("assets", {}).items():
        if role not in {"PM", "ME", "DX", "MX", "FX"} and not role.startswith("OPT"):
            raise ValueError(f"Rol desconocido: {role}")
        channels = []
        for c in a["channels"]:
            if c["label"] not in labels or "://" in c["path"]:
                raise ValueError("Mapa de canales o ruta no válidos; solo archivos locales")
            index = _integer(c["file_channel"], "file_channel")
            if index < 0:
                raise ValueError("Índice de canal negativo")
            channels.append(ChannelSource((path.parent / c["path"]).resolve(), index, c["label"]))
        if not channels or len({c.label for c in channels}) != len(channels):
            raise ValueError("Etiquetas de canales vacías o duplicadas")
        if len({(c.path, c.file_channel) for c in channels}) != len(channels):
            raise ValueError("Un canal físico está asignado dos veces")
        if role in {"PM", "ME"} and set(c.label for c in channels) != set(labels):
            raise ValueError("PM/M&E requieren mapa completo declarado")
        channels.sort(key=lambda c: labels.index(c.label))
        assets[role] = Asset(role, str(a["version"]), tuple(channels), _integer(a["start_sample"], "start_sample"))
    if not {"PM", "ME", "DX"} <= assets.keys():
        raise ValueError("Se requieren PM, ME y DX")
    span = tuple(raw["program_span"])
    if len(span) != 2 or any(type(x) is not int for x in span) or span[1] <= span[0]:
        raise ValueError("Región de programa no válida")
    fps = Fraction(str(raw["fps"])) if raw.get("fps") is not None else None
    if fps is not None and fps <= 0:
        raise ValueError("FPS debe ser positivo")
    start_frame = raw.get("start_frame")
    if start_frame is not None:
        _integer(start_frame, "start_frame")
    exceptions=raw.get("exceptions", [])
    if not isinstance(exceptions,list):
        raise ValueError("Excepciones debe ser una lista")
    for exception in exceptions:
        interval=exception.get("span",[])
        if exception.get("kind") != "language" or exception.get("policy") != "allowed" or not exception.get("reason"):
            raise ValueError("Excepción requiere language, allowed y motivo explícito")
        if len(interval)!=2 or any(type(x) is not int for x in interval) or interval[1]<=interval[0]:
            raise ValueError("Rango de excepción inválido")
        if not exception.get("channels") or not set(exception["channels"])<=set(labels):
            raise ValueError("Canales de excepción inválidos")
    if type(raw.get("synthetic",False)) is not bool:
        raise ValueError("synthetic debe ser booleano")
    return Manifest(str(raw["project_id"]), raw["layout"], assets, span, fps,
                    start_frame, bool(raw.get("drop_frame", False)), exceptions,raw.get("synthetic",False))
