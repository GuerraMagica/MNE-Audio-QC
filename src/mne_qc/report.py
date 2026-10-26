from dataclasses import asdict, is_dataclass
from fractions import Fraction
import html
import json
import math
from pathlib import Path
from urllib.parse import quote

import soundfile as sf

from .audio import fingerprint, iter_blocks
from .domain import Manifest, RunResult, Span, StopToken
from .timeline import format_timecode


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (Path, Fraction)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def identities_unchanged(identities: dict, stop: StopToken, verify_hash=False) -> bool:
    for name, identity in identities.items():
        p = Path(name)
        if identity.get("status") != "complete" or not p.is_file():
            return False
        s = p.stat()
        if (s.st_size, s.st_mtime_ns) != (identity["size_bytes"], identity["mtime_ns"]):
            return False
        if verify_hash and fingerprint(p, stop).get("sha256") != identity.get("sha256"):
            return False
    return not stop.cancelled()


def make_previews(result: RunResult, manifest: Manifest, output: Path, stop: StopToken, limit=20):
    if not identities_unchanged(result.input_identity, stop, verify_hash=True):
        raise ValueError("Fuente modificada o identidad incompleta; previews bloqueados")
    generated = []
    records = []
    try:
        for stage in result.stages:
            for finding in stage.findings:
                if len(records) >= limit or stop.cancelled():
                    break
                if finding.span is None:
                    continue
                origin = max(manifest.program_span[0], finding.span.start - 48000)
                end = min(manifest.program_span[1], origin + 10*48000, finding.span.end + 48000)
                if end <= origin:
                    continue
                record = {"finding_id": finding.id, "start": origin, "end": end, "files": {},
                          "auxiliary": True, "normalized": False, "layout": manifest.layout}
                for role in ("PM", "ME", "DX"):
                    if not identities_unchanged(result.input_identity, stop):
                        raise ValueError("Fuente modificada durante previews")
                    a = manifest.assets[role]
                    try:
                        samples = next(iter_blocks(a, Span(origin, end), 10*48000)).samples
                    except ValueError:
                        continue  # No padding of a missing guide or source region.
                    target = output / f"preview-{len(records):03d}-{role}.wav"
                    sf.write(target, samples, 48000, subtype="PCM_24")
                    generated.append(target)
                    record["files"][role] = target.name
                records.append(record)
        if not identities_unchanged(result.input_identity, stop, verify_hash=True):
            raise ValueError("Fuente modificada o cancelación; previews invalidados")
    except BaseException:
        for p in generated:
            p.unlink(missing_ok=True)  # Only auxiliary files created by this invocation.
        raise
    result.environment["previews"] = records
    result.environment["preview_limit"] = limit
    return records


def write_report(result: RunResult, manifest: Manifest, output: Path) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run.json", {"schema_version": 1, "manifest": manifest, "result": result})
    findings = [f for s in result.stages for f in s.findings]
    review_path = output / "review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {"schema_version": 1, "decisions": {}}
    for f in findings:
        review["decisions"].setdefault(f.id, {"decision": "pending", "note": ""})
    write_json(review_path, review)
    esc = lambda x: html.escape(str(x), quote=True)
    preview_map = {p["finding_id"]: p for p in result.environment.get("previews", [])}
    rows = []
    for stage in result.stages:
        rows.append(f'<section><h2>{esc(stage.stage)} · {esc(stage.status)}</h2><p>{esc(stage.reason or "")}</p>'
                    f'<p>Cobertura calculada: {esc([(s.start,s.end) for s in stage.coverage])}</p>')
        for f in stage.findings:
            position = "Archivo / sin posición localizada"
            if f.span:
                position = f"{f.span.start}–{f.span.end} muestras · {f.span.start/48000:.3f}–{f.span.end/48000:.3f} s"
                if manifest.fps is not None and manifest.start_frame is not None:
                    position += " · " + format_timecode(f.span.start, manifest.fps, manifest.start_frame, manifest.drop_frame)
            evidence = json.dumps(jsonable(f.evidence), ensure_ascii=False, indent=2)
            rows.append(f'<article><h3>{esc(f.kind)} · {esc(", ".join(f.channels))}</h3>'
                        f'<p>{esc(f.state)} · {esc(position)}</p><pre>{esc(evidence)}</pre>')
            for role, name in preview_map.get(f.id, {}).get("files", {}).items():
                rows.append(f'<p>{esc(role)} · preview auxiliar <audio controls preload="none" src="{quote(name)}"></audio></p>')
            rows.append(f'<label>Revisión <select data-id="{esc(f.id)}"><option value="pending">Pendiente</option>'
                        '<option value="confirmed">Incidencia confirmada</option><option value="accepted">Diferencia aceptada</option>'
                        '<option value="ambiguous">Ambiguo</option></select></label> '
                        f'<input data-note="{esc(f.id)}" placeholder="Nota de revisión" aria-label="Nota de revisión">'
                        '</article>')
        rows.append('</section>')
    text = f'''<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>M&E QC · {esc(manifest.project_id)}</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 24px;background:#f5f6f8;color:#172331}}h1{{margin-bottom:8px}}section,article{{background:white;padding:18px;margin:16px 0;border:1px solid #dce1e7;border-radius:10px}}article{{border-left:4px solid #ba841a}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f6f8;padding:12px}}button,input,select{{font:inherit;padding:6px}}audio{{vertical-align:middle}}.note{{background:#fff3d2;padding:16px}}input{{max-width:90%;width:340px}}</style>
<h1>M&E QC Inspector</h1><p>{esc(manifest.project_id)} · {esc(manifest.layout)} · cómputo: {esc(result.status)}</p>
<p class="note">Resultado experimental: terminar el análisis no equivale a aprobación de la entrega. Sin alertas no demuestra ausencia de errores. La discernibilidad requiere escucha del M&E original. El navegador puede no reproducir correctamente el multicanal.</p>
<p>Identidad de fuentes verificada: {esc(result.environment.get("source_identity_verified", False))}. Datos y audio permanecen locales.</p>
<button id="export">Guardar revisión JSON</button>
<label> Cargar revisión anterior <input type="file" id="import" accept=".json"></label>
{''.join(rows)}
<h2>Fuentes originales</h2><ul>{''.join(f'<li>{esc(r)}: {esc(c.label)} · {esc(c.path)}</li>' for r,a in manifest.assets.items() for c in a.channels)}</ul>
<p>Las mediciones completas y versiones están en run.json. Los previews, cuando existen, conservan nivel y están limitados a 10 segundos.</p>
<script>
document.getElementById('export').onclick=()=>{{const decisions={{}};document.querySelectorAll('select[data-id]').forEach(s=>{{const note=[...document.querySelectorAll('input[data-note]')].find(n=>n.dataset.note===s.dataset.id);decisions[s.dataset.id]={{decision:s.value,note:note.value}};}});const u=URL.createObjectURL(new Blob([JSON.stringify({{schema_version:1,decisions}},null,2)],{{type:'application/json'}}));const a=document.createElement('a');a.href=u;a.download='review.json';a.click();URL.revokeObjectURL(u);}};
document.getElementById('import').onchange=async(e)=>{{try{{const r=JSON.parse(await e.target.files[0].text());document.querySelectorAll('select[data-id]').forEach(s=>{{const d=r.decisions?.[s.dataset.id];if(d){{s.value=d.decision;const n=[...document.querySelectorAll('input[data-note]')].find(n=>n.dataset.note===s.dataset.id);n.value=d.note||'';}}}});}}catch(_){{alert('No se puede leer la revisión JSON');}}}};
</script></html>'''
    target = output / "report.html"
    target.write_text(text, encoding="utf-8")
    return target
