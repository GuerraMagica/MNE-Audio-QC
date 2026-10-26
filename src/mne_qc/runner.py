from dataclasses import replace
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import resource
import time

from .audio import fingerprint
from .compare import compare_me, DEFAULTS
from .domain import Manifest, RunResult, SpeechConfig, StageResult, StopToken
from .manifest import manifest_identity
from .reconstruction import describe_me_residual, reconstruct
from .report import identities_unchanged, jsonable, make_previews, write_report
from .speech import scan_speech
from .technical import check_technical
from .process import run_local


def run_analysis(manifest: Manifest, output: Path, speech: SpeechConfig | None, stop: StopToken,
                 *, previews=False, both_speech_modes=False, dsp_config=None) -> RunResult:
    output = Path(output).resolve()
    sources = sorted({c.path for a in manifest.assets.values() for c in a.channels})
    if any(output == p or output in p.parents for p in sources):
        raise ValueError("El directorio de salida contiene entradas; elija uno nuevo")
    output.mkdir(parents=True, exist_ok=False)
    result = RunResult("partial", environment={"python": platform.python_version(), "platform": platform.platform(),
        "package_version": "0.1.0", "source_identity_verified": False,
        "project_id":manifest.project_id,"layout":manifest.layout,"program_span":list(manifest.program_span),
        "synthetic":manifest.synthetic,
        "manifest_sha256":manifest_identity(manifest),
        "mode":"complete" if {"MX","FX"} <= manifest.assets.keys() else "reduced",
        "profile_source": "Netflix branded public guidance 2026-10-26; title applicability requires review",
        "requested_speech": "both" if both_speech_modes else speech.mode if speech else "off"})
    implementation = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        implementation.update(path.name.encode()+path.read_bytes())
    analysis_config = {"dsp":{**DEFAULTS,**(dsp_config or {})},"speech":jsonable(speech),"both_speech_modes":both_speech_modes}
    result.environment.update(implementation_sha256=implementation.hexdigest(),analysis_config=analysis_config,
        analysis_config_sha256=hashlib.sha256(json.dumps(analysis_config,sort_keys=True).encode()).hexdigest(),
        dependencies={name:importlib.metadata.version(name) for name in ("numpy","scipy","soundfile")})
    tools={}
    for tool in ("ffmpeg","ffprobe"):
        try:
            stdout,_=run_local([tool,"-version"],stop,10)
            tools[tool]=stdout.splitlines()[0]
        except (OSError,RuntimeError,TimeoutError,InterruptedError) as e:
            tools[tool]=f"unavailable: {e}"
    result.environment["tool_versions"]=tools
    started = time.monotonic()
    for p in sources:
        try:
            result.input_identity[str(p)] = fingerprint(p, stop)
        except OSError as e:
            result.input_identity[str(p)] = {"path": str(p), "status": "failed", "sha256": None, "reason": str(e)}
        write_report(result, manifest, output)
        if stop.cancelled():
            result.stages.append(StageResult("input_identity", "partial", reason="Hash cancelado"))
            write_report(result, manifest, output)
            return result
    if any(i.get("status") != "complete" for i in result.input_identity.values()):
        result.status = "failed"
        result.stages.append(StageResult("input_identity", "failed", reason="No se verificaron las fuentes"))
        write_report(result, manifest, output)
        return result
    steps = [("reconstruction", lambda: reconstruct(manifest, stop)),
             ("me_residual", lambda: describe_me_residual(manifest, stop)),
             ("technical", lambda: check_technical(manifest, stop)),
             ("comparison", lambda: compare_me(manifest, dsp_config or {}, stop))]
    if speech:
        modes = ["fixed", "vad"] if both_speech_modes else [speech.mode]
        for mode in modes:
            steps.append((f"speech_{mode}", lambda mode=mode: scan_speech(manifest, replace(speech, mode=mode), stop)))
    else:
        steps.append(("speech", lambda: StageResult("speech", "not_run", reason="IA desactivada; no se ha buscado lenguaje")))
    for name, action in steps:
        stage_start=time.monotonic()
        if stop.cancelled():
            result.stages.append(StageResult(name, "not_run", reason="Cancelado antes de esta etapa"))
        else:
            try:
                result.stages.append(action())
            except (OSError, ValueError, RuntimeError, TimeoutError) as e:
                result.stages.append(StageResult(name, "failed", reason=str(e)))
        for finding in result.stages[-1].findings:
            if finding.kind != "language" or finding.span is None:
                continue
            for exception in manifest.exceptions:
                if exception.get("kind")=="language" and exception.get("policy")=="allowed" and \
                   set(finding.channels)<=set(exception.get("channels",[])) and \
                   exception["span"][0]<=finding.span.start and exception["span"][1]>=finding.span.end:
                    finding.evidence.update(policy_exempt=True,title_exception=exception)
        result.stages[-1].measurements["stage_wall_seconds"]=time.monotonic()-stage_start
        write_report(result, manifest, output)
    verified = identities_unchanged(result.input_identity, stop, verify_hash=True)
    result.environment["source_identity_verified"] = verified
    if not verified:
        for stage in result.stages:
            if stage.status == "complete":
                stage.status = "partial"
                stage.reason = "Identidad final no verificada: cancelación o fuente modificada"
    if verified and previews:
        try:
            make_previews(result, manifest, output, stop)
        except (ValueError, OSError) as e:
            result.environment["preview_error"] = str(e)
            result.environment["source_identity_verified"] = False
    bad = any(s.status not in {"complete", "not_run"} for s in result.stages)
    result.status = "complete" if verified and not bad and not stop.cancelled() and not result.environment.get("preview_error") else "partial"
    result.environment["elapsed_seconds"] = time.monotonic() - started
    factor=1 if platform.system()=="Darwin" else 1024
    result.environment["memory_observation"]={
        "self_max_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*factor,
        "children_max_rss_bytes":resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*factor,
        "scope":"cumulative process maxima; not simultaneous total or GPU memory"}
    write_report(result, manifest, output)
    return result
