from pathlib import Path
from unittest.mock import patch

import pytest

from mne_qc.domain import CANDIDATE, Finding, RunResult, Span, StageResult, StopToken
from mne_qc.manifest import load_manifest
from mne_qc.report import make_previews, write_report
from mne_qc.runner import run_analysis


def test_failure_preserves_completed_stages(case, tmp_path):
    p, _, _ = case
    with patch("mne_qc.runner.check_technical", side_effect=RuntimeError("deliberate tool failure")):
        result = run_analysis(load_manifest(p), tmp_path / "out", None, StopToken())
    assert result.status == "partial"
    assert result.stages[0].status == "complete"
    assert any(s.status == "failed" for s in result.stages)
    assert (tmp_path / "out" / "run.json").is_file()


def test_cancel_preserves_coverage(case, tmp_path):
    p, _, _ = case
    stop = StopToken()
    def cancel_after_first(*args):
        stop.cancel()
        return StageResult("reconstruction", "complete", [Span(0, 24000)])
    with patch("mne_qc.runner.reconstruct", side_effect=cancel_after_first):
        result = run_analysis(load_manifest(p), tmp_path / "cancel", None, stop)
    assert result.status == "partial"
    assert result.stages[0].coverage == [Span(0,24000)]


def test_source_changed_blocks_preview(case, tmp_path):
    p, _, _ = case
    m = load_manifest(p)
    from mne_qc.audio import fingerprint
    identities = {str(c.path): fingerprint(c.path, StopToken()) for a in m.assets.values() for c in a.channels}
    source = m.assets["ME"].channels[0].path
    source.touch()
    r = RunResult("complete", [StageResult("comparison", "complete", findings=[
        Finding("x", "possible_omission", CANDIDATE, Span(0,1000), ("L",))])], identities)
    out = tmp_path / "previews"
    out.mkdir()
    with pytest.raises(ValueError, match="Fuente"):
        make_previews(r, m, out, StopToken())
    assert not list(out.glob("*.wav"))


def test_output_never_overwrites_input_or_previous_run(case, tmp_path):
    p, _, _ = case
    out = tmp_path / "existing"
    out.mkdir()
    (out/"sentinel").write_text("keep")
    with pytest.raises(FileExistsError): run_analysis(load_manifest(p), out, None, StopToken())
    assert (out/"sentinel").read_text() == "keep"


def test_html_escapes_names_and_asr_text(case, tmp_path):
    p, _, _ = case
    m = load_manifest(p)
    m.project_id = '<script>alert("x")</script>'
    r = RunResult("complete", [StageResult("speech_fixed", "complete", findings=[
        Finding("1", "language", CANDIDATE, Span(0,1000), ("L",), {"text_hypothesis": '<img src=x onerror="alert(1)">'})])])
    path = write_report(r, m, tmp_path)
    text = path.read_text()
    assert "<script>alert" not in text
    assert "&lt;script&gt;" in text and "&lt;img" in text


def test_no_alerts_is_not_pass(case, tmp_path):
    p, _, _ = case
    text = write_report(RunResult("complete"), load_manifest(p), tmp_path).read_text()
    assert "no equivale a aprobación" in text


def test_speech_off_is_visibly_not_run(case, tmp_path):
    p, _, _ = case
    with patch("mne_qc.runner.check_technical", return_value=StageResult("technical", "complete", [Span(0,24000)])):
        r = run_analysis(load_manifest(p), tmp_path/"off", None, StopToken())
    assert any(s.stage == "speech" and s.status == "not_run" for s in r.stages)
    assert r.environment["source_identity_verified"] is True


def test_source_identity_preserved_on_normal_run(case, tmp_path):
    p, _, _ = case
    from mne_qc.audio import fingerprint
    source = p.parent/"PM.wav"
    before = fingerprint(source, StopToken())["sha256"]
    with patch("mne_qc.runner.check_technical", return_value=StageResult("technical", "complete")):
        run_analysis(load_manifest(p), tmp_path/"normal", None, StopToken())
    assert fingerprint(source, StopToken())["sha256"] == before


def test_title_exception_preserves_language_detection(case,tmp_path):
    p,_,_=case
    m=load_manifest(p)
    m.exceptions=[{"kind":"language","span":[0,24000],"channels":["L"],"policy":"allowed","reason":"Licensed song"}]
    from mne_qc.domain import SpeechConfig
    import sys
    language=StageResult("speech_fixed","complete",[Span(0,24000)],
        [Finding("lyric","language",CANDIDATE,Span(1000,2000),("L",),{"text_hypothesis":"hello"})])
    with patch("mne_qc.runner.check_technical",return_value=StageResult("technical","complete")), \
         patch("mne_qc.runner.scan_speech",return_value=language):
        r=run_analysis(m,tmp_path/"exception",SpeechConfig(Path(sys.executable),tmp_path/"fake"),StopToken())
    f=next(s for s in r.stages if s.stage=="speech_fixed").findings[0]
    assert f.kind=="language" and f.evidence["policy_exempt"] is True


def test_runtime_measurements_are_observed_and_scoped(case,tmp_path):
    p,_,_=case
    with patch("mne_qc.runner.check_technical",return_value=StageResult("technical","complete")):
        r=run_analysis(load_manifest(p),tmp_path/"timed",None,StopToken())
    assert r.environment["memory_observation"]["self_max_rss_bytes"]>0
    assert "ffmpeg" in r.environment["tool_versions"]
    assert all(s.measurements["stage_wall_seconds"]>=0 for s in r.stages)
