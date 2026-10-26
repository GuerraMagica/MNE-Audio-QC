from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from mne_qc.domain import SpeechConfig, Span, StopToken
from mne_qc.manifest import load_manifest
from mne_qc.process import run_local
from mne_qc.speech import map_segments, merge_detections, parse_whisper_json, scan_speech


def config(tmp_path, mode="fixed"):
    model = tmp_path / "fake-model.bin"
    model.write_bytes(b"test-double-not-real-weights")
    vad = tmp_path / "fake-vad.bin"
    vad.write_bytes(b"test-double")
    return SpeechConfig(Path(sys.executable), model, vad, mode=mode)


def test_window_boundary_mapping_and_dedup():
    raw = [{"start_ms": 25000, "end_ms": 27000, "text": "una frase"}]
    a = map_segments(raw, 0, 30*48000, "Rs")
    b = map_segments([{"start_ms": 0, "end_ms": 2000, "text": "una frase"}], 25*48000, 30*48000, "Rs")
    merged = merge_detections(a+b)
    assert len(merged) == 1
    assert merged[0].span == Span(25*48000, 27*48000)


def test_parser_uses_offsets_in_ms():
    r = parse_whisper_json({"transcription": [{"offsets": {"from": 120, "to": 330}, "text": "Hola"}]})
    assert r == [{"start_ms": 120, "end_ms": 330, "text": "Hola"}]
    with pytest.raises(ValueError): parse_whisper_json({"unexpected": []})


def test_surround_channel_never_skipped(case, tmp_path):
    p, _, _ = case
    with patch("mne_qc.speech._transcribe_window", return_value=[{"start_ms": 100, "end_ms": 300, "text": "hola"}]) as fn:
        r = scan_speech(load_manifest(p), config(tmp_path), StopToken())
    labels = {c for f in r.findings for c in f.channels}
    assert "R" in labels
    if load_manifest(p).layout == "5.1": assert "Rs" in labels
    assert fn.call_count >= len(labels)


def test_voice_absent_from_dx_still_candidate(case, tmp_path):
    p, _, _ = case
    responses = [{"start_ms": 100, "end_ms": 300, "text": "radio"}]
    def backend(path, cfg, stop): return [] if "DX" in str(path) else responses
    with patch("mne_qc.speech._transcribe_window", side_effect=backend):
        r = scan_speech(load_manifest(p), config(tmp_path), StopToken())
    assert r.findings and all(f.evidence["dx_text_match"] is False for f in r.findings)


def test_hallucinated_text_not_confirmed_issue(case, tmp_path):
    p, _, _ = case
    with patch("mne_qc.speech._transcribe_window", return_value=[{"start_ms": 0, "end_ms": 200, "text": "inventado"}]):
        r = scan_speech(load_manifest(p), config(tmp_path), StopToken())
    assert all(f.state == "posible incidencia" for f in r.findings)
    assert all(f.evidence["intelligibility_confirmed"] is False for f in r.findings)


def test_missing_binary_or_weights_not_evaluable(case, tmp_path):
    p, _, _ = case
    cfg = SpeechConfig(tmp_path / "none", tmp_path / "missing")
    assert scan_speech(load_manifest(p), cfg, StopToken()).status == "not_evaluable"


def test_timeout_and_cancellation_terminate_child():
    with pytest.raises(TimeoutError):
        run_local([sys.executable, "-c", "import time; time.sleep(10)"], StopToken(), timeout=0.05)
    stop = StopToken()
    stop.cancel()
    with pytest.raises(InterruptedError): run_local([sys.executable, "-c", "pass"], stop)


def test_vad_and_fixed_have_distinct_coverage(case, tmp_path):
    p, _, _ = case
    with patch("mne_qc.speech._transcribe_window", return_value=[]):
        a = scan_speech(load_manifest(p), config(tmp_path), StopToken())
        b = scan_speech(load_manifest(p), config(tmp_path, "vad"), StopToken())
    assert a.stage == "speech_fixed" and b.stage == "speech_vad"
    assert a.coverage == b.coverage
    assert b.measurements["coverage_semantics"] == "input_scanned_vad_may_reject_speech"


def test_no_network_or_automatic_download(case, tmp_path):
    p, _, _ = case
    with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
        r = scan_speech(load_manifest(p), SpeechConfig(tmp_path/"none", tmp_path/"none2"), StopToken())
    assert r.status == "not_evaluable"


def test_child_failure_keeps_previous_detections(case, tmp_path):
    p, _, _ = case
    calls = 0
    def backend(*args):
        nonlocal calls
        calls += 1
        if calls > 1: raise RuntimeError("tool failure")
        return [{"start_ms": 10, "end_ms": 100, "text": "hola"}]
    with patch("mne_qc.speech._transcribe_window", side_effect=backend):
        r = scan_speech(load_manifest(p), config(tmp_path), StopToken())
    assert r.status == "partial" and r.findings


def test_explicit_cpu_backend_is_forwarded_to_cli(tmp_path):
    from mne_qc.speech import _transcribe_window
    cfg=replace(config(tmp_path),use_gpu=False)
    audio=tmp_path/"control.wav"
    def backend(args,*unused):
        output=Path(args[args.index("-of")+1]+".json")
        output.write_text(json.dumps({"transcription":[]}))
        return "",""
    with patch("mne_qc.speech.run_local",side_effect=backend) as fn:
        assert _transcribe_window(audio,cfg,StopToken())==[]
    assert "-ng" in fn.call_args.args[0]


def test_non_speech_model_label_is_preserved_but_not_called_language(case,tmp_path):
    p,_,_=case
    with patch("mne_qc.speech._transcribe_window",return_value=[{"start_ms":0,"end_ms":10000,"text":"[MÚSICA]"}]):
        r=scan_speech(load_manifest(p),config(tmp_path),StopToken())
    assert not r.findings
    assert r.measurements["non_speech_annotations"][0]["text"]=="[MÚSICA]"


def test_words_next_to_annotation_remain_language_candidate(case,tmp_path):
    p,_,_=case
    with patch("mne_qc.speech._transcribe_window",return_value=[{"start_ms":0,"end_ms":200,"text":"[MÚSICA] Hola"}]):
        r=scan_speech(load_manifest(p),config(tmp_path),StopToken())
    assert r.findings and r.findings[0].kind=="language"
