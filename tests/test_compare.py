import numpy as np
import soundfile as sf

from mne_qc.compare import compare_me
from mne_qc.domain import StopToken
from mne_qc.manifest import load_manifest


def test_missing_music_impact_ambience(case):
    p, arrays, _ = case
    sf.write(p.parent / "ME.wav", np.zeros_like(arrays["ME"]), 48000, subtype="PCM_24")
    r = compare_me(load_manifest(p), {}, StopToken())
    assert any(f.kind == "possible_omission" for f in r.findings)
    assert all(f.state == "posible incidencia" for f in r.findings)


def test_partial_missing_event(case):
    p, arrays, _ = case
    me = arrays["ME"].copy()
    me[7000:13000] = 0
    sf.write(p.parent / "ME.wav", me, 48000, subtype="PCM_24")
    r = compare_me(load_manifest(p), {}, StopToken())
    assert any(f.kind in {"possible_omission", "additional_or_replacement"} for f in r.findings)


def test_foley_replacement_not_automatic_error(case):
    p, arrays, _ = case
    sf.write(p.parent / "ME.wav", arrays["MX"] + np.roll(arrays["FX"], 180, axis=0), 48000, subtype="PCM_24")
    r = compare_me(load_manifest(p), {}, StopToken())
    assert all(f.state == "posible incidencia" for f in r.findings)


def test_dx_step_missing_despite_null(case):
    p, arrays, _ = case
    step = arrays["FX"]
    sf.write(p.parent / "DX.wav", arrays["DX"] + step, 48000, subtype="PCM_24")
    sf.write(p.parent / "FX.wav", np.zeros_like(step), 48000, subtype="PCM_24")
    sf.write(p.parent / "ME.wav", arrays["MX"], 48000, subtype="PCM_24")
    r = compare_me(load_manifest(p), {}, StopToken())
    assert any(f.kind == "dx_content_review" for f in r.findings)


def test_pfx_during_speech_retained_for_review(case):
    p, _, _ = case
    r = compare_me(load_manifest(p), {}, StopToken())
    assert any(f.kind == "dx_content_review" for f in r.findings)
    assert r.measurements["dx_separation_validated"] is False


def test_channel_relocation_not_global_omission(case):
    p, arrays, _ = case
    me = arrays["ME"].copy()
    me[:, 1] += me[:, 0]
    me[:, 0] = 0
    sf.write(p.parent / "ME.wav", me, 48000, subtype="PCM_24")
    r = compare_me(load_manifest(p), {}, StopToken())
    assert any(f.kind == "spatial_change" for f in r.findings)
    assert not any(f.state == "error técnico comprobado" for f in r.findings)


def test_optional_is_not_evidence_of_main_body_presence(case):
    p, arrays, _ = case
    m = load_manifest(p)
    m.assets["OPT_A"] = m.assets["FX"]
    sf.write(p.parent / "ME.wav", np.zeros_like(arrays["ME"]), 48000, subtype="PCM_24")
    r = compare_me(m, {}, StopToken())
    assert any(f.kind == "possible_omission" for f in r.findings)


def test_reduced_is_exploratory(case):
    p, _, _ = case
    m = load_manifest(p)
    del m.assets["FX"]
    r = compare_me(m, {}, StopToken())
    assert r.measurements["mode"] == "reduced"
    assert r.measurements["omission_claim"] == "exploratory"
