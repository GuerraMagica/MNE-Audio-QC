from unittest.mock import patch

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from mne_qc.domain import StopToken
from mne_qc.manifest import load_manifest
from mne_qc.technical import check_technical, measure_true_peak


def test_pcm48_24_profile(case):
    p, _, _ = case
    r = check_technical(load_manifest(p), StopToken())
    assert r.status == "complete"
    assert not r.findings


def test_unknown_bits_not_pass(case):
    p, arrays, _ = case
    sf.write(p.parent / "ME.wav", arrays["ME"], 48000, subtype="FLOAT")
    r = check_technical(load_manifest(p), StopToken())
    assert any(f.kind == "format" for f in r.findings)


def test_true_peak_above_minus2_reported(case):
    p, arrays, _ = case
    signal = np.ones_like(arrays["ME"]) * 0.95
    sf.write(p.parent / "ME.wav", signal, 48000, subtype="PCM_24")
    r = check_technical(load_manifest(p), StopToken())
    assert any(f.kind == "true_peak_exceeded" for f in r.findings)


def test_true_peak_boundary_is_measured(tmp_path):
    p = tmp_path / "tone.wav"
    t = np.arange(48000)
    data = 10**(-2 / 20) * np.sin(2 * np.pi * 1000 * t / 48000)
    sf.write(p, data, 48000, subtype="PCM_24")
    value = measure_true_peak(p, 0, StopToken())
    assert abs(value + 2) <= 0.2


def test_intersample_reference(tmp_path):
    p = tmp_path / "inter.wav"
    t = np.arange(48000)
    data = 0.9 * np.sin(2 * np.pi * 12000 * t / 48000 + np.pi / 4)
    # Smooth edges avoid boundary transients in the reference interpolator.
    data *= np.minimum(1, np.minimum(t / 1000, (47999 - t) / 1000))
    sf.write(p, data, 48000, subtype="PCM_24")
    ref = 20 * np.log10(np.max(np.abs(resample_poly(data, 64, 1))))
    measured = measure_true_peak(p, 0, StopToken())
    assert abs(measured - ref) <= 0.2
    assert measured > 20 * np.log10(np.max(np.abs(data))) + 2


def test_ffmpeg_failure_not_zero_peak(case):
    p, _, _ = case
    with patch("mne_qc.technical.measure_true_peak", side_effect=RuntimeError("tool error")):
        r = check_technical(load_manifest(p), StopToken())
    assert r.status == "partial"
    assert r.measurements["peaks"][0]["value_dbtp"] is None


def test_each_channel_and_lfe_scanned(case):
    p, _, _ = case
    m = load_manifest(p)
    with patch("mne_qc.technical.measure_true_peak", return_value=-20) as fn:
        r = check_technical(m, StopToken())
    assert fn.call_count == len(m.assets) * len(m.assets["PM"].channels)
    if m.layout == "5.1":
        assert any(v["label"] == "LFE" for v in r.measurements["peaks"])
