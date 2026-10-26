from dataclasses import replace

import numpy as np
import soundfile as sf

from conftest import make_case
from mne_qc.domain import ChannelSource, Span, StopToken
from mne_qc.manifest import load_manifest
from mne_qc.reconstruction import describe_me_residual, reconstruct
from mne_qc.timeline import program_coverage


def test_exact_sum_cancels(case):
    p, _, _ = case
    m = load_manifest(p)
    r = reconstruct(m, StopToken())
    assert r.status == "complete"
    assert r.measurements["peak_abs_by_channel"] == [0.0] * len(m.assets["PM"].channels)
    assert r.coverage == [Span(*m.program_span)]


def test_one_sample_offset_is_reported_not_fixed(case):
    p, arrays, _ = case
    sf.write(p.parent / "DX.wav", np.roll(arrays["DX"], 1, axis=0), 48000, subtype="PCM_24")
    r = reconstruct(load_manifest(p), StopToken())
    assert max(r.measurements["peak_abs_by_channel"]) > 0
    assert r.measurements["alignment_diagnostics"]["DX"]["offset_samples"] in (-1, 1)
    assert r.measurements["raw_residual"] is True


def test_polarity_gain_and_swap_are_candidates(tmp_path):
    p, arrays, _ = make_case(tmp_path)
    # Make channel content distinguishable, then swap the complete stem.
    dx = arrays["DX"].copy()
    dx[:, 1] = np.roll(dx[:, 1], 173) * 0.5
    pm = arrays["MX"] + arrays["FX"] + dx
    sf.write(tmp_path / "PM.wav", pm, 48000, subtype="PCM_24")
    sf.write(tmp_path / "DX.wav", dx[:, ::-1], 48000, subtype="PCM_24")
    r = reconstruct(load_manifest(p), StopToken())
    assert any(f.kind == "channel_or_processing_difference" for f in r.findings)
    sf.write(tmp_path / "DX.wav", -dx, 48000, subtype="PCM_24")
    r = reconstruct(load_manifest(p), StopToken())
    assert r.measurements["alignment_diagnostics"]["DX"]["polarity"] == -1


def test_silence_has_no_reliable_alignment(case):
    p, arrays, _ = case
    for role, a in arrays.items():
        sf.write(p.parent / f"{role}.wav", np.zeros_like(a), 48000, subtype="PCM_24")
    r = reconstruct(load_manifest(p), StopToken())
    assert r.measurements["alignment_diagnostics"]["DX"]["offset_samples"] is None


def test_leader_and_tail_coverage_visible(case):
    p, _, _ = case
    m = load_manifest(p)
    m.assets["DX"] = replace(m.assets["DX"], start_sample=100)
    c = program_coverage(m)
    assert c["assets"]["DX"]["missing"] == [[0, 100]]
    assert c["assets"]["DX"]["outside_program"] == [[24000, 24100]]
    assert reconstruct(m, StopToken()).status == "partial"


def test_mono_dx_5_1_is_not_evaluable(tmp_path):
    p, _, _ = make_case(tmp_path, "5.1")
    m = load_manifest(p)
    m.assets["DX"] = replace(m.assets["DX"], channels=(ChannelSource(p.parent / "DX.wav", 2, "C"),))
    assert reconstruct(m, StopToken()).status == "not_evaluable"


def test_reduced_mode_has_no_original_reconstruction(case):
    p, _, _ = case
    m = load_manifest(p)
    del m.assets["MX"]
    del m.assets["FX"]
    assert reconstruct(m, StopToken()).status == "not_evaluable"
    assert describe_me_residual(m, StopToken()).status == "complete"


def test_null_counterexamples_never_approve_me(case):
    p, arrays, _ = case
    # A production step is in DX; the PM has it, original FX does not.
    step = arrays["FX"]
    sf.write(p.parent / "DX.wav", arrays["DX"] + step, 48000, subtype="PCM_24")
    sf.write(p.parent / "FX.wav", np.zeros_like(step), 48000, subtype="PCM_24")
    sf.write(p.parent / "ME.wav", arrays["MX"], 48000, subtype="PCM_24")
    bad = describe_me_residual(load_manifest(p), StopToken())
    assert max(bad.measurements["peak_abs_by_channel"]) == 0
    assert bad.measurements["me_verdict"] == "requires_review"
    sf.write(p.parent / "ME.wav", arrays["MX"] + step * 0.5, 48000, subtype="PCM_24")
    good = describe_me_residual(load_manifest(p), StopToken())
    assert max(good.measurements["peak_abs_by_channel"]) > 0
    assert good.measurements["me_verdict"] == "requires_review"


def test_cancelled_reconstruction_preserves_coverage(case):
    p, _, _ = case
    stop = StopToken()
    stop.cancel()
    r = reconstruct(load_manifest(p), stop)
    assert r.status == "partial" and not r.coverage
