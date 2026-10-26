import json
import struct

import numpy as np
import pytest
import soundfile as sf

from conftest import make_case
from mne_qc.audio import fingerprint, inspect_audio, iter_blocks
from mne_qc.domain import Span, StopToken
from mne_qc.manifest import load_manifest


def test_discrete_matches_interleaved(tmp_path):
    a, _, _ = make_case(tmp_path / "a", "5.1")
    b, _, _ = make_case(tmp_path / "b", "5.1", discrete=True)
    x = np.concatenate([v.samples for v in iter_blocks(load_manifest(a).assets["PM"], Span(0, 24000), 777)])
    y = np.concatenate([v.samples for v in iter_blocks(load_manifest(b).assets["PM"], Span(0, 24000), 777)])
    assert x.shape == y.shape == (24000, 6)
    assert np.array_equal(x, y)


@pytest.mark.parametrize("bad", ["duplicate", "unknown", "url", "negative_index"])
def test_duplicate_or_unknown_channel_rejected(case, bad):
    path, _, m = case
    c = m["assets"]["ME"]["channels"]
    if bad == "duplicate": c[1]["label"] = c[0]["label"]
    if bad == "unknown": c[0]["label"] = "guess"
    if bad == "url": c[0]["path"] = "https://example.com/private.wav"
    if bad == "negative_index": c[0]["file_channel"] = -1
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError): load_manifest(path)


def test_wrong_rate_or_bits_diagnosed(tmp_path):
    p = tmp_path / "wrong.wav"
    sf.write(p, np.zeros(100), 44100, subtype="PCM_16")
    info = inspect_audio(p)
    assert (info.sample_rate, info.declared_bits) == (44100, 16)


def test_short_discrete_channel_not_zero_padded(tmp_path):
    p, _, _ = make_case(tmp_path, discrete=True)
    sf.write(tmp_path / "PM_R.wav", np.zeros(100), 48000, subtype="PCM_24")
    with pytest.raises(ValueError, match="cobertura|duración|muestras"):
        list(iter_blocks(load_manifest(p).assets["PM"], Span(0, 24000)))


def test_truncated_wav_rejected(tmp_path):
    p = tmp_path / "bad.wav"
    sf.write(p, np.ones(1000) * 0.1, 48000, subtype="PCM_24")
    p.write_bytes(p.read_bytes()[:-60])
    with pytest.raises(ValueError, match="trunc|tamaño"):
        inspect_audio(p)


def test_rf64_and_bwf_metadata(tmp_path):
    rf = tmp_path / "rf.wav"
    sf.write(rf, np.zeros(100), 48000, format="RF64", subtype="PCM_24")
    assert inspect_audio(rf).frames == 100
    p = tmp_path / "bwf.wav"
    sf.write(p, np.zeros(100), 48000, subtype="PCM_24")
    raw = p.read_bytes()
    bext = bytearray(602)
    struct.pack_into("<Q", bext, 338, 2**32 + 48000)
    extra = b"bext" + struct.pack("<I", len(bext)) + bext
    result = raw[:12] + extra + raw[12:]
    result = result[:4] + struct.pack("<I", len(result) - 8) + result[8:]
    p.write_bytes(result)
    assert inspect_audio(p).bwf_time_reference == 2**32 + 48000


def test_cancelled_fingerprint_is_explicit(tmp_path):
    p = tmp_path / "a"
    p.write_bytes(b"x" * 10)
    stop = StopToken()
    stop.cancel()
    result = fingerprint(p, stop)
    assert result["sha256"] is None
    assert result["status"] == "cancelled"


def test_float_bits_are_not_pcm_precision(tmp_path):
    p = tmp_path / "float.wav"
    sf.write(p, np.zeros(100), 48000, subtype="FLOAT")
    assert inspect_audio(p).declared_bits is None
