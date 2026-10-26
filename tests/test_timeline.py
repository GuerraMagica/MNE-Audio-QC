from fractions import Fraction

import pytest

from mne_qc.timeline import format_timecode


@pytest.mark.parametrize("fps,sample,expected", [(Fraction(24),48000,"00:00:01:00"),
    (Fraction(25),48000,"00:00:01:00"), (Fraction(24000,1001),48048,"00:00:01:00")])
def test_timecode_24_25_23976(fps, sample, expected):
    assert format_timecode(sample, fps, 0, False) == expected


def test_dropframe_2997_boundaries():
    fps = Fraction(30000,1001)
    def sample(frame): return round(Fraction(frame * 48000, 1) / fps)
    assert format_timecode(sample(1800), fps, 0, True) == "00:01:00;02"
    assert format_timecode(sample(17982), fps, 0, True) == "00:10:00;00"
    with pytest.raises(ValueError): format_timecode(0, Fraction(25), 0, True)


def test_no_fps_reports_samples_only():
    with pytest.raises(ValueError): format_timecode(48000, None, 0, False)
