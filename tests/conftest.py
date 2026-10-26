import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

LABELS = {"2.0": ["L", "R"], "5.1": ["L", "R", "C", "LFE", "Ls", "Rs"]}


def make_case(root: Path, layout="2.0", frames=24000, discrete=False):
    root.mkdir(parents=True, exist_ok=True)
    labels = LABELS[layout]
    t = np.arange(frames)
    # Quantized, exactly representable PCM24 with safe headroom.
    mx = np.tile((np.round(80 * np.sin(t * 0.06)) / 32768)[:, None], (1, len(labels)))
    fx = np.zeros_like(mx)
    fx[frames // 3: frames // 3 + 120] = 0.0625
    dx = np.tile((np.round(160 * np.sin(t * 0.031)) / 32768)[:, None], (1, len(labels)))
    arrays = {"MX": mx, "FX": fx, "DX": dx, "ME": mx + fx, "PM": mx + fx + dx}
    assets = {}
    for role, data in arrays.items():
        channels = []
        if discrete:
            for i, label in enumerate(labels):
                name = f"{role}_{label}.wav"
                sf.write(root / name, data[:, i], 48000, subtype="PCM_24")
                channels.append({"path": name, "file_channel": 0, "label": label})
        else:
            name = f"{role}.wav"
            sf.write(root / name, data, 48000, subtype="PCM_24")
            channels = [{"path": name, "file_channel": i, "label": label} for i, label in enumerate(labels)]
        assets[role] = {"version": "v1", "start_sample": 0, "channels": channels}
    manifest = {"schema_version": 1, "project_id": "synthetic", "layout": layout,
                "assets": assets, "program_span": [0, frames], "exceptions": []}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, arrays, manifest


@pytest.fixture(params=["2.0", "5.1"])
def case(tmp_path, request):
    return make_case(tmp_path, request.param)
