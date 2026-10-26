"""Small deterministic signals, deliberately not a speech recognition benchmark."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import soundfile as sf
from mne_qc.domain import LAYOUTS


def make_fixtures(output: Path, layout="2.0", defect=False) -> Path:
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    labels = LAYOUTS[layout]
    t = np.arange(96000)
    mx = np.column_stack([np.round(500*np.sin(t*(0.05+0.005*i)))/32768 for i in range(len(labels))])
    dx = np.column_stack([np.round(300*np.sin(t*(0.035+0.003*i)))/32768 for i in range(len(labels))])
    fx = np.zeros_like(mx)
    fx[30000:30480] = 0.125
    arrays = {"DX":dx,"MX":mx,"FX":fx,"ME":mx+fx,"PM":dx+mx+fx}
    if defect:
        arrays["ME"] = arrays["ME"].copy()
        arrays["ME"][24000:48000] = 0
    assets = {}
    for role,a in arrays.items():
        sf.write(output/f"{role}.wav",a,48000,subtype="PCM_24")
        assets[role] = {"version":"synthetic-v1","start_sample":0,"channels":[
            {"path":f"{role}.wav","file_channel":i,"label":label} for i,label in enumerate(labels)]}
    manifest = {"schema_version":1,"synthetic":True,"project_id":"synthetic-defect" if defect else "synthetic-exact",
                "layout":layout,"program_span":[0,96000],"fps":"24","start_frame":0,"drop_frame":False,
                "exceptions":[],"assets":assets}
    path = output/"manifest.json"
    path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return path


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--layout",choices=LAYOUTS,default="2.0")
    p.add_argument("--defect",action="store_true")
    a=p.parse_args()
    print(make_fixtures(a.output,a.layout,a.defect))
