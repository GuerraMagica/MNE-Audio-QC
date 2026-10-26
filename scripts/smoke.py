"""Real DSP/FFmpeg smoke over generated files. No model or production audio."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

from scripts.make_fixtures import make_fixtures
from mne_qc.domain import Annotation, Span, StopToken
from mne_qc.evaluation import evaluate_runs
from mne_qc.manifest import load_manifest
from mne_qc.report import write_json
from mne_qc.runner import run_analysis


def smoke(output: Path):
    output.mkdir(parents=True,exist_ok=False)
    runs=[]
    annotations=[]
    for layout in ("2.0","5.1"):
        for defect in (False,True):
            tag=layout+("-defect" if defect else "-exact")
            manifest=load_manifest(make_fixtures(output/(tag+"-inputs"),layout,defect))
            result=run_analysis(manifest,output/tag,None,StopToken(),previews=defect)
            assert result.status=="complete",(tag,result.status)
            reconstruction=next(s for s in result.stages if s.stage=="reconstruction")
            assert max(reconstruction.measurements["peak_abs_by_channel"])==0
            assert any(s.stage=="speech" and s.status=="not_run" for s in result.stages)
            assert result.environment["source_identity_verified"]
            if defect:
                comparison=next(s for s in result.stages if s.stage=="comparison")
                assert any(f.kind=="possible_omission" for f in comparison.findings)
                runs.append(result)
                annotations.append(Annotation(tag,manifest.project_id,layout,"omission",Span(24000,48000),("L",),injected=True))
            print(f"{tag}: DSP completo; originales verificados; IA no ejecutada")
    stop=StopToken(); stop.cancel()
    manifest=load_manifest(output/"2.0-exact-inputs"/"manifest.json")
    interrupted=run_analysis(manifest,output/"cancelled",None,stop)
    assert interrupted.status=="partial"
    evaluation=evaluate_runs(runs,annotations,{}, {})
    assert all(v=="insufficient_evidence" for v in evaluation.gates.values())
    write_json(output/"synthetic-evaluation.json",evaluation)
    print("Cancelación conservada; evaluación sintética insuficiente para validar eficacia real")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    smoke(p.parse_args().output)
