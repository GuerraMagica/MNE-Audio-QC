"""Opt-in real local ASR integration over an explicitly supplied synthetic phrase.

Does not synthesize/download anything or inspect production folders. Generates
known contamination in Rs (5.1) and a phrase crossing the 30 s boundary (2.0).
"""
import argparse
import json
from pathlib import Path
import sys
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

import numpy as np
import soundfile as sf

from mne_qc.audio import fingerprint
from mne_qc.domain import LAYOUTS, SpeechConfig, StopToken
from mne_qc.evaluation import evaluate_runs
from mne_qc.manifest import load_manifest
from mne_qc.report import write_json
from mne_qc.runner import run_analysis


def _fixture(root,phrase,layout,position):
    root.mkdir(parents=True,exist_ok=False)
    labels=LAYOUTS[layout]
    start=round(position*48000)
    end=start+len(phrase)
    frames=end+48000
    dx=np.zeros((frames,len(labels)))
    me=np.zeros_like(dx)
    dx_channel=labels.index("C") if layout=="5.1" else 0
    target="Rs" if layout=="5.1" else "L"
    dx[start:end,dx_channel]=phrase*0.5
    me[start:end,labels.index(target)]=phrase*0.5
    arrays={"PM":dx,"DX":dx,"ME":me,"MX":np.zeros_like(dx),"FX":np.zeros_like(dx)}
    assets={}
    for role,data in arrays.items():
        sf.write(root/f"{role}.wav",data,48000,subtype="PCM_24")
        assets[role]={"version":"synthetic-asr-v1","start_sample":0,"channels":[
            {"path":f"{role}.wav","file_channel":i,"label":label} for i,label in enumerate(labels)]}
    path=root/"manifest.json"
    path.write_text(json.dumps({"schema_version":1,"synthetic":True,"project_id":"asr-smoke-"+layout,
        "layout":layout,"program_span":[0,frames],"fps":"24","start_frame":0,
        "exceptions":[],"assets":assets},indent=2))
    return load_manifest(path),target,start,end


def _plain(text):
    return "".join(c for c in unicodedata.normalize("NFKD",text.casefold()) if not unicodedata.combining(c))


def smoke(phrase_path,output,binary,model,vad_model,cpu):
    info=sf.info(phrase_path)
    if info.samplerate!=48000 or info.channels!=1 or not 1<=info.duration<=20:
        raise ValueError("La frase sintética debe ser mono/48kHz y durar entre 1 y 20 segundos")
    phrase,_=sf.read(phrase_path,dtype="float64")
    output.mkdir(parents=True,exist_ok=False)
    cfg=SpeechConfig(binary.resolve(),model.resolve(),vad_model.resolve(),use_gpu=not cpu)
    summaries=[]; runs=[]
    for layout,position in (("5.1",2),("2.0",27)):
        manifest,target,start,end=_fixture(output/(layout+"-inputs"),phrase,layout,position)
        print(f"Iniciando {layout}: voz conocida en {target}, {start/48000:.3f}–{end/48000:.3f} s; fixed + VAD",flush=True)
        result=run_analysis(manifest,output/layout,cfg,StopToken(),both_speech_modes=True,previews=True)
        assert result.status=="complete",[(s.stage,s.status,s.reason) for s in result.stages]
        assert result.environment["source_identity_verified"]
        stages=[]
        for stage in result.stages:
            if not stage.stage.startswith("speech_"):
                continue
            assert stage.status=="complete" and stage.measurements["backend_identity_verified"]
            relevant=[f for f in stage.findings if target in f.channels]
            assert any("prueba" in _plain(f.evidence["text_hypothesis"]) for f in relevant),(layout,stage.stage,"phrase missing")
            if layout=="2.0":
                endings=[f for f in relevant if "escena" in _plain(f.evidence["text_hypothesis"])]
                assert endings and any(abs(f.span.end-end)<=48000 for f in endings),(stage.stage,"window mapping error",[(f.span,f.evidence["text_hypothesis"]) for f in relevant])
            unexpected=sorted({c for f in stage.findings for c in f.channels if c!=target})
            record={"variant":stage.stage,"findings":len(stage.findings),"unexpected_language_channels":unexpected,
                "non_speech_annotations":len(stage.measurements["non_speech_annotations"]),
                "coverage":stage.coverage,"elapsed_seconds":stage.measurements["elapsed_seconds"],
                "texts":[f.evidence["text_hypothesis"] for f in relevant],
                "hypotheses":[{"span":[f.span.start,f.span.end],
                    "texts":f.evidence.get("text_hypotheses",[f.evidence["text_hypothesis"]])} for f in relevant],
                "ranges_outside_known_voice":[[f.span.start,f.span.end] for f in relevant
                    if f.span.start<start-48000 or f.span.end>end+48000]}
            stages.append(record)
            print(f"{layout} {stage.stage}: {len(stage.findings)} hipótesis; otros canales={unexpected}",flush=True)
        summaries.append({"layout":layout,"expected_channel":target,"voice_span":[start,end],
            "programme_seconds":manifest.program_span[1]/48000,"wall_seconds":result.environment["elapsed_seconds"],
            "memory_observation":result.environment["memory_observation"],"stages":stages})
        runs.append(result)
    evaluation=evaluate_runs(runs,[],{}, {})
    assert all(g=="insufficient_evidence" for g in evaluation.gates.values())
    receipt={"schema_version":1,"status":"controlled_integration_complete","use_gpu_requested":not cpu,
        "phrase_identity":fingerprint(phrase_path,StopToken()),"runs":summaries,
        "efficacy_evaluation":evaluation,
        "limitations":["One local synthetic voice; not film recall or intelligibility validation",
                       "No real production audio, long-program benchmark or Pro Tools listening"]}
    write_json(output/"validation.json",receipt)
    print("Integración ASR real completada; eficacia cinematográfica: evidencia insuficiente",flush=True)
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phrase",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--binary",type=Path,default=ROOT/"vendor/whisper.cpp/build/bin/whisper-cli")
    p.add_argument("--model",type=Path,default=ROOT/"models/ggml-small.bin")
    p.add_argument("--vad-model",type=Path,default=ROOT/"models/ggml-silero-v6.2.0.bin")
    p.add_argument("--cpu",action="store_true")
    a=p.parse_args()
    smoke(a.phrase.resolve(),a.output,a.binary,a.model,a.vad_model,a.cpu)
