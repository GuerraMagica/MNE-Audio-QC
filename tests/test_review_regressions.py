from copy import deepcopy
import hashlib
import json
from unittest.mock import patch

from conftest import make_case
from mne_qc.audio import iter_blocks as real_iter_blocks
from mne_qc.compare import compare_me
from mne_qc.domain import Annotation, CANDIDATE, Detection, Finding, RunResult, Span, StageResult, StopToken
from mne_qc.evaluation import evaluate_runs
from mne_qc.manifest import load_manifest
from mne_qc.runner import run_analysis
from mne_qc.speech import merge_detections


def eligible_example(short_runs=False):
    duration=1800*48000
    corpus={"entries":[],"frozen_protocol":{
        "implementation_sha256":"impl","analysis_config_sha256":["cfg"],
        "evaluation_config_sha256":hashlib.sha256(json.dumps({"matching_slack_ms":250},sort_keys=True).encode()).hexdigest(),
        "backend_sha256":["binary","model"],"manifest_sha256":{}},
        "technical_controls_passed":True,"unaided_review_minutes":60,
        "assisted_review_minutes":{"language|2.0|complete|fixed":20}}
    corpus["entries"].append({"project_id":"cal","source_id":"work-cal","layout":"2.0","partition":"calibration",
        "span":[0,duration],"source_span":[0,duration],"original_fragment_id":"cal-original"})
    runs=[]; annotations=[]
    for pid in ("p","q"):
        end=48000 if short_runs else duration
        corpus["entries"].append({"project_id":pid,"source_id":"work-"+pid,"layout":"2.0","partition":"evaluation",
            "span":[0,duration],"source_span":[0,duration],"original_fragment_id":pid+"-original",
            "annotations_complete":True,"source_hashes":{pid+".wav":"audio"}})
        corpus["frozen_protocol"]["manifest_sha256"][pid+"|2.0"]="manifest-"+pid
        findings=[]
        for i in range(30):
            span=Span(i*1000+100,i*1000+500)
            annotations.append(Annotation(pid+str(i),pid,"2.0","language",span,("L",)))
            findings.append(Finding(pid+str(i),"language",CANDIDATE,span,("L",),{"text_hypothesis":str(i)}))
        stage=StageResult("speech_fixed","complete",[Span(0,end)],findings,
            {"mode":"fixed","backend_identity_verified":True,"backend_identity":{"binary":{"status":"complete","sha256":"binary"},
             "model":{"status":"complete","sha256":"model"}}})
        runs.append(RunResult("complete",[stage],{pid+".wav":{"sha256":"audio","status":"complete"}},
            {"project_id":pid,"layout":"2.0","mode":"complete","program_span":[0,end],
             "source_identity_verified":True,"implementation_sha256":"impl","analysis_config_sha256":"cfg",
             "manifest_sha256":"manifest-"+pid}))
    return runs,annotations,corpus


def test_positive_protocol_control_can_continue_limited_validation():
    runs,gold,corpus=eligible_example()
    assert set(evaluate_runs(runs,gold,corpus,{}).gates.values())=={"continue_limited_validation"}


def test_declared_hour_cannot_be_proven_by_two_seconds():
    runs,gold,corpus=eligible_example(short_runs=True)
    r=evaluate_runs(runs,gold,corpus,{})
    assert "corpus_run_coverage_mismatch" in r.eligibility["reasons"]
    assert set(r.gates.values())=={"insufficient_evidence"}


def test_variants_of_original_fragment_do_not_duplicate_minutes():
    runs,gold,corpus=eligible_example()
    # Same source material, different project ids: elapsed programme is still 30 min.
    corpus["entries"][2].update(source_id="work-p",source_span=[0,1800*48000],original_fragment_id="p-original")
    r=evaluate_runs(runs,gold,corpus,{})
    assert r.eligibility["duration_seconds"]["evaluation"]==1800


def test_matching_decisions_are_frozen_before_holdout():
    runs,gold,corpus=eligible_example()
    r=evaluate_runs(runs,gold,corpus,{"matching_slack_ms":60000})
    assert "evaluation_protocol_mismatch" in r.eligibility["reasons"]


def test_actual_model_bytes_and_manifest_decisions_are_frozen():
    runs,gold,corpus=eligible_example()
    runs[0].stages[0].measurements["backend_identity"]["model"]["sha256"]="different-bytes"
    r=evaluate_runs(runs,gold,corpus,{})
    assert "backend_identity_mismatch" in next(iter(r.eligibility["by_bucket"].values()))["reasons"]
    runs,gold,corpus=eligible_example()
    runs[0].environment["manifest_sha256"]="changed-channel-map-or-exception"
    assert "manifest_identity_mismatch" in evaluate_runs(runs,gold,corpus,{}).eligibility["reasons"]


def test_annotations_must_be_inside_the_evaluated_corpus():
    runs,gold,corpus=eligible_example()
    gold[0].span=Span(1801*48000,1802*48000)
    assert "annotation_outside_corpus" in evaluate_runs(runs,gold,corpus,{}).eligibility["reasons"]


def test_read_failure_preserves_completed_dsp_blocks(tmp_path):
    p,_,_=make_case(tmp_path/"inputs",frames=96000)
    def broken(asset,span,block_frames=65536):
        for i,b in enumerate(real_iter_blocks(asset,span,block_frames)):
            if i==1: raise ValueError("injected read failure after first block")
            yield b
    with patch("mne_qc.reconstruction.iter_blocks",side_effect=broken):
        r=run_analysis(load_manifest(p),tmp_path/"run",None,StopToken())
    for stage in (s for s in r.stages if s.stage in {"reconstruction","me_residual"}):
        assert stage.status=="partial"
        assert stage.coverage==[Span(0,24000)]
        assert stage.measurements["windows"]
        assert stage.measurements["peak_abs_by_channel"]==[0.0,0.0]


def test_comparison_read_failure_keeps_first_window(tmp_path):
    p,_,_=make_case(tmp_path/"inputs",frames=96000)
    def broken(asset,span,block_frames=65536):
        if span.start>=12000: raise ValueError("injected later window failure")
        yield from real_iter_blocks(asset,span,block_frames)
    with patch("mne_qc.compare.iter_blocks",side_effect=broken):
        r=compare_me(load_manifest(p),{},StopToken())
    assert r.status=="partial" and r.coverage==[Span(0,24000)]
    assert r.measurements["windows"] and r.findings


def test_omission_is_not_absorbed_by_dx_activity_queue():
    findings=[Finding("dx"+str(i),"dx_content_review",CANDIDATE,Span(i*12000,i*12000+24000),("L",),{"reason":"DX active"}) for i in range(10)]
    findings.append(Finding("loss","possible_omission",CANDIDATE,Span(60000,62000),("L",),{"reason":"localized loss"}))
    stage=StageResult("comparison","complete",[Span(0,144000)],findings,{"mode":"complete"})
    run=RunResult("complete",[stage],environment={"project_id":"p","layout":"2.0","mode":"complete","program_span":[0,144000]})
    gold=[Annotation("gold","p","2.0","omission",Span(60000,62000),("L",))]
    value=next(iter(evaluate_runs([run],gold,{},{}).metrics.values()))
    assert value["tp"]==1 and value["fp"]==1


def test_changed_asr_text_is_one_event_with_both_hypotheses():
    a=Detection(Span(0,10000),("L",),"hola",{"window":0})
    b=Detection(Span(100,10100),("R",),"ola",{"window":1})
    merged=merge_detections([a,b])
    assert len(merged)==1
    assert set(merged[0].evidence["text_hypotheses"])=={"hola","ola"}
    stage=StageResult("speech_fixed","complete",[Span(0,48000)],[
        Finding("a","language",CANDIDATE,a.span,a.channels,{"text_hypothesis":"hola"}),
        Finding("b","language",CANDIDATE,b.span,b.channels,{"text_hypothesis":"ola"})])
    run=RunResult("complete",[stage],environment={"project_id":"p","layout":"2.0","program_span":[0,48000]})
    value=next(iter(evaluate_runs([run],[Annotation("gold","p","2.0","language",Span(0,10000),("L",))],{},{}).metrics.values()))
    assert value["tp"]==1 and value["fp"]==0


def test_model_mutation_during_inference_invalidates_stage(tmp_path):
    from pathlib import Path
    import sys
    from mne_qc.domain import SpeechConfig
    from mne_qc.speech import scan_speech
    p,_,_=make_case(tmp_path/"inputs")
    model=tmp_path/"fake-model"; model.write_bytes(b"before")
    cfg=SpeechConfig(Path(sys.executable),model)
    def backend(*args):
        model.write_bytes(b"changed during inference")
        return [{"start_ms":10,"end_ms":100,"text":"hola"}]
    with patch("mne_qc.speech._transcribe_window",side_effect=backend):
        result=scan_speech(load_manifest(p),cfg,StopToken())
    assert result.status=="partial"
    assert result.measurements["backend_identity_verified"] is False
