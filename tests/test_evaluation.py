from mne_qc.domain import Annotation, Finding, RunResult, Span, StageResult, CANDIDATE
from mne_qc.evaluation import evaluate_runs


def run(findings, status="complete", layout="2.0", mode="complete", variant="fixed"):
    stage = StageResult(f"speech_{variant}", status, [Span(0,48000)], findings, {"mode": variant})
    return RunResult(status, [stage], environment={"project_id": "p", "layout": layout,
        "mode": mode, "program_span": [0,48000], "source_identity_verified": True})


def a(id="a", span=Span(1000,3000), label="positive", **kwargs):
    return Annotation(id, "p", "2.0", "language", span, ("L",), label=label, **kwargs)


def f(id="f", span=Span(1000,3000), channels=("L",)):
    return Finding(id, "language", CANDIDATE, span, channels, {"text_hypothesis": "hola"})


def metrics(result): return next(iter(result.metrics.values()))


def test_duplicate_channels_count_once():
    r = evaluate_runs([run([f(),f("other",channels=("R",))])], [a()], {}, {})
    assert metrics(r)["tp"] == 1 and metrics(r)["fp"] == 0


def test_one_to_one_matching():
    r = evaluate_runs([run([f()])], [a(),a("second",Span(32000,34000))], {}, {})
    assert metrics(r)["tp"] == 1 and metrics(r)["fn"] == 1


def test_no_positive_labels_is_not_100_recall():
    r = evaluate_runs([run([])], [], {}, {})
    assert metrics(r)["recall"] is None
    assert set(r.gates.values()) == {"insufficient_evidence"}


def test_shared_title_leaks_across_split():
    corpus = {"entries": [
        {"project_id":"p", "source_id":"same", "layout":"2.0", "partition":"calibration", "span":[0,48000]},
        {"project_id":"p", "source_id":"same", "layout":"2.0", "partition":"evaluation", "span":[0,48000]}]}
    r = evaluate_runs([run([f()])], [a()], corpus, {})
    assert "source_leakage" in r.eligibility["reasons"]


def test_incomplete_coverage_ineligible():
    r = evaluate_runs([run([f()],status="partial")], [a()], {}, {})
    assert all(g == "insufficient_evidence" for g in r.gates.values())
    assert "incomplete_analysis" in r.eligibility["reasons"]


def test_each_format_and_mode_has_own_gate():
    r = evaluate_runs([run([f()]),run([],layout="5.1",mode="reduced")], [a()], {}, {})
    assert len(r.metrics) == 2
    assert all(g == "insufficient_evidence" for g in r.gates.values())


def test_ambiguous_and_allowed_lyrics_not_negatives():
    r = evaluate_runs([run([f()])], [a(policy_issue=False),a("amb",Span(10000,12000),label="ambiguous")], {}, {})
    assert metrics(r)["tp"] == 1
    assert metrics(r)["ambiguous_annotations"] == 1
    assert metrics(r)["policy_false_alerts"] == 1


def test_under_dialogue_subgroup_cannot_be_hidden():
    omission = Annotation("o","p","2.0","omission",Span(32000,33000),("L",), injected=True)
    r = evaluate_runs([run([])], [omission], {"under_dialogue_ids":["o"]}, {})
    assert any(v["under_dialogue"]["fn"] == 1 for v in r.metrics.values())


def test_long_alert_cannot_claim_short_event():
    r = evaluate_runs([run([f(span=Span(0,48000))])], [a(span=Span(10000,11000))], {}, {})
    assert metrics(r)["tp"] == 0


def test_variants_do_not_inflate_programme_hours():
    r = evaluate_runs([run([f()]),run([f()],variant="vad")], [a()], {}, {})
    assert all(v["hours"] == 1/3600 for v in r.metrics.values())


def test_nominal_complete_cannot_hide_short_coverage():
    r = run([f()])
    r.stages[0].coverage = [Span(0,100)]
    out = evaluate_runs([r],[a()],{}, {})
    assert "function_coverage_incomplete" in next(iter(out.eligibility["by_bucket"].values()))["reasons"]


def test_global_eligibility_respects_bucket_evidence():
    corpus = {"entries":[
        {"project_id":"c1","source_id":"s1","layout":"2.0","partition":"calibration","span":[0,1800*48000]},
        {"project_id":"p","source_id":"s2","layout":"2.0","partition":"evaluation","span":[0,1800*48000],"annotations_complete":True,"source_hashes":{"x":"hash"}},
        {"project_id":"p2","source_id":"s3","layout":"2.0","partition":"evaluation","span":[0,1800*48000],"annotations_complete":True,"source_hashes":{"x":"hash"}}],
        "frozen_protocol":{"implementation_sha256":"impl","analysis_config_sha256":["cfg"]},"technical_controls_passed":True}
    r = run([f()]); r.input_identity={"x":{"sha256":"hash"}}
    r.environment.update(implementation_sha256="impl",analysis_config_sha256="cfg")
    from copy import deepcopy
    r2=deepcopy(r); r2.environment["project_id"]="p2"
    out=evaluate_runs([r,r2],[a()],corpus,{})
    assert out.eligibility["eligible"] is False


def test_frozen_protocol_must_match_actual_implementation():
    r=run([f()]); r.environment.update(implementation_sha256="actual",analysis_config_sha256="actual-config")
    corpus={"frozen_protocol":{"implementation_sha256":"different","analysis_config_sha256":["other"]}}
    out=evaluate_runs([r],[a()],corpus,{})
    assert "protocol_identity_mismatch" in out.eligibility["reasons"]


def test_synthetic_run_cannot_qualify_as_real_evidence():
    r=run([f()]); r.environment["synthetic"]=True
    assert "synthetic_input" in evaluate_runs([r],[a()],{},{}).eligibility["reasons"]
