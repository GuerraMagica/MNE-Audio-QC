"""Event evaluation: arithmetic metrics are separate from evidence eligibility."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math

from .domain import Annotation, EvaluationResult, Finding, RunResult, Span, StageResult

EVALUATION_DEFAULTS={"matching_slack_ms":250}


def _union_length(spans):
    total, end = 0, None
    for a, b in sorted(spans):
        if end is None or a > end:
            total += b-a
        elif b > end:
            total += b-end
        end = max(end if end is not None else b, b)
    return total


def _covered(ranges, expected):
    return expected[1]>expected[0] and _union_length([
        (max(a,expected[0]),min(b,expected[1])) for a,b in ranges
        if min(b,expected[1])>max(a,expected[0])])==expected[1]-expected[0]


def _span_valid(span):
    return isinstance(span,(list,tuple)) and len(span)==2 and all(type(v) is int for v in span) and span[1]>span[0]


def _wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k/n
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    margin = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0,center-margin),min(1,center+margin)]


def _deduplicate(predictions):
    merged = []
    for pid, f in sorted(predictions, key=lambda x:(x[0], x[1].span.start, x[1].id)):
        for old_pid, old in reversed(merged):
            overlap = min(old.span.end,f.span.end)-max(old.span.start,f.span.start)
            shorter = min(old.span.end-old.span.start,f.span.end-f.span.start)
            if pid == old_pid and old.kind==f.kind and overlap >= shorter*0.5:
                old.span = Span(min(old.span.start,f.span.start),max(old.span.end,f.span.end))
                old.channels = tuple(dict.fromkeys(old.channels+f.channels))
                old.evidence["contributors"].append({"id":f.id,"span":[f.span.start,f.span.end],
                                                      "channels":list(f.channels),"evidence":deepcopy(f.evidence)})
                old.evidence["policy_exempt"]=bool(old.evidence.get("policy_exempt")) and bool(f.evidence.get("policy_exempt"))
                break
        else:
            # Evaluation must never mutate original detector output.
            evidence=deepcopy(f.evidence)
            evidence["contributors"]=[{"id":f.id,"span":[f.span.start,f.span.end],
                                       "channels":list(f.channels),"evidence":deepcopy(f.evidence)}]
            merged.append((pid,Finding(f.id,f.kind,f.state,Span(f.span.start,f.span.end),f.channels,evidence)))
    return merged


def _match(predictions, annotations, slack):
    choices = []
    for pi, (pid,p) in enumerate(predictions):
        for ai,a in enumerate(annotations):
            if pid != a.project_id:
                continue
            intersection = max(0,min(p.span.end,a.span.end+slack)-max(p.span.start,a.span.start-slack))
            if intersection >= (p.span.end-p.span.start)*0.5:
                choices.append((-intersection,a.id,p.id,pi,ai))
    matches, usedp, useda = [], set(), set()
    for _,_,_,pi,ai in sorted(choices):
        if pi not in usedp and ai not in useda:
            usedp.add(pi)
            useda.add(ai)
            matches.append((pi,ai))
    return matches,usedp,useda


def evaluate_runs(runs: list[RunResult], annotations: list[Annotation], corpus: dict, config: dict) -> EvaluationResult:
    if set(config)-set(EVALUATION_DEFAULTS):
        raise ValueError("Configuración de evaluación desconocida")
    config={**EVALUATION_DEFAULTS,**config}
    evaluation_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    entries = corpus.get("entries", [])
    evaluation_ids = {e["project_id"] for e in entries if e.get("partition") == "evaluation"}
    calibration_sources = {e["source_id"] for e in entries if e.get("partition") == "calibration"}
    evaluation_sources = {e["source_id"] for e in entries if e.get("partition") == "evaluation"}
    reasons = []
    for e in entries:
        if not _span_valid(e.get("span")):
            raise ValueError("Corpus: rango de programa inválido")
        if not e.get("original_fragment_id") or not _span_valid(e.get("source_span")):
            reasons.append("missing_original_fragment_identity")
        elif e["source_span"][1]-e["source_span"][0] != e["span"][1]-e["span"][0]:
            reasons.append("original_fragment_duration_mismatch")
    if calibration_sources & evaluation_sources:
        reasons.append("source_leakage")
    if len(calibration_sources | evaluation_sources) < 3:
        reasons.append("fewer_than_three_sources")
    durations = {}
    for partition in ("calibration","evaluation"):
        grouped = defaultdict(list)
        for e in entries:
            if e.get("partition") == partition:
                # Source coordinates identify original programme across project ids,
                # layouts and derived variants. Missing coordinates remain ineligible.
                grouped[e["source_id"]].append(e.get("source_span",e["span"]))
        durations[partition] = sum(_union_length(s) for s in grouped.values())/48000
    if durations["calibration"] < 1800 or durations["evaluation"] < 3600:
        reasons.append("insufficient_programme_duration")
    if not entries or not all(e.get("annotations_complete") is True for e in entries if e.get("partition") == "evaluation"):
        reasons.append("incomplete_annotations")
    protocol=corpus.get("frozen_protocol")
    if not isinstance(protocol,dict) or not protocol:
        reasons.append("protocol_not_frozen")
    elif any(r.environment.get("implementation_sha256") != protocol.get("implementation_sha256") or
             r.environment.get("analysis_config_sha256") not in protocol.get("analysis_config_sha256",[])
             for r in runs):
        reasons.append("protocol_identity_mismatch")
    if isinstance(protocol,dict):
        if protocol.get("evaluation_config_sha256") != evaluation_hash:
            reasons.append("evaluation_protocol_mismatch")
        for r in runs:
            key=f'{r.environment.get("project_id")}|{r.environment.get("layout")}'
            if not r.environment.get("manifest_sha256") or r.environment["manifest_sha256"] != protocol.get("manifest_sha256",{}).get(key):
                reasons.append("manifest_identity_mismatch")
    if any(r.environment.get("synthetic") for r in runs):
        reasons.append("synthetic_input")
    if not corpus.get("technical_controls_passed"):
        reasons.append("technical_controls_not_documented")
    if not runs or any(r.status != "complete" for r in runs):
        reasons.append("incomplete_analysis")
    if not runs or any(not r.environment.get("source_identity_verified") for r in runs):
        reasons.append("unverified_sources")
    if len({a.id for a in annotations}) != len(annotations):
        reasons.append("duplicate_annotation_ids")
    for e in entries:
        if e.get("partition") != "evaluation":
            continue
        matching = [r for r in runs if r.environment.get("project_id") == e["project_id"] and r.environment.get("layout") == e["layout"]]
        if not matching:
            reasons.append("missing_evaluation_run")
            continue
        expected_hashes = e.get("source_hashes")
        if not expected_hashes or any({k:v.get("sha256") for k,v in r.input_identity.items()} != expected_hashes for r in matching):
            reasons.append("annotation_source_identity_unverified")
        if any(not _covered([r.environment.get("program_span",[0,0])],e["span"]) for r in matching):
            reasons.append("corpus_run_coverage_mismatch")
        annotated_ranges=[entry["span"] for entry in entries if entry.get("partition")=="evaluation" and
                          entry["project_id"]==e["project_id"] and entry["layout"]==e["layout"]]
        if any(not _covered(annotated_ranges,r.environment.get("program_span",[0,0])) for r in matching):
            reasons.append("corpus_run_coverage_mismatch")
    if entries:
        for annotation in annotations:
            ranges=[e["span"] for e in entries if e.get("partition")=="evaluation" and
                    e["project_id"]==annotation.project_id and e["layout"]==annotation.layout]
            if not _covered(ranges,(annotation.span.start,annotation.span.end)):
                reasons.append("annotation_outside_corpus")
    buckets = {}
    for run in runs:
        env = run.environment
        pid,layout = env.get("project_id"),env.get("layout")
        if not pid or layout not in {"2.0","5.1"}:
            reasons.append("missing_run_context")
            continue
        if entries and pid not in evaluation_ids:
            continue
        mode = env.get("mode", next((s.measurements.get("mode") for s in run.stages if s.stage == "comparison"), "complete"))
        function_stages = [("language",s) for s in run.stages if s.stage.startswith("speech_")]
        function_stages += [("omission",s) for s in run.stages if s.stage == "comparison"]
        # A function with annotations but no executed stage still has FNs, not silence.
        for kind in {a.kind for a in annotations if a.project_id == pid and a.layout == layout}:
            if not any(k == kind for k,_ in function_stages):
                function_stages.append((kind,StageResult("missing", "not_run")))
        for function,stage in function_stages:
            variant = stage.measurements.get("mode",stage.stage.removeprefix("speech_")) if function == "language" else "dsp"
            key = f"{function}|{layout}|{mode}|{variant}"
            bucket = buckets.setdefault(key,{"function":function,"layout":layout,"mode":mode,"variant":variant,
                "projects":set(),"spans":defaultdict(list),"predictions":[],"stages":[],"stage_context":[]})
            bucket["projects"].add(pid)
            bucket["spans"][pid].append(env.get("program_span", [0,0]))
            bucket["stages"].append(stage)
            bucket["stage_context"].append((stage,env.get("program_span",[0,0])))
            accepted_kinds = {"language"} if function == "language" else {"possible_omission","dx_content_review"}
            bucket["predictions"].extend((pid,f) for f in stage.findings if f.kind in accepted_kinds and f.span)
    metrics,gates,per_bucket = {},{},{ }
    slack = round(config.get("matching_slack_ms",250)*48)
    if slack < 0:
        raise ValueError("Holgura temporal negativa")
    under_ids = set(corpus.get("under_dialogue_ids",[]))
    for key,b in buckets.items():
        gold_all = [a for a in annotations if a.project_id in b["projects"] and a.layout == b["layout"] and a.kind == b["function"]]
        gold = [a for a in gold_all if a.label == "positive"]
        ambiguous = [a for a in gold_all if a.label == "ambiguous"]
        pred = _deduplicate(b["predictions"])
        matches,usedp,useda = _match(pred,gold,slack)
        ambiguous_predictions = set()
        for pi,p in enumerate(pred):
            if pi not in usedp and _match([p],ambiguous,slack)[0]:
                ambiguous_predictions.add(pi)
        tp,fp,fn = len(matches),len(pred)-len(usedp)-len(ambiguous_predictions),len(gold)-len(useda)
        hours = sum(_union_length(v) for v in b["spans"].values())/48000/3600
        error_start = [abs(pred[pi][1].span.start-gold[ai].span.start)/48 for pi,ai in matches]
        error_end = [abs(pred[pi][1].span.end-gold[ai].span.end)/48 for pi,ai in matches]
        def subgroup(ids):
            indices = {i for i,a in enumerate(gold) if a.id in ids}
            n = len(indices)
            matched = len(indices & useda)
            return {"tp":matched,"fn":n-matched,"recall":matched/n if n else None,"n":n}
        policy_fp = fp + sum(gold[ai].policy_issue is False and not pred[pi][1].evidence.get("policy_exempt",False) for pi,ai in matches)
        value = {"tp":tp,"fp":fp,"fn":fn,"precision":tp/(tp+fp) if tp+fp else None,
            "recall":tp/(tp+fn) if tp+fn else None,"recall_wilson_95":_wilson(tp,tp+fn),
            "precision_wilson_95":_wilson(tp,tp+fp),"hours":hours,"false_alarms_per_hour":fp/hours if hours else None,
            "policy_false_alerts":policy_fp,"policy_false_alerts_per_hour":policy_fp/hours if hours else None,
            "ambiguous_annotations":len(ambiguous),"ambiguous_predictions":len(ambiguous_predictions),
            "onset_error_ms":error_start,"end_error_ms":error_end,"under_dialogue":subgroup(under_ids),
            "injected":subgroup({a.id for a in gold if a.injected}),
            "natural":subgroup({a.id for a in gold if not a.injected}),
            "matching_trace":[{"prediction_id":pred[pi][1].id,"annotation_id":gold[ai].id,
                "contributors":pred[pi][1].evidence.get("contributors",[])} for pi,ai in matches]}
        metrics[key] = value
        own = list(dict.fromkeys(reasons))
        if len(gold) < 30:
            own.append("fewer_than_30_events")
        if any(s.status != "complete" or expected[1] <= expected[0] or
               _union_length([(max(v.start,expected[0]),min(v.end,expected[1])) for v in s.coverage
                              if min(v.end,expected[1]) > max(v.start,expected[0])]) != expected[1]-expected[0]
               for s,expected in b["stage_context"]):
            own.append("function_coverage_incomplete")
        if b["function"]=="language":
            allowed=set(protocol.get("backend_sha256",[])) if isinstance(protocol,dict) else set()
            for stage in b["stages"]:
                identity=stage.measurements.get("backend_identity",{})
                required=3 if stage.stage=="speech_vad" else 2
                if stage.measurements.get("backend_identity_verified") is not True or len(identity)<required or any(v.get("status")!="complete" or not v.get("sha256") or
                    v["sha256"] not in allowed for v in identity.values()):
                    own.append("backend_identity_mismatch")
                    break
        if b["function"] == "omission" and b["mode"] != "complete":
            own.append("reduced_omissions_exploratory")
        if b["function"] == "omission" and not value["under_dialogue"]["n"]:
            own.append("missing_under_dialogue_examples")
        review_minutes = corpus.get("assisted_review_minutes",{}).get(key)
        baseline_minutes = corpus.get("unaided_review_minutes")
        if review_minutes is None or baseline_minutes is None:
            own.append("review_benefit_not_measured")
        if own:
            gates[key] = "insufficient_evidence"
        else:
            target = 0.95 if b["function"] == "language" else 0.90
            passed = value["recall"] is not None and value["recall"] >= target and value["false_alarms_per_hour"] <= 6
            if b["function"] == "omission":
                passed = passed and value["under_dialogue"]["recall"] >= 0.90
            gates[key] = "continue_limited_validation" if passed and review_minutes < baseline_minutes else "targets_not_met"
        per_bucket[key] = {"eligible": not own,"reasons":own}
    return EvaluationResult(metrics,{"eligible": bool(buckets) and not reasons and all(v["eligible"] for v in per_bucket.values()),
        "reasons":list(dict.fromkeys(reasons)),"by_bucket":per_bucket,"duration_seconds":durations,
        "protocol_config_sha256":evaluation_hash,
        "scope":"exploratory_continuation_only_not_delivery_approval"},gates)


def load_run(raw: dict) -> RunResult:
    r = raw["result"]
    stages = []
    for s in r["stages"]:
        findings = [Finding(f["id"],f["kind"],f["state"],Span(**f["span"]) if f.get("span") else None,
                            tuple(f["channels"]),f.get("evidence",{})) for f in s["findings"]]
        stages.append(StageResult(s["stage"],s["status"],[Span(**v) for v in s["coverage"]],findings,
                                  s.get("measurements",{}),s.get("reason")))
    return RunResult(r["status"],stages,r["input_identity"],r["environment"])
