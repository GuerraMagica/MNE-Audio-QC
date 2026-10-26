import json

from mne_qc.cli import main
from scripts.make_fixtures import make_fixtures


def test_generator_and_real_cli(tmp_path):
    path = make_fixtures(tmp_path / "inputs", "2.0", defect=True)
    out = tmp_path / "run"
    assert main(["inspect", str(path)]) == 0
    assert main(["analyze", str(path), "--output", str(out), "--previews"]) == 0
    raw = json.loads((out/"run.json").read_text())
    assert raw["result"]["environment"]["source_identity_verified"] is True
    assert raw["result"]["environment"]["project_id"] == "synthetic-defect"
    assert any(s["findings"] for s in raw["result"]["stages"] if s["stage"] == "comparison")
    assert (out/"review.json").is_file() and list(out.glob("preview-*.wav"))


def test_cli_evaluate_is_explicitly_exploratory(tmp_path):
    path = make_fixtures(tmp_path / "inputs", "2.0")
    out = tmp_path / "run"
    assert main(["analyze",str(path),"--output",str(out)]) == 0
    annotation = tmp_path / "annotations.json"
    annotation.write_text(json.dumps({"schema_version":1,"annotations":[]}))
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"schema_version":1,"entries":[]}))
    target = tmp_path / "evaluation.json"
    assert main(["evaluate","--runs",str(out/"run.json"),"--annotations",str(annotation),
                 "--corpus",str(corpus),"--output",str(target)]) == 2
    result = json.loads(target.read_text())
    assert all(g == "insufficient_evidence" for g in result["gates"].values())


def test_cli_cpu_option_is_explicit_and_recorded(tmp_path):
    from unittest.mock import patch
    from mne_qc.domain import RunResult
    path=make_fixtures(tmp_path/"inputs")
    with patch("mne_qc.cli.run_analysis",return_value=RunResult("complete")) as run:
        assert main(["analyze",str(path),"--output",str(tmp_path/"out"),"--speech","fixed",
            "--binary",str(tmp_path/"binary"),"--model",str(tmp_path/"model"),"--no-gpu"])==0
    assert run.call_args.args[2].use_gpu is False
