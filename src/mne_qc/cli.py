import argparse
import json
from pathlib import Path
import signal

from .domain import SpeechConfig, StopToken
from .manifest import load_manifest, manifest_identity
from .report import jsonable, write_json
from .runner import run_analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M&E QC experimental y local; no certifica entregas")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Validar manifiesto y cobertura")
    inspect.add_argument("manifest", type=Path)
    analyze = sub.add_parser("analyze", help="Analizar sin modificar originales")
    analyze.add_argument("manifest", type=Path)
    analyze.add_argument("--output", type=Path, required=True, help="Directorio nuevo")
    analyze.add_argument("--speech", choices=["off", "fixed", "vad", "both"], default="off")
    analyze.add_argument("--binary", type=Path)
    analyze.add_argument("--model", type=Path)
    analyze.add_argument("--vad-model", type=Path)
    analyze.add_argument("--language", default="es")
    analyze.add_argument("--no-gpu",action="store_true",help="Ejecutar ASR en CPU, de forma explícita")
    analyze.add_argument("--previews", action="store_true")
    analyze.add_argument("--dsp-config", type=Path)
    evaluate = sub.add_parser("evaluate",help="Métricas experimentales, sin aprobación de entrega")
    evaluate.add_argument("--runs",type=Path,nargs="+",required=True)
    evaluate.add_argument("--annotations",type=Path,required=True)
    evaluate.add_argument("--corpus",type=Path,required=True)
    evaluate.add_argument("--config",type=Path)
    evaluate.add_argument("--output",type=Path,required=True)
    args = parser.parse_args(argv)
    stop = StopToken()
    previous = signal.signal(signal.SIGINT, lambda *_: stop.cancel())
    try:
        if args.command == "evaluate":
            from .domain import Annotation, Span
            from .evaluation import evaluate_runs, load_run
            runs=[load_run(json.loads(p.read_text())) for p in args.runs]
            raw=json.loads(args.annotations.read_text())
            annotations=[Annotation(a["id"],a["project_id"],a["layout"],a["kind"],
                Span(*a["span"]),tuple(a["channels"]),a.get("label","positive"),
                a.get("injected",False),a.get("policy_issue")) for a in raw["annotations"]]
            result=evaluate_runs(runs,annotations,json.loads(args.corpus.read_text()),
                                 json.loads(args.config.read_text()) if args.config else {})
            if args.output.exists():
                raise FileExistsError("La evaluación de salida ya existe")
            args.output.parent.mkdir(parents=True,exist_ok=True)
            write_json(args.output,result)
            print(json.dumps(jsonable(result),ensure_ascii=False,indent=2))
            return 0 if result.gates and all(g=="continue_limited_validation" for g in result.gates.values()) else 2
        manifest = load_manifest(args.manifest)
        if args.command == "inspect":
            from .timeline import program_coverage
            print(json.dumps(jsonable({**program_coverage(manifest),"manifest_sha256":manifest_identity(manifest)}), ensure_ascii=False, indent=2))
            return 0
        speech = None
        if args.speech != "off":
            if args.binary is None or args.model is None:
                parser.error("--speech requiere --binary y --model locales; no hay descargas automáticas")
            speech = SpeechConfig(args.binary, args.model, args.vad_model, args.language,
                                  "fixed" if args.speech == "both" else args.speech,use_gpu=not args.no_gpu)
        cfg = json.loads(args.dsp_config.read_text()) if args.dsp_config else None
        result = run_analysis(manifest, args.output, speech, stop, previews=args.previews,
                              both_speech_modes=args.speech == "both", dsp_config=cfg)
        print(json.dumps({"status": result.status, "report": str(args.output.resolve()/"report.html"),
                          "meaning": "estado del cómputo; no aprobación de la entrega"}, ensure_ascii=False))
        return 0 if result.status == "complete" else 2
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as e:
        print(f"No se puede completar el análisis: {e}")
        return 2
    finally:
        signal.signal(signal.SIGINT, previous)
