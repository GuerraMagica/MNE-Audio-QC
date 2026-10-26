from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from threading import Event

import numpy as np

LAYOUTS = {"2.0": ("L", "R"), "5.1": ("L", "R", "C", "LFE", "Ls", "Rs")}
CANDIDATE = "posible incidencia"
TECHNICAL = "error técnico comprobado"


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError("Rango vacío o invertido")


@dataclass(frozen=True)
class ChannelSource:
    path: Path
    file_channel: int
    label: str


@dataclass(frozen=True)
class Asset:
    role: str
    version: str
    channels: tuple[ChannelSource, ...]
    start_sample: int = 0


@dataclass
class Manifest:
    project_id: str
    layout: str
    assets: dict[str, Asset]
    program_span: tuple[int, int]
    fps: Fraction | None = None
    start_frame: int | None = None
    drop_frame: bool = False
    exceptions: list[dict] = field(default_factory=list)
    synthetic: bool = False


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    declared_bits: int | None
    subtype: str
    frames: int
    channels: int
    bwf_time_reference: int | None = None


@dataclass
class Block:
    start: int
    samples: np.ndarray


@dataclass
class Finding:
    id: str
    kind: str
    state: str
    span: Span | None
    channels: tuple[str, ...]
    evidence: dict = field(default_factory=dict)


@dataclass
class StageResult:
    stage: str
    status: str
    coverage: list[Span] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)
    reason: str | None = None


@dataclass
class RunResult:
    status: str
    stages: list[StageResult] = field(default_factory=list)
    input_identity: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SpeechConfig:
    binary: Path
    model: Path
    vad_model: Path | None = None
    language: str = "es"
    mode: str = "fixed"
    threads: int = 4
    timeout_seconds: float = 180.0
    use_gpu: bool = True


@dataclass
class Detection:
    span: Span
    channels: tuple[str, ...]
    text: str
    evidence: dict = field(default_factory=dict)


@dataclass
class Annotation:
    id: str
    project_id: str
    layout: str
    kind: str
    span: Span
    channels: tuple[str, ...]
    label: str = "positive"
    injected: bool = False
    policy_issue: bool | None = None


@dataclass
class EvaluationResult:
    metrics: dict
    eligibility: dict
    gates: dict


class StopToken:
    def __init__(self):
        self._event = Event()

    def cancel(self):
        self._event.set()

    def cancelled(self) -> bool:
        return self._event.is_set()
