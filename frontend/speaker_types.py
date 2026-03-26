from dataclasses import dataclass, field


@dataclass
class SpeakerBox:
    label: str
    source_lang: str = "auto"
    is_active: bool = False
    lines: list[str] = field(default_factory=list)
    last_active_ts: float = 0.0
    keep_until_ts: float | None = None
