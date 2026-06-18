"""Runtime settings for laptop and Raspberry Pi execution."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeSettings:
    """Configuration values shared by the pipeline components."""

    confidence_threshold: float = 0.85
    max_sentence_latency_ms: int = 2000
    max_word_latency_ms: int = 500
    models_dir: Path = Path("models")
    data_dir: Path = Path("data")
    api_correction_enabled: bool = False
    device_profile: str = "laptop"

    def validate(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.max_word_latency_ms <= 0:
            raise ValueError("max_word_latency_ms must be positive")
        if self.max_sentence_latency_ms <= 0:
            raise ValueError("max_sentence_latency_ms must be positive")
        if self.device_profile not in {"laptop", "raspberry_pi"}:
            raise ValueError("device_profile must be 'laptop' or 'raspberry_pi'")
