from dataclasses import dataclass
import numpy as np


@dataclass
class AudioChunk:
    """A finalized segment of speech audio."""
    samples: np.ndarray  # float32, shape (N,), 16kHz mono
    sample_rate: int = 16000


@dataclass
class TranscriptionResult:
    """Output from the STT engine."""
    text: str       # Transcribed text (empty string if no speech)
    language: str   # ISO 639-1 detected language code (e.g. "en")

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass
class Payload:
    """JSON structure broadcast over WebSocket."""
    original_lang: str              # ISO 639-1 source language
    original_text: str              # Transcribed text
    translations: dict[str, str]    # {"en": ..., "el": ..., "es": ...}

    def to_dict(self) -> dict:
        return {
            "original_lang": self.original_lang,
            "original_text": self.original_text,
            "translations": self.translations,
        }
