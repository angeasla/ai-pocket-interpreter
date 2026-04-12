"""Speech-to-Text engine wrapping faster-whisper."""

import logging

import numpy as np
from faster_whisper import WhisperModel

from models import AudioChunk, TranscriptionResult

logger = logging.getLogger(__name__)


class STTEngine:
    """Wraps faster-whisper for transcription and language detection.

    The model is loaded once via :meth:`load` at application startup and
    reused for all subsequent transcriptions.
    """

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: WhisperModel | None = None

    def load(self) -> None:
        """Load the Whisper model once at startup."""
        logger.info(
            "Loading Whisper model '%s' on %s (%s)…",
            self.model_size,
            self.device,
            self.compute_type,
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("Whisper model loaded successfully.")

    def transcribe(
        self,
        audio_chunk: AudioChunk,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe an AudioChunk and detect the source language.

        Parameters
        ----------
        audio_chunk:
            Finalized speech audio from the VAD.
        initial_prompt:
            Optional trailing text from the *previous* transcription.
            Passed straight to ``faster-whisper`` so the decoder has
            cross-chunk context, reducing hallucinations on hard cuts.

        Returns an empty :class:`TranscriptionResult` when the chunk
        contains no recognisable speech.
        """
        if self._model is None:
            raise RuntimeError("STTEngine.load() must be called before transcribe()")

        samples = audio_chunk.samples
        if samples is None or samples.size == 0:
            return TranscriptionResult(text="", language="")

        # Ensure float32 as required by faster-whisper
        audio_np = samples.astype(np.float32)

        # Build transcribe kwargs; only include initial_prompt when present.
        transcribe_kwargs: dict = {"beam_size": 5}
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt

        segments, info = self._model.transcribe(audio_np, **transcribe_kwargs)
        text = "".join(segment.text for segment in segments).strip()
        language = info.language or ""

        if not text:
            return TranscriptionResult(text="", language="")

        return TranscriptionResult(text=text, language=language)
