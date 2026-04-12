"""Voice Activity Detection processor using Silero-VAD.

Implements a dynamic silence-threshold state machine to handle continuous,
long-form speech on memory-constrained edge devices.  Three zones prevent
OOM errors while preserving natural sentence boundaries:

  Zone 1 (0–6 s):   >= 0.8 s silence  – ideal syntactic cut (sentence end)
  Zone 2 (6–12 s):  >= 0.3 s silence  – rescue cut (clause / breath pause)
  Zone 3 (> 12 s):  >= 0.05 s silence – emergency hard cap (word boundary)
"""

import asyncio
import enum
import logging
import struct

import numpy as np
import torch

from models import AudioChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dynamic threshold zone boundaries (seconds of accumulated speech)
# ---------------------------------------------------------------------------
_ZONE1_MAX = 6.0    # Ideal syntactic cut window
_ZONE2_MAX = 12.0   # Rescue cut window

_ZONE1_SILENCE = 0.8   # Wait for a full sentence pause
_ZONE2_SILENCE = 0.3   # Catch respiratory / clause pauses
_ZONE3_SILENCE = 0.05  # Cut at the very next non-speech frame


class _State(enum.Enum):
    IDLE = "idle"
    SPEAKING = "speaking"


class VADProcessor:
    """Wraps Silero-VAD and implements a speech/silence state machine.

    Consumes raw audio frames from *audio_queue*, classifies each frame as
    speech or non-speech, accumulates speech frames into an AudioChunk, and
    emits finalized chunks onto *chunk_queue* when silence exceeds a
    *dynamic* threshold that tightens as the chunk grows longer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        sample_rate: int = 16000,
        silence_threshold_sec: float = 1.0,  # kept for API compat; unused by dynamic logic
    ) -> None:
        self.model = model
        self.sample_rate = sample_rate
        # Legacy attribute retained for backward compatibility / tests.
        self.silence_threshold_sec = silence_threshold_sec

    # ------------------------------------------------------------------
    # Dynamic threshold logic
    # ------------------------------------------------------------------

    def _dynamic_silence_threshold(self, speech_duration_sec: float) -> float:
        """Return the silence threshold for the current speech duration.

        Zone 1 (0–6 s):   0.8 s  – wait for a natural sentence break.
        Zone 2 (6–12 s):  0.3 s  – catch clause / breath pauses.
        Zone 3 (> 12 s):  0.05 s – emergency cut at next non-speech frame.
        """
        if speech_duration_sec <= _ZONE1_MAX:
            return _ZONE1_SILENCE
        if speech_duration_sec <= _ZONE2_MAX:
            return _ZONE2_SILENCE
        return _ZONE3_SILENCE

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(
        self,
        audio_queue: asyncio.Queue,
        chunk_queue: asyncio.Queue,
    ) -> None:
        """Consume audio frames from *audio_queue*, emit AudioChunks."""
        state = _State.IDLE
        speech_frames: list[np.ndarray] = []
        silence_frames = 0
        speech_sample_count = 0  # total speech samples accumulated

        while True:
            try:
                frame: bytes = await audio_queue.get()
            except asyncio.CancelledError:
                # Finalize any remaining speech on shutdown.
                if state == _State.SPEAKING and speech_frames:
                    chunk = self._build_chunk(speech_frames)
                    await chunk_queue.put(chunk)
                return

            is_speech = self._classify_frame(frame)
            float_frame = self._bytes_to_float32(frame)
            frame_samples = len(float_frame)
            frame_duration = frame_samples / self.sample_rate

            if state == _State.IDLE:
                if is_speech:
                    state = _State.SPEAKING
                    speech_frames = [float_frame]
                    speech_sample_count = frame_samples
                    silence_frames = 0
                # else: discard silence-only frames (Req 2.5)

            elif state == _State.SPEAKING:
                if is_speech:
                    speech_frames.append(float_frame)
                    speech_sample_count += frame_samples
                    silence_frames = 0
                else:
                    silence_frames += 1
                    # Include trailing silence for natural audio boundaries.
                    speech_frames.append(float_frame)
                    speech_sample_count += frame_samples

                    # Compute current speech duration and pick the threshold.
                    speech_duration = speech_sample_count / self.sample_rate
                    threshold = self._dynamic_silence_threshold(speech_duration)

                    if silence_frames * frame_duration >= threshold:
                        chunk = self._build_chunk(speech_frames)
                        await chunk_queue.put(chunk)
                        logger.debug(
                            "Chunk emitted: %.2f s speech, threshold=%.2f s",
                            speech_duration,
                            threshold,
                        )
                        speech_frames = []
                        silence_frames = 0
                        speech_sample_count = 0
                        state = _State.IDLE

    def _classify_frame(self, frame: bytes) -> bool:
        """Return True if *frame* contains speech according to Silero-VAD."""
        float_samples = self._bytes_to_float32(frame)
        tensor = torch.from_numpy(float_samples)
        with torch.no_grad():
            confidence = self.model(tensor, self.sample_rate).item()
        return confidence >= 0.5

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bytes_to_float32(frame: bytes) -> np.ndarray:
        """Convert raw paInt16 bytes to float32 numpy array in [-1, 1]."""
        n_samples = len(frame) // 2
        samples = struct.unpack(f"<{n_samples}h", frame)
        return np.array(samples, dtype=np.float32) / 32768.0

    @staticmethod
    def _build_chunk(frames: list[np.ndarray]) -> AudioChunk:
        """Concatenate accumulated float32 frames into an AudioChunk."""
        return AudioChunk(samples=np.concatenate(frames))
