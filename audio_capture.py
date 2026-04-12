"""Audio capture module using PyAudio in callback mode."""

import logging
import queue

import pyaudio

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures audio from the default microphone using PyAudio callback mode."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        audio_queue: queue.Queue = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = audio_queue if audio_queue is not None else queue.Queue()
        self._pa: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None

    def start(self) -> None:
        """Open PyAudio stream in callback mode; non-blocking."""
        try:
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._callback,
            )
            self._stream.start_stream()
        except OSError as e:
            logger.error("Microphone unavailable: %s", e)
            if self._pa is not None:
                self._pa.terminate()
                self._pa = None
            raise

    def stop(self) -> None:
        """Stop and close the PyAudio stream."""
        if self._stream is not None:
            if self._stream.is_active():
                self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def _callback(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: dict,
        status: int,
    ) -> tuple:
        """PyAudio callback: enqueues raw bytes onto audio_queue."""
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
