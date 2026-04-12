"""Unit tests for AudioCapture class."""

import queue
from unittest.mock import patch, MagicMock

import pytest

from audio_capture import AudioCapture


class TestAudioCaptureInit:
    def test_default_parameters(self):
        ac = AudioCapture()
        assert ac.sample_rate == 16000
        assert ac.chunk_size == 512
        assert isinstance(ac.audio_queue, queue.Queue)

    def test_custom_parameters(self):
        q = queue.Queue()
        ac = AudioCapture(sample_rate=44100, chunk_size=1024, audio_queue=q)
        assert ac.sample_rate == 44100
        assert ac.chunk_size == 1024
        assert ac.audio_queue is q

    def test_none_audio_queue_creates_default(self):
        ac = AudioCapture(audio_queue=None)
        assert isinstance(ac.audio_queue, queue.Queue)


class TestAudioCaptureCallback:
    def test_callback_enqueues_data(self):
        q = queue.Queue()
        ac = AudioCapture(audio_queue=q)
        raw_data = b"\x00\x01\x02\x03"

        result = ac._callback(raw_data, 512, {}, 0)

        assert q.get_nowait() == raw_data
        assert result == (None, 0)  # pyaudio.paContinue == 0

    def test_callback_enqueues_multiple_frames(self):
        q = queue.Queue()
        ac = AudioCapture(audio_queue=q)

        ac._callback(b"frame1", 512, {}, 0)
        ac._callback(b"frame2", 512, {}, 0)
        ac._callback(b"frame3", 512, {}, 0)

        assert q.get_nowait() == b"frame1"
        assert q.get_nowait() == b"frame2"
        assert q.get_nowait() == b"frame3"


class TestAudioCaptureStart:
    @patch("audio_capture.pyaudio.PyAudio")
    def test_start_opens_stream_with_correct_params(self, mock_pyaudio_cls):
        mock_pa = MagicMock()
        mock_stream = MagicMock()
        mock_pa.open.return_value = mock_stream
        mock_pyaudio_cls.return_value = mock_pa

        q = queue.Queue()
        ac = AudioCapture(sample_rate=16000, chunk_size=512, audio_queue=q)
        ac.start()

        mock_pa.open.assert_called_once()
        call_kwargs = mock_pa.open.call_args[1]
        assert call_kwargs["format"] == 8  # pyaudio.paInt16
        assert call_kwargs["channels"] == 1
        assert call_kwargs["rate"] == 16000
        assert call_kwargs["input"] is True
        assert call_kwargs["frames_per_buffer"] == 512
        assert call_kwargs["stream_callback"] == ac._callback
        mock_stream.start_stream.assert_called_once()

    @patch("audio_capture.pyaudio.PyAudio")
    def test_start_raises_on_microphone_unavailable(self, mock_pyaudio_cls):
        mock_pa = MagicMock()
        mock_pa.open.side_effect = OSError("No microphone found")
        mock_pyaudio_cls.return_value = mock_pa

        ac = AudioCapture()
        with pytest.raises(OSError, match="No microphone found"):
            ac.start()

        mock_pa.terminate.assert_called_once()


class TestAudioCaptureStop:
    @patch("audio_capture.pyaudio.PyAudio")
    def test_stop_closes_stream_and_terminates(self, mock_pyaudio_cls):
        mock_pa = MagicMock()
        mock_stream = MagicMock()
        mock_stream.is_active.return_value = True
        mock_pa.open.return_value = mock_stream
        mock_pyaudio_cls.return_value = mock_pa

        ac = AudioCapture()
        ac.start()
        ac.stop()

        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_pa.terminate.assert_called_once()

    def test_stop_when_not_started(self):
        ac = AudioCapture()
        # Should not raise
        ac.stop()
