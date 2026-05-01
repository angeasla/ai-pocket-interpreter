"""FastAPI application wiring all pipeline components together."""

import asyncio
import logging
import queue
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from audio_capture import AudioCapture
from connection_manager import ConnectionManager
from language_mapper import LanguageMapper
from models import Payload
from stt_engine import STTEngine
from translation_engine import TranslationEngine
from vad_processor import VADProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware auto-detection
# ---------------------------------------------------------------------------

# Minimum free VRAM (bytes) required to load NLLB-200 on GPU.
# NLLB int8 needs ~1.3 GB; we require 1.5 GB free to leave headroom.
_NLLB_VRAM_REQUIRED = 1_500 * 1024 * 1024  # 1.5 GB


@dataclass(frozen=True)
class HardwareProfile:
    """Detected hardware capabilities and the compute settings derived from them."""
    stt_device: str        # "cuda" | "cpu" — for faster-whisper
    stt_compute_type: str  # compute_type for faster-whisper
    nllb_device: str       # "cuda" | "cpu" — for CTranslate2 NLLB
    nllb_compute_type: str  # compute_type for CTranslate2 NLLB


def detect_hardware() -> HardwareProfile:
    """Detect the best available execution backend and return matching settings.

    STT (Whisper) is placed on CUDA whenever available — it is the latency-
    critical path.  NLLB-200 is only placed on CUDA if there is enough free
    VRAM after Whisper loads (~1.5 GB headroom required); otherwise it falls
    back to CPU int8 which is fast enough for translation workloads.

    GPU tiers:
      • Ampere+ (cc >= 8.0, e.g. RTX 30xx/40xx) — float16 / int8_float16
      • Older CUDA (Pascal/Turing/Volta, cc < 8.0) — int8 on both
      • No CUDA — everything on CPU int8
    """
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA not available — running fully on CPU (int8). "
            "Performance will be reduced."
        )
        return HardwareProfile(
            stt_device="cpu", stt_compute_type="int8",
            nllb_device="cpu", nllb_compute_type="int8",
        )

    device_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_vram  = torch.cuda.get_device_properties(0).total_memory
    total_gb    = total_vram / (1024 ** 3)

    # Choose STT compute type based on architecture.
    if major >= 8:
        stt_compute = "float16"
        arch_label  = "Ampere+ (float16)"
    else:
        stt_compute = "int8"
        arch_label  = "Pascal/Turing/Volta (int8)"

    # Decide where NLLB goes: GPU only if enough total VRAM.
    # GTX 1060 3 GB: Whisper int8 ~1.5 GB → only ~1.5 GB left, too tight.
    # We use total VRAM as a proxy; if < 6 GB, offload NLLB to CPU.
    if total_vram >= 6 * 1024 ** 3:
        nllb_device   = "cuda"
        nllb_compute  = "int8_float16" if major >= 8 else "int8"
        nllb_location = "GPU"
    else:
        nllb_device   = "cpu"
        nllb_compute  = "int8"
        nllb_location = "CPU (VRAM too small for both models)"

    logger.info(
        "CUDA detected: %s (cc %s.%s, %.1f GB VRAM) — arch=%s",
        device_name, major, minor, total_gb, arch_label,
    )
    logger.info(
        "Model placement: Whisper → GPU (%s) | NLLB → %s (%s)",
        stt_compute, nllb_location, nllb_compute,
    )

    return HardwareProfile(
        stt_device="cuda",    stt_compute_type=stt_compute,
        nllb_device=nllb_device, nllb_compute_type=nllb_compute,
    )

app = FastAPI()
manager = ConnectionManager()
language_mapper = LanguageMapper()

# Global references for pipeline components
stt_engine: STTEngine | None = None
translation_engine: TranslationEngine | None = None


async def bridge_queue(sync_queue: queue.Queue, async_queue: asyncio.Queue) -> None:
    """Bridge a thread-safe queue.Queue into an asyncio.Queue.

    Polls the sync queue in a non-blocking loop, yielding control back to the
    event loop between checks so other coroutines can run.
    """
    loop = asyncio.get_event_loop()
    while True:
        try:
            item = await loop.run_in_executor(None, sync_queue.get)
            await async_queue.put(item)
        except asyncio.CancelledError:
            return


# ---------------------------------------------------------------------------
# Context-overlap helpers
# ---------------------------------------------------------------------------
_CONTEXT_WORD_LIMIT = 12  # keep the last ~12 words as sliding-window prompt


def _tail_words(text: str, n: int = _CONTEXT_WORD_LIMIT) -> str:
    """Return the last *n* whitespace-delimited words of *text*."""
    words = text.split()
    return " ".join(words[-n:]) if words else ""


async def pipeline_consumer(chunk_queue: asyncio.Queue) -> None:
    """Read AudioChunks from chunk_queue, run STT + translation, broadcast.

    Maintains a sliding context window (last ~12 words) that is passed as
    ``initial_prompt`` to the next STT call so Whisper can recover context
    lost by dynamic VAD cuts (Zone 2 / Zone 3).
    """
    loop = asyncio.get_event_loop()

    # Sliding-window state --------------------------------------------------
    prev_context: str = ""      # trailing words from the previous chunk
    prev_language: str = ""     # language of the previous chunk

    while True:
        try:
            audio_chunk = await chunk_queue.get()
        except asyncio.CancelledError:
            return

        # STT (blocking GPU call) — pass context from previous chunk.
        transcription = await loop.run_in_executor(
            None,
            lambda ac=audio_chunk, ctx=prev_context: stt_engine.transcribe(
                ac, initial_prompt=ctx or None
            ),
        )

        # Skip empty transcriptions (Req 3.4) and clear context.
        if transcription.is_empty:
            logger.debug("Empty transcription, clearing context.")
            prev_context = ""
            prev_language = ""
            continue

        # Language change detection — reset context on switch.
        if prev_language and transcription.language != prev_language:
            logger.info(
                "Language changed (%s → %s), clearing context.",
                prev_language,
                transcription.language,
            )
            prev_context = ""

        # Update sliding-window context for the next chunk.
        prev_context = _tail_words(transcription.text)
        prev_language = transcription.language

        # Map language code (Req 4.3)
        flores_code = language_mapper.map(transcription.language)
        if flores_code is None:
            logger.warning(
                "Unmapped language code '%s', skipping translation.",
                transcription.language,
            )
            continue

        # Translation (blocking GPU call)
        translations = await loop.run_in_executor(
            None, translation_engine.translate, transcription.text, flores_code
        )

        # Build and broadcast payload
        payload = Payload(
            original_lang=transcription.language,
            original_text=transcription.text,
            translations=translations,
        )
        await manager.broadcast(payload.to_dict())


@app.on_event("startup")
async def startup() -> None:
    """Load all models, start audio capture, and launch pipeline coroutines."""
    global stt_engine, translation_engine

    # --- Detect hardware and derive compute settings ---
    hw = detect_hardware()

    # --- Load Silero-VAD model ---
    try:
        logger.info("Loading Silero-VAD model…")
        vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
        logger.info("Silero-VAD model loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load Silero-VAD model: %s", exc)
        sys.exit(1)

    # --- Load STT Engine ---
    try:
        stt_engine = STTEngine(
            device=hw.stt_device,
            compute_type=hw.stt_compute_type,
        )
        stt_engine.load()
    except Exception as exc:
        logger.error("Failed to load STT engine: %s", exc)
        sys.exit(1)

    # --- Load Translation Engine ---
    try:
        translation_engine = TranslationEngine(
            model_path="models/nllb-200-distilled-1.3b-ct2",
            tokenizer_name="facebook/nllb-200-distilled-1.3B",
            device=hw.nllb_device,
            compute_type=hw.nllb_compute_type,
        )
        translation_engine.load()
    except Exception as exc:
        logger.error("Failed to load Translation engine: %s", exc)
        sys.exit(1)

    # --- Create queues ---
    audio_sync_queue: queue.Queue = queue.Queue()
    audio_async_queue: asyncio.Queue = asyncio.Queue()
    chunk_queue: asyncio.Queue = asyncio.Queue()

    # --- Start AudioCapture ---
    try:
        capture = AudioCapture(audio_queue=audio_sync_queue)
        capture.start()
        logger.info("Audio capture started.")
    except OSError as exc:
        logger.error("Microphone unavailable: %s", exc)
        sys.exit(1)

    # --- Create VADProcessor ---
    vad_processor = VADProcessor(model=vad_model)

    # --- Launch background coroutines ---
    asyncio.create_task(bridge_queue(audio_sync_queue, audio_async_queue))
    asyncio.create_task(vad_processor.run(audio_async_queue, chunk_queue))
    asyncio.create_task(pipeline_consumer(chunk_queue))

    logger.info("All models loaded and pipeline started.")
    logger.info("Application available at http://localhost:8000")


@app.get("/")
async def serve_frontend() -> HTMLResponse:
    """Serve the single-file frontend HTML."""
    frontend_path = Path(__file__).parent / "frontend.html"
    html_content = frontend_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for live caption streaming."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; client doesn't send data
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
