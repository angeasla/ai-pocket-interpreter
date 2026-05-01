# AI Pocket Interpreter

**Privacy-First, 100% Offline** real-time speech-to-text and multi-language translation — runs entirely on edge hardware with no cloud dependency.

Speak into your microphone in any supported language and see live captions translated into English, Greek, and Spanish streamed to your browser over WebSockets.

---

## Overview

AI Pocket Interpreter captures live audio, detects speech boundaries, transcribes with OpenAI Whisper, translates via Meta's NLLB-200, and pushes results to a browser UI — all locally, all in real time.

Key highlights:

- **Zero internet required** at runtime. Audio never leaves your device.
- **Dynamic VAD chunking** prevents OOM on continuous speech (adaptive silence thresholds across three zones).
- **Context-overlap sliding window** feeds trailing words to Whisper so mid-sentence cuts don't break transcription quality.
- **Single-process async architecture** — FastAPI event loop coordinates audio capture, GPU inference, and WebSocket broadcast.

---

## Architecture

```
Microphone → Silero-VAD → faster-whisper (STT) → Language Mapper → NLLB-200 (Translation) → WebSocket → Browser
```

| Component | Role |
|---|---|
| **AudioCapture** | PyAudio callback stream → thread-safe queue |
| **VADProcessor** | Silero-VAD speech/silence state machine with dynamic thresholds |
| **STTEngine** | faster-whisper `large-v3-turbo` transcription + language detection |
| **LanguageMapper** | ISO 639-1 → FLORES-200 code mapping |
| **TranslationEngine** | CTranslate2 NLLB-200 distilled-1.3B (int8) |
| **ConnectionManager** | WebSocket client tracking and JSON broadcast |
| **Frontend** | Single-file HTML/CSS/JS — no external resources |

---

## Hardware Requirements

Designed for edge devices. The application **automatically detects** the best available backend at startup — no configuration needed.

| Hardware | Device | Compute Type | Notes |
|---|---|---|---|
| Nvidia GPU (≥ 6 GB VRAM) | `cuda` | float16 / int8_float16 | Recommended for real-time use |
| Intel N100 / any CPU | `cpu` | int8 / int8 | Fully supported, slower throughput |

Minimum system requirements:

| Resource | Minimum |
|---|---|
| CPU | x86-64 with AVX2 (Intel N100 or better) |
| RAM | 8 GB |
| GPU VRAM | 6 GB (CUDA) — optional, falls back to CPU automatically |
| Storage | ~4 GB for model weights |

### VRAM Budget (GPU mode)

| Model | Quantization | Est. VRAM |
|---|---|---|
| Whisper large-v3-turbo | float16 | ~1.6 GB |
| NLLB-200 distilled-1.3B | int8_float16 | ~1.3 GB |
| Silero-VAD | float16 | ~10 MB |
| **Total** | | **~3.0 GB** |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/ai-pocket-interpreter.git
cd ai-pocket-interpreter

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install system dependency for PyAudio
sudo apt-get install portaudio19-dev   # Debian/Ubuntu

# 4. Install Python dependencies
pip install -r requirements.txt
```

---

## Model Download

The AI weights are **not** included in the repository. Download them into a `models/` directory before first run.

### NLLB-200 (CTranslate2, int8)

```bash
pip install huggingface_hub

# Download the pre-converted CTranslate2 model
huggingface-cli download JustFrederik/nllb-200-distilled-1.3B-ct2-int8 \
    --local-dir models/nllb-200-distilled-1.3b-ct2
```

### Whisper large-v3-turbo

`faster-whisper` downloads and caches the model automatically on first run. No manual step needed — just ensure you have internet for the initial download, then it works offline forever.

### Silero-VAD

Downloaded automatically via `torch.hub` on first run and cached locally.

---

## Usage

```bash
# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000

# Open in your browser
# http://localhost:8000
```

Speak into your microphone. Live captions appear in the browser with translations into English, Greek, and Spanish.

---

## Project Structure

```
ai-pocket-interpreter/
├── main.py                 # FastAPI app, startup, pipeline wiring
├── audio_capture.py        # Microphone capture (PyAudio callback)
├── vad_processor.py        # Dynamic VAD state machine
├── stt_engine.py           # Whisper STT with context overlap
├── language_mapper.py      # ISO 639-1 → FLORES-200 mapping
├── translation_engine.py   # NLLB-200 translation (CTranslate2)
├── connection_manager.py   # WebSocket client management
├── models.py               # Data classes (AudioChunk, Payload, etc.)
├── frontend.html           # Self-contained browser UI
├── requirements.txt        # Python dependencies
└── models/                 # AI weights (not in repo — see above)
```

---

## License

MIT
