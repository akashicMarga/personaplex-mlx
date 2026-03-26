# CLAUDE.md

## Project Overview

PersonaPlex-MLX — MLX inference for NVIDIA PersonaPlex (full-duplex conversational speech model) on Apple Silicon. Port of the NVIDIA PersonaPlex model to Apple's MLX framework.

## Tech Stack

- Python 3.10–3.12 (3.12 recommended)
- MLX (>=0.26.0, <0.27) for Apple Silicon inference
- Audio: rustymimi (tokenizer), sounddevice (I/O), sphn (phonemes)
- Model weights from HuggingFace: `nvidia/personaplex-7b-v1`

## Project Structure

- `personaplex_mlx/models/` — Model architecture (lm.py), generation (generate.py), MIMI tokenizer (mimi.py), TTS (tts.py)
- `personaplex_mlx/modules/` — Building blocks: transformer, conv, quantization, conditioning, KV cache, SEANet
- `personaplex_mlx/utils/` — Sampling, HF loaders
- `personaplex_mlx/local.py` — Realtime terminal chat interface
- `personaplex_mlx/local_web.py` — WebSocket-based web interface (port 8998)
- `personaplex_mlx/offline.py` — WAV-to-WAV offline inference
- `personaplex_mlx/persona_utils.py` — Model loading, voice prompt resolution, HF utilities

## Entry Points

```
personaplex-local      → personaplex_mlx.local:main
personaplex-local-web  → personaplex_mlx.local_web:main
personaplex-offline    → personaplex_mlx.offline:main
```

## Setup & Run

```bash
pip install -e ".[dev]"
personaplex-local          # Terminal chat
personaplex-local-web      # Web interface at :8998
personaplex-offline        # WAV-to-WAV
```

## Linting & Type Checking

```bash
flake8 .                   # Linting (max-line-length=120, ignore E203,E704)
pyright                    # Type checking
```

No test suite exists currently.

## Code Conventions

- Max line length: 120
- All files carry Kyutai copyright + MIT license header
- Heavy use of type hints and dataclasses
- snake_case for functions/variables, PascalCase for classes
- `from __future__ import annotations` in all modules
- MLX patterns: `@mx.compile`, proper dtype management (`mx.float32`, `mx.float16`)
