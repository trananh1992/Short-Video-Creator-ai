# Parakeet Transcription Design

Date: 2026-07-20
Status: Approved

## Goal

Replace `whisper-timestamped` (whisper-small.en, CPU) with NVIDIA Parakeet TDT 0.6B v3
for word-level caption timestamps. Must run on macOS, Windows, and Linux.

## Why

- Parakeet TDT v3: ~6.3% avg WER vs ~8-9% for whisper-small.en.
- TDT architecture models token durations natively → stable word timestamps
  (whisper-timestamped infers them from cross-attention; drifts on long/noisy audio).
- Much faster than whisper on CPU; ONNX int8 runs everywhere.

## Backend decision

**onnx-asr** (`pip install onnx-asr[cpu,hub]`) running the ONNX export of
`parakeet-tdt-0.6b-v3` (int8 quantization).

- Cross-platform: Mac/Windows/Linux, x86 + ARM, CPU by default (CUDA etc. optional).
- Minimal deps: numpy, onnxruntime, huggingface-hub.
- Auto-downloads model from Hugging Face on first load (HF cache).
- Token-level timestamps supported.

Fallback if onnx-asr's timestamp API proves unusable during implementation:
use `sherpa-onnx` directly with the `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`
model — same strategy, more config boilerplate.

Rejected alternatives:
- Hybrid MLX (Mac) + ONNX (others): two code paths, two failure surfaces.
- NeMo: Linux-first, Windows needs WSL2, heavy dependency tree.

## Architecture

### New module: `transcriber.py`

Single public function:

```python
def transcribe_words(audio_path: str) -> list[dict]:
    # returns [{'timestamp': (start: float, end: float), 'text': str}, ...]
```

Return shape is exactly what `VideoCreation.create_transcription()` produces today,
so caption logic in `main.py` is untouched.

Internals:
- Module-level model cache: model loads once per worker process (current code
  reloads whisper per video — fix that waste for free).
- Recognize with timestamps enabled.
- Merge BPE tokens into words at `▁` (word-boundary) markers:
  - word start = first token's start time
  - word end = last token's end time if the API provides ends; otherwise
    next word's start clamped to a small max duration; final word gets
    start + estimated duration. Pin exact rule against real API output
    during implementation.
- Standalone smoke-run entrypoint: `python transcriber.py path/to/audio.wav`
  prints the word list.

### Changes to existing files

- `config.py`: replace `MODEL_NAME = 'whisper-small.en'` with the Parakeet ONNX
  model ID and add `QUANTIZATION = 'int8'`. Keep `LANGUAGE` (v3 is multilingual;
  param may be unused by onnx-asr — keep for future use).
- `main.py`:
  - `create_transcription()`: keep temp-audio-extraction flow but write
    16 kHz mono WAV (ASR-native) instead of MP3; call
    `transcriber.transcribe_words()`; delete whisper import.
  - Delete `clone_respository()`, `check_command()`, and the
    `if not os.path.exists(MODEL_NAME)` download block — onnx-asr handles
    model download. Git/git-LFS no longer prerequisites.
- `requirements.txt`: remove `whisper-timestamped`, `openai-whisper`;
  add `onnx-asr[cpu,hub]` (pinned).
- `README.md`: update model section (remove git-LFS instructions, describe
  auto-download, note whisper-timestamped no longer used).
- `.gitignore` / repo: local `whisper-small.en/` dir no longer needed.

## Error handling

- Empty/no-speech audio → `transcribe_words` returns `[]`;
  `add_captions_to_video` already handles the empty case.
- Model download failure → let the underlying error propagate with a log line
  naming Hugging Face and suggesting a network check.

## Testing

No test framework in repo (stays that way).

1. Smoke: `python transcriber.py <wav>` on a short English clip — words present,
   timestamps monotonic, start < end.
2. End-to-end: run `main.py` on one input video; eyeball output captions for sync.

## Trade-offs accepted

- Mac loses potential MLX (Apple GPU) speed in exchange for one code path.
  int8 ONNX on CPU still far faster than current whisper-on-CPU.
- Parakeet ~1% WER behind leaderboard leaders (Canary-Qwen, Granite) — those
  lack native word timestamps, which matter more for caption sync.
