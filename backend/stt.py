"""Speech to text for the game: the player holds a key, talks, and the line is
typed for them.

Whisper base.en (openai/whisper-base.en, 74M parameters) through transformers,
which the project already depends on -- no extra package. Lazy and
thread-safe like the fact extractor; loaded once at server startup so the
first spoken line does not pay the load.

Why transcription is on the server, not in Godot: the game has no speech
model, and the server already holds the GPU/CPU budget. The transcript then
goes through /chat exactly like a typed line, so every NPC stage -- intent,
memory, news, the reply guard -- treats spoken and typed players the same.

Audio arrives as the game captures it: 32-bit float samples at the engine's
mix rate (usually 48 kHz), mono or interleaved stereo. It is downmixed and
resampled to Whisper's 16 kHz here.

Silence is not sent to the model. Given near-silent audio Whisper still writes
something ("Thank you.", "you"), and an NPC answering a line the player never
said is worse than "didn't catch that".
"""

import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

MODEL = "openai/whisper-base.en"
TARGET_RATE = 16000
MIN_SECONDS = 0.3        # a tap of the key, not speech
MAX_SECONDS = 30.0       # Whisper's window; longer is cut
SILENCE_RMS = 0.004      # below this the clip is treated as silence (float samples, -1..1)
# Vocabulary hint passed as Whisper's previous-text prompt. It is the setting,
# not an answer: department abbreviations are otherwise run together
# ("cse d" -> "CSCD"). With it, "CSE, D." on the same clip; unchanged elsewhere
# on five synthetic test lines. Names outside it ("Yugabharathi") still vary.
PROMPT = "Students at a college open day in India: CSE, ECE, EEE, IT, AI and DS, section A, B, C, D."


def to_mono_16k(samples: np.ndarray, sample_rate: int, channels: int) -> np.ndarray:
    """Interleaved float samples -> mono float32 at 16 kHz."""
    audio = np.asarray(samples, dtype=np.float32)
    if channels > 1:
        audio = audio[: len(audio) - len(audio) % channels].reshape(-1, channels).mean(axis=1)
    audio = audio[: int(MAX_SECONDS * sample_rate)]
    if sample_rate != TARGET_RATE and len(audio):
        import torch
        import torchaudio.functional as F
        audio = F.resample(torch.from_numpy(audio), sample_rate, TARGET_RATE).numpy()
    return audio


def decode(raw: bytes, fmt: str) -> np.ndarray:
    if fmt == "s16":
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if fmt == "f32":
        return np.frombuffer(raw[: len(raw) - len(raw) % 4], dtype="<f4")
    raise ValueError("unknown sample format %r (expected 'f32' or 's16')" % fmt)


class Transcriber:
    def __init__(self, model: str = MODEL):
        self.model_name = model
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._lock = threading.Lock()
        self._failed = False

    def _load(self):
        if self._model is not None or self._failed:
            return self._model
        with self._lock:
            if self._model is not None or self._failed:
                return self._model
            try:
                import torch
                import transformers
                from transformers import WhisperForConditionalGeneration, WhisperProcessor
                transformers.logging.set_verbosity_error()  # generate() config chatter on every line
                # The NPC models run on CPU (llama.cpp, n_gpu_layers=0), so the GPU is free.
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if self._device == "cuda" else torch.float32
                self._processor = WhisperProcessor.from_pretrained(self.model_name, local_files_only=True)
                self._model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_name, local_files_only=True, dtype=dtype).to(self._device).eval()
            except Exception as exc:  # noqa: BLE001 - speech input is optional
                log.warning("speech-to-text unavailable (%s): players can still type", exc)
                self._failed = True
                self._model = None
            return self._model

    @property
    def available(self) -> bool:
        return self._load() is not None

    @property
    def source(self) -> str:
        return ("%s on %s" % (self.model_name, self._device)) if self._model is not None else ""

    def transcribe(self, audio16k: np.ndarray) -> dict:
        """{"text", "transcribe_ms", "duration_ms", "skipped"}; skipped is
        "short" or "silence" when the model was not run."""
        duration_ms = round(len(audio16k) * 1000 / TARGET_RATE, 1)
        out = {"text": "", "transcribe_ms": 0.0, "duration_ms": duration_ms, "skipped": ""}
        if len(audio16k) < MIN_SECONDS * TARGET_RATE:
            out["skipped"] = "short"
            return out
        if float(np.sqrt(np.mean(np.square(audio16k)))) < SILENCE_RMS:
            out["skipped"] = "silence"
            return out
        model = self._load()
        if model is None:
            raise RuntimeError("speech-to-text model not available")
        import torch
        start = time.perf_counter()
        features = self._processor(audio16k, sampling_rate=TARGET_RATE, return_tensors="pt").input_features
        features = features.to(self._device, dtype=model.dtype)
        prompt_ids = torch.tensor(self._processor.get_prompt_ids(PROMPT)).to(self._device)
        with self._lock, torch.inference_mode():
            ids = model.generate(features, max_new_tokens=96, prompt_ids=prompt_ids)
        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if text.startswith(PROMPT):   # the decoded ids include the prompt
            text = text[len(PROMPT):].strip()
        out["text"] = text
        out["transcribe_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return out


# One per process.
default = Transcriber()
