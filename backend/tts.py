"""Piper text-to-speech for the NPC demo.

Design notes, and why speech is a separate call from /chat.

The game asks for text first and audio second. Generation takes ~330 ms and
synthesis another ~170 ms, so bundling them would hold the subtitle back until
both finished and put every spoken line over the project's 500 ms real-time
target. Splitting them means the line appears on screen at generation latency
-- the number RQ4 is about, unchanged -- and the voice follows shortly after.
Report the two separately; do not add them together and call it text latency.

Synthesis is cheap relative to playback. Measured on this machine with
`en_GB-alan-medium`: median 168 ms of compute for a median 4.57 s of audio, a
real-time factor of 0.031. Nothing here needs streaming or chunking at NPC
line lengths.

Voices are ~63 MB ONNX models loaded once and kept resident. Loading costs
about 1.3 s, and the first synthesis on a freshly loaded voice costs roughly
2 s of ONNX warm-up before settling to the median above, so voices are warmed
at startup rather than on a player's first line.
"""

import io
import logging
import re
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VOICES_DIR = Path(__file__).resolve().parent / "voices"

# One distinct speaker per NPC, keyed by the display name in
# NpcDirector.SPAWNS. An NPC with no entry, or whose voice file is missing,
# simply has no audio -- the game keeps working, silently.
VOICE_BY_NPC = {
    "Officer Reyes": "en_GB-northern_english_male-medium",
    "Prof. Adeyemi": "en_GB-alan-medium",
    "Halvorsen": "en_US-lessac-medium",
    "Ms. Okafor": "en_GB-jenny_dioco-medium",
    "Nadia": "en_US-amy-medium",
}

DEFAULT_VOICE = "en_GB-alan-medium"


# --- prosody ---------------------------------------------------------------
#
# Every Piper voice ships with the same inference defaults (length_scale 1.0,
# noise_scale 0.667, noise_w 0.8), so out of the box all five NPCs share one
# delivery. These three dials are what VITS exposes:
#
#   length_scale   > 1 is slower. 1.0 reads as slightly rushed for spoken
#                  dialogue; a few percent slower sounds conversational.
#   noise_scale    variation in pitch and timbre -- higher is livelier,
#                  lower is flatter and more controlled.
#   noise_w_scale  variation in phoneme duration -- the rhythm. Too low and
#                  every syllable is the same length, which is most of what
#                  makes TTS sound mechanical.
#
# pause_scale multiplies the silences inserted between sentences (see
# PAUSE_MS), so a deliberate speaker also *waits* longer, not just talks
# slower.

@dataclass(frozen=True)
class SpeechStyle:
    length_scale: float = 1.06
    noise_scale: float = 0.667
    noise_w_scale: float = 0.85
    pause_scale: float = 1.0


# Tuned by ear against each character's role, not measured -- there is no
# objective target for "sounds like a head of department". Change freely.
STYLE_BY_NPC = {
    # steady and plain; says what needs saying
    "Officer Reyes": SpeechStyle(length_scale=1.05, noise_scale=0.62, noise_w_scale=0.80, pause_scale=1.0),
    # thoughtful, a little slower, some lift in the voice
    "Prof. Adeyemi": SpeechStyle(length_scale=1.10, noise_scale=0.70, noise_w_scale=0.88, pause_scale=1.15),
    # deliberate and controlled; the longest pauses of anyone
    "Halvorsen": SpeechStyle(length_scale=1.12, noise_scale=0.58, noise_w_scale=0.75, pause_scale=1.3),
    # warm and unhurried
    "Ms. Okafor": SpeechStyle(length_scale=1.10, noise_scale=0.72, noise_w_scale=0.92, pause_scale=1.2),
    # brisk and chatty, serving customers between sentences
    "Nadia": SpeechStyle(length_scale=0.98, noise_scale=0.76, noise_w_scale=0.92, pause_scale=0.85),
}
DEFAULT_STYLE = SpeechStyle()

# Extra silence added after a sentence, by how it ended. Piper already leaves
# roughly 200 ms at each sentence boundary (measured: ~170 ms trailing plus
# ~30 ms leading); these are added on top, taking a full stop to ~430 ms,
# which is in the range of ordinary conversational pausing.
PAUSE_MS = {
    ".": 230,
    "!": 200,
    "?": 280,
    "…": 420,     # a trailing-off pause, longer than a full stop
    ",": 0,       # espeak already pauses at commas within a sentence
}
LEAD_IN_MS = 60   # so the first syllable does not start on the very first sample
FADE_MS = 8       # ramps at each join, so concatenated chunks do not click


# --- text normalisation ----------------------------------------------------
#
# Found by probing what Piper actually does with lines the NPCs produce:
#
#   "Prof. Adeyemi"   read as two sentences, "prof." <pause> "Adeyemi" --
#                     the full stop after the abbreviation ends the sentence
#                     in the middle of a name.
#   "Ms. Okafor"      read as "M. S." <pause> "Okafor".
#   "4pm... Ms."      the ellipsis produced no pause at all and the next
#                     sentence was glued on: "four P M, M S."
#   "CS2011"          read as a year: "C S two thousand and eleven".
#
# All of these are fixed here, before synthesis, rather than in the NPCs'
# text: the dialogue on screen should keep reading "Prof." and "CS2011".

_ABBREVIATIONS = [
    (r"\bProf\.", "Professor"),
    (r"\bDr\.", "Doctor"),
    (r"\bMrs\.", "Missus"),
    (r"\bMr\.", "Mister"),
    (r"\bMs\.?(?=\s)", "Miz"),
    (r"\bSt\.(?=\s+[A-Z])", "Saint"),
    (r"\be\.g\.", "for example"),
    (r"\bi\.e\.", "that is"),
    (r"\betc\.", "et cetera"),
    (r"\bvs\.?", "versus"),
]


def _say_code_digits(digits: str) -> str:
    """"2011" -> "20 11" (twenty eleven); "1002" -> "10 oh 2" (ten oh two)."""
    if len(digits) != 4:
        return " ".join(digits)
    head, tail = digits[:2], digits[2:]
    if tail == "00":
        tail_spoken = "hundred"
    elif tail[0] == "0":
        tail_spoken = "oh " + tail[1]
    else:
        tail_spoken = tail
    return "%s %s" % (head.lstrip("0") or "0", tail_spoken)


def normalize_for_speech(text: str) -> str:
    """Rewrites a display line into something the phonemiser reads naturally."""
    s = text.strip()
    for pattern, replacement in _ABBREVIATIONS:
        s = re.sub(pattern, replacement, s)

    # Module / room codes: letters then four digits. "CS2011" -> "C S 20 11".
    s = re.sub(r"\b([A-Z]{2,4})(\d{4})\b",
               lambda m: "%s %s" % (" ".join(m.group(1)), _say_code_digits(m.group(2))), s)

    # Ellipses: normalise both forms to one character, so the splitter can
    # give it its own (longer) pause instead of letting espeak drop it.
    s = s.replace("...", "…")

    # A dash mid-sentence is a pause, and espeak renders a comma cleanly.
    s = re.sub(r"\s*[—–]\s*", ", ", s)

    return re.sub(r"\s+", " ", s)


# Sentence ends: terminal punctuation followed by space or end of string. By
# the time this runs the abbreviations are expanded, so "Prof." no longer
# looks like the end of a sentence.
#
# Only a full stop *followed* by a digit is protected, which is exactly a
# decimal point ("3.5"). An earlier version refused to split after any digit,
# which looked equivalent and was not: "I earn 52,000. I love it." and "CS3040…
# and" both lost their pause, because the character before the stop was a
# number -- and pay and course-code answers end in numbers constantly.
_SENTENCE = re.compile(r"(.+?(?:[!?…]|\.(?!\d)|$))(?:\s+|$)")


def split_sentences(text: str) -> list:
    """Returns [(sentence, terminal_punctuation), ...]."""
    out = []
    for m in _SENTENCE.finditer(text):
        sentence = m.group(1).strip()
        if not sentence:
            continue
        end = sentence[-1] if sentence[-1] in PAUSE_MS else "."
        out.append((sentence, end))
    return out

log = logging.getLogger("npc.tts")


class VoicePool:
    """Loads Piper voices on demand and keeps them resident.

    Unlike the language-model pool there is no eviction: the voices are small
    next to the adapters, and a demo has at most a handful.
    """

    def __init__(self, voices_dir: Path = VOICES_DIR):
        self.voices_dir = voices_dir
        self._voices = {}
        self._PiperVoice = None
        self.available = self._probe()

    def _probe(self) -> bool:
        try:
            from piper import PiperVoice  # noqa: F401
        except Exception as exc:
            log.warning("piper not importable, NPC speech disabled: %s", exc)
            return False
        if not self.voices_dir.is_dir():
            log.warning("no voices directory at %s, NPC speech disabled", self.voices_dir)
            return False
        return True

    def installed(self) -> list:
        """Voice names with a model file actually present on disk."""
        if not self.voices_dir.is_dir():
            return []
        return sorted(p.stem for p in self.voices_dir.glob("*.onnx"))

    def voice_for(self, npc_name: str | None, requested: str | None) -> str | None:
        """Resolve a voice name, preferring an explicit request, then the NPC's
        assigned voice, then the default -- skipping any that are not on disk."""
        installed = set(self.installed())
        for candidate in (requested, VOICE_BY_NPC.get(npc_name or ""), DEFAULT_VOICE):
            if candidate and candidate in installed:
                return candidate
        return next(iter(sorted(installed)), None)

    def get(self, name: str):
        if name in self._voices:
            return self._voices[name]

        if self._PiperVoice is None:
            from piper import PiperVoice
            self._PiperVoice = PiperVoice

        path = self.voices_dir / (name + ".onnx")
        if not path.exists():
            raise FileNotFoundError("voice model not found: %s" % path)

        start = time.perf_counter()
        voice = self._PiperVoice.load(path)
        log.info("loaded voice %s in %.0f ms", name, (time.perf_counter() - start) * 1000)
        self._voices[name] = voice
        return voice

    def warm(self, names=None) -> None:
        """Load and run one throwaway synthesis per voice.

        The first inference on a fresh ONNX session costs roughly 2 s against a
        steady-state median of ~170 ms. Paying that at startup keeps it off the
        first line a player hears.
        """
        for name in (names if names is not None else self.installed()):
            try:
                self.synthesize("Ready.", name)
            except Exception as exc:
                log.warning("could not warm voice %s: %s", name, exc)

    def synthesize(self, text: str, name: str, style: SpeechStyle = DEFAULT_STYLE) -> dict:
        """Returns 16-bit mono PCM plus timing.

        Speaks the line one sentence at a time rather than handing Piper the
        whole string, so the gap after each sentence can be chosen from how it
        ended (see PAUSE_MS) and scaled by the character's style. Piper's own
        splitting cannot do that: it never tells us what punctuation ended a
        chunk, and it silently drops ellipses.

        Raw PCM rather than a WAV container: Godot builds an AudioStreamWAV
        straight from the samples, so a header would only have to be parsed
        and thrown away on the other side.
        """
        from piper import SynthesisConfig

        voice = self.get(name)
        config = SynthesisConfig(
            length_scale=style.length_scale,
            noise_scale=style.noise_scale,
            noise_w_scale=style.noise_w_scale,
        )
        spoken = normalize_for_speech(text)
        sentences = split_sentences(spoken) or [(spoken, ".")]

        start = time.perf_counter()
        rate = None
        pieces = []
        for i, (sentence, end) in enumerate(sentences):
            audio = []
            for chunk in voice.synthesize(sentence, syn_config=config):
                rate = chunk.sample_rate
                audio.append(chunk.audio_float_array)
            if not audio:
                continue
            pieces.append(_fade(np.concatenate(audio), rate))
            if i < len(sentences) - 1:
                gap = PAUSE_MS.get(end, PAUSE_MS["."]) * style.pause_scale
                pieces.append(np.zeros(int(rate * gap / 1000.0), dtype=np.float32))
        synth_ms = round((time.perf_counter() - start) * 1000, 1)

        if rate is None:
            raise ValueError("piper produced no audio for %r" % text)

        lead = np.zeros(int(rate * LEAD_IN_MS / 1000.0), dtype=np.float32)
        samples = np.concatenate([lead] + pieces)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

        return {
            "pcm": pcm,
            "sample_rate": rate,
            "channels": 1,
            "synth_ms": synth_ms,
            "duration_ms": round(1000.0 * len(samples) / rate, 1),
            "voice": name,
            "spoken_text": spoken,
            "sentences": len(sentences),
        }


def _fade(audio: np.ndarray, rate: int) -> np.ndarray:
    """Short linear ramps at both ends of a piece of audio.

    Joining separately synthesised sentences end to end puts a hard edge at
    each seam, which is audible as a click even when the edge is near zero.
    A few milliseconds of ramp is inaudible as a fade but removes the click.
    """
    n = min(int(rate * FADE_MS / 1000.0), len(audio) // 2)
    if n <= 0:
        return audio
    out = audio.astype(np.float32, copy=True)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def style_for(npc_name: str | None) -> SpeechStyle:
    return STYLE_BY_NPC.get(npc_name or "", DEFAULT_STYLE)
