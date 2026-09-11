"""Download Piper voices for the NPC demo, with size verification and retry.

Piper's own `download_voice` does not check that the file it wrote is the size
the index says it should be. On a slow link this fails silently: we ended up
with a 6.6 MB file where 63.1 MB was expected, and the only symptom was
`InvalidProtobuf: Protobuf parsing failed` at load time, several steps later.
Verifying here turns that into an immediate, obvious error.

Usage:
    .venv/Scripts/python.exe backend/fetch_voices.py
    .venv/Scripts/python.exe backend/fetch_voices.py en_US-amy-medium
"""
import json
import sys
import urllib.request
from pathlib import Path

VOICES_JSON = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
               "voices.json?download=true")
FILE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/{path}?download=true"
VOICES_DIR = Path(__file__).resolve().parent / "voices"

# One distinct speaker per NPC. Names must match VOICE_BY_NPC in tts.py.
DEFAULT_VOICES = [
    "en_GB-alan-medium",
    "en_GB-jenny_dioco-medium",
    "en_US-amy-medium",
    "en_GB-northern_english_male-medium",
    "en_US-lessac-medium",
]

RETRIES = 3


def _fetch(url: str, dest: Path, expected: int | None) -> None:
    for attempt in range(1, RETRIES + 1):
        with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        got = dest.stat().st_size
        if expected is None or got == expected:
            return
        print("    attempt %d: got %d bytes, expected %d - retrying"
              % (attempt, got, expected))
    raise IOError("%s: still wrong size after %d attempts (%d != %d)"
                  % (dest.name, RETRIES, dest.stat().st_size, expected))


def download(name: str, index: dict) -> None:
    entry = index.get(name)
    if entry is None:
        raise KeyError("voice %r is not in the Piper index" % name)

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    for remote_path, meta in entry["files"].items():
        if not (remote_path.endswith(".onnx") or remote_path.endswith(".onnx.json")):
            continue
        suffix = ".onnx.json" if remote_path.endswith(".onnx.json") else ".onnx"
        dest = VOICES_DIR / (name + suffix)
        expected = meta.get("size_bytes")

        if dest.exists() and (expected is None or dest.stat().st_size == expected):
            print("  have %s" % dest.name)
            continue

        print("  fetching %s (%.1f MB)..." % (dest.name, (expected or 0) / 1e6))
        _fetch(FILE_URL.format(path=remote_path), dest, expected)
        print("    ok, %d bytes" % dest.stat().st_size)


def main() -> int:
    wanted = sys.argv[1:] or DEFAULT_VOICES
    print("loading Piper voice index...")
    index = json.loads(urllib.request.urlopen(VOICES_JSON, timeout=120).read())

    failed = []
    for name in wanted:
        print("%s" % name)
        try:
            download(name, index)
        except Exception as exc:
            failed.append(name)
            print("  FAILED: %s: %s" % (type(exc).__name__, exc))

    print("\n%d/%d voices ready in %s" % (len(wanted) - len(failed), len(wanted), VOICES_DIR))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
