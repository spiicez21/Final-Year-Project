"""
Week 11 (Docs/TODO.md): latency / peak-RAM / adapter-storage benchmarks.
Peak RAM measured via psutil, sampling this process's RSS on a background
thread throughout model load + generation (not just before/after snapshots,
which would miss a transient peak mid-load).

Usage:
    python evaluation/run_benchmarks.py
"""

import os
import statistics
import sys
import threading
import time
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
GGUF_MODELS_DIR = REPO_ROOT / "training" / "gguf_models"
ADAPTERS_DIR = REPO_ROOT / "training" / "adapters"
MERGED_DIR = REPO_ROOT / "training" / "merged_models"

CONDITIONS = [
    ("A: baseline (no adapter)", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", None),
    ("B: police officer adapter (merged GGUF)", "modern_r16_a32_policeofficer-Q4_K_M.gguf", "modern_r16_a32_policeofficer"),
]

PROMPTS = [
    ("police officer", "Excuse me, can you tell me how to get to the train station?"),
    ("police officer", "Someone just stole my bike, can you help?"),
    ("police officer", "Is it okay to park here for a few minutes?"),
]

SYSTEM_TEMPLATE = ("You are a {archetype} NPC in a modern city. Respond in a natural, contemporary "
                    "voice consistent with your role. Never break character.")


def _add_torch_cuda_dlls_to_path():
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")


class PeakRSSMonitor:
    """Samples this process's RSS on a background thread so a transient peak
    (e.g. during model load) isn't missed by a simple before/after snapshot."""

    def __init__(self, interval=0.1):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            rss = self.process.memory_info().rss
            self.peak_bytes = max(self.peak_bytes, rss)
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def adapter_storage_mb(adapter_dir_name: str) -> float:
    path = ADAPTERS_DIR / adapter_dir_name / "adapter_model.safetensors"
    return path.stat().st_size / 1e6 if path.exists() else None


def gguf_storage_mb(gguf_filename: str) -> float:
    path = GGUF_MODELS_DIR / gguf_filename
    return path.stat().st_size / 1e6 if path.exists() else None


def main():
    _add_torch_cuda_dlls_to_path()
    from llama_cpp import Llama

    print(f"{'Condition':45s} {'Peak RSS (MB)':>15s} {'Mean latency (ms)':>20s} {'Storage':>15s}")
    print("-" * 100)

    for label, gguf_filename, adapter_name in CONDITIONS:
        with PeakRSSMonitor() as monitor:
            model_path = GGUF_MODELS_DIR / gguf_filename
            llm = Llama(model_path=str(model_path), n_ctx=2048, verbose=False, n_gpu_layers=0,
                        n_threads=os.cpu_count(), n_batch=512, chat_format="zephyr")

            latencies = []
            for archetype, prompt in PROMPTS:
                system = SYSTEM_TEMPLATE.format(archetype=archetype)
                messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
                start = time.perf_counter()
                llm.create_chat_completion(messages=messages, max_tokens=40, temperature=0.0)
                latencies.append((time.perf_counter() - start) * 1000)

            del llm

        storage = f"{gguf_storage_mb(gguf_filename):.1f}MB (GGUF)"
        if adapter_name:
            adapter_mb = adapter_storage_mb(adapter_name)
            storage += f" / {adapter_mb:.1f}MB (adapter)"

        print(f"{label:45s} {monitor.peak_bytes / 1e6:15.1f} {statistics.mean(latencies):20.1f} {storage:>15s}")

    print("\nReference (archived, not re-measured this run): full fine-tune = 11,860MB on disk (Docs/TODO.md).")
    print("Note: peak RSS includes the whole Python process (interpreter + llama.cpp model + KV cache),")
    print("not model weights alone — this is the honest 'what does running this actually cost' number.")


if __name__ == "__main__":
    main()
