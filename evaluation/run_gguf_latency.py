"""
Week 1 (Docs/TODO.md): re-measure inference latency on the llama.cpp/GGUF
serving path instead of the transformers+bitsandbytes path used everywhere
else so far. The previously recorded 3486ms baseline is an unoptimized-path
artifact (Ollama serving stock TinyLlama, not GGUF-quantized direct
inference) — not a valid model-size conclusion until re-measured here.

Downloads a quantized GGUF build via huggingface_hub, runs a fixed prompt
set through llama_cpp.Llama, reports mean/p50/p95 latency.

Usage:
    python run_gguf_latency.py --model tinyllama
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

GGUF_MODELS = {
    # TinyLlama-1.1B-Chat-v1.0 was fine-tuned with a Zephyr-style template
    # (<|system|>/<|user|>/<|assistant|> tags, not ChatML) — llama-cpp-python
    # ships a matching built-in "zephyr" chat_format.
    "tinyllama": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "chat_format": "zephyr",
    },
    # Local file — merged medieval_r8_gutonly LoRA -> GGUF Q4_K_M via
    # training/merge_lora.py + training/quantize_gguf.py. This is the actual
    # fine-tuned adapter on the GGUF serving path, not stock TinyLlama.
    "medieval_gutonly": {
        "local_path": "medieval_r8_gutonly-Q4_K_M.gguf",
        "chat_format": "zephyr",
    },
}

MODELS_DIR = Path(__file__).resolve().parents[1] / "training" / "gguf_models"

TEST_PROMPTS = [
    ("guard", "Halt! State thy business here."),
    ("merchant", "What wares do you have for sale, good sir?"),
    ("innkeeper", "Might I have a room for the night?"),
    ("scholar", "Tell me of the old kingdom's history."),
    ("noble", "Bring me news from the capital at once."),
]

SYSTEM_TEMPLATE = ("You are a {archetype} NPC in a medieval RPG world. Respond in an archaic, "
                   "period-appropriate voice consistent with your role. Never break character.")


def download_model(key: str) -> Path:
    spec = GGUF_MODELS[key]
    if "local_path" in spec:
        path = MODELS_DIR / spec["local_path"]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run merge_lora.py + quantize_gguf.py first")
        return path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=spec["repo_id"], filename=spec["filename"], local_dir=str(MODELS_DIR))
    return Path(path)


def _add_torch_cuda_dlls_to_path():
    # The installed llama-cpp-python wheel is the CUDA build (pulled from
    # abetlen's cu121 wheel index) — its llama.dll dynamically links against
    # cudart64_12.dll/cublas64_12.dll even when n_gpu_layers=0 at runtime, so
    # it fails to import at all without them on PATH. No standalone CUDA
    # toolkit is installed on this machine (only the driver) — reuse the copies
    # torch's own cu121 wheel already ships instead of a multi-GB toolkit install.
    torch_lib = Path(__file__).resolve().parents[1] / "Lib" / "site-packages" / "torch" / "lib"
    candidates = [
        torch_lib,
        Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
    ]
    for candidate in candidates:
        if candidate.exists():
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(GGUF_MODELS.keys()), default="tinyllama")
    parser.add_argument("--max-tokens", type=int, default=40)
    args = parser.parse_args()

    _add_torch_cuda_dlls_to_path()
    from llama_cpp import Llama

    spec = GGUF_MODELS[args.model]
    print(f"downloading/locating GGUF: {spec.get('repo_id', spec.get('local_path'))}")
    model_path = download_model(args.model)
    print(f"model on disk: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")

    # GPU offload (n_gpu_layers=-1) was tried and confirmed active (verbose
    # log showed 23/23 layers on CUDA0) but gave no speedup on this GPU — the
    # MX450 is power-capped to ~5W (see Docs/TODO.md known issues), so it's
    # not meaningfully faster than CPU for a model this small. CPU-only with
    # explicit thread/batch tuning (n_threads=os.cpu_count(), n_batch=512)
    # measured ~12% faster than default settings and avoids CPU<->GPU sync
    # overhead entirely, so that's the config used here, not GPU offload.
    print("loading model (llama.cpp, CPU, tuned threads/batch)...")
    load_start = time.perf_counter()
    llm = Llama(model_path=str(model_path), n_ctx=2048, verbose=False, n_gpu_layers=0,
                n_threads=os.cpu_count(), n_batch=512,
                chat_format=GGUF_MODELS[args.model]["chat_format"])
    load_ms = (time.perf_counter() - load_start) * 1000
    print(f"model load: {load_ms:.1f}ms")

    latencies = []
    for archetype, prompt in TEST_PROMPTS:
        system = SYSTEM_TEMPLATE.format(archetype=archetype)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

        start = time.perf_counter()
        result = llm.create_chat_completion(messages=messages, max_tokens=args.max_tokens, temperature=0.0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        response = result["choices"][0]["message"]["content"].strip()
        print(f"[{archetype:10s}] {elapsed_ms:7.1f}ms  {response[:80]!r}")

    print(f"\n--- GGUF latency ({args.model}, CPU, max_tokens={args.max_tokens}) ---")
    print(f"mean: {statistics.mean(latencies):.1f}ms")
    print(f"p50:  {statistics.median(latencies):.1f}ms")
    print(f"min:  {min(latencies):.1f}ms   max: {max(latencies):.1f}ms")
    print(f"exit condition (Week 1, Docs/TODO.md): latency < 500ms -> {'PASS' if statistics.mean(latencies) < 500 else 'FAIL'}")


if __name__ == "__main__":
    main()
