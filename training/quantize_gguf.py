"""
Quantize an f16 GGUF to Q4_K_M using llama-cpp-python's bound
llama_model_quantize() directly — no compiled llama-quantize.exe needed
(no C/C++ toolchain on this machine, see Docs/TODO.md known issues).

Usage:
    python training/quantize_gguf.py training/gguf_models/medieval_r8_gutonly-f16.gguf
"""

import argparse
import ctypes
import os
import sys
from pathlib import Path


def _add_torch_cuda_dlls_to_path():
    # llama-cpp-python here is the CUDA build (installed from abetlen's
    # cu121 wheel index for the GGUF latency work) — its llama.dll needs
    # cudart64_12.dll/cublas64_12.dll on PATH even for CPU-only use like
    # quantization. No standalone CUDA toolkit on this machine, so reuse
    # torch's own cu121 wheel's bundled copies instead. See Docs/TODO.md.
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")


_add_torch_cuda_dlls_to_path()
import llama_cpp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_gguf")
    parser.add_argument("--outtype", default="q4_k_m", choices=["q4_k_m", "q8_0", "q4_0"])
    args = parser.parse_args()

    ftype_map = {
        "q4_k_m": llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M,
        "q8_0": llama_cpp.LLAMA_FTYPE_MOSTLY_Q8_0,
        "q4_0": llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_0,
    }

    in_path = Path(args.input_gguf)
    out_path = in_path.with_name(in_path.name.replace("-f16", f"-{args.outtype.upper()}"))

    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = ftype_map[args.outtype]

    print(f"quantizing {in_path.name} -> {out_path.name} ({args.outtype})")
    result = llama_cpp.llama_model_quantize(
        str(in_path).encode("utf-8"), str(out_path).encode("utf-8"), ctypes.byref(params)
    )
    if result != 0:
        raise RuntimeError(f"llama_model_quantize failed, code {result}")

    print(f"done -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
