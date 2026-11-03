#!/usr/bin/env python3
"""Fix the Metal shader in llama.cpp master that breaks the build on macOS.

LFM3 support landed in llama.cpp master (PR #31252), but the Metal backend fails to
compile with:

    error: constexpr function never produces a constant expression
    inline constexpr bfloat4 ggml_metal_unary_cast<bfloat4, float4>(...)

MSL cannot constant-fold the bfloat4 constructor, so the two bf16 cast
specializations must not be `constexpr`. This drops the keyword. Idempotent: safe to
run on an already patched tree, and a no-op if upstream has fixed it another way.

Usage: python3 scripts/fix-metal-bf16.py ~/llama.cpp-src
"""
import os
import sys

REL = "ggml/src/ggml-metal/kernels/unary.metal"
BAD = "inline constexpr bfloat4 ggml_metal_unary_cast"
GOOD = "inline bfloat4 ggml_metal_unary_cast"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/llama.cpp-src")
    path = os.path.join(root, REL)
    if not os.path.exists(path):
        print(f"skip: {path} not found (upstream may have moved it)")
        return 0

    src = open(path, encoding="utf-8").read()
    n = src.count(BAD)
    if n == 0:
        print(f"ok: nothing to patch in {REL} (already fixed)")
        return 0

    open(path, "w", encoding="utf-8").write(src.replace(BAD, GOOD))
    print(f"patched {n} bf16 cast specialization(s) in {REL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
