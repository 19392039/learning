#!/usr/bin/env python3
"""Diagnose the Python package and compiled extension used by the ATK worker."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys


MODULES = (
    "torch",
    "torch_npu",
    "atk",
    "flash_attn_npu_v3",
    "flash_attn_npu_3",
    "flash_attn_npu_v3.flash_attn_interface",
)


def main() -> int:
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    print(f"FLASH_ATTN_NPU_REPO: {os.environ.get('FLASH_ATTN_NPU_REPO', '<unset>')}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '<unset>')}")

    failures = 0
    for name in MODULES:
        try:
            spec = importlib.util.find_spec(name)
        except Exception as exc:
            print(f"[FAIL] find_spec {name}: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        if spec is None:
            print(f"[MISS] {name}")
            failures += 1
        else:
            print(f"[ OK ] {name}: {spec.origin}")

    try:
        interface = importlib.import_module(
            "flash_attn_npu_v3.flash_attn_interface"
        )
        func = interface.flash_attn_func
        print(f"flash_attn_func file: {inspect.getsourcefile(func)}")
        print(f"flash_attn_func signature: {inspect.signature(func)}")
    except Exception as exc:
        print(f"[FAIL] import API: {type(exc).__name__}: {exc}")
        failures += 1

    try:
        extension = importlib.import_module("flash_attn_npu_3")
        exports = [name for name in dir(extension) if not name.startswith("_")]
        print(f"extension exports: {exports}")
        print(f"has fwd: {hasattr(extension, 'fwd')}")
        print(f"has bwd: {hasattr(extension, 'bwd')}")
    except Exception as exc:
        print(f"[FAIL] import extension: {type(exc).__name__}: {exc}")
        failures += 1

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
