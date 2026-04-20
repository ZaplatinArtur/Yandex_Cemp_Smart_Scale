#!/usr/bin/env python3
"""
Quantize an ONNX model (dynamic quantization of weights).

Default target is the project's embedder: "assets/models/fruit_embedder_final.onnx".

Usage examples:
  python scripts/quantize_onnx.py -i assets/models/fruit_embedder_final.onnx
  python scripts/quantize_onnx.py -i assets/models/fruit_embedder_final.onnx -o assets/models/fruit_embedder_final_int8.onnx --per-channel

Requirements:
  pip install onnxruntime onnxruntime-tools

This script performs weight-only dynamic quantization (no calibration dataset required).
"""
from __future__ import annotations

import argparse
import sys
import shutil
import time
import inspect
from pathlib import Path

try:
    # quantize_dynamic and QuantType are available in onnxruntime.quantization
    from onnxruntime.quantization import quantize_dynamic, QuantType
except Exception as exc:  # pragma: no cover - runtime dependency
    print("\nERROR: onnxruntime.quantization import failed:\n", exc)
    print("Install with: pip install onnxruntime onnxruntime-tools")
    raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic quantization for ONNX models (weights -> INT8/UINT8)")
    p.add_argument("-i", "--input", type=Path, default=Path("assets/models/fruit_embedder_final.onnx"),
                   help="Input ONNX model path (default: assets/models/fruit_embedder_final.onnx)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output path for quantized model (default: input_int8.onnx)")
    p.add_argument("--weight-type", choices=("qint8", "quint8"), default="qint8",
                   help="Quantized weight type: qint8 (signed) or quint8 (unsigned)")
    p.add_argument("--per-channel", action="store_true", help="Use per-channel quantization for weights (best for conv/linear)")
    p.add_argument("--optimize", action="store_true", help="Run ONNX model optimization step where supported")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    return p.parse_args()


def make_unique_output(src: Path, requested: Path | None, overwrite: bool) -> Path:
    """Return a safe, non-colliding output path that will not overwrite the source.

    If requested is None the output will be created next to the source and have
    suffix `_int8`. If the candidate exists and overwrite is False, a numeric
    suffix will be appended until an unused path is found.
    """
    base_dir = requested.parent if requested else src.parent
    base_name = requested.stem if requested else f"{src.stem}_int8"
    suffix = requested.suffix if requested else src.suffix

    candidate = base_dir / (base_name + suffix)
    i = 0
    while True:
        try:
            cand_resolved = candidate.resolve()
        except Exception:
            cand_resolved = candidate
        try:
            src_resolved = src.resolve()
        except Exception:
            src_resolved = src

        # never allow the destination to be identical to the source
        if cand_resolved == src_resolved:
            i += 1
            candidate = base_dir / f"{base_name}_{i}{suffix}"
            continue

        if candidate.exists() and not overwrite:
            i += 1
            candidate = base_dir / f"{base_name}_{i}{suffix}"
            continue

        return candidate


def main() -> None:
    args = parse_args()
    src: Path = args.input
    if not src.exists():
        print(f"Input model not found: {src}")
        raise SystemExit(2)

    # choose a safe output path (do not overwrite source)
    dst = make_unique_output(src, args.output, args.overwrite)

    weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8

    print(f"Quantizing: {src} -> {dst}")
    print(f"  weight_type={args.weight_type}, per_channel={args.per_channel}, optimize={args.optimize}")

    # create a lightweight backup next to original (only if not present)
    try:
        backup = src.with_name(src.name + ".bak")
        if not backup.exists():
            print(f"Creating backup: {backup}")
            shutil.copy2(src, backup)
    except Exception as e:
        print("Warning: failed to create backup:", e)

    # quantize into a temporary file first, then atomically move into place
    tmp_out = dst.with_name(dst.name + ".tmp")
    if tmp_out.exists():
        try:
            tmp_out.unlink()
        except Exception:
            pass

    # build kwargs according to quantize_dynamic signature
    sig = inspect.signature(quantize_dynamic)
    kwargs = {}
    if "model_input" in sig.parameters:
        kwargs["model_input"] = str(src)
        kwargs["model_output"] = str(tmp_out)
    else:
        # older signatures accept positional args; we'll call with positional later
        pass

    if "weight_type" in sig.parameters:
        kwargs["weight_type"] = weight_type

    if "per_channel" in sig.parameters:
        kwargs["per_channel"] = args.per_channel

    if "optimize_model" in sig.parameters:
        kwargs["optimize_model"] = args.optimize

    try:
        if kwargs:
            quantize_dynamic(**kwargs)
        else:
            # fallback to positional call
            quantize_dynamic(str(src), str(tmp_out), weight_type=weight_type, per_channel=args.per_channel)
    except Exception as exc:
        # cleanup any partial output and surface a helpful message
        if tmp_out.exists():
            try:
                tmp_out.unlink()
            except Exception:
                pass
        print("\nQuantization failed:")
        print(exc)
        print("\nNote: original model has not been overwritten. See backup next to the original if created.")
        print("You can try: --per-channel, different onnxruntime version, or run shape inference manually.")
        raise SystemExit(1)

    # move temporary file into final destination
    try:
        # prefer atomic replace
        tmp_out.replace(dst)
    except Exception:
        shutil.move(str(tmp_out), str(dst))

    print("Quantization finished. Output:", dst)


if __name__ == "__main__":
    main()
