#!/usr/bin/env python3
"""
Quantize a YOLO ONNX model.

Supports two modes:
 - dynamic: fast weight-only dynamic quantization (no calibration)
 - static: full static quantization using a calibration dataset (requires --calib-data)

This script never overwrites the original model unless --overwrite is provided.
It writes a new file next to the source (suffix `_int8`, `_int8_1`, etc.) and
writes to a temporary file first.

Examples:
  python scripts/quantize_yolo.py -i assets/models/yolo.onnx
  python scripts/quantize_yolo.py -i assets/models/yolo.onnx --mode static --calib-data data/calib_images --per-channel

Requirements:
  pip install onnxruntime onnxruntime-tools pillow numpy

Note: static quantization requires reasonable calibration images that match
the model's input preprocessing. The included simple reader resizes images
to the model input spatial size (if known) and normalizes to [0,1]. You may
need to adapt preprocessing to match your YOLO export.
"""
from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    from onnxruntime.quantization import (
        quantize_dynamic,
        quantize_static,
        QuantType,
        QuantFormat,
    )
except Exception:  # pragma: no cover - runtime dependency
    # quantize_static may not be available in minimal installs
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_static = None
    QuantFormat = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantize YOLO ONNX model (dynamic|static)")
    p.add_argument("-i", "--input", type=Path, default=Path("assets/models/yolo.onnx"), help="Input ONNX model")
    p.add_argument("-o", "--output", type=Path, default=None, help="Desired output path (script will avoid overwriting)")
    p.add_argument("--mode", choices=("dynamic", "static"), default="dynamic", help="Quantization mode")
    p.add_argument("--per-channel", action="store_true", help="Use per-channel quantization where supported")
    p.add_argument("--weight-type", choices=("qint8", "quint8"), default="qint8")
    p.add_argument("--calib-data", type=Path, default=None, help="Calibration images folder (for static mode)")
    p.add_argument("--calib-max", type=int, default=128, help="Max calibration images to use (static mode)")
    p.add_argument("--optimize", action="store_true", help="Run model optimization step where supported")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting output file if it already exists")
    return p.parse_args()


def make_unique_output(src: Path, requested: Optional[Path], overwrite: bool) -> Path:
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
        if cand_resolved == src_resolved:
            i += 1
            candidate = base_dir / f"{base_name}_{i}{suffix}"
            continue
        if candidate.exists() and not overwrite:
            i += 1
            candidate = base_dir / f"{base_name}_{i}{suffix}"
            continue
        return candidate


class YOLOCalibrationDataReader:
    """Simple calibration data reader for static quantization.

    It resizes images to the desired `size` and returns a dict mapping the
    model input name to a numpy array with shape (1, C, H, W) and dtype float32.
    """

    def __init__(self, folder: Path, input_name: str, size: tuple[int, int] = (640, 640), max_images: int | None = None):
        self.input_name = input_name
        images = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if max_images:
            images = images[:max_images]
        self.files = images
        self.size = size
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self.files):
            return None
        p = self.files[self._idx]
        self._idx += 1
        img = Image.open(p).convert("RGB")
        img = img.resize(self.size, Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        # return shape (1, C, H, W)
        arr = arr.transpose(2, 0, 1)[None, ...]
        return {self.input_name: arr}


def main() -> None:
    args = parse_args()
    src = args.input
    if not src.exists():
        print(f"Input model not found: {src}")
        raise SystemExit(2)

    dst = make_unique_output(src, args.output, args.overwrite)
    weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8

    print(f"Quantizing ({args.mode}): {src} -> {dst}")

    tmp_out = dst.with_name(dst.name + ".tmp")
    if tmp_out.exists():
        try:
            tmp_out.unlink()
        except Exception:
            pass

    if args.mode == "dynamic":
        sig = inspect.signature(quantize_dynamic)
        kwargs = {}
        if "model_input" in sig.parameters:
            kwargs["model_input"] = str(src)
            kwargs["model_output"] = str(tmp_out)
        else:
            # positional fallback will be used
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
                quantize_dynamic(str(src), str(tmp_out), weight_type=weight_type, per_channel=args.per_channel)
        except Exception as exc:
            if tmp_out.exists():
                try:
                    tmp_out.unlink()
                except Exception:
                    pass
            print("Quantization (dynamic) failed:\n", exc)
            raise SystemExit(1)

    else:  # static
        if quantize_static is None:
            print("Static quantization is not available in this environment. Install onnxruntime-tools.")
            raise SystemExit(2)
        if not args.calib_data or not args.calib_data.exists():
            print("Static quantization requires --calib-data pointing to a folder with images.")
            raise SystemExit(2)

        try:
            import onnxruntime as ort
        except Exception as exc:
            print("onnxruntime is required for static quantization and input discovery:\n", exc)
            raise SystemExit(2)

        sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
        input_meta = sess.get_inputs()[0]
        input_name = input_meta.name
        # attempt to infer spatial dims
        shape = input_meta.shape
        h = 640
        w = 640
        try:
            if len(shape) >= 4:
                maybe_h = shape[2]
                maybe_w = shape[3]
                if isinstance(maybe_h, int) and isinstance(maybe_w, int):
                    h, w = int(maybe_h), int(maybe_w)
        except Exception:
            pass

        reader = YOLOCalibrationDataReader(args.calib_data, input_name, size=(h, w), max_images=args.calib_max)

        sig = inspect.signature(quantize_static)
        qkwargs = {}
        if "model_input" in sig.parameters:
            qkwargs["model_input"] = str(src)
            qkwargs["model_output"] = str(tmp_out)
        if "calibration_data_reader" in sig.parameters:
            qkwargs["calibration_data_reader"] = reader
        elif "calibration" in sig.parameters:
            qkwargs["calibration"] = reader
        if "quant_format" in sig.parameters and QuantFormat is not None:
            qkwargs["quant_format"] = QuantFormat.QOperator
        if "per_channel" in sig.parameters:
            qkwargs["per_channel"] = args.per_channel
        if "activation_type" in sig.parameters:
            # use unsigned activations by default
            qkwargs["activation_type"] = QuantType.QUInt8
        if "weight_type" in sig.parameters:
            qkwargs["weight_type"] = weight_type
        if "optimize_model" in sig.parameters:
            qkwargs["optimize_model"] = args.optimize

        try:
            quantize_static(**qkwargs)
        except Exception as exc:
            if tmp_out.exists():
                try:
                    tmp_out.unlink()
                except Exception:
                    pass
            print("Quantization (static) failed:\n", exc)
            print("Original model preserved; consider adjusting calibration images or using --per-channel.")
            raise SystemExit(1)

    # move tmp into final dst
    try:
        tmp_out.replace(dst)
    except Exception:
        shutil.move(str(tmp_out), str(dst))

    print("Quantization finished. Output:", dst)


if __name__ == "__main__":
    main()
