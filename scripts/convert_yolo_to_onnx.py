#!/usr/bin/env python3
"""
Convert a YOLO `.pt` model to ONNX.

This script first tries to use the `ultralytics` package (recommended). If that's
not available it attempts a generic `torch.jit` / `torch.onnx` export when the
.pt file contains a scripted module or a saved `nn.Module` instance.

Usage:
  python scripts/convert_yolo_to_onnx.py \
    --weights "C:\\Users\\artur\\Desktop\\Yandex_Cemp\\Yandex_Cemp_Smart_Scale\\assets\\models\\yolo11n-seg.pt" \
    --out ./yolo.onnx --imgsz 640 --opset 16 --dynamic
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Optional


def _find_new_onnx(before: set[str], search_dirs: list[Path]) -> Optional[Path]:
    after = set(p.resolve() for d in search_dirs for p in d.rglob("*.onnx") if p.exists())
    new = after - before
    return Path(next(iter(new))) if new else None


def export_with_ultralytics(weights: Path, out: Path, imgsz: int, opset: int, dynamic: bool, simplify: bool, device: str, verbose: bool) -> bool:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:  # pragma: no cover - runtime dependency
        if verbose:
            print("ultralytics not available:", e)
        return False

    if verbose:
        print("Using ultralytics.YOLO to export the model (if supported)")

    before = set(p.resolve() for p in Path('.').rglob('*.onnx'))
    try:
        model = YOLO(str(weights))
        # newer versions support many kwargs; pass common ones and fall back if signature differs
        try:
            model.export(format='onnx', imgsz=imgsz, opset=opset, simplify=simplify, dynamic=dynamic, device=device)
        except TypeError:
            model.export(format='onnx', imgsz=imgsz, opset=opset)
    except Exception as e:
        print("ultralytics export failed:", e, file=sys.stderr)
        return False

    # Give filesystem a tiny moment to flush
    time.sleep(0.1)

    # Look for newly created ONNX files
    candidate = _find_new_onnx(before, [Path('.'), weights.parent])
    if candidate is None:
        # try some common ultralytics export run folders
        candidate = _find_new_onnx(before, [Path('runs')])

    if candidate is None:
        print("No ONNX file found after ultralytics export.", file=sys.stderr)
        return False

    try:
        candidate.rename(out)
    except Exception:
        shutil.copy2(candidate, out)

    print(f"ONNX model saved to {out}")
    return True


def export_with_torch(weights: Path, out: Path, imgsz: int, opset: int, dynamic: bool, simplify: bool, device: str, verbose: bool) -> bool:
    try:
        import torch
    except Exception as e:  # pragma: no cover - runtime dependency
        print("PyTorch is not installed; please install torch.", file=sys.stderr)
        if verbose:
            print(e, file=sys.stderr)
        return False

    device_obj = torch.device('cpu' if device == 'cpu' else device)

    model = None
    # Try loading a scripted module first
    try:
        model = torch.jit.load(str(weights), map_location=device_obj)
        if verbose:
            print("Loaded scripted model with torch.jit.load")
    except Exception:
        # Try torch.load and hope it's an nn.Module instance
        try:
            ckpt = torch.load(str(weights), map_location=device_obj)
            if isinstance(ckpt, torch.nn.Module):
                model = ckpt
                if verbose:
                    print("Loaded nn.Module instance from checkpoint")
            elif isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], torch.nn.Module):
                model = ckpt['model']
                if verbose:
                    print("Found 'model' nn.Module inside checkpoint dict")
            else:
                print("Checkpoint doesn't contain a torch.nn.Module instance; cannot export with generic torch exporter.", file=sys.stderr)
                return False
        except Exception as e:
            print("Failed to load model with torch:", e, file=sys.stderr)
            return False

    model.eval()
    try:
        model.to(device_obj)
    except Exception:
        if verbose:
            print("Failed to move model to requested device, continuing on CPU")
        model.to(torch.device('cpu'))

    batch = 1
    dummy = torch.zeros(batch, 3, imgsz, imgsz, dtype=torch.float32, device=device_obj)

    input_names = ['images']
    output_names = ['output']
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'}, 'output': {0: 'batch'}}

    try:
        torch.onnx.export(
            model,
            dummy,
            str(out),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    except Exception as e:
        print("torch.onnx.export failed:", e, file=sys.stderr)
        return False

    print(f"ONNX model saved to {out}")

    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            onnx_model = onnx.load(str(out))
            model_simp, check = onnx_simplify(onnx_model)
            if check:
                onnx.save(model_simp, str(out))
                print("ONNX model simplified")
            else:
                print("ONNX simplifier check failed; original file kept", file=sys.stderr)
        except Exception as e:
            print("ONNX simplification failed (install onnx and onnx-simplifier to enable):", e, file=sys.stderr)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a YOLO .pt model to ONNX")
    default_weights = r"C:\Users\artur\Desktop\Yandex_Cemp\Yandex_Cemp_Smart_Scale\assets\models\yolo11n-seg.pt"
    parser.add_argument('--weights', '-w', default=default_weights, help='Path to .pt weights')
    parser.add_argument('--out', '-o', default=None, help='Output ONNX path')
    parser.add_argument('--imgsz', '-s', type=int, default=640, help='Image size (square)')
    parser.add_argument('--opset', '-p', type=int, default=16, help='ONNX opset version')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic axes for height/width')
    parser.add_argument('--simplify', action='store_true', help='Run onnx-simplifier after export')
    parser.add_argument('--device', default='cpu', help='Device for torch export (cpu or cuda:0)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"Weights not found: {weights}", file=sys.stderr)
        sys.exit(2)

    out = Path(args.out) if args.out else weights.with_suffix('.onnx')
    out.parent.mkdir(parents=True, exist_ok=True)

    # Try ultralytics first (best results for common YOLO releases)
    if export_with_ultralytics(weights, out, args.imgsz, args.opset, args.dynamic, args.simplify, args.device, args.verbose):
        return

    # Fallback to generic torch-based export (requires scripted or module instance)
    if export_with_torch(weights, out, args.imgsz, args.opset, args.dynamic, args.simplify, args.device, args.verbose):
        return

    print("Failed to export model. Install 'ultralytics' or provide a scripted .pt model.", file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
    main()
