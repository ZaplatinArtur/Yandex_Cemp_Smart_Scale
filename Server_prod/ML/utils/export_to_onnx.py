#!/usr/bin/env python3
"""
Export a FruitEmbedder model checkpoint to ONNX.

Usage (run from workspace root):
  pip install -r Yandex_Cemp_Smart_Scale/requirements.txt  # if needed
  pip install transformers onnx onnxruntime Pillow
  python Yandex_Cemp_Smart_Scale/Server_prod/ML/export_to_onnx.py --checkpoint fruit_embedder_final.pth --output fruit_embedder_final.onnx

The script tries to handle several checkpoint formats:
 - full checkpoint dict with keys like 'head_state_dict' and 'backbone_state_dict'
 - dict containing 'state_dict' or 'model_state_dict'
 - raw state_dict (saved by model.state_dict())

If export fails due to unsupported ops, try increasing/decreasing --opset or use the HuggingFace/Optimum export helpers.
"""

import os
import sys
import argparse
import torch

# Ensure local imports from this directory work when running from repo root
sys.path.append(os.path.dirname(__file__))

from vector_model import FruitEmbedder
from transformers import AutoImageProcessor


def build_model_from_checkpoint(checkpoint_path, device=torch.device('cpu')):
    loaded = torch.load(checkpoint_path, map_location=device)

    # Case A: raw state_dict (mapping param_name->tensor)
    if isinstance(loaded, dict) and any(isinstance(v, torch.Tensor) for v in loaded.values()) and not any(k in loaded for k in ('head_state_dict','backbone_state_dict','state_dict','model_state_dict','backbone_name')):
        print("Detected raw state_dict -> will try to load into model (non-strict if needed)")
        backbone_name = "facebook/dinov2-small"
        embedding_dim = 256
        model = FruitEmbedder(backbone_name=backbone_name, embedding_dim=embedding_dim)
        try:
            model.load_state_dict(loaded)
        except Exception as e:
            print(f"  Warning: strict load failed: {e}. Trying non-strict load.")
            model.load_state_dict(loaded, strict=False)
        return model, backbone_name, embedding_dim

    # Case B: checkpoint dict with structured keys
    if isinstance(loaded, dict):
        backbone_name = loaded.get('backbone_name', 'facebook/dinov2-small')
        embedding_dim = loaded.get('embedding_dim', 256)
        model = FruitEmbedder(backbone_name=backbone_name, embedding_dim=embedding_dim)

        # Try common keys
        if 'head_state_dict' in loaded:
            try:
                model.head.load_state_dict(loaded['head_state_dict'])
            except Exception as e:
                print(f"  Warning: head_state_dict load failed: {e}. Trying non-strict load.")
                model.head.load_state_dict(loaded['head_state_dict'], strict=False)

        if 'backbone_state_dict' in loaded:
            try:
                model.backbone.load_state_dict(loaded['backbone_state_dict'])
            except Exception as e:
                print(f"  Warning: backbone_state_dict load failed: {e} (ignoring)")

        # If the checkpoint keeps a full model/state dict under common keys
        for key in ('state_dict', 'model_state_dict'):
            if key in loaded:
                try:
                    model.load_state_dict(loaded[key])
                except Exception as e:
                    print(f"  Warning: model.load_state_dict('{key}') failed: {e} (trying non-strict)")
                    model.load_state_dict(loaded[key], strict=False)

        return model, backbone_name, embedding_dim

    raise RuntimeError("Unsupported checkpoint format: expected state_dict or checkpoint dict.")


def export_to_onnx(model: torch.nn.Module, output_path: str, opset: int = 16):
    model_cpu = model.to('cpu')
    model_cpu.eval()

    # Typical DINOv2 preproc yields 224x224; use that by default
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    with torch.no_grad():
        try:
            torch.onnx.export(
                model_cpu,
                (dummy,),
                output_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=['pixel_values'],
                output_names=['embedding'],
                dynamic_axes={'pixel_values': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}},
            )
            print(f"ONNX saved: {output_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")
            raise


def try_validate_onnx(output_path: str, dummy_tensor: torch.Tensor = None):
    try:
        import onnx
        m = onnx.load(output_path)
        onnx.checker.check_model(m)
        print("ONNX model is valid (onnx.checker)")
    except Exception as e:
        print("ONNX validation failed:", e)

    if dummy_tensor is None:
        return

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        inp_name = sess.get_inputs()[0].name
        out = sess.run(None, {inp_name: dummy_tensor.numpy()})
        print("ONNXRuntime inference ok; output shape:", out[0].shape)
    except Exception as e:
        print("onnxruntime inference failed:", e)


def main():
    parser = argparse.ArgumentParser(description='Export FruitEmbedder checkpoint to ONNX')
    parser.add_argument('--checkpoint', required=True, help='path to .pth checkpoint')
    parser.add_argument('--output', default='model.onnx', help='output ONNX file')
    parser.add_argument('--opset', type=int, default=16, help='ONNX opset version')
    parser.add_argument('--device', default='cpu', help='device for loading model (cpu/cuda)')
    parser.add_argument('--validate', action='store_true', help='run basic ONNX validation (requires onnx/onnxruntime)')

    args = parser.parse_args()

    device = torch.device(args.device if args.device in ('cpu', 'cuda') else 'cpu')

    print(f"Loading checkpoint: {args.checkpoint} (device={device})")
    model, backbone_name, embedding_dim = build_model_from_checkpoint(args.checkpoint, device=device)
    print(f"Model constructed: backbone={backbone_name}, embedding_dim={embedding_dim}")

    print("Exporting to ONNX...")
    export_to_onnx(model, args.output, opset=args.opset)

    if args.validate:
        try:
            dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)
            try_validate_onnx(args.output, dummy)
        except Exception:
            pass


if __name__ == '__main__':
    main()
