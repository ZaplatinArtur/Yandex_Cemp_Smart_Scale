"""
Detector stub

This module provides a minimal `Detector` class that forwards a photo
to the `FruitSearchEngine` defined in `vector_model.py` and prints the
search results (including metadata). Use this as a placeholder while
the real detector/hardware integration is implemented.

Usage (from Server_prod/ML):
  python detector.py --model "C:\\Users\\artur\\Desktop\\Yandex_Cemp\\fruit_embedder_final.pth" \
	--db "Server_prod/DB/fruits.db" --image "C:\\path\\to\\img.jpg" --top 5
"""

import os
import argparse
import json
from pathlib import Path
from typing import Optional

try:
	import vector_model
	FruitSearchEngine = vector_model.FruitSearchEngine
	_json_default = vector_model._json_default
except Exception:
	# fallback to direct import if module path differs
	from vector_model import FruitSearchEngine, _json_default


class Detector:
	"""Minimal detector that forwards images to the search engine.

	Methods:
	  - detect(image, top_k): returns search results from `FruitSearchEngine`.
	  - detect_and_print(image, top_k): prints results (including metadata).
	"""

	def __init__(self, model_path: str, db_path: Optional[str] = None,
				 db_backend: str = "faiss", device: str = "auto", top_k: int = 5):
		# Resolve DB path relative to project root (same behaviour as vector_model.main)
		if db_path and not os.path.isabs(db_path):
			project_root = Path(__file__).resolve().parents[2]
			db_path = str(project_root.joinpath(db_path))

		self.model_path = model_path
		self.db_path = db_path
		self.top_k = top_k

		print(f"Detector: initializing engine (model={model_path}, db={db_path})")
		self.engine = FruitSearchEngine(model_path, db_path, device=device, db_backend=db_backend)

	def detect(self, image, top_k: Optional[int] = None):
		"""Run search for given image (path or PIL.Image) and return results."""
		if top_k is None:
			top_k = self.top_k
		return self.engine.search(image, top_k=top_k)

	def detect_and_print(self, image, top_k: Optional[int] = None):
		results = self.detect(image, top_k=top_k)
		if not results:
			print("No results")
			return results

		print(f"Top-{len(results)} results:")
		for i, r in enumerate(results, 1):
			meta = r.get("metadata", {}) or {}
			cat = meta.get("category", "")
			pid = meta.get("product_id", "")
			name = r.get("name") or cat or pid or r.get("id")
			pid_str = f" product_id:{pid}" if pid else ""
			db_idx = r.get("db_index")
			idx_str = f" index:{db_idx}" if db_idx is not None else ""
			meta_str = json.dumps(meta, ensure_ascii=False, default=_json_default)
			print(f"  {i}. [{r['score']:.3f}] {name} — {r.get('id')}{pid_str}{idx_str}")
			print(f"     metadata: {meta_str}")
		return results


def _resolve_db_path(db_arg: str) -> str:
	if os.path.isabs(db_arg):
		return db_arg
	project_root = Path(__file__).resolve().parents[2]
	return str(project_root.joinpath(db_arg))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Detector stub — forward image to vector_model search")
	parser.add_argument("--model", default="C:\\Users\\artur\\Desktop\\Yandex_Cemp\\fruit_embedder_final.pth",
						help="Path to model checkpoint")
	parser.add_argument("--db", default=os.path.join("Server_prod", "DB", "fruits.db"),
						help="Path to DB file (relative to project root or absolute)")
	parser.add_argument("--db-backend", choices=["faiss"], default="faiss")
	parser.add_argument("--image", required=True, help="Path to query image")
	parser.add_argument("--top", type=int, default=5)

	args = parser.parse_args()

	db_path = _resolve_db_path(args.db)
	det = Detector(args.model, db_path=db_path, db_backend=args.db_backend, top_k=args.top)
	det.detect_and_print(args.image, top_k=args.top)
