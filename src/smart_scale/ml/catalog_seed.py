from __future__ import annotations

import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LABEL_ALIASES = {
    "cabbage_cauliflowre": "cabbage_cauliflower",
    "pepper_sweet_elderberry": "pepper_sweet_elonged",
    "pomegranate_pomelo": "pomegranate_pomegranate",
    "tomato_cherry_yello": "tomato_cherry_yellow",
    "watermelon_elonge": "watermelon_elonged",
}


def split_sort_label(label: str) -> tuple[str, str]:
    normalized = normalize_catalog_label(label)
    if "_" not in normalized:
        return normalized, normalized
    product_type, product_sort = normalized.split("_", 1)
    return product_type, product_sort


def normalize_catalog_label(label: str) -> str:
    normalized = label.strip().lower()
    return LABEL_ALIASES.get(normalized, normalized)


def load_price_catalog(source: str | Path | None) -> dict[str, float]:
    if source is None:
        return {}

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {normalize_catalog_label(str(key)): float(value) for key, value in payload.items()}
    if suffix == ".csv":
        return _load_prices_from_csv(path)
    if suffix == ".py":
        return _load_prices_from_python(path)

    raise ValueError(f"Неподдерживаемый формат прайс-листа: {path.suffix}")


class PackEatCatalogSeeder:
    def __init__(self, embedder: Any, store: Any) -> None:
        self.embedder = embedder
        self.store = store

    def build(
        self,
        dataset_dir: str | Path,
        price_source: str | Path,
        *,
        samples_per_sort: int = 5,
        batch_size: int = 16,
    ) -> int:
        dataset_root = Path(dataset_dir)
        if not dataset_root.exists():
            raise FileNotFoundError(dataset_root)
        if samples_per_sort <= 0:
            raise ValueError("samples_per_sort должен быть больше нуля.")

        price_catalog = load_price_catalog(price_source)
        if not price_catalog:
            raise ValueError("Прайс-лист пуст или не загружен.")

        discovered = self._discover_packeat_images(dataset_root, set(price_catalog))
        missing_labels = sorted(label for label in price_catalog if label not in discovered)
        if missing_labels:
            preview = ", ".join(missing_labels[:10])
            raise ValueError(
                "В датасете не найдены изображения для некоторых сортов: "
                + preview
                + (" ..." if len(missing_labels) > 10 else "")
            )

        insufficient = {
            label: len(paths)
            for label, paths in discovered.items()
            if label in price_catalog and len(paths) < samples_per_sort
        }
        if insufficient:
            details = ", ".join(f"{label}={count}" for label, count in sorted(insufficient.items())[:10])
            raise ValueError(
                "Для некоторых сортов недостаточно изображений: "
                + details
                + (" ..." if len(insufficient) > 10 else "")
            )

        dataset_items: list[tuple[str, Path, dict[str, Any]]] = []
        for label in sorted(price_catalog):
            price_rub_per_kg = float(price_catalog[label])
            product_type, product_sort = split_sort_label(label)
            for sample_index, image_path in enumerate(discovered[label][:samples_per_sort], start=1):
                product_id = f"{label}:{sample_index:02d}"
                dataset_items.append(
                    (
                        product_id,
                        image_path,
                        {
                            "product_id": product_id,
                            "product_type": product_type,
                            "product_sort": product_sort,
                            "price_rub_per_kg": price_rub_per_kg,
                            "source_path": str(image_path),
                        },
                    )
                )

        if not dataset_items:
            if hasattr(self.store, "replace_catalog"):
                dim = int(getattr(self.store, "dim", getattr(self.embedder, "embedding_dim", 0)))
                self.store.replace_catalog([], np.empty((0, dim), dtype=np.float32), [])
            else:
                self.store.clear()
            return 0

        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(dataset_items), batch_size):
            chunk = dataset_items[start:start + batch_size]
            image_paths = [image_path for _, image_path, _ in chunk]
            embeddings = self.embedder.embed_batch(image_paths, batch_size=min(batch_size, len(chunk)))
            all_embeddings.append(np.asarray(embeddings, dtype=np.float32))

        ids = [product_id for product_id, _, _ in dataset_items]
        metadata_list = [metadata for _, _, metadata in dataset_items]
        embeddings_matrix = np.vstack(all_embeddings)

        if hasattr(self.store, "replace_catalog"):
            self.store.replace_catalog(ids, embeddings_matrix, metadata_list)
        else:
            self.store.clear()
            self.store.add_batch(ids, embeddings_matrix, metadata_list)

        return len(dataset_items)

    def _discover_packeat_images(self, dataset_root: Path, labels: set[str]) -> dict[str, list[Path]]:
        search_roots = [dataset_root / "classification"] if (dataset_root / "classification").exists() else [dataset_root]
        label_to_images: dict[str, list[Path]] = defaultdict(list)

        for search_root in search_roots:
            for image_path in search_root.rglob("*"):
                if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                    continue

                lowered_parts = [part.lower() for part in image_path.parts]
                if any("segment" in part or "mask" in part for part in lowered_parts):
                    continue

                label = self._resolve_label(image_path, labels)
                if label is None:
                    continue

                label_to_images[label].append(image_path)

        return {
            label: sorted(paths, key=lambda path: str(path.relative_to(dataset_root)).lower())
            for label, paths in label_to_images.items()
        }

    def _resolve_label(self, image_path: Path, labels: set[str]) -> str | None:
        for part in reversed(image_path.parts):
            normalized = normalize_catalog_label(part)
            if normalized in labels:
                return normalized
        return None


def _load_prices_from_csv(path: Path) -> dict[str, float]:
    prices: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = normalize_catalog_label(str(row.get("label") or row.get("sort") or row.get("product_sort") or ""))
            price = row.get("price") or row.get("price_rub_per_kg")
            if not label or price in {None, ""}:
                continue
            prices[label] = float(str(price).replace(",", "."))
    return prices


def _load_prices_from_python(path: Path) -> dict[str, float]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            continue
        if statement.targets[0].id != "prices":
            continue
        payload = ast.literal_eval(statement.value)
        if not isinstance(payload, dict):
            raise ValueError("Ожидался словарь prices = {...}.")
        return {normalize_catalog_label(str(key)): float(value) for key, value in payload.items()}

    raise ValueError("В python-файле не найден словарь prices = {...}.")
