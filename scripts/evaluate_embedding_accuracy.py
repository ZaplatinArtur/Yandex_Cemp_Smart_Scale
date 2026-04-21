from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smart_scale.config import get_settings
from smart_scale.ml.catalog_seed import SUPPORTED_IMAGE_EXTENSIONS, normalize_catalog_label, split_sort_label
from smart_scale.ml.detection import ProductLocalizer
from smart_scale.ml.embedding import DinoV2Embedder
from smart_scale.ml.vector_store import FileVectorStore


@dataclass(frozen=True)
class EvaluationMetrics:
    dataset_dir: str
    model_name: str
    embedding_dim: int
    yolo_enabled: bool
    detection_model: str | None
    train_images: int
    test_images: int
    train_classes: int
    test_classes: int
    train_masked_images: int
    test_masked_images: int
    train_localized_images: int
    test_localized_images: int
    top_k: int
    top1_accuracy: float
    topk_accuracy: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DINOv2 KNN accuracy on a train/test image dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset root. Defaults to SMART_SCALE_DATASET_DIR.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--use-yolo", action="store_true", help="Run YOLO localization before embedding images.")
    parser.add_argument(
        "--detection-model",
        type=Path,
        default=None,
        help="YOLO model path. Defaults to SMART_SCALE_DETECTION_MODEL resolution.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    settings = get_settings()
    dataset_dir = args.dataset_dir or settings.dataset_dir
    detection_model = args.detection_model or settings.detection_model_path

    metrics = evaluate_accuracy(
        dataset_dir=dataset_dir,
        model_name=settings.embedding_model_name,
        embedding_dim=settings.embedding_dim,
        batch_size=args.batch_size,
        top_k=args.top_k,
        use_yolo=args.use_yolo,
        detection_model=detection_model,
        confidence_threshold=args.confidence_threshold,
        mask_threshold=args.mask_threshold,
    )

    if args.json:
        print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))
        return

    print(f"dataset_dir: {metrics.dataset_dir}")
    print(f"model_name: {metrics.model_name}")
    print(f"embedding_dim: {metrics.embedding_dim}")
    print(f"yolo_enabled: {metrics.yolo_enabled}")
    print(f"detection_model: {metrics.detection_model}")
    print(f"train_images: {metrics.train_images}")
    print(f"test_images: {metrics.test_images}")
    print(f"train_classes: {metrics.train_classes}")
    print(f"test_classes: {metrics.test_classes}")
    print(f"train_masked_images: {metrics.train_masked_images}")
    print(f"test_masked_images: {metrics.test_masked_images}")
    print(f"train_localized_images: {metrics.train_localized_images}")
    print(f"test_localized_images: {metrics.test_localized_images}")
    print(f"top1_accuracy: {metrics.top1_accuracy:.6f}")
    print(f"top{metrics.top_k}_accuracy: {metrics.topk_accuracy:.6f}")


def evaluate_accuracy(
    *,
    dataset_dir: Path,
    model_name: str,
    embedding_dim: int,
    batch_size: int,
    top_k: int,
    use_yolo: bool,
    detection_model: Path,
    confidence_threshold: float,
    mask_threshold: float,
) -> EvaluationMetrics:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    train_items = _discover_split(dataset_dir, "train")
    test_items = _discover_split(dataset_dir, "test")
    if not train_items:
        raise RuntimeError(f"No train images found under {dataset_dir}.")
    if not test_items:
        raise RuntimeError(f"No test images found under {dataset_dir}.")

    embedder = DinoV2Embedder(
        model_name=model_name,
        embedding_dim=embedding_dim,
    )
    embedder.warmup()
    localizer = None
    if use_yolo:
        localizer = ProductLocalizer(
            model_path=detection_model,
            confidence_threshold=confidence_threshold,
            mask_threshold=mask_threshold,
        )
        if not localizer.is_ready:
            raise RuntimeError(
                f"YOLO localizer is not ready for {detection_model}."
                + (f" Reason: {localizer.failure_reason}" if localizer.failure_reason else "")
            )

    store = FileVectorStore(dim=embedding_dim, snapshot_path=dataset_dir / ".embedding_eval_unused.pkl")
    train_masked_images = 0
    train_localized_images = 0
    for start in range(0, len(train_items), batch_size):
        chunk = train_items[start:start + batch_size]
        images, batch_stats = _prepare_batch_images(chunk, localizer)
        train_masked_images += batch_stats["masked_images"]
        train_localized_images += batch_stats["localized_images"]
        embeddings = embedder.embed_batch(images, batch_size=batch_size)
        ids = [f"{label}/train/{start + offset:06d}" for offset, (label, _) in enumerate(chunk)]
        metadata = [_metadata_for_label(label) for label, _ in chunk]
        store.add_batch(ids, embeddings, metadata)

    correct_top1 = 0
    correct_topk = 0
    test_masked_images = 0
    test_localized_images = 0
    for start in range(0, len(test_items), batch_size):
        chunk = test_items[start:start + batch_size]
        images, batch_stats = _prepare_batch_images(chunk, localizer)
        test_masked_images += batch_stats["masked_images"]
        test_localized_images += batch_stats["localized_images"]
        embeddings = embedder.embed_batch(images, batch_size=batch_size)
        for (expected_label, _), embedding in zip(chunk, embeddings, strict=True):
            matches = store.search(embedding, top_k=top_k)
            predicted_labels = [match.name for match in matches]
            if predicted_labels and predicted_labels[0] == expected_label:
                correct_top1 += 1
            if expected_label in predicted_labels:
                correct_topk += 1

    test_count = len(test_items)
    return EvaluationMetrics(
        dataset_dir=str(dataset_dir),
        model_name=model_name,
        embedding_dim=embedding_dim,
        yolo_enabled=use_yolo,
        detection_model=str(detection_model) if use_yolo else None,
        train_images=len(train_items),
        test_images=test_count,
        train_classes=len({label for label, _ in train_items}),
        test_classes=len({label for label, _ in test_items}),
        train_masked_images=train_masked_images,
        test_masked_images=test_masked_images,
        train_localized_images=train_localized_images,
        test_localized_images=test_localized_images,
        top_k=top_k,
        top1_accuracy=correct_top1 / test_count,
        topk_accuracy=correct_topk / test_count,
    )


def _discover_split(dataset_dir: Path, split: str) -> list[tuple[str, Path]]:
    split_root = _resolve_split_root(dataset_dir, split)
    items: list[tuple[str, Path]] = []
    for label_dir in sorted((path for path in split_root.iterdir() if path.is_dir()), key=lambda path: path.name.lower()):
        label = normalize_catalog_label(label_dir.name)
        for image_path in _iter_images(label_dir):
            items.append((label, image_path))
    return items


def _resolve_split_root(dataset_dir: Path, split: str) -> Path:
    candidates = [
        dataset_dir / "classification" / split,
        dataset_dir / split,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find split '{split}'. Expected one of: " + ", ".join(str(candidate) for candidate in candidates)
    )


def _iter_images(directory: Path) -> Iterable[Path]:
    return (
        path
        for path in sorted(directory.rglob("*"), key=lambda item: str(item).lower())
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _prepare_batch_images(
    chunk: list[tuple[str, Path]],
    localizer: ProductLocalizer | None,
) -> tuple[list[Path | Image.Image], dict[str, int]]:
    if localizer is None:
        return [path for _, path in chunk], {"masked_images": 0, "localized_images": 0}

    images: list[Image.Image] = []
    masked_images = 0
    localized_images = 0
    for _, image_path in chunk:
        image = Image.open(image_path).convert("RGB")
        crop = localizer.localize(image)
        images.append(crop.image)
        if crop.mask_applied:
            masked_images += 1
        if crop.bbox != (0, 0, image.size[0], image.size[1]):
            localized_images += 1

    return images, {"masked_images": masked_images, "localized_images": localized_images}


def _metadata_for_label(label: str) -> dict[str, object]:
    product_type, product_sort = split_sort_label(label)
    return {
        "product_type": product_type,
        "product_sort": product_sort,
        "price_rub_per_kg": 0.0,
    }


if __name__ == "__main__":
    main()
