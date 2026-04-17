# ml/embedder.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path


class ImageEmbedder:
    """Модуль получения эмбеддингов изображений для умных весов."""

    def __init__(self, device: str = None):
        # Автоопределение устройства
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # MobileNetV3-Small без классификационной головы
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier = nn.Identity()
        model.eval()
        self.model = model.to(self.device)

        # Препроцессинг — стандартный для ImageNet-моделей
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _load_image(self, source) -> Image.Image:
        """
        Принимает разные форматы входа:
        - str / Path — путь к файлу
        - bytes — сырые байты (например, от камеры)
        - PIL.Image — уже открытое изображение
        - numpy.ndarray — массив (BGR от OpenCV)
        """
        if isinstance(source, (str, Path)):
            image = Image.open(source).convert("RGB")
        elif isinstance(source, bytes):
            from io import BytesIO
            image = Image.open(BytesIO(source)).convert("RGB")
        elif isinstance(source, np.ndarray):
            # OpenCV даёт BGR, конвертируем в RGB
            image = Image.fromarray(source[:, :, ::-1]).convert("RGB")
        elif isinstance(source, Image.Image):
            image = source.convert("RGB")
        else:
            raise TypeError(f"Неподдерживаемый формат: {type(source)}")
        return image

    @torch.no_grad()
    def get_embedding(self, source) -> np.ndarray:
        """Получить эмбеддинг одного изображения. Возвращает numpy array (576,)."""
        image = self._load_image(source)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model(tensor).squeeze(0).cpu().numpy()
        return embedding

    @torch.no_grad()
    def get_embeddings_batch(self, sources: list) -> np.ndarray:
        """Батч эмбеддингов. Возвращает numpy array (N, 576)."""
        tensors = []
        for src in sources:
            image = self._load_image(src)
            tensors.append(self.transform(image))
        batch = torch.stack(tensors).to(self.device)
        embeddings = self.model(batch).cpu().numpy()
        return embeddings



import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Косинусное сходство между двумя векторами."""
    dot = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm == 0:
        return 0.0
    return float(dot / norm)

# --- Пример использования ---
if __name__ == "__main__":
    embedder = ImageEmbedder()

    # Из файла
    emb = embedder.get_embedding("C:/Users/artur/Desktop/Yandex_Cemp/photo_2026-04-17_15-49-54.jpg")
    emb2 = embedder.get_embedding("C:/Users/artur/Desktop/Yandex_Cemp/photo_2026-04-17_15-49-54 (2).jpg")
    score = cosine_similarity(emb, emb2)
    print(f"Сходство: {score:.3f}")
    

    