import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from torchvision import transforms
import numpy as np


class FruitEmbedder(nn.Module):
    def __init__(self, backbone_name="facebook/dinov2-small",
                 backbone_dim=384, embedding_dim=256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        embedding = self.head(cls_token)
        return F.normalize(embedding, p=2, dim=1)


# --- Загрузка модели ---
checkpoint = torch.load("C:/Users/artur/Desktop/Yandex_Cemp/fruit_embedder_final.pth", map_location="cpu")

model = FruitEmbedder(
    backbone_name=checkpoint["backbone_name"],
    embedding_dim=checkpoint["embedding_dim"],
)
model.head.load_state_dict(checkpoint["head_state_dict"])
if checkpoint.get("backbone_state_dict"):
    model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
model.eval()

processor = AutoImageProcessor.from_pretrained(checkpoint["backbone_name"])

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])


# --- Извлечение эмбеддинга ---
def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = model(inputs["pixel_values"])
    return emb.squeeze().numpy()


# --- Сравнение двух фото ---
def compare(path1, path2):
    e1 = get_embedding(path1)
    e2 = get_embedding(path2)
    similarity = np.dot(e1, e2)  # cosine sim (уже L2-норм.)
    return float(similarity)


# --- Поиск по базе ---
def build_database(image_folder):
    embeddings = []
    paths = []
    for fname in os.listdir(image_folder):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(image_folder, fname)
            emb = get_embedding(path)
            embeddings.append(emb)
            paths.append(path)
    return np.array(embeddings), paths


def search(query_path, embeddings, paths, top_k=5):
    query = get_embedding(query_path)
    scores = np.dot(embeddings, query)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(paths[i], scores[i]) for i in top_idx]

score = compare("C:/Users/artur/Desktop/Yandex_Cemp/r0_10.jpg", "C:/Users/artur/Desktop/Yandex_Cemp/r0_118.jpg")
print(f"Сходство: {score:.4f}")