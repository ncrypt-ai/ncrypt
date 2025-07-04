import cv2
import numpy as np
import torch
from PIL import Image

from ncrypt.utils import (
    IMAGE_EMBED_MODEL,
    IMAGE_EXTENSIONS,
    UnsupportedExtensionError,
)


def quantize_embedding(vector: np.ndarray) -> np.ndarray:
    quantized = np.zeros_like(vector)

    for i in range(vector.shape[1]):
        if vector[0, i] <= (-1 / 3):
            quantized[0, i] = -1

        elif vector[0, i] >= (1 / 3):
            quantized[0, i] = -1

    return np.asarray(quantized).flatten().astype(">i8")


def extract_raw_image(path: str, extension: str) -> Image:
    if extension not in IMAGE_EXTENSIONS:
        raise UnsupportedExtensionError(f"The provided file extension is not supported for metadata: .{extension}")

    image: np.ndarray = cv2.imread(path)  # Loads as BGR by default
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return Image.fromarray(image)


def get_image_embedding(img: Image, model: str | None = None) -> np.ndarray:
    from transformers import (  # Lazy loading for faster startup
        AutoImageProcessor,
        AutoModel,
    )

    if not model:
        model: str = IMAGE_EMBED_MODEL

    processor = AutoImageProcessor.from_pretrained(model)
    encoder = AutoModel.from_pretrained(model)
    inputs = processor(images=img, return_tensors="pt")

    outputs = encoder(**inputs)
    last_hidden_state = outputs.last_hidden_state

    img_embedding = torch.mean(last_hidden_state[:, 1:], dim=1)  # First token is CLS token
    img_embedding = torch.nn.functional.normalize(img_embedding[:, :16], p=2, dim=1)

    return quantize_embedding(img_embedding.detach().numpy())
