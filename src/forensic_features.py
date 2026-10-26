from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image

MODEL_IMAGE_SIZE = 256
INPUT_CHANNELS = 7
FORGERY_TYPES = ("Splicing", "Copy-Move", "Deepfake/AI-Generated")


def letterbox_image(image: np.ndarray, size: int = MODEL_IMAGE_SIZE, mask: bool = False) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_AREA
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    shape = (size, size) if image.ndim == 2 else (size, size, image.shape[2])
    canvas = np.zeros(shape, dtype=image.dtype)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas


def make_ela_map(rgb_image: np.ndarray, quality: int = 90) -> np.ndarray:
    image = Image.fromarray(rgb_image.astype(np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32)
    difference = np.abs(rgb_image.astype(np.float32) - recompressed)
    return np.clip(difference * (10.0 / 255.0), 0.0, 1.0)


def make_srm_map(rgb_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    filters = (
        np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0,
        np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32),
    )
    residuals = [np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)) for kernel in filters]
    return np.clip(np.mean(residuals, axis=0) * 4.0, 0.0, 1.0)


def prepare_model_input(rgb_image: np.ndarray, size: int = MODEL_IMAGE_SIZE) -> torch.Tensor:
    rgb = letterbox_image(rgb_image.astype(np.uint8), size=size).astype(np.float32) / 255.0
    ela = make_ela_map(np.round(rgb * 255.0).astype(np.uint8))
    srm = make_srm_map(np.round(rgb * 255.0).astype(np.uint8))[..., np.newaxis]
    features = np.concatenate((rgb, ela, srm), axis=2)
    return torch.from_numpy(features.transpose(2, 0, 1).copy()).float()