from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

try:
    import supervision as sv
except ImportError:
    sv = None

from src.forensic_features import FORGERY_TYPES, MODEL_IMAGE_SIZE, prepare_model_input
from src.unet import UNet


@lru_cache(maxsize=2)
def load_model(checkpoint_path: str) -> tuple[UNet, int, torch.device]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    image_size = int(checkpoint.get("image_size", MODEL_IMAGE_SIZE))
    model = UNet()
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, image_size, torch.device("cpu")


def _restore_letterboxed_map(values: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    size = values.shape[0]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    cropped = values[top:top + resized_height, left:left + resized_width]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def extract_bounding_boxes(mask: np.ndarray) -> list[list[int]]:
    if sv is not None:
        count, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        components = np.asarray([labels == label for label in range(1, count)], dtype=bool)
        if not len(components):
            return []
        boxes = sv.mask_to_xyxy(components, coordinate_convention="exclusive")
        return [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in boxes]

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [
        [int(x), int(y), int(x + width), int(y + height)]
        for x, y, width, height, area in stats[1:count]
        if area > 0
    ]


def draw_bounding_boxes(image_bgr: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    annotated = image_bgr.copy()
    for index, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 215, 255), 2)
        cv2.putText(annotated, f"Region {index}", (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 110), 2, cv2.LINE_AA)
    return annotated


def predict_with_model(image_bgr: np.ndarray, checkpoint_path: str) -> dict[str, object]:
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(checkpoint_path)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    model, image_size, device = load_model(str(Path(checkpoint_path).resolve()))
    features = prepare_model_input(image_rgb, size=image_size).unsqueeze(0).to(device)
    features.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    outputs = model(features)

    class_probabilities = torch.softmax(outputs["classification_logits"], dim=1)[0]
    predicted_class = int(torch.argmax(class_probabilities).item())
    class_probabilities[predicted_class].backward()

    mask_probability = torch.sigmoid(outputs["segmentation_logits"])[0, 0].detach().cpu().numpy()
    mask_probability = _restore_letterboxed_map(mask_probability, image_bgr.shape[:2])
    mask = mask_probability >= 0.5 if predicted_class == 1 else np.zeros_like(mask_probability, dtype=bool)
    type_probabilities = torch.softmax(outputs["forgery_type_logits"], dim=1)[0].detach().cpu().numpy()

    activations = model.cam_features
    if activations is None or activations.grad is None:
        cam = np.zeros(image_bgr.shape[:2], dtype=np.float32)
    else:
        weights = activations.grad.mean(dim=(2, 3), keepdim=True)
        cam_tensor = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam_tensor = functional.interpolate(cam_tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
        cam = _restore_letterboxed_map(cam_tensor[0, 0].detach().cpu().numpy(), image_bgr.shape[:2])
        maximum = float(cam.max())
        if maximum > 0:
            cam /= maximum

    boxes = extract_bounding_boxes(mask)
    return {
        "mask": mask.astype(np.uint8) * 255,
        "confidence": float(class_probabilities[predicted_class].item()),
        "is_forged": bool(predicted_class),
        "forgery_type": FORGERY_TYPES[int(np.argmax(type_probabilities))] if predicted_class else "Not applicable",
        "type_confidence": float(np.max(type_probabilities)),
        "boxes": boxes,
        "cam": cam,
        "method": "ELA + SRM multi-task U-Net",
    }


def make_gradcam_overlay(image_bgr: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(np.clip(cam, 0.0, 1.0) * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.55, heatmap, 0.45, 0.0)