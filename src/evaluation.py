from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.detector import detect_copy_move


METRIC_NAMES = ("accuracy", "precision", "recall", "f1_score")


def evaluate_traditional_detector(uploads_dir: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Evaluate the available detector against a reproducible synthetic benchmark."""
    uploads_dir = Path(uploads_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources: list[np.ndarray] = []
    for path in sorted(uploads_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", "jpeg", ".jpeg", ".bmp", ".webp"}:
            image = cv2.imread(str(path))
            if image is not None and image.shape[0] >= 80 and image.shape[1] >= 80:
                scale = min(1.0, 320.0 / max(image.shape[:2]))
                sources.append(cv2.resize(image, (max(80, round(image.shape[1] * scale)), max(80, round(image.shape[0] * scale)))))
        if len(sources) >= 4:
            break

    if not sources:
        dummy = np.full((128, 128, 3), 128, dtype=np.uint8)
        sources = [dummy]

    truth: list[int] = []
    predictions: list[int] = []
    for index, source in enumerate(sources):
        truth.append(0)
        predictions.append(_detector_prediction(source))
        truth.append(1)
        predictions.append(_detector_prediction(_copy_move_variant(source)))

    y_true = np.asarray(truth, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    classification = _binary_metrics(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    metrics = {
        "sample_count": int(len(y_true)),
        "dataset": "Built-in copy-move benchmark (synthetic copy-paste and pristine variants)",
        "benchmark_type": "traditional-detector",
        "classification": classification,
        "segmentation": None,
        "confusion_matrix": matrix,
    }
    save_evaluation_outputs(metrics, output_dir)
    return metrics


def _detector_prediction(image: np.ndarray) -> int:
    result = detect_copy_move(image)
    return int(result.status.lower() != "no suspicious copy-move region detected" and np.count_nonzero(result.mask) > 0)


def _copy_move_variant(image: np.ndarray) -> np.ndarray:
    output = cv2.resize(image.copy(), (min(image.shape[1], 512), min(image.shape[0], 512)))
    height, width = output.shape[:2]
    patch_height = max(12, height // 4)
    patch_width = max(12, width // 4)
    source_y = max(0, height // 8)
    source_x = max(0, width // 8)
    source = output[source_y:source_y + patch_height, source_x:source_x + patch_width]
    target_y = min(height - patch_height, height // 2)
    target_x = min(width - patch_width, width // 2)
    output[target_y:target_y + patch_height, target_x:target_x + patch_width] = source
    return output


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.int64).reshape(-1)
    y_pred = y_pred.astype(np.int64).reshape(-1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }


def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, object]:
    image_truth: list[np.ndarray] = []
    image_prediction: list[np.ndarray] = []
    pixel_truth: list[np.ndarray] = []
    pixel_prediction: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for images, masks, image_labels, _ in loader:
            outputs = model(images.to(device))
            image_prediction.extend(torch.argmax(outputs["classification_logits"], dim=1).cpu().numpy())
            image_truth.extend(image_labels.cpu().numpy())
            predicted_masks = torch.sigmoid(outputs["segmentation_logits"]) >= 0.5
            predicted_masks = predicted_masks.cpu().numpy()[:, 0].astype(np.uint8)
            true_masks = masks.cpu().numpy()[:, 0] > 0.5
            pixel_prediction.append(predicted_masks.reshape(predicted_masks.shape[0], -1))
            pixel_truth.append(true_masks.reshape(true_masks.shape[0], -1))

    y_true_image = np.asarray(image_truth)
    y_pred_image = np.asarray(image_prediction)
    y_true_pixels = np.concatenate(pixel_truth).reshape(-1)
    y_pred_pixels = np.concatenate(pixel_prediction).reshape(-1)

    image_metrics = _binary_metrics(y_true_image, y_pred_image)
    segmentation_metrics = _binary_metrics(y_true_pixels, y_pred_pixels)
    matrix = confusion_matrix(y_true_image, y_pred_image, labels=[0, 1])
    return {
        "sample_count": len(loader.dataset),
        "classification": image_metrics,
        "segmentation": segmentation_metrics,
        "confusion_matrix": matrix.tolist(),
    }


def save_evaluation_outputs(metrics: dict[str, object], output_dir: str | Path, epoch_losses: list[float] | None = None) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    classification_path = output_dir / "classification_metrics.png"
    names = ["Accuracy", "Precision", "Recall", "F1-score"]
    values = [float(metrics["classification"][name]) for name in METRIC_NAMES]
    _save_metric_plot(names, values, classification_path, "Image Classification Metrics")

    confusion_path = output_dir / "confusion_matrix.png"
    matrix = np.asarray(metrics["confusion_matrix"])
    _save_confusion_plot(matrix, confusion_path)

    plots = {"metrics_json": str(metrics_path), "metrics_plot": str(classification_path), "confusion_plot": str(confusion_path)}
    if epoch_losses:
        loss_path = output_dir / "training_loss.png"
        _save_loss_plot(epoch_losses, loss_path)
        plots["training_loss_plot"] = str(loss_path)
    return plots


def _save_metric_plot(names: list[str], values: list[float], path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(names, values, color=["#2563eb", "#0891b2", "#059669", "#7c3aed"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_confusion_plot(matrix: np.ndarray, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["Real", "Forged"])
    axis.set_yticks([0, 1], labels=["Real", "Forged"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_title("Confusion Matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontweight="bold", color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_detection_visualization(confidence_percent: float, area_percent: float, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    axes[0].bar(["Confidence"], [confidence_percent / 100], color="#2563eb")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Prediction Confidence")
    axes[0].text(0, confidence_percent / 100 + 0.03, f"{confidence_percent:.1f}%", ha="center", fontweight="bold")
    axes[1].bar(["Forged region"], [area_percent / 100], color="#dc2626")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Detected Area")
    axes[1].text(0, area_percent / 100 + 0.03, f"{area_percent:.1f}%", ha="center", fontweight="bold")
    for axis in axes:
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_loss_plot(losses: list[float], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(range(1, len(losses) + 1), losses, marker="o", color="#2563eb")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training loss")
    axis.set_title("Training Loss Curve")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
