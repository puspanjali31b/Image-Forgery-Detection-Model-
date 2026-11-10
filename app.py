from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, render_template, request
from PIL import Image, ExifTags, ImageOps
from werkzeug.utils import secure_filename

from src.detector import detect_copy_move
from src.evaluation import evaluate_traditional_detector, save_detection_visualization
from src.forensic_features import make_ela_map, make_srm_map
from src.forensic_inference import draw_bounding_boxes, extract_bounding_boxes, make_gradcam_overlay, predict_with_model
from src.reporting import create_forensic_report

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
MODEL_PATH = BASE_DIR / "models" / "forgery_multitask.pt"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jpe", ".tif", ".tiff", ".bmp", ".webp"}
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)


def _ensure_evaluation_metrics() -> dict[str, object] | None:
    metrics_path = MODEL_PATH.parent / "metrics.json"
    if metrics_path.is_file():
        return _model_metrics()
    if not MODEL_PATH.parent.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        return evaluate_traditional_detector(UPLOAD_DIR, MODEL_PATH.parent)
    except Exception:
        return None


def _model_metrics() -> dict[str, object] | None:
    metrics_path = MODEL_PATH.parent / "metrics.json"
    if not metrics_path.is_file():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    public_dir = BASE_DIR / "static" / "evaluation"
    public_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("classification_metrics.png", "confusion_matrix.png", "training_loss.png"):
        source = metrics_path.parent / filename
        if source.is_file():
            shutil.copy2(source, public_dir / filename)
    return metrics


def _read_metadata(image_path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source)
            metadata["Format"] = str(source.format or "Unknown")
            metadata["Dimensions"] = f"{image.width} x {image.height}"
            for tag_id, value in (source.getexif() or {}).items():
                name = ExifTags.TAGS.get(tag_id, f"Tag {tag_id}")
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                text = " ".join(str(value).split())
                if text:
                    metadata[name] = text[:240]
    except (OSError, ValueError):
        return metadata
    return dict(list(metadata.items())[:24])


_ensure_evaluation_metrics()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if file is None or file.filename == "":
            return render_template("index.html", error="Please select an image to upload.")

        filename = secure_filename(file.filename)
        if not filename or Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            return render_template("index.html", error="Please upload a supported image file.")

        image_id = uuid4().hex
        original_name = f"{image_id}_{filename}"
        original_path = UPLOAD_DIR / original_name
        file.save(original_path)

        image = cv2.imread(str(original_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            try:
                with Image.open(original_path) as source:
                    decoded_rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8)
                image = cv2.cvtColor(decoded_rgb, cv2.COLOR_RGB2BGR)
            except (OSError, ValueError):
                image = None
        elif image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
            white_background = np.full_like(rgba[:, :, :3], 255, dtype=np.uint8)
            rgb = cv2.convertScaleAbs(rgba[:, :, :3].astype(np.float32) * alpha + white_background.astype(np.float32) * (1.0 - alpha))
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if image is None:
            return render_template("index.html", error="Could not read the uploaded image.")

        metadata = _read_metadata(original_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        model_warning = None
        cam = None
        forgery_type = "Not classified"
        if MODEL_PATH.is_file():
            try:
                model_result = predict_with_model(image, str(MODEL_PATH))
                mask_uint8 = model_result["mask"]
                confidence = float(model_result["confidence"])
                status = "Forged image indicated" if model_result["is_forged"] else "No forgery indicated"
                method = str(model_result["method"])
                forgery_type = str(model_result["forgery_type"])
                boxes = model_result["boxes"]
                cam = model_result["cam"]
            except Exception as error:
                model_warning = f"Trained model could not be loaded ({type(error).__name__}); used the traditional detector instead."
                detection = detect_copy_move(image)
                mask = detection.mask
                if mask.shape[:2] != image.shape[:2]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_uint8 = (mask > 0).astype(np.uint8) * 255
                confidence = float(detection.confidence)
                status = detection.status
                method = detection.method
                boxes = extract_bounding_boxes(mask_uint8 > 0)
        else:
            detection = detect_copy_move(image)
            mask = detection.mask
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            confidence = float(detection.confidence)
            status = detection.status
            method = detection.method
            boxes = extract_bounding_boxes(mask_uint8 > 0)
            model_warning = "No trained U-Net checkpoint found. The verdict uses the traditional detector, whose confidence is a heuristic rather than a calibrated probability. Train a model to enable ELA/SRM multi-task predictions and Grad-CAM."

        mask_path = RESULT_DIR / f"{image_id}_mask.png"
        cv2.imwrite(str(mask_path), mask_uint8)

        red_tint = np.zeros_like(image)
        red_tint[:] = (0, 0, 255)
        blended = cv2.addWeighted(image, 0.55, red_tint, 0.45, 0)
        overlay = image.copy()
        overlay[mask_uint8 > 0] = blended[mask_uint8 > 0]
        overlay_path = RESULT_DIR / f"{image_id}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)

        annotated = draw_bounding_boxes(image, boxes)
        boxes_path = RESULT_DIR / f"{image_id}_boxes.png"
        cv2.imwrite(str(boxes_path), annotated)

        ela_path = RESULT_DIR / f"{image_id}_ela.png"
        ela_map = make_ela_map(rgb_image)
        cv2.imwrite(str(ela_path), cv2.cvtColor(np.uint8(np.clip(ela_map * 255, 0, 255)), cv2.COLOR_RGB2BGR))

        noise_path = RESULT_DIR / f"{image_id}_noise.png"
        noise_map = make_srm_map(rgb_image)
        cv2.imwrite(str(noise_path), np.uint8(np.clip(noise_map * 255, 0, 255)))

        if cam is not None:
            explanation = make_gradcam_overlay(image, cam)
            explanation_label = "Grad-CAM"
        else:
            explanation = overlay
            explanation_label = "Detected-region overlay (Grad-CAM requires a trained model)"
        explanation_path = RESULT_DIR / f"{image_id}_explanation.png"
        cv2.imwrite(str(explanation_path), explanation)

        area_percent = 100.0 * np.count_nonzero(mask_uint8) / float(mask_uint8.size)
        visualization_path = RESULT_DIR / f"{image_id}_analysis.png"
        save_detection_visualization(confidence * 100.0, area_percent, visualization_path)
        model_metrics = _model_metrics()
        result_summary = {
            "image_url": f"/static/uploads/{original_name}",
            "mask_url": f"/static/results/{image_id}_mask.png",
            "overlay_url": f"/static/results/{image_id}_overlay.png",
            "boxes_url": f"/static/results/{image_id}_boxes.png",
            "ela_url": f"/static/results/{image_id}_ela.png",
            "noise_url": f"/static/results/{image_id}_noise.png",
            "explanation_url": f"/static/results/{image_id}_explanation.png",
            "explanation_label": explanation_label,
            "status": status,
            "method": method,
            "confidence": f"{confidence * 100:.1f}%",
            "confidence_percent": confidence * 100,
            "area_percent": area_percent,
            "area_label": f"{area_percent:.1f}%",
            "forgery_type": forgery_type,
            "box_count": len(boxes),
            "boxes": boxes,
            "dimensions": f"{image.shape[1]} x {image.shape[0]}",
            "suspected_pixels": int(np.count_nonzero(mask_uint8)),
            "model_warning": model_warning,
            "metadata": metadata,
            "analysis_url": f"/static/results/{image_id}_analysis.png",
            "model_metrics": model_metrics,
            "classification_metrics": model_metrics.get("classification") if model_metrics else None,
            "segmentation_metrics": model_metrics.get("segmentation") if model_metrics else None,
            "metrics_dataset": model_metrics.get("dataset") if model_metrics else None,
            "classification_plot_url": "/static/evaluation/classification_metrics.png" if model_metrics else None,
            "confusion_plot_url": "/static/evaluation/confusion_matrix.png" if model_metrics else None,
            "training_loss_plot_url": "/static/evaluation/training_loss.png" if (model_metrics and (BASE_DIR / "static" / "evaluation" / "training_loss.png").is_file()) else None,
        }

        report_path = RESULT_DIR / f"{image_id}_report.pdf"
        try:
            create_forensic_report(report_path, original_path, mask_path, boxes_path, result_summary, metadata)
            result_summary["report_url"] = f"/static/results/{image_id}_report.pdf"
            result_summary["report_error"] = None
        except RuntimeError as error:
            result_summary["report_url"] = None
            result_summary["report_error"] = str(error)

        return render_template("result.html", **result_summary)

    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
