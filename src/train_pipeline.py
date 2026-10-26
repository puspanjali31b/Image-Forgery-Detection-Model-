from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps

from src.forensic_features import FORGERY_TYPES, MODEL_IMAGE_SIZE, letterbox_image, prepare_model_input
from src.unet import UNet, dice_loss

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class ForgeryDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, mask_dir: str, labels_csv: str | None = None, image_size: int = MODEL_IMAGE_SIZE):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.image_files = sorted(path for path in self.image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        self.type_labels: dict[str, str] = {}
        if labels_csv:
            with Path(labels_csv).open(newline="", encoding="utf-8-sig") as label_file:
                for row in csv.DictReader(label_file):
                    filename = row.get("filename", "").strip()
                    forgery_type = row.get("forgery_type", "").strip()
                    if filename and forgery_type:
                        self.type_labels[Path(filename).stem.lower()] = forgery_type

        if not self.image_files:
            raise ValueError(f"No training images found in {self.image_dir}")
        missing_masks = [path.name for path in self.image_files if self._find_mask(path) is None]
        if missing_masks:
            raise ValueError(f"Missing masks for {len(missing_masks)} image(s), starting with: {missing_masks[0]}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _find_mask(self, image_path: Path) -> Path | None:
        for extension in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            candidate = self.mask_dir / f"{image_path.stem}{extension}"
            if candidate.is_file():
                return candidate
        return None

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = self.image_files[idx]
        mask_path = self._find_mask(image_path)
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for {image_path.name}")

        with Image.open(image_path) as source:
            image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as source_mask:
            mask = np.asarray(ImageOps.exif_transpose(source_mask).convert("L"), dtype=np.uint8)

        mask = letterbox_image(mask, size=self.image_size, mask=True)
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        image_tensor = prepare_model_input(image, size=self.image_size)
        image_label = torch.tensor(int(mask_tensor.any().item()), dtype=torch.long)
        type_label = self.type_labels.get(image_path.stem.lower(), "")
        normalized_type = type_label.strip().lower().replace("_", "-")
        type_index = next(
            (index for index, name in enumerate(FORGERY_TYPES) if normalized_type == name.lower().replace("_", "-")),
            -1,
        )
        return image_tensor, mask_tensor, image_label, torch.tensor(type_index, dtype=torch.long)


def train_unet(
    data_dir: str,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    labels_csv: str | None = None,
    checkpoint_path: str = "models/forgery_multitask.pt",
    image_size: int = MODEL_IMAGE_SIZE,
) -> UNet:
    train_dir = Path(data_dir) / "train"
    train_dataset = ForgeryDataset(
        str(train_dir / "images"),
        str(train_dir / "masks"),
        labels_csv=labels_csv,
        image_size=image_size,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    segmentation_criterion = torch.nn.BCEWithLogitsLoss()
    classification_criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, masks, image_labels, type_labels in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            image_labels = image_labels.to(device)
            type_labels = type_labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            segmentation_logits = outputs["segmentation_logits"]
            loss = segmentation_criterion(segmentation_logits, masks) + dice_loss(segmentation_logits, masks)
            loss = loss + classification_criterion(outputs["classification_logits"], image_labels)
            valid_types = type_labels >= 0
            if valid_types.any():
                loss = loss + functional.cross_entropy(outputs["forgery_type_logits"][valid_types], type_labels[valid_types])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss / len(train_loader):.4f}")

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "image_size": image_size}, checkpoint)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the multi-task image forgery U-Net.")
    parser.add_argument("data_dir", help="Dataset root containing train/images and train/masks")
    parser.add_argument("--labels-csv", help="Optional CSV with filename and forgery_type columns")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint", default="models/forgery_multitask.pt")
    args = parser.parse_args()
    train_unet(
        args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        labels_csv=args.labels_csv,
        checkpoint_path=args.checkpoint,
    )
