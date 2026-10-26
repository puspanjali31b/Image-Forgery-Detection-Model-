# Copy-Move Image Forgery Detection

This project is an image-forensics screening prototype with a Flask dashboard, a traditional copy-move baseline, and a trainable multi-task U-Net. The learned model uses RGB, Error Level Analysis (ELA), and SRM-inspired noise residual channels to predict a pixel mask, real/forged class, and forgery type.

## Features

- Flask web interface with drag-and-drop image upload
- Copy-move forgery detection using OpenCV SIFT + matching
- ELA and SRM-inspired noise visualization
- Multi-task U-Net with segmentation, real/forged, and forgery-type heads
- Supervision bounding boxes, Grad-CAM, EXIF metadata, and PDF report
- Explicit fallback to the traditional detector until trained weights exist

## Project structure

- `app.py` – Flask application entry point
- `src/detector.py` – copy-move detection engine
- `src/unet.py` – multi-task U-Net
- `src/forensic_features.py` – RGB, ELA, and noise-residual preprocessing
- `src/forensic_inference.py` – checkpoint inference, Grad-CAM, and boxes
- `src/train_pipeline.py` – dataset and training entry point
- `src/reporting.py` – downloadable PDF report generation
- `templates/` – HTML pages
- `static/` – CSS and generated detection outputs
- `requirements.txt` – dependencies

## Quick start

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   python app.py
   ```

4. Open the browser at:

   ```text
   http://127.0.0.1:5000
   ```

5. Open the URL and upload an image. Without a checkpoint, the app reports that it used the traditional SIFT/template-matching detector.

## Model and detection approach

The app uses a hybrid design:

- Traditional fallback: SIFT feature extraction + descriptor matching + translation consistency + mask generation
- Learned path: U-Net receives seven channels (RGB, three ELA channels, one noise-residual channel) and emits a segmentation mask plus image-level classification and forgery-type logits

ELA, noise maps, and metadata are shown for every upload. The model verdict and Grad-CAM are only used when a compatible trained checkpoint is available at `models/forgery_multitask.pt`.

## Training the U-Net

Prepare the training split in this layout. Images and masks must share a filename stem; masks are grayscale, where nonzero pixels mark manipulated regions.

```text
data/
   train/
      images/
         image_001.jpg
      masks/
         image_001.png
      labels.csv
```

The optional `labels.csv` contains `filename,forgery_type` columns. Supported type labels are `Splicing`, `Copy-Move`, and `Deepfake/AI-Generated`; missing labels are ignored for the type loss. Image-level real/forged labels are derived from whether each mask contains any marked pixels.

Train and save the default checkpoint:

```bash
python -m src.train_pipeline data --labels-csv data/train/labels.csv --epochs 20 --batch-size 4
```

The checkpoint is written to `models/forgery_multitask.pt`. Restart or refresh the Flask app after training. You can choose another output using `--checkpoint path/to/model.pt`.

## Outputs and limitations

- The dashboard shows the input image, predicted mask, highlighted regions, bounding boxes, ELA map, noise residual, explanation view, and available metadata.
- Grad-CAM is generated from the trained model's classification head. Before training, the explanation image is the traditional detector overlay, and the UI says so.
- Confidence and area percentages describe model outputs, not calibrated forensic probabilities. The traditional fallback's confidence is a heuristic.
- The PDF includes the original, mask, box annotation, metadata, and summary. It is an automated screening report, not a certified forensic opinion or proof of authenticity.
- Supervision creates boxes from connected components of the binary mask. ReportLab is required for PDF downloads.

## Datasets for research extension

The guide recommends starting with:

- CoMoFoD
- MICC-F220
- MICC-F2000
- CASIA (carefully filtered to copy-move samples)

The code is structured so you can add your dataset directory and train the U-Net after preparing image-mask pairs.

## Notes

- This is a prototype, not a guarantee of forensic validity for every image.
- The model should be treated as a suspicious-region detector and not as proof of image authenticity.
- The app stores uploaded files and output masks in the project workspace for demo purposes.

## Credits

This implementation is based on the project guidance provided for a copy-move forgery detection system using deep learning and digital forensics.
