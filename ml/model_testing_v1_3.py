# -*- coding: utf-8 -*-
"""
Local test runner for the Keras dental model (VS Code friendly).
Usage examples (from ED2_TEST root, venv active):
  python -m ml.model_testing_v1_3 --image sample_data/xray1.png
  python -m ml.model_testing_v1_3 --folder sample_data/
  python -m ml.model_testing_v1_3 --image my.png --model ml/dental_unet_disease.keras --labels ml/condition_labels.json
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Make sure we can import the local 'ml' package no matter where we run from
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import your local inference helpers
# These must be defined in ED2_TEST/ml/model_inference.py
from ml.model_inference import InferenceModel, resize_pad_square  # type: ignore


def process_one_image(infer: InferenceModel, image_path: Path, out_dir: Path) -> Path:
    """Runs masking + prediction + overlay + legend for a single image."""
    # Step A: Resize/mask to 512×512
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    masked_img = resize_pad_square(img, 512)
    masked_name = out_dir / f"masked_{image_path.name}"
    cv2.imwrite(str(masked_name), masked_img)

    # Step B: Predict and get overlay from your InferenceModel
    overlay_bgr, legend = infer.save_overlay(str(masked_name), str(out_dir / f"overlay_{image_path.name}"))

    # Step C: Draw legend directly on overlay image
    overlay_with_text = overlay_bgr.copy()
    y0 = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # white text
    thickness = 2

    # Sort legend by confidence (highest first)
    sorted_items = sorted(legend.items(), key=lambda x: x[1], reverse=True)
    cv2.putText(overlay_with_text, "Predicted Conditions:", (10, y0), font, font_scale, (0, 255, 255), thickness)
    y = y0 + 25
    for label, conf in sorted_items:
        text = f"{label}: {conf:.1f}%"
        cv2.putText(overlay_with_text, text, (10, y), font, font_scale, color, thickness)
        y += 25

    # Step D: Display results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
    plt.title("512×512 Masked Input")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay_with_text, cv2.COLOR_BGR2RGB))
    plt.title("Highlighted Overlay + Legend")
    plt.axis("off")
    plt.show()

    # Step E: Save with legend
    output_name = out_dir / f"overlay_with_legend_{image_path.name}"
    cv2.imwrite(str(output_name), overlay_with_text)
    print(f"✅ Saved: {output_name}")
    return output_name


def main():
    parser = argparse.ArgumentParser(description="Local tester for dental_unet_disease.keras")
    parser.add_argument("--model", default=str(ROOT / "ml" / "dental_unet_disease.keras"),
                        help="Path to .keras model file")
    parser.add_argument("--labels", default=str(ROOT / "ml" / "condition_labels.json"),
                        help="Path to labels JSON file")
    parser.add_argument("--image", help="Path to one image")
    parser.add_argument("--folder", help="Folder of images (png/jpg/jpeg)")
    parser.add_argument("--out", default=str(ROOT / "outputs"),
                        help="Output directory")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    labels_path = Path(args.labels).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels JSON not found: {labels_path}")

    # 2️⃣ Load the model
    infer = InferenceModel(str(model_path), labels_path=str(labels_path))

    # Collect images
    images: list[Path] = []
    if args.image:
        images.append(Path(args.image))
    if args.folder:
        folder = Path(args.folder)
        images.extend(sorted(folder.glob("*.png")))
        images.extend(sorted(folder.glob("*.jpg")))
        images.extend(sorted(folder.glob("*.jpeg")))
    if not images:
        raise SystemExit("No images provided. Use --image path/to/img.png or --folder path/to/dir")

    # 3️⃣ Run prediction(s)
    for img_path in images:
        process_one_image(infer, img_path, out_dir)


if __name__ == "__main__":
    main()