# tools/export_keras.py
from pathlib import Path
import tensorflow as tf

IMG_SIZE    = 512
CHANNELS    = 3
BASE_CH     = 32
DEPTH       = 3
NUM_CLASSES = 1

ROOT = Path(__file__).resolve().parents[1]
ML   = ROOT / "ml"
H5_PATH   = ML / "model.weights.h5"
KERAS_OUT = ML / "dental_unet_disease.keras"

assert H5_PATH.exists(), f"Missing weights: {H5_PATH}"

from ml.model_inference import build_unet

print(f"[export] Building UNet size={IMG_SIZE} ch={CHANNELS} base={BASE_CH} depth={DEPTH} classes={NUM_CLASSES}")
model = build_unet(IMG_SIZE, CHANNELS, NUM_CLASSES, BASE_CH, DEPTH)

print(f"[export] Loading weights: {H5_PATH}")
model.load_weights(str(H5_PATH))

if KERAS_OUT.exists() and KERAS_OUT.stat().st_size == 0:
    KERAS_OUT.unlink()

print(f"[export] Saving .keras  {KERAS_OUT}")
model.save(str(KERAS_OUT), include_optimizer=False)
print(f"[export] Done. Size = {KERAS_OUT.stat().st_size} bytes")
