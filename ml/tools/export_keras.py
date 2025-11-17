# tools/export_keras.py
from pathlib import Path
import sys
import tensorflow as tf

# --- Make project root importable so "ml" resolves ---
ROOT = Path(__file__).resolve().parents[1]      # .../ED2_TEST
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Now "ml" can be imported
from ml.model_inference import build_unet  # uses same layer names: down*, up*, output

# ---- Hyperparams must MATCH the training of model.weights.h5 ----
IMG_SIZE    = 512
CHANNELS    = 3
BASE_CH     = 32
DEPTH       = 3          # <- Your weight errors showed bottleneck=256, so depth=3
NUM_CLASSES = 1

ML_DIR     = ROOT / "ml"
H5_PATH    = ML_DIR / "model.weights.h5"
KERAS_OUT  = ML_DIR / "dental_unet_disease.keras"

assert H5_PATH.exists(), f"Missing weights file: {H5_PATH}"

print(f"[export] Building UNet size={IMG_SIZE} ch={CHANNELS} base={BASE_CH} depth={DEPTH} classes={NUM_CLASSES}")
model = build_unet(IMG_SIZE, CHANNELS, NUM_CLASSES, BASE_CH, DEPTH)

print(f"[export] Loading weights: {H5_PATH}")
model.load_weights(str(H5_PATH))   # will raise if any shape/name mismatch

# If an old zero-byte file exists, remove it first
if KERAS_OUT.exists() and KERAS_OUT.stat().st_size == 0:
    KERAS_OUT.unlink()

print(f"[export] Saving .keras -> {KERAS_OUT}")
model.save(str(KERAS_OUT), include_optimizer=False)
print(f"[export] Done. Size = {KERAS_OUT.stat().st_size} bytes")
