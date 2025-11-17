# ml/tools/inspect_model.py
from pathlib import Path
import sys
import tensorflow as tf

# This file lives in ml/tools, so:
#   parents[0] -> .../ED2_TEST/ml/tools
#   parents[1] -> .../ED2_TEST/ml
#   parents[2] -> .../ED2_TEST  (project root)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print(f"[inspect] Project root added to sys.path: {ROOT}")

from ml.model_inference import get_model


def main():
    print("[inspect] Loading model via ml.model_inference.get_model() ...")
    model = get_model()

    print("\n[inspect] Model summary:")
    # This prints a table of layers in the console
    model.summary()

    print("\n[inspect] Input shape :", model.input_shape)
    print("[inspect] Output shape:", model.output_shape)


if __name__ == "__main__":
    main()
