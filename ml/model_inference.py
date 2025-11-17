# ml/model_inference.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import time

import h5py
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

L = tf.keras.layers
M = tf.keras.models

# ---------- Singleton caches ----------
_MODEL: Optional[tf.keras.Model] = None
_LAST_BG: Optional[np.ndarray] = None  # last 512x512 RGB copy for display

# ---------- Paths ----------
HERE = Path(__file__).resolve().parent          # .../ED2_TEST/ml
APP_ROOT = HERE.parent                          # .../ED2_TEST
STATIC_OUT = APP_ROOT / "static" / "outputs"

KERAS_PATH  = HERE / "dental_unet_disease.keras"
H5_PATH     = HERE / "model.weights.h5"
CONFIG_PATH = HERE / "config.json"
LABELS_PATH = HERE / "condition_labels.json"

# ---------- Helpers ----------
def _read_json(p: Path, default):
    """Safe JSON reader that also tolerates UTF-8 BOMs."""
    try:
        with p.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return default


CONFIG: Dict[str, Any] = _read_json(CONFIG_PATH, {})
LABELS_RAW = _read_json(LABELS_PATH, [])


def _normalize_labels(raw) -> List[str]:
    """Allow labels to be list or dict {'0': 'Cond0', ...}."""
    if isinstance(raw, dict):
        try:
            # sort by integer key if possible
            return [raw[k] for k in sorted(raw.keys(), key=lambda x: int(x))]
        except Exception:
            # fallback: dict iteration order
            return list(raw.values())
    elif isinstance(raw, list):
        return raw
    else:
        return []


LABELS: List[str] = _normalize_labels(LABELS_RAW)

# ---------- Defaults / config ----------
DEFAULT_IMG_SIZE      = 512
DEFAULT_CHANNELS      = 3
DEFAULT_THRESHOLD     = 0.5
DEFAULT_BASE_CH       = 64
DEFAULT_DEPTH         = 3
DEFAULT_NUM_CLASSES   = int(CONFIG.get("num_classes", 1))  # used only for fallback UNet

IMG_SIZE    = int(CONFIG.get("img_size", DEFAULT_IMG_SIZE))
N_CHANNELS  = int(CONFIG.get("channels", DEFAULT_CHANNELS))
THRESH      = float(CONFIG.get("mask_threshold", DEFAULT_THRESHOLD))
BASE_CH     = int(CONFIG.get("base_channels", DEFAULT_BASE_CH))
DEPTH       = int(CONFIG.get("depth", DEFAULT_DEPTH))
NUM_CLASSES = int(CONFIG.get("num_classes", DEFAULT_NUM_CLASSES))

# ---------- Colors / class names ----------
def _distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate n visually distinct RGB colors.
    Index 0 is reserved for 'background'.
    """
    import colorsys
    if n <= 0:
        return []
    cols = [(0, 0, 0)]  # bg
    k = max(1, n - 1)
    for i in range(k):
        h = (i / k) % 1.0
        s = 0.65
        v = 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return cols


# These names are for the **condition head** classes.
CLASS_NAMES: List[str] = LABELS if LABELS else [f"class_{i}" for i in range(NUM_CLASSES)]

# Initial colors (will be overridden in InferenceModel if LABELS are reloaded)
CLASS_COLORS: List[Tuple[int, int, int]] = _distinct_colors(max(1, len(CLASS_NAMES)))

# ---------- UNet (fallback, not used for your .keras) ----------
def _conv_block(x, c, name):
    x = L.Conv2D(c, 3, padding="same", activation="relu", name=name + "_c1")(x)
    x = L.Conv2D(c, 3, padding="same", activation="relu", name=name + "_c2")(x)
    return x


def build_unet(
    img_size: int,
    channels: int,
    num_classes: int,
    base_ch: int = 32,
    depth: int = 4,
) -> tf.keras.Model:
    inp = L.Input(shape=(img_size, img_size, channels), name="input")
    skips = []
    x = inp
    c = base_ch

    # Down path
    for d in range(depth):
        x = _conv_block(x, c, name=f"down{d}")
        skips.append(x)
        x = L.MaxPool2D(2, name=f"pool{d}")(x)
        c *= 2

    # Bottleneck
    x = _conv_block(x, c, name="bottleneck")

    # Up path
    for d in reversed(range(depth)):
        c //= 2
        x = L.Conv2DTranspose(c, 2, strides=2, padding="same", name=f"up{d}")(x)
        x = L.Concatenate(name=f"concat{d}")([x, skips[d]])
        x = _conv_block(x, c, name=f"upblk{d}")

    # Output head
    if num_classes == 1:
        out = L.Conv2D(1, 1, activation="sigmoid", name="output")(x)
    else:
        out = L.Conv2D(num_classes, 1, activation="softmax", name="output")(x)

    return M.Model(inp, out, name="UNet")

# ---------- (optional) Inspect H5 to guess specs ----------
def _infer_specs_from_h5(h5_path: Path, depth: int) -> Tuple[int, int]:
    """
    Infer (base_ch, num_classes) by inspecting an H5 checkpoint.
    (Kept for compatibility; not strictly required for your flow.)
    """
    if not h5_path.exists() or h5_path.stat().st_size == 0:
        raise FileNotFoundError(f"Checkpoint not found or empty: {h5_path}")

    upblk_key = f"upblk{depth-1}_c1"
    out_filters = None
    num_classes = None

    with h5py.File(str(h5_path), "r") as f:
        def walk(g, path=""):
            nonlocal out_filters, num_classes
            for k, v in g.items():
                if isinstance(v, h5py.Group):
                    walk(v, f"{path}/{k}" if path else k)
                else:
                    if k == "kernel:0":
                        shape = v[()].shape  # (kh, kw, inC, outC)
                        if "output" in path and num_classes is None:
                            num_classes = int(shape[-1])
                        if upblk_key in path and out_filters is None:
                            out_filters = int(shape[-1])
        walk(f)

    if out_filters is None:
        with h5py.File(str(h5_path), "r") as f:
            max_seen = None
            def walk2(g, path=""):
                nonlocal max_seen
                for k, v in g.items():
                    if isinstance(v, h5py.Group):
                        walk2(v, f"{path}/{k}" if path else k)
                    else:
                        if k == "kernel:0" and "upblk" in path and "_c1" in path:
                            val = int(v[()].shape[-1])
                            max_seen = val if (max_seen is None or val > max_seen) else max_seen
            walk2(f)
            out_filters = max_seen or 256

    if num_classes is None:
        num_classes = 1

    base_ch = int(out_filters // (2 ** (depth - 1)))
    base_ch = max(base_ch, 1)
    print(f"[model] H5 says upblk{depth-1}_c1 has {out_filters} filters -> base_ch={base_ch}, num_classes={num_classes}")
    return base_ch, num_classes

# ---------- Model loader ----------
def _load_model() -> tf.keras.Model:
    """
    Prefer loading the full .keras model. If not present, build UNet and load
    weights from H5.
    """
    # 1) Prefer a valid .keras if it exists and is non-empty
    if KERAS_PATH.exists() and KERAS_PATH.stat().st_size > 0:
        print(f"[model] Loading FULL Keras model: {KERAS_PATH}")
        return tf.keras.models.load_model(KERAS_PATH, compile=False)

    # 2) Build ORIGINAL architecture (fallback; not used with your multi-head .keras)
    print(
        f"[model] Building UNet with original params: "
        f"img_size={IMG_SIZE}, channels={N_CHANNELS}, "
        f"base_ch={BASE_CH}, depth={DEPTH}, num_classes={NUM_CLASSES}"
    )
    model = build_unet(
        img_size=IMG_SIZE,
        channels=N_CHANNELS,
        num_classes=NUM_CLASSES,
        base_ch=BASE_CH,
        depth=DEPTH,
    )

    # 3) Load weights (strict first, then tolerant fallback)
    if H5_PATH.exists() and H5_PATH.stat().st_size > 0:
        try:
            print(f"[model] Loading weights (strict): {H5_PATH}")
            model.load_weights(str(H5_PATH))
            print("[model] Weights loaded strictly.")
        except Exception as e:
            print(f"[model] Strict load failed ({e}); retrying with skip_mismatch=True …")
            model.load_weights(str(H5_PATH), skip_mismatch=True)
            print("[model] Weights loaded with skip_mismatch=True (partial).")
    else:
        print(f"[model] WARNING: weights file not found or empty: {H5_PATH}")

    # 4) Optionally cache a .keras
    try:
        if not KERAS_PATH.exists() or KERAS_PATH.stat().st_size == 0:
            model.save(KERAS_PATH, include_optimizer=False)
            print(f"[model] Cached model to {KERAS_PATH}")
    except Exception:
        pass

    return model


def get_model() -> tf.keras.Model:
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL

# ---------- Pre/Post ----------
def resize_pad_square(
    img_bgr: np.ndarray,
    out_size: int = 512,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Letterbox-resize an image to out_size × out_size with padding.
    """
    h, w = img_bgr.shape[:2]
    s = out_size
    scale = min(s / h, s / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((s, s, 3), pad_color, dtype=resized.dtype)
    y0 = (s - nh) // 2
    x0 = (s - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _prep_from_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Normalize and resize input, cache a 512×512 RGB copy for overlays.
    Uses letterbox padding (resize_pad_square) to mimic training.
    """
    global _LAST_BG
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty image")

    img512 = resize_pad_square(img_bgr, IMG_SIZE)
    rgb512 = cv2.cvtColor(img512, cv2.COLOR_BGR2RGB)
    _LAST_BG = rgb512.copy()          # RGB 512×512, used as “masked input”
    arr = (rgb512.astype(np.float32) / 255.0)
    return arr[None, ...]             # (1, H, W, 3)


def _save_png_uint8(arr: np.ndarray, stem: str) -> str:
    """
    Save a uint8 RGB or grayscale image under static/outputs and return its URL.
    """
    STATIC_OUT.mkdir(parents=True, exist_ok=True)
    fname = f"{stem}_{int(time.time() * 1000)}.png"
    outp = STATIC_OUT / fname
    Image.fromarray(arr).save(outp)
    return f"/static/outputs/{fname}"

# ---------- Core postprocessing ----------
def _smooth_binary_mask(mask: np.ndarray, min_frac: float = 5e-4) -> np.ndarray:
    """
    mask: (H, W) binary 0/1
    Returns a cleaned mask with small specks removed and morphological smoothing.
    """
    h, w = mask.shape
    if mask.sum() == 0:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    # Remove very small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    cleaned = np.zeros_like(m)
    min_area = int(min_frac * h * w)
    for lab in range(1, num_labels):  # skip background 0
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == lab] = 1

    return cleaned


def _select_condition_head_output(y: Any) -> np.ndarray:
    """
    Your model has 3 heads:
        quadrant_head   (None,512,512,5)
        tooth_head      (None,512,512,9)
        condition_head  (None,512,512,5)

    This function extracts the **condition_head** output.

    y can be:
        - a single np.ndarray (if model has 1 output)
        - a list/tuple of outputs (quadrant, tooth, condition)
    """
    if isinstance(y, (list, tuple)):
        # Expected order from your model summary:
        # [quadrant_head, tooth_head, condition_head]
        if len(y) >= 3:
            return np.asarray(y[2])
        return np.asarray(y[-1])
    return np.asarray(y)


def _postprocess(y: Any, orig_size=None) -> Dict[str, Any]:
    """
    MULTI-CLASS postprocessing for the **condition** head output:
      y_cond: (1,H,W,C_cond) where C_cond >= 2

    We keep everything at model resolution (512×512 letterboxed) so that
    segmentation aligns perfectly with the stored _LAST_BG.
    """
    global _LAST_BG, CLASS_NAMES, CLASS_COLORS

    # --- extract condition head and apply softmax once ---
    y_cond = _select_condition_head_output(y)             # (1,H,W,C)
    y_cond = tf.nn.softmax(y_cond, axis=-1).numpy()       # probs in [0,1]

    assert y_cond.ndim == 4 and y_cond.shape[0] == 1, f"Expected (1,H,W,C) output, got {y_cond.shape}"
    h, w, C = y_cond.shape[1:]  # model H, W (e.g., 512×512)

    # ---- argmax over classes ----
    cls_map = np.argmax(y_cond[0], axis=-1).astype(np.uint8)  # (H,W)

    # We **do not** resize to orig_size. Stay in model space.
    tgt_w, tgt_h = w, h

    # ---- restore model-space masked input background ----
    if _LAST_BG is not None:
        bg = cv2.resize(_LAST_BG, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
    else:
        bg = np.zeros((tgt_h, tgt_w, 3), dtype=np.uint8)

    overlay = bg.copy()

    # ---- per-class mean probability over predicted region ----
    prob_conf: List[float] = []
    for i in range(C):
        mask = (cls_map == i)
        if mask.sum() == 0:
            prob_conf.append(0.0)
        else:
            p = float(y_cond[0, :, :, i][mask].mean())  # in [0,1]
            prob_conf.append(p)

    # ---- ensure we have enough colors / names ----
    if len(CLASS_COLORS) < C:
        CLASS_COLORS = _distinct_colors(C)
    if len(CLASS_NAMES) < C:
        CLASS_NAMES = CLASS_NAMES + [f"class_{i}" for i in range(len(CLASS_NAMES), C)]

    # ---- color overlay for non-background classes ----
    alpha = 0.40
    for i in range(1, C):  # skip background class 0
        mask = (cls_map == i)
        if not mask.any():
            continue
        color = CLASS_COLORS[i] if i < len(CLASS_COLORS) else (255, 0, 0)
        overlay[mask] = (
            (1 - alpha) * overlay[mask] +
            alpha * np.array(color, dtype=np.uint8)
        ).astype(np.uint8)

    # ---- draw contour outlines for each class ----
    for i in range(1, C):
        m = (cls_map == i).astype(np.uint8) * 255
        if m.sum() == 0:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 2)

    # ---- legend rows ----
    legend_rows = []
    for i in range(1, C):  # skip background 0
        legend_rows.append({
            "label": CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}",
            "p": prob_conf[i],
            "idx": i,
        })

    # sort by probability
    legend_rows.sort(key=lambda r: r["p"], reverse=True)
    # drop near-zero
    legend_rows = [r for r in legend_rows if r["p"] > 0.001]

    if not legend_rows:
        legend_rows = [{"label": "No detectable condition", "p": 0.0, "idx": 0}]

    # ---- legend panel drawing ----
    panel_h = 35 + 30 * len(legend_rows)
    cv2.rectangle(overlay, (0, 0), (420, panel_h), (0, 0, 0), -1)

    cv2.putText(
        overlay,
        "Predicted Conditions:",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for i, row in enumerate(legend_rows):
        name = row["label"]
        pct = row["p"] * 100.0
        color = CLASS_COLORS[row["idx"]] if row["idx"] < len(CLASS_COLORS) else (255, 255, 255)
        y = 55 + i * 30

        # color box
        cv2.rectangle(
            overlay,
            (10, y - 18),
            (30, y + 2),
            color,
            -1,
        )

        # text
        cv2.putText(
            overlay,
            f"{name}: {pct:.1f}%",
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

    # ---- save artifacts ----
    # masked image: 512×512 pano (what you see on the left)
    if _LAST_BG is not None:
        masked_vis = cv2.resize(_LAST_BG, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
    else:
        masked_vis = np.zeros_like(overlay)

    mask_url    = _save_png_uint8(masked_vis, "maskMC")
    overlay_url = _save_png_uint8(overlay, "overlayMC")

    pred_label = "no finding"
    pred_conf = 0.0

    top3_simple = [
        {
            "label": r["label"],
            "p": float(r["p"]),
            "text": f"{r['label']}: {r['p']*100:.1f}%",
        }
        for r in legend_rows[:3]
    ]

    return {
        "prediction": pred_label,
        "confidence": pred_conf,
        "masked_url": mask_url,
        "overlay_url": overlay_url,
        "top3": top3_simple,
    }


# ---------- Public predict() wrapper ----------
def predict(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Main inference function used by the website.

    Args:
        img_bgr: OpenCV-style BGR image array.

    Returns:
        { "results": [ postprocessed_dict ] }
    """
    model = get_model()

    # Preprocess for the model (512x512, letterboxed) and cache _LAST_BG
    x = _prep_from_bgr(img_bgr)

    # Run inference
    y = model.predict(x, verbose=0)

    # Postprocess in model (512×512) space
    res = _postprocess(y, orig_size=None)

    return {"results": [res]}

# ---------- InferenceModel for Colab-style testing ----------
class InferenceModel:
    """
    Thin wrapper used by model_testing_v1_3.py

    Keeps the same API:
      infer = InferenceModel(model_path, labels_path)
      overlay_bgr, legend_dict = infer.save_overlay(in_path, out_path)
    """

    def __init__(
        self,
        model_path: str | Path = KERAS_PATH,
        labels_path: str | Path | None = LABELS_PATH,
    ):
        # (Re)load labels if provided
        if labels_path:
            p = Path(labels_path)
            if p.exists():
                try:
                    global LABELS, CLASS_NAMES, CLASS_COLORS
                    raw = _read_json(p, LABELS_RAW)
                    LABELS = _normalize_labels(raw)
                    CLASS_NAMES = LABELS if LABELS else [f"class_{i}" for i in range(NUM_CLASSES)]
                    CLASS_COLORS = _distinct_colors(max(1, len(CLASS_NAMES)))
                except Exception as e:
                    print("[labels] Failed to load labels:", e)

        # Force-load model once so predict() uses the same weights
        _ = get_model()

    def _predict_on_file(self, image_path: str) -> Dict[str, Any]:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return predict(img)["results"][0]

    def save_overlay(self, image_path: str, out_path: str):
        """
        Run prediction on a single image_path and save final overlay to out_path.
        Returns:
            overlay_bgr (np.ndarray),
            legend (Dict[str, float]) — label -> percentage
        """
        res = self._predict_on_file(image_path)

        overlay_url = res.get("overlay_url") or res.get("masked_url")
        legend_pairs = res.get("top3") or []
        legend = {row["label"]: float(row["p"]) * 100 for row in legend_pairs}

        if overlay_url:
            fs = APP_ROOT / overlay_url.lstrip("/")
            if fs.exists():
                overlay = cv2.imread(str(fs), cv2.IMREAD_COLOR)
            else:
                overlay = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else:
            overlay = cv2.imread(image_path, cv2.IMREAD_COLOR)

        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), overlay)

        return overlay, legend
