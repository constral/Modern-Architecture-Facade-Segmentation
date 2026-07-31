#!/usr/bin/env python3
# infer_minimal.py - minimal segmentation inference (no checks/fallbacks)

from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf

# ---------------- CONFIG (edit as needed) ----------------
MODEL_PATH = Path("models/unet_basic/20260113_170247_best_wmiou_0.8449.keras")
INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("outputs")
OVERLAY_ALPHA = 0.45


# colors for map
LABEL_TO_COLOR = {
    0: (0, 0, 0),
    1: (128, 0, 128),
    2: (128, 0, 0),
    3: (0, 128, 0),
    4: (128, 128, 0),
    5: (0, 0, 128),
}
NUM_CLASSES = len(LABEL_TO_COLOR)




# model loading and input shape
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
inp_shape = model.inputs[0].shape
in_h = int(inp_shape[1])
in_w = int(inp_shape[2])


# parse input images
image_paths = sorted([p for p in INPUT_DIR.glob("*") if p.suffix == ".jpg"])

for img_path in image_paths:
    img = Image.open(img_path).convert("RGB")
    basename = img_path.stem

    # copy original to compare output
    img.save(OUTPUT_DIR / f"{basename}_original.jpg")

    # preprocess input jpgs
    orig_w, orig_h = img.size
    img_rs = img.resize((in_w, in_h), resample=Image.BILINEAR)
    arr = np.array(img_rs).astype(np.float32) / 255.0
    input = np.expand_dims(arr, axis=0)

    # infer 
    preds = model.predict(input)


	

	lab = np.argmax(preds[0], axis=-1).astype(np.uint8)
    # reduce to (H,W) label map
    # if preds.ndim == 4:
    #     lab = np.argmax(preds[0], axis=-1).astype(np.uint8)
    # elif preds.ndim == 3:
    #     lab = np.argmax(preds, axis=-1).astype(np.uint8)
    # else:
    #     # fallback: try argmax on last axis of first element
    #     lab = np.argmax(preds[0], axis=-1).astype(np.uint8)

    # upsample to original size
    mask_img = Image.fromarray(lab, mode="L").resize((orig_w, orig_h), resample=Image.NEAREST)
    mask_arr_up = np.array(mask_img).astype(np.uint8)

    # colorize mask
    h, w = mask_arr_up.shape
    color_arr = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl, col in LABEL_TO_COLOR.items():
        color_arr[mask_arr_up == lbl] = col
    color_img = Image.fromarray(color_arr)
    color_img.save(OUTPUT_DIR / f"{basename}_mask.png")