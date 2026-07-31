#!/usr/bin/env python3
"""
prepare_samples.py

Run from the folder that contains `samples/`.

Reads:
  samples/train/images/*.jpg(.jpeg)
  samples/train/masks/*.png
  samples/train/jsons/*.json

and same for samples/val.

Writes:
  samples_ready/train/images/*.jpg
  samples_ready/train/masks/*.png   (single-channel label images, 0..5)
  samples_ready/train/jsons/*.json
  and same for val.

For each input sample it produces 4 processed variants:
  <basename>_orig.png/_orig.jpg
  <basename>_flip...
  <basename>_hscale...
  <basename>_vscale...

Processing steps (for each variant):
  - convert mask RGB -> integer labels if needed
  - optional horizontal flip
  - scale (if requested) using bilinear for images, nearest for masks
  - pad to square (centered), pad value 0 for both image and mask
  - resize to 256x256 (image: bilinear, mask: nearest)
"""
from pathlib import Path
from PIL import Image
import numpy as np
import os
import shutil
import sys

# color -> label mapping (exact as provided)
COLOR_TO_LABEL = {
    (0,0,0): 0,
    (128,0,128): 1,
    (128,0,0): 2,
    (0,128,0): 3,
    (128,128,0): 4,
    (0,0,128): 5,
}

# augmentation variants (deterministic)
VARIANTS = [
    ("orig", False, 1.0, 1.0),
    ("flip", True, 1.0, 1.0),
    ("hscale", False, 1.10, 0.90),   # wider, slightly shorter
    ("vscale", False, 0.90, 1.10),   # narrower, slightly taller
]

OUT_BASE = Path("samples_ready")
IN_BASE = Path("samples_sorted")
TARGET = 256

def ensure_dirs():
    for split in ("train", "val"):
        for typ in ("images", "masks", "jsons"):
            p = OUT_BASE / split / typ
            p.mkdir(parents=True, exist_ok=True)

def find_samples(split):
    images_dir = IN_BASE / split / "images"
    masks_dir  = IN_BASE / split / "masks"
    jsons_dir  = IN_BASE / split / "jsons"
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        imgs.extend(sorted(images_dir.glob(ext)))
    basenames = [p.stem for p in imgs]
    samples = []
    for b in basenames:
        img_path = None
        for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
            p = images_dir / (b + ext)
            if p.exists():
                img_path = p
                break
        mask_path = masks_dir / (b + ".png")
        json_path = jsons_dir / (b + ".json")
        samples.append((b, img_path, mask_path, json_path))
    return samples

def convert_mask_to_label(pil_mask):
    """
    Accepts a PIL Image.
    If mode == 'L' assume it's already single-channel labels -> return as np.uint8.
    If mode == 'RGB' or 'RGBA' convert colors -> labels using mapping.
    Returns a 2D numpy array dtype uint8 with values 0..5.
    """
    if pil_mask.mode == 'L':
        arr = np.array(pil_mask, dtype=np.uint8)
        return arr
    rgb = pil_mask.convert('RGB')
    arr = np.array(rgb, dtype=np.uint8)
    h, w, _ = arr.shape
    label = np.zeros((h, w), dtype=np.uint8)
    # vectorized mapping
    # construct view for comparisons
    for color, lbl in COLOR_TO_LABEL.items():
        matches = (arr[:,:,0] == color[0]) & (arr[:,:,1] == color[1]) & (arr[:,:,2] == color[2])
        label[matches] = lbl
    return label

def pad_to_square(img, fill=0):
    w, h = img.size
    side = max(w, h)
    # create new image and paste centered
    if img.mode == 'L':
        new = Image.new('L', (side, side), color=fill)
    else:
        new = Image.new('RGB', (side, side), color=(fill, fill, fill))
    left = (side - w) // 2
    top  = (side - h) // 2
    new.paste(img, (left, top))
    return new

def process_and_save(basename, img_path, mask_path, json_path, split):
    """
    For one sample, perform the 4 deterministic variants and save outputs.
    """
    # load image
    img = Image.open(img_path).convert('RGB')
    # load mask: if missing -> skip (user said masks exist)
    if not mask_path.exists():
        print(f"SKIP {basename}: missing mask {mask_path}")
        return 0

    mask_pil_raw = Image.open(mask_path)
    # convert mask to label numpy array
    label_arr = convert_mask_to_label(mask_pil_raw)  # 2D uint8
    # wrap into PIL Image mode 'L' for transformations
    mask = Image.fromarray(label_arr, mode='L')

    saved = 0
    for suffix, do_flip, sx, sy in VARIANTS:
        # start from originals each time
        img_variant = img
        mask_variant = mask

        # flip
        if do_flip:
            img_variant = img_variant.transpose(Image.FLIP_LEFT_RIGHT)
            mask_variant = mask_variant.transpose(Image.FLIP_LEFT_RIGHT)

        # scale (sx, sy)
        if not (sx == 1.0 and sy == 1.0):
            orig_w, orig_h = img_variant.size
            new_w = max(1, int(round(orig_w * sx)))
            new_h = max(1, int(round(orig_h * sy)))
            img_variant = img_variant.resize((new_w, new_h), resample=Image.BILINEAR)
            mask_variant = mask_variant.resize((new_w, new_h), resample=Image.NEAREST)

        # pad to square (center)
        img_variant = pad_to_square(img_variant, fill=0)
        mask_variant = pad_to_square(mask_variant, fill=0)

        # final resize to TARGET x TARGET
        img_variant = img_variant.resize((TARGET, TARGET), resample=Image.BILINEAR)
        mask_variant = mask_variant.resize((TARGET, TARGET), resample=Image.NEAREST)

        # save
        out_img_name = f"{basename}_{suffix}.jpg"
        out_mask_name = f"{basename}_{suffix}.png"

        out_img_path = OUT_BASE / split / "images" / out_img_name
        out_mask_path = OUT_BASE / split / "masks"  / out_mask_name

        img_variant.save(out_img_path, quality=95)
        # ensure mask is single-channel 'L' and dtype uint8
        if mask_variant.mode != 'L':
            mask_variant = mask_variant.convert('L')
        mask_variant.save(out_mask_path)
        saved += 1

    # copy json if exists
    if json_path and json_path.exists():
        dst_json = OUT_BASE / split / "jsons" / (basename + ".json")
        shutil.copy2(json_path, dst_json)

    return saved

def main():
    if not IN_BASE.exists():
        print("ERROR: 'samples/' directory not found in current folder.")
        sys.exit(1)
    ensure_dirs()
    total_saved = 0
    for split in ("train", "val"):
        samples = find_samples(split)
        print(f"[{split}] Found {len(samples)} candidate samples (images found).")
        count = 0
        skip = 0
        for (b, img_p, mask_p, json_p) in samples:
            if img_p is None or not img_p.exists():
                print(f"  SKIP {b}: missing image")
                skip += 1
                continue
            if not mask_p.exists():
                print(f"  SKIP {b}: missing mask")
                skip += 1
                continue
            n = process_and_save(b, img_p, mask_p, json_p, split)
            if n == 0:
                skip += 1
            else:
                count += 1
                total_saved += n
            if (count + skip) % 100 == 0:
                print(f"  processed {count}  skipped {skip}")
        print(f"[{split}] processed={count}  skipped={skip}  variants_saved={count * len(VARIANTS)}")
    print("Total saved files (masks+images per variant counted per-sample):", total_saved)
    print("Done. Output root:", OUT_BASE.resolve())

if __name__ == "__main__":
    main()
