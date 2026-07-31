import argparse, os, shutil, random, glob

def list_basenames(samples_dir):
    # find JPG/JPEG files and return basenames without extension
    files = glob.glob(os.path.join(samples_dir, "*.jpg")) + glob.glob(os.path.join(samples_dir, "*.jpeg"))
    basenames = [os.path.splitext(os.path.basename(p))[0] for p in files]
    return sorted(set(basenames))

def ensure_dirs(base_out):
    for split in ("train", "val"):
        for sub in ("images", "masks", "jsons"):
            os.makedirs(os.path.join(base_out, split, sub), exist_ok=True)

def copy_triplet(samples_dir, out_dir, base, split):
    src_img_j = os.path.join(samples_dir, base + ".jpg")
    src_img_j2 = os.path.join(samples_dir, base + ".jpeg")
    src_mask = os.path.join(samples_dir, base + ".png")
    src_json = os.path.join(samples_dir, base + ".json")

    img_src = src_img_j if os.path.exists(src_img_j) else (src_img_j2 if os.path.exists(src_img_j2) else None)
    if not img_src or not os.path.exists(src_mask):
        print(f"SKIP (missing image/mask): {base}")
        return False

    shutil.copy2(img_src, os.path.join(out_dir, split, "images", os.path.basename(img_src)))
    shutil.copy2(src_mask, os.path.join(out_dir, split, "masks", os.path.basename(src_mask)))
    if os.path.exists(src_json):
        shutil.copy2(src_json, os.path.join(out_dir, split, "jsons", os.path.basename(src_json)))
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("samples_dir")
    p.add_argument("out_dir")
    p.add_argument("val_fraction", type=float)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    basenames = list_basenames(args.samples_dir)
    if not basenames:
        print("No jpg/jpeg images found in samples_dir.")
        return
    random.seed(args.seed)
    random.shuffle(basenames)
    n_val = int(round(len(basenames) * args.val_fraction))
    val_list = basenames[:n_val]
    train_list = basenames[n_val:]

    ensure_dirs(args.out_dir)

    c_train = sum(1 for b in train_list if copy_triplet(args.samples_dir, args.out_dir, b, "train"))
    c_val   = sum(1 for b in val_list   if copy_triplet(args.samples_dir, args.out_dir, b, "val"))

    print(f"TOTAL={len(basenames)}  TRAIN={c_train}  VAL={c_val}")

if __name__ == "__main__":
    main()
