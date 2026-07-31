# force cpu before tensorflow is imported by hiding all gpus
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""





from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf

# GPU:
# BASE_FILTERS = 32
# CPU:
BASE_FILTERS = 8
BATCH_SIZE = 8
IMG_SIZE = (256, 256)
NUM_CLASSES = 6
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE_RLR = 3
PATIENCE_ES = 9
MIN_LR = 1e-6
MODELS_ROOT = Path("models")
MODELS_ROOT.mkdir(parents=True, exist_ok=True)
tf.random.set_seed(1337)
np.random.seed(69)






# Use small filters (3×3 pixels) to scan the image and find features
# Applies ReLU to add non-linearity and help the model to learn better
def conv_block(x, filters, kernel_size=3):
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

# Increase image size to get back the original image size
def decoder_block(x, skip):
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    return x



# standard U-Net encoder-decoder
# encoder shrinks the image to capture features, then decoder upsamples to restore size using skip connections (more conv blocks) to keep details
def build_unet_basic(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES, base_filters=BASE_FILTERS):

    inp = tf.keras.Input(shape=input_shape)

    # Encoder

    c1 = conv_block(inp, base_filters)
    p1 = tf.keras.layers.MaxPool2D()(c1)

    c2 = conv_block(p1, base_filters * 2)
    p2 = tf.keras.layers.MaxPool2D()(c2)

    c3 = conv_block(p2, base_filters * 4)
    p3 = tf.keras.layers.MaxPool2D()(c3)

    c4 = conv_block(p3, base_filters * 8)
    p4 = tf.keras.layers.MaxPool2D()(c4)



    # Bottleneck links the encoder and decoder

    b = conv_block(p4, base_filters * 16)



    # Decoder

    u4 = decoder_block(b, c4)
    # Use convolution layers again to clean up and refine the output
    d4 = conv_block(u4, base_filters * 8)

    u3 = decoder_block(d4, c3)
    d3 = conv_block(u3, base_filters * 4)

    u2 = decoder_block(d3, c2)
    d2 = conv_block(u2, base_filters * 2)

    u1 = decoder_block(d2, c1)
    d1 = conv_block(u1, base_filters)

    seg_out = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax", name="seg_out")(d1)

    return tf.keras.Model(inputs=inp, outputs=seg_out, name="unet_basic")



# custom callback to:
# - compute per-class IoU and weighted-mean IoU on the whole validation dataset at the end of each epoch
# - save the model when WMIoU improves
class ValIoUCallback(tf.keras.callbacks.Callback):
    """
    Validation callback computing WMIoU and mean IoU. Robustly handles
    val_dataset y_batch being tensor/list/tuple/dict and multi-scale targets.
    """
    def __init__(self, val_dataset, arch_name, num_classes=NUM_CLASSES):
        super().__init__()
        self.val_dataset = val_dataset
        self.num_classes = int(num_classes)
        self.arch_name = str(arch_name)
        self.best_wmiou = -1.0

    def _select_seg_from_preds(self, preds):
        """Given model.predict_on_batch output, select segmentation array (B,H,W,C) where C==num_classes."""
        if isinstance(preds, (list, tuple)):
            for p in preds:
                a = np.array(p)
                if a.ndim == 4 and a.shape[-1] == self.num_classes:
                    return a
            # fallback to first element
            return np.array(preds[0])
        return np.array(preds)

    def _find_matching_true_flat(self, pred_labels_flat_len, y_batch):
        """
        Find among y_batch entries the one whose flattened length equals pred_labels_flat_len.
        y_batch may be tensor, list/tuple, or dict.
        Returns flattened int32 numpy array.
        """
        # If it's a plain tensor-like
        if isinstance(y_batch, tf.Tensor) or not isinstance(y_batch, (list, tuple, dict)):
            arr = np.array(y_batch)
            if arr.ndim == 4 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            return arr.ravel().astype(np.int32)

        # If list/tuple, search entries
        if isinstance(y_batch, (list, tuple)):
            for el in y_batch:
                a = np.array(el)
                if a.ndim == 4 and a.shape[-1] == 1:
                    a_flat_len = a[...,0].ravel().size
                    if a_flat_len == pred_labels_flat_len:
                        return a[...,0].ravel().astype(np.int32)
                else:
                    a_flat_len = a.ravel().size
                    if a_flat_len == pred_labels_flat_len:
                        return a.ravel().astype(np.int32)
            # fallback: try first element
            a = np.array(y_batch[0])
            if a.ndim == 4 and a.shape[-1] == 1:
                return a[...,0].ravel().astype(np.int32)
            return a.ravel().astype(np.int32)

        # If dict, check values
        if isinstance(y_batch, dict):
            for k, v in y_batch.items():
                a = np.array(v)
                if a.ndim == 4 and a.shape[-1] == 1:
                    a_flat_len = a[...,0].ravel().size
                    if a_flat_len == pred_labels_flat_len:
                        return a[...,0].ravel().astype(np.int32)
                else:
                    a_flat_len = a.ravel().size
                    if a_flat_len == pred_labels_flat_len:
                        return a.ravel().astype(np.int32)
            # fallback: pick first value
            first = next(iter(y_batch.values()))
            a = np.array(first)
            if a.ndim == 4 and a.shape[-1] == 1:
                return a[...,0].ravel().astype(np.int32)
            return a.ravel().astype(np.int32)

        # ultimate fallback
        return np.array(y_batch).ravel().astype(np.int32)

    def on_epoch_end(self, epoch, logs=None):
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        for batch in self.val_dataset:
            x_batch, y_batch = batch
            preds = self.model.predict_on_batch(x_batch)
            preds_arr = self._select_seg_from_preds(preds)

            # build predicted labels (flattened)
            if preds_arr.ndim == 4 and preds_arr.shape[-1] == self.num_classes:
                pred_labels = np.argmax(preds_arr, axis=-1).ravel().astype(np.int32)
            elif preds_arr.ndim == 4 and preds_arr.shape[-1] == 1:
                pred_labels = preds_arr[...,0].ravel().astype(np.int32)
            elif preds_arr.ndim == 3:
                # (B,H,W) or (H,W,C)
                if preds_arr.shape[-1] == self.num_classes:
                    pred_labels = np.argmax(preds_arr, axis=-1).ravel().astype(np.int32)
                else:
                    pred_labels = preds_arr.ravel().astype(np.int32)
            else:
                pred_labels = preds_arr.ravel().astype(np.int32)

            # match true labels by flattened length
            true_labels = self._find_matching_true_flat(pred_labels.size, y_batch)

            if pred_labels.shape[0] != true_labels.shape[0]:
                raise RuntimeError(f"Prediction/label size mismatch: {pred_labels.shape[0]} vs {true_labels.shape[0]}")

            idx = true_labels * self.num_classes + pred_labels
            binc = np.bincount(idx, minlength=self.num_classes * self.num_classes)
            cm += binc.reshape((self.num_classes, self.num_classes))

        intersection = np.diag(cm).astype(float)
        true_pixels = cm.sum(axis=1).astype(float)
        pred_pixels = cm.sum(axis=0).astype(float)
        union = true_pixels + pred_pixels - intersection

        iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union>0)
        valid = union > 0
        mean_iou = float(np.mean(iou[valid])) if np.any(valid) else 0.0

        total_true_pixels = float(np.sum(true_pixels))
        if total_true_pixels > 0.0:
            weighted_sum = float(np.sum(true_pixels * iou))
            wm_iou = weighted_sum / total_true_pixels
        else:
            wm_iou = 0.0

        logs = logs or {}
        logs["val_mean_iou"] = mean_iou
        logs["val_wm_iou"] = wm_iou
        print(f"\n[ValIoU] epoch={epoch+1} mean_iou={mean_iou:.4f} wmIoU={wm_iou:.4f} per_class={iou.tolist()}")

        if wm_iou > self.best_wmiou:
            self.best_wmiou = wm_iou
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_wm = f"{wm_iou:.4f}"
            arch_dir = MODELS_ROOT / self.arch_name
            arch_dir.mkdir(parents=True, exist_ok=True)
            base_name = f"{ts}_best_wmiou_{safe_wm}"
            save_path = arch_dir / f"{base_name}.keras"

            try:
                print(" Saving .keras ->", save_path)
                self.model.save(str(save_path))
                print(" .keras saved.")
            except Exception as e:
                print(" Warning: failed to save .keras:", e)

            summary_path = arch_dir / f"{base_name}.txt"
            try:
                with open(summary_path, "w") as fh:
                    fh.write(f"arch: {self.arch_name}\n")
                    fh.write(f"timestamp: {ts}\n")
                    fh.write(f"WMIoU: {wm_iou:.6f}\n")
                    fh.write(f"mean_IoU: {mean_iou:.6f}\n")
                    fh.write("per_class:\n")
                    for i, val in enumerate(iou.tolist()):
                        fh.write(f"  class_{i}: IoU={val:.6f}  true_px={int(true_pixels[i])}  inter={int(intersection[i])}  union={int(union[i])}\n")
                print(" WMIoU summary written ->", summary_path)
            except Exception as e:
                print(" Warning: failed writing WMIoU summary:", e)






def main():

	# dataset paths

    base = Path("samples")
    train_img_dir = base / "train" / "images"
    train_mask_dir = base / "train" / "masks"
    val_img_dir = base / "val" / "images"
    val_mask_dir = base / "val" / "masks"

    train_imgs = sorted([str(p) for p in train_img_dir.glob("*.jpg")])
    train_masks = sorted([str(p) for p in train_mask_dir.glob("*.png")])
    val_imgs = sorted([str(p) for p in val_img_dir.glob("*.jpg")])
    val_masks = sorted([str(p) for p in val_mask_dir.glob("*.png")])




    # preprocess original jpgs to float32
    def parse_image(img_p):
        img = tf.io.read_file(img_p)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, IMG_SIZE, method="bilinear")
        img = tf.ensure_shape(img, [IMG_SIZE[0], IMG_SIZE[1], 3])
        return img

    # preprocess masks to integers
    def parse_mask(mask_p):
        m = tf.io.read_file(mask_p)
        mask = tf.io.decode_png(m, channels=1, dtype=tf.uint8)
        mask = tf.squeeze(mask, axis=-1)
        mask = tf.cast(tf.expand_dims(mask, -1), tf.float32)
        mask = tf.image.resize(mask, IMG_SIZE, method="nearest")
        mask = tf.cast(tf.squeeze(mask, axis=-1), tf.int32)
        mask = tf.ensure_shape(mask, [IMG_SIZE[0], IMG_SIZE[1]])
        return mask

    def parse_pair(img_p, mask_p):
        return parse_image(img_p), parse_mask(mask_p)

    # training dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_imgs, train_masks))
    # shuffle so we don't get them sorted by filename
    train_ds = train_ds.shuffle(buffer_size=len(train_imgs), seed=69)
    train_ds = train_ds.map(lambda a, b: parse_pair(a, b), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices((val_imgs, val_masks))
    # shuffle so we don't get them sorted by filename
    val_ds = val_ds.shuffle(buffer_size=len(val_imgs), seed=69)
    val_ds = val_ds.map(lambda a, b: parse_pair(a, b), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)





    # prepare model

    model = build_unet_basic()
    compile_model_for_training(model)
    model.summary()

    cb_rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=PATIENCE_RLR, min_lr=MIN_LR, verbose=1)
    cb_es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE_ES, restore_best_weights=True, verbose=1)
    # custom IOU callback, see above
    cb_miou = ValIoUCallback(val_dataset=val_ds, arch_name="unet_basic", num_classes=NUM_CLASSES)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        callbacks=[cb_rlr, cb_es, cb_miou],
        verbose=2,
    )

    # save final model
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch_dir = MODELS_ROOT / "unet_basic"
    arch_dir.mkdir(parents=True, exist_ok=True)
    final_path = arch_dir / f"{ts}_final.keras"
    try:
        model.save(str(final_path))
        print("Saved final model to:", final_path)
    except Exception as e:
        print("Warning: failed saving final model:", e)



if __name__ == "__main__":
    main()
