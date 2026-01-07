import os
import cv2
import numpy as np
from tqdm import tqdm

# ========== CONFIG ==========
RAW_ROOT = "data/raw/Otto"
OUT_ROOT = "data/processed/fusion"

ACCEL_DIR = os.path.join(RAW_ROOT, "Accelerometer")
ACOU_DIR  = os.path.join(RAW_ROOT, "Acoustic")

IMG_SIZE = (224, 224)
VALID_EXT = (".png", ".jpg", ".jpeg")
# ============================


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_and_resize(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img


def fuse_channel(img_accel, img_acou):
    """
    img_accel: H x W x 3
    img_acou : H x W x 3
    return    : H x W x 6
    """
    return np.concatenate([img_accel, img_acou], axis=-1)


def main():
    fault_types = sorted(os.listdir(ACCEL_DIR))

    for fault in fault_types:
        accel_fault = os.path.join(ACCEL_DIR, fault)
        acou_fault  = os.path.join(ACOU_DIR, fault)

        if not os.path.isdir(accel_fault):
            continue

        subfolders = sorted(os.listdir(accel_fault))

        for sub in subfolders:
            accel_sub = os.path.join(accel_fault, sub)
            acou_sub  = os.path.join(acou_fault, sub)

            out_sub = os.path.join(OUT_ROOT, fault, sub)
            ensure_dir(out_sub)

            files = [f for f in os.listdir(accel_sub) if f.endswith(VALID_EXT)]

            for fname in tqdm(files, desc=f"Fusion {fault}/{sub}", leave=False):
                accel_path = os.path.join(accel_sub, fname)
                acou_path  = os.path.join(acou_sub, fname)

                if not os.path.exists(acou_path):
                    print(f"[WARN] Missing acoustic file: {acou_path}")
                    continue

                img_accel = load_and_resize(accel_path)
                img_acou  = load_and_resize(acou_path)

                fused = fuse_channel(img_accel, img_acou)

                out_name = os.path.splitext(fname)[0] + ".npy"
                np.save(os.path.join(out_sub, out_name), fused)

    print("✅ Channel fusion completed!")


if __name__ == "__main__":
    main()
