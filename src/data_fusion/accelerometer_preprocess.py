import os
import cv2
import numpy as np
from tqdm import tqdm

# ========== CONFIG ==========
RAW_ROOT = "data/raw/Otto/Accelerometer"
OUT_ROOT = "data/processed/accelerometer"

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


def main():
    fault_types = sorted(os.listdir(RAW_ROOT))

    for fault in fault_types:
        fault_path = os.path.join(RAW_ROOT, fault)
        if not os.path.isdir(fault_path):
            continue

        for sub in sorted(os.listdir(fault_path)):
            sub_path = os.path.join(fault_path, sub)
            if not os.path.isdir(sub_path):
                continue

            out_sub = os.path.join(OUT_ROOT, fault, sub)
            ensure_dir(out_sub)

            files = [f for f in os.listdir(sub_path) if f.endswith(VALID_EXT)]

            for fname in tqdm(files, desc=f"Resize {fault}/{sub}", leave=False):
                in_path = os.path.join(sub_path, fname)

                img = load_and_resize(in_path)

                out_name = os.path.splitext(fname)[0] + ".npy"
                out_path = os.path.join(out_sub, out_name)

                np.save(out_path, img)

    print("✅ Resize accelerometer completed!")


if __name__ == "__main__":
    main()
