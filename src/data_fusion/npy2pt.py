import os
import numpy as np
import torch
from tqdm import tqdm

# ========== CONFIG ==========
SRC_ROOTS = {
    "data/processed/accelerometer": "data/processed/accelerometer_pt",
    "data/processed/fusion": "data/processed/fusion_pt",
}
# ============================


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def convert_npy_to_pt(src_root, dst_root):
    for root, _, files in os.walk(src_root):
        rel_path = os.path.relpath(root, src_root)
        out_dir = os.path.join(dst_root, rel_path)
        ensure_dir(out_dir)

        for fname in tqdm(files, desc=f"Converting {rel_path}", leave=False):
            if not fname.endswith(".npy"):
                continue

            npy_path = os.path.join(root, fname)
            pt_path = os.path.join(
                out_dir, fname.replace(".npy", ".pt")
            )

            arr = np.load(npy_path)

            # H W C -> C H W
            if arr.ndim == 3:
                arr = torch.from_numpy(arr).permute(2, 0, 1)
            else:
                arr = torch.from_numpy(arr)

            torch.save(arr, pt_path)


def main():
    for src, dst in SRC_ROOTS.items():
        print(f"\n🚀 Converting {src} → {dst}")
        convert_npy_to_pt(src, dst)

    print("\n✅ All .npy converted to .pt successfully!")


if __name__ == "__main__":
    main()
