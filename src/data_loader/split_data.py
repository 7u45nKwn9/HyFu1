import random
import shutil
from pathlib import Path


def split_dataset(
    acc_root,
    fus_root,
    out_root,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42
):
    random.seed(seed)

    acc_root = Path(acc_root)
    fus_root = Path(fus_root)
    out_root = Path(out_root)

    for class_dir in acc_root.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for sub_dir in class_dir.iterdir():
            if not sub_dir.is_dir():
                continue

            sub_name = sub_dir.name
            acc_files = sorted(sub_dir.glob("*.pt"))

            if len(acc_files) == 0:
                continue

            random.shuffle(acc_files)

            n = len(acc_files)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            split_map = {
                "train": acc_files[:n_train],
                "val": acc_files[n_train:n_train + n_val],
                "test": acc_files[n_train + n_val:]
            }

            for split, files in split_map.items():
                for acc_file in files:
                    acc_dst = (
                        out_root / split /
                        "accelerometer_pt" /
                        class_name / sub_name
                    )
                    fus_dst = (
                        out_root / split /
                        "fusion_pt" /
                        class_name / sub_name
                    )

                    acc_dst.mkdir(parents=True, exist_ok=True)
                    fus_dst.mkdir(parents=True, exist_ok=True)

                    # copy accelerometer
                    shutil.copy2(acc_file, acc_dst / acc_file.name)

                    # copy fusion (path đối xứng)
                    fus_file = (
                        fus_root /
                        class_name / sub_name / acc_file.name
                    )

                    if not fus_file.exists():
                        raise RuntimeError(
                            f"Missing fusion file: {fus_file}"
                        )

                    shutil.copy2(fus_file, fus_dst / acc_file.name)

    print("DONE: split train / val / test successfully.")


if __name__ == "__main__":
    split_dataset(
        acc_root="data/processed/accelerometer_pt",
        fus_root="data/processed/fusion_pt",
        out_root="data/split"
    )
