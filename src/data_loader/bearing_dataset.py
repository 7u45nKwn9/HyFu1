from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class FusionDataset(Dataset):

    FAULT_MAP = {
        "1_Healthy": 0,
        "2_Inner": 1,
        "3_Outer": 2,
        "4_Ball": 3,
        "5_Cage": 4,
    }

    def __init__(self, fusion_root: str, transform=True):
        self.root = Path(fusion_root)
        self.use_transform = transform

        self.transform = transforms.Normalize(
            mean=IMAGENET_MEAN * 2,
            std=IMAGENET_STD * 2
        )

        self.samples = self._scan()
        print(f"[FusionDataset] {self.root} → {len(self.samples)} samples")

    def _scan(self):
        samples = []

        for class_name, label in self.FAULT_MAP.items():
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue

            for sample_dir in class_dir.iterdir():
                if not sample_dir.is_dir():
                    continue

                for pt_file in sample_dir.glob("*.pt"):
                    samples.append((pt_file, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        x = torch.load(path)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        assert x.shape[0] == 6, f"Expected 6 channels, got {x.shape}"

        if self.use_transform:
            x = self.transform(x)

        y = torch.tensor(label, dtype=torch.long)
        return x, y
