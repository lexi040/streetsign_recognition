"""
filtram datasetul sa contina doar clsele relevante pt proiect
  - speed limit 30 km/h  (GTSRB class 1)
  - speed limit 50 km/h  (GTSRB class 2)
  - speed limit 80 km/h  (GTSRB class 5)
  - yield                (GTSRB class 13)
  - stop                 (GTSRB class 14)
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image


#remapam claseel
GTSRB_TO_LOCAL = {1: 0, 2: 1, 5: 2, 13: 3, 14: 4}

CLASS_NAMES = [
    "Speed_30",   # 0
    "Speed_50",   # 1
    "Speed_80",   # 2
    "Yield",      # 3
    "Stop",       # 4
]

NUM_CLASSES = len(CLASS_NAMES)

MEAN = [0.3337, 0.3064, 0.3171]
STD  = [0.2672, 0.2564, 0.2629]

IMAGE_SIZE = 48 


#transformari

#augmentare de date pentru antrenament
def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


#transofrmari pentru validare/test
def get_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])



class FilteredGTSRB(Dataset):
    #luam datasetul GTSRB si filtram clasele relevante si remapam etichetele

    def __init__(self, root: str, split: str = "train", transform=None, download: bool = True):
        base = datasets.GTSRB(root=root, split=split, download=download)

        #colectare indice si etichete pt clase
        self.samples = []
        for img_path, gtsrb_label in base._samples:
            if gtsrb_label in GTSRB_TO_LOCAL:
                self.samples.append((img_path, GTSRB_TO_LOCAL[gtsrb_label]))

        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label



def get_loaders(data_root: str, batch_size: int = 64, num_workers: int = 4):
    #returneaza dataloader-ele pentru train, val, test si class weights pentru loss

    train_ds = FilteredGTSRB(data_root, split="train", transform=get_train_transform())
    test_ds  = FilteredGTSRB(data_root, split="test",  transform=get_val_transform())

    # Split 20 % of train into a validation set
    n_val = max(1, int(0.2 * len(train_ds)))
    n_train = len(train_ds) - n_val
    train_subset, val_subset = torch.utils.data.random_split(
        train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    
    val_subset.dataset = FilteredGTSRB(data_root, split="train", transform=get_val_transform())

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,      batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    #calc class weights
    labels = [lbl for _, lbl in train_subset]
    counts = torch.zeros(NUM_CLASSES)
    for lbl in labels:
        counts[lbl] += 1
    weights = 1.0 / counts.clamp(min=1)
    weights = weights / weights.sum()

    return train_loader, val_loader, test_loader, weights
