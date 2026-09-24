"""
data_utils.py
Shared data pipeline. Import get_dataloaders() from every notebook
so all three models train and test on the exact same split.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ImageNet stats: the transfer models were pretrained on ImageNet,
# so inputs must be normalized the same way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _build_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def get_dataloaders(data_dir, img_size=224, batch_size=32, seed=42,
                    val_size=0.15, test_size=0.15, num_workers=2):
    train_tf, eval_tf = _build_transforms(img_size)

    # Two views of the same folder: augmented for train, clean for val/test.
    train_ds_full = datasets.ImageFolder(data_dir, transform=train_tf)
    eval_ds_full  = datasets.ImageFolder(data_dir, transform=eval_tf)

    class_names = train_ds_full.classes
    targets = np.array(train_ds_full.targets)
    indices = np.arange(len(targets))

    # Stratified split: carve out test first, then val from what's left.
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=targets, random_state=seed
    )
    val_ratio = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio,
        stratify=targets[train_val_idx], random_state=seed
    )

    train_set = Subset(train_ds_full, train_idx)   # augmented view
    val_set   = Subset(eval_ds_full,  val_idx)     # clean view
    test_set  = Subset(eval_ds_full,  test_idx)    # clean view

    # Class weights from TRAIN labels only, to fight the imbalance.
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=targets[train_idx],
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names, class_weights