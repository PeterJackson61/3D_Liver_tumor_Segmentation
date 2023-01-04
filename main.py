from pathlib import Path

import torchio as tio
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import numpy as np
from segmenter import Segmenter
from model import UNet
import os


def change_img_to_label_path(path):
    """
    Replace data with mask to get the masks
    """
    parts = list(path.parts)
    parts[parts.index("imagesTr")] = "labelsTr"
    return Path(*parts)

def train_val_dataset():
    path = Path("Liver_dataset/imagesTr/")
    subjects_paths = list(path.glob("liver_*"))
    subjects = []

    for subject_path in subjects_paths:
        label_path = change_img_to_label_path(subject_path)
        subject = tio.Subject({"CT": tio.ScalarImage(subject_path), "Label": tio.LabelMap(label_path)})
        subjects.append(subject)
    for subject in subjects:
        assert subject["CT"].orientation == ("R", "A", "S")
    process = tio.Compose([
        tio.CropOrPad((256, 256, 200)),
        tio.RescaleIntensity((-1, 1))
    ])

    augmentation = tio.RandomAffine(scales=(0.9, 1.1), degrees=(-10, 10))
    val_transform = process
    train_transform = tio.Compose([process, augmentation])
    train_dataset = tio.SubjectsDataset(subjects[:105], transform=train_transform)
    val_dataset = tio.SubjectsDataset(subjects[105:], transform=val_transform)

    return train_dataset, val_dataset

def train_val_loader():
    train_dataset, val_dataset = train_val_dataset()
    sampler = tio.data.LabelSampler(patch_size=96, label_name="Label", label_probabilities={0: 0.2, 1: 0.3, 2: 0.5})
    train_patches_queue = tio.Queue(
        train_dataset,
        max_length=40,
        samples_per_volume=5,
        sampler=sampler,
        num_workers=4,
    )

    val_patches_queue = tio.Queue(
        val_dataset,
        max_length=40,
        samples_per_volume=5,
        sampler=sampler,
        num_workers=4,
    )
    batch_size = 2

    train_loader = torch.utils.data.DataLoader(train_patches_queue, batch_size=batch_size, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_patches_queue, batch_size=batch_size, num_workers=0)

    return train_loader, val_loader
if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    train_loader, val_loader = train_val_loader()
    model = Segmenter()
    checkpoint_callback = ModelCheckpoint(
            monitor='Val Loss',
            save_top_k=10,
            mode='min')
    gpus = 1
    trainer = pl.Trainer(gpus=gpus, logger=TensorBoardLogger(save_dir="./logs"), log_every_n_steps=1,
                             callbacks=checkpoint_callback,
                             max_epochs=100)
    trainer.fit(model, train_loader, val_loader)