# -*- coding: utf-8 -*-
"""
Module Name: train_model.py
Description: Train the model (from command line)

Copyright (C) 2025 J.Cincotta

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
from itertools import chain
import torch
import timm
import torchvision.transforms as T

import torch
import torch.nn as nn
import torch.optim as optim

from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.train import ArcFaceLoss, BasicTrainer
from welfareobs.utils.config import Config
from welfareobs.detectron.welfareobs_dataset import WelfareObsDataset
from welfareobs.utils.padded_square_transform import PaddedSquareTransform
from welfareobs.detectron.feature_net import FeatureNet

import os
import pandas as pd
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

from tqdm import tqdm


# Set data for similarity-aware and random splits
config: Config = Config("/project/config-dino.json")
sets = [o.strip() for o in config["configs"].split(",")]
for ptr in sets:
    learning_rate = config.as_float(f"{ptr}.learning-rate")
    use_opt = config[f"{ptr}.optimizer"]
    dimensions = config.as_int(f"{ptr}.dimensions")
    name = config[f"{ptr}.name"]
    device = config[f"{ptr}.device"]
    outpath = f"/project/data/results/{name}"
    writer = SummaryWriter(log_dir=outpath)
    model_name = config[f"{ptr}.model-name"]
    print(f"Processing {ptr} to {outpath}.")
    os.makedirs(outpath, exist_ok=True)
    transform = T.Compose([
        PaddedSquareTransform(fill=0, padding_mode="edge"),    
        T.Resize(
            size=(dimensions,dimensions),
            interpolation=T.InterpolationMode.BILINEAR,
            max_size=None,
            antialias=True
        ),
        T.ToTensor(),  # Convert a PIL Image or ndarray to tensor and scale the values 0->255 to 0.0->1.0
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # output[channel] = (input[channel] - mean[channel]) / std[channel] (this is the mapping for ImageNet RGB)
    ])

    # WelfareObsDataset, WildlifeDataset and ImageDataset all return RGB images 
    dataset=WelfareObsDataset(
        root=config[f"{ptr}.root"],
        annotations_file=config[f"{ptr}.annotations-filename"],
        transform = transform,
        img_load="bbox", # "bbox_mask",
        col_path="path",
        col_label="identity",
        load_label=True
    )
    validation_dataset=WelfareObsDataset(
        root=config[f"{ptr}.root"],
        annotations_file=config[f"{ptr}.validation-filename"],
        transform = transform,
        img_load="bbox", # "bbox_mask",
        col_path="path",
        col_label="identity",
        load_label=True
    )

    model = FeatureNet(
        model_name = model_name,
        num_classes = config.as_int(f"{ptr}.target-classes"),
        image_dimensions = dimensions,
        hidden_dim = config.as_int(f"{ptr}.hidden-layer-width"),
        depth = config.as_int(f"{ptr}.hidden-layer-depth"),
        dropout = config.as_float(f"{ptr}.hidden-layer-dropout"),
        device = device
    )
    
    # extract embeddings from everything
    print(f"Extracting embeddings from {config[f"{ptr}.annotations-filename"]}...")
    x, y = model.extract(dataset)
    print(f"Extracting embeddings from {config[f"{ptr}.validation-filename"]}...")
    x_val, y_val = model.extract(validation_dataset)   
    criterion = nn.CrossEntropyLoss()
    optimizer = None
    if use_opt == "SGD":
        optimizer = SGD(params=model.parameters(), lr=learning_rate, momentum=config.as_float(f"{ptr}.learning-rate-momentum"))
    elif use_opt == "ASGD":
        optimizer = ASGD(params=model.parameters(), lr=learning_rate, lambd=config.as_float(f"{ptr}.learning-rate-lambd"))
    elif use_opt == "Adam":
        optimizer = Adam(params=model.parameters(), lr=learning_rate)
    
    for epoch in tqdm(range(config.as_int(f"{ptr}.trainer-epochs")), desc=f"Epoch {epoch + 1}: ", mininterval=1, ncols=100):
        model.train()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_out = model(x_val)
            val_loss = criterion(val_out, y_val)
            val_acc = (val_out.argmax(dim=1) == y_val).float().mean()
        writer.add_scalar("Train/Loss", loss.item(), epoch + 1)
        writer.add_scalar("Validation/Loss", val_loss.item(), epoch + 1)
        writer.add_scalar("Validation/Accuracy", val_acc.item(), epoch + 1)

    model.save(os.path.join(outpath,"checkpoint.pth"))

    















