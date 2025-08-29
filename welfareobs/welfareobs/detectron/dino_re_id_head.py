# -*- coding: utf-8 -*-
"""
Module Name: re_id_head.py
Description: ReID Detectron2 head implements deep features from https://github.com/wildlifelabs/wildlife-tools
             from the paper by Cermak et al., "WildlifeDatasets: An Open-Source Toolkit for Animal Re-Identification"


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
import torch
import torch.nn as nn
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity 
from wildlife_tools.similarity.cosine import cosine_similarity
from wildlife_tools.inference import KnnClassifier
from wildlife_tools.data import WildlifeDataset, FeatureDataset
from welfareobs.detectron.feature_net import FeatureNet
import timm
import numpy as np


class DinoReIdHead(nn.Module):
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 image_dimensions: int,
                 checkpoint_filename: str|None = None,
                 hidden_layer_width: int = 256,
                 hidden_layer_depth: int = 2,
                 hidden_layer_dropout: float = 0.3,
                 device: str = "cuda"
                 ):
        super().__init__()
        # we expose this for pre-run validation only
        print(f"Using device: {device}")
        self.intermediate_model = FeatureNet(
            model_name = model_name,
            num_classes = num_classes,
            image_dimensions = image_dimensions,
            hidden_dim = hidden_layer_width,
            depth = hidden_layer_depth,
            dropout = hidden_layer_dropout,
            device = device
        )
        if checkpoint_filename is not None:
            self.intermediate_model.load(checkpoint_filename)

    def forward(self, x, labels=None):
        ex = self.intermediate_model.extractor(x)
        return self.intermediate_model(x)

