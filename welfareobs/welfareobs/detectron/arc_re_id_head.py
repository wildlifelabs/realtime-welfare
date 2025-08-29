# -*- coding: utf-8 -*-
"""
Module Name: arc_re_id_head.py
Description: ReID Detectron2 head that only uses ArcFaceLoss clusters
             adapted from the paper by Cermak et al., "WildlifeDatasets: An Open-Source Toolkit for Animal Re-Identification"


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
import timm
import numpy as np


class ArcReIdHead(nn.Module):
    def __init__(self,
                 input_dim: int,
                 model_name: str,
                 checkpoint_filename: str|None = None,
                 batch_size: int = 128,
                 num_workers: int = 1,
                 device: str = "cuda"
                ):
        super().__init__()
        # we expose this for pre-run validation only
        print(f"Using device: {device}")
        self.input_dim = input_dim
        self.intermediate_model = timm.create_model(model_name, pretrained=True, num_classes=0)
        if checkpoint_filename is not None:
            self.intermediate_model.load_state_dict(torch.load(checkpoint_filename, weights_only=False, map_location=torch.device(device))['model'])

    def forward(self, x, labels=None):
        out = self.intermediate_model(x)
        logits = self.intermediate_model.objective.loss.get_logits(out)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return torch.argmax(probs, dim=1)

