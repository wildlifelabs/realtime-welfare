# -*- coding: utf-8 -*-
"""
Module Name: 
Description: 

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
import torch.nn.functional as F
from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from welfareobs.detectron.re_id_head import ReIdHead
from detectron2.structures import Boxes, ImageList, Instances
from typing import Optional, List
import json
# from welfareobs.utils.performance_monitor import PerformanceMonitor
import matplotlib.pyplot as plt
import numpy as np
from welfareobs.utils.bgr_transform import BGRTransform 
import torchvision.transforms as T


class ReIdROIHeads(StandardROIHeads):
    def __init__(
            self,
            *,
            reid_head: nn.Module | None = None,
            classes_to_reid: list = [],
            **kwargs,
    ):
        """
        NOTE: interface copied from detectron2 source and extended

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        self.reid_head: ReIdHead = reid_head
        self.classes_to_reid: list = classes_to_reid
        self.reid_tx = T.Compose([
            T.Resize(
                size=(self.reid_head.input_dim, self.reid_head.input_dim),
                interpolation=T.InterpolationMode.BILINEAR,
                max_size=None,
                antialias=True
            ),  # Resize the input image to the given size
            BGRTransform(),  # since our data source is BGR this will reverge it back to RGB
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # renormalise fragments as RGB      
        ])

    def forward(
            self,
            images: ImageList,
            features: dict[str, torch.Tensor],
            proposals: list[Instances],
            targets: list[Instances]|None= None,
    ) -> (list[Instances], dict[str, torch.Tensor]):
        instances, losses = super().forward(images, features, proposals, targets)
        for ptr in range(len(instances)):
            # debug source image
            # self.dump_image(images[ptr])
            # debug instances
            # self.dump_instance(instances[ptr])
            mask_logits = instances[ptr].pred_masks  # Shape: (N, 1, H, W)
            reid_embeddings = [torch.tensor(-1)] * mask_logits.shape[0]
            reid_proposals = {}
            for i in range(mask_logits.shape[0]):  # Process each detected instance
                # Don't bother trying to reid anything we are not interested in
                c = instances[ptr].pred_classes[i]  
                if c not in self.classes_to_reid:
                    continue
                # convert to a set of proposals (multiple individuals)
                x1, y1, x2, y2 = instances[ptr].pred_boxes[i].tensor[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cropped_region = images[ptr][:,y1:y2, x1:x2]
                # cropped_region = F.interpolate(
                #     cropped_region.unsqueeze(0),
                #     size=(self.reid_head.input_dim, self.reid_head.input_dim),
                #     mode="bilinear",
                #     align_corners=False
                # )
                reid_proposals[i] = self.reid_tx(cropped_region).unsqueeze(0) # Add zero batch to left
                # debug reid proposals
                # self.dump_crop(reid_proposals[i])
            if len(reid_proposals.keys()) > 0:
                tmp_embeddings = self.reid_head.forward([(reid_proposals[o],0) for o in reid_proposals.keys()])
                for index, offset in enumerate(reid_proposals.keys()):
                    reid_embeddings[offset] = torch.tensor(int(tmp_embeddings[index]))
                # We add a new field to Detectron instances object - this needs to be a Tensor loaded into the GPU!
                instances[ptr].set("reid_embeddings", torch.stack(reid_embeddings).to("cuda"))
            # else:
            #     instances[ptr].set("reid_embeddings", [])
        # output
        return instances, losses

    def dump_instance(self, output):
        print(f"Available fields in the result: {','.join([o for o in output.get_fields().keys()])}")
        _classes = list(output.get("pred_classes").cpu().numpy())
        if len(_classes) > 0:
            _boxes = list(output.get("pred_boxes").to("cpu"))
            _masks = list(output.get("pred_masks").cpu().numpy())
            _scores = list(output.get("scores").cpu().numpy())
            print("Boxes:")
            for c, b, s in zip(_classes, _boxes, _scores):
                print(f"Class: {c} Score: {s} = {b}")

    def dump_crop(self, image):    # Convert ImageList to a batch of tensors
        self.debug_np("Crop -> input", image)
        img = (image.cpu().squeeze(0).permute(1,2,0).numpy())[ :, : ,:]
        img = np.interp(img, (img.min(), img.max()), (0, 255)).astype(np.uint8)
        plt.imshow( img )
        plt.show()
    
    def dump_image(self, image):    # Convert ImageList to a batch of tensors
        self.debug_np("Image -> input", image)
        img = (image.cpu().permute(1, 2, 0).numpy())[ :, :, [2, 1, 0]]
        img = np.interp(img, (img.min(), img.max()), (0, 255)).astype(np.uint8)
        plt.imshow( img )
        plt.show()

    def debug_np(self, label, src):
        print(f"{label} details: Max={src.max()} Min={src.min()} shape={src.shape}")
