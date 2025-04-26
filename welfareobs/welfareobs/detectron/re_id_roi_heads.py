import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from welfareobs.detectron.re_id_head import ReIdHead
from detectron2.structures import Boxes, ImageList, Instances
from typing import Optional, List
import json
from welfareobs.utils.performance_monitor import PerformanceMonitor


class ReIdROIHeads(StandardROIHeads):
    def __init__(
            self,
            *,
            reid_head: nn.Module | None = None,
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
        self.__pm_seg: PerformanceMonitor = PerformanceMonitor(
            label="ReIdROIHeads:Segmentation",
            history_size=100
        )
        self.__re_seg: PerformanceMonitor = PerformanceMonitor(
            label="ReIdROIHeads:Reidentification",
            history_size=100
        )

    def forward(
            self,
            images: ImageList,
            features: dict[str, torch.Tensor],
            proposals: list[Instances],
            targets: list[Instances]|None= None,
    ) -> (list[Instances], dict[str, torch.Tensor]):
        self.__pm_seg.track_start()
        instances, losses = super().forward(images, features, proposals, targets)
        self.__pm_seg.track_end()
        print(str(self.__pm_seg))
        for ptr in range(len(instances)):
            # Extract mask features from the mask branch
            mask_logits = instances[ptr].pred_masks  # Shape: (N, 1, H, W)
            reid_embeddings = []
            for i in range(mask_logits.shape[0]):  # Process each detected instance
                x1, y1, x2, y2 = instances[ptr].pred_boxes[i].tensor[0]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cropped_region = images[ptr][:,y1:y2, x1:x2]
                cropped_region = F.interpolate(
                    cropped_region.unsqueeze(0),
                    size=(self.reid_head.input_dim, self.reid_head.input_dim),
                    mode="bilinear",
                    align_corners=False
                )
                self.__re_seg.track_start()
                reid_embedding = self.reid_head(cropped_region)
                self.__re_seg.track_end()
                print(str(self.__re_seg))
                if reid_embedding is not None:
                    reid_embeddings.append(torch.tensor([int(x) for x in reid_embedding]))
            if len(reid_embeddings) > 0:
                # We add a new field to Detectron instances object - this needs to be a Tensor loaded into the GPU!
                instances[ptr].set("reid_embeddings", torch.stack(reid_embeddings).to("cuda"))
        # output
        return instances, losses

