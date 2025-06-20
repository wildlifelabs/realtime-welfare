# -*- coding: utf-8 -*-
"""
Module Name: detectron_configuration.py
Description: lazy-called (modern) main configuration for our model

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
import os
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone import BasicStem, FPN, ResNet
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    FastRCNNConvFCHead,
)

from welfareobs.detectron.re_id_head import ReIdHead
from welfareobs.detectron.re_id_roi_heads import ReIdROIHeads
from wildlife_tools.data import FeatureDataset


# this comes from Detectron2
constants = dict(
    imagenet_rgb256_mean=[123.675, 116.28, 103.53],
    imagenet_rgb256_std=[58.395, 57.12, 57.375],
    imagenet_bgr256_mean=[103.530, 116.280, 123.675],
    # When using pre-trained models in Detectron1 or any MSRA models,
    # std has been absorbed into its conv1 weights, so the std needs to be set 1.
    # Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
    imagenet_bgr256_std=[1.0, 1.0, 1.0],
    # imagenet_bgr256_std=[57.375, 57.120, 58.395] <-- these have been baked into imagenet
)


def get_configuration(
        root: str,
        backbone: str = "hf-hub:BVRA/wildlife-mega-L-384",
        dimensions: int = 384,
        device: str = "cuda"
):
    return L(GeneralizedRCNN)(
        backbone=L(FPN)(
            bottom_up=L(ResNet)(
                stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
                stages=L(ResNet.make_default_stages)(
                    depth=101,
                    stride_in_1x1=True,
                    norm="FrozenBN",
                ),
                out_features=["res2", "res3", "res4", "res5"],
            ),
            in_features="${.bottom_up.out_features}",
            out_channels=256,
            top_block=L(LastLevelMaxPool)(),
        ),
        proposal_generator=L(RPN)(
            in_features=["p2", "p3", "p4", "p5", "p6"],
            head=L(StandardRPNHead)(in_channels=256, num_anchors=3),
            anchor_generator=L(DefaultAnchorGenerator)(
                sizes=[[32], [64], [128], [256], [512]],
                aspect_ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64],
                offset=0.0,
            ),
            anchor_matcher=L(Matcher)(
                thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
            ),
            box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_topk=(2000, 1000),
            post_nms_topk=(1000, 1000),
            nms_thresh=0.7,
        ),
        roi_heads=L(ReIdROIHeads)(
            num_classes=80,
            batch_size_per_image=512,
            positive_fraction=0.25,
            proposal_matcher=L(Matcher)(
                thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
            ),
            box_in_features=["p2", "p3", "p4", "p5"],
            box_pooler=L(ROIPooler)(
                output_size=7,
                scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
                sampling_ratio=0,
                pooler_type="ROIAlignV2",
            ),
            box_head=L(FastRCNNConvFCHead)(
                input_shape=ShapeSpec(channels=256, height=7, width=7),
                conv_dims=[],
                fc_dims=[1024, 1024],
            ),
            box_predictor=L(FastRCNNOutputLayers)(
                input_shape=ShapeSpec(channels=1024),
                test_score_thresh=0.05,
                box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
                num_classes="${..num_classes}",
            ),
            mask_in_features=["p2", "p3", "p4", "p5"],
            mask_pooler=L(ROIPooler)(
                output_size=14,
                scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
                sampling_ratio=0,
                pooler_type="ROIAlignV2",
            ),
            mask_head=L(MaskRCNNConvUpsampleHead)(
                input_shape=ShapeSpec(channels=256, width=14, height=14),
                num_classes="${..num_classes}",
                conv_dims=[256, 256, 256, 256, 256],
            ),
            reid_head=L(ReIdHead)(
                input_dim=dimensions,
                model_name=backbone,
                checkpoint_filename=os.path.join(root, "checkpoint.pth"),
                batch_size=1,
                num_workers=1,
                device=device,
                features_database=L(FeatureDataset.from_file)(
                    path=os.path.join(root, "similarity.pkl")
                )
            ),
            device=device,
            classes_to_reid=[23, 24, 25]  #23 on CUDA - somehow CPU breaks this.
        ),
        pixel_mean=constants["imagenet_bgr256_mean"],
        pixel_std=constants["imagenet_bgr256_std"],
        input_format="BGR",
    )
