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
from typing import Optional
from pyarrow import timestamp
from welfareobs.detectron.detectron_configuration import get_configuration
from welfareobs.detectron.detectron_calls import image_tensor, predict
from detectron2.config import instantiate
from detectron2.checkpoint import DetectionCheckpointer
from welfareobs.handlers.abstract_handler import AbstractHandler
from welfareobs.models.frame import Frame
from welfareobs.models.individual import Individual
from welfareobs.utils.config import Config

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog
from detectron2.structures import Instances, Boxes


class DetectionHandler(AbstractHandler):
    """
    INPUT: single image frame in an array wrapper
    OUTPUT: array of individuals List[Individual]
    JSON config param is a config filename with hyperparameters

    detection config file looks like this:
        {
          "input-dimensions": "384",
          "reid-dimensions": "384",
          "reid-model-root": "/project/data/results/wod-md",
          "reid-timm-backbone": "hf-hub:BVRA/wildlife-mega-L-384",
          "segmentation-checkpoint": "/project/data/detectron2_models/mask_rcnn_R_101_FPN_3x/model_final_a3ec72.pkl"
          "debug-enable": "True"
        }    
    
    """
    def __init__(self, name: str, inputs: list[str], param: str):
        super().__init__(name, inputs, param)
        self.__model = None
        self.__current_frames = None
        self.__input_dimensions: int = 0
        self.__reid_dimensions: int = 0
        self.__reid_model_root: str = ""
        self.__reid_timm_backbone: str = ""
        self.__segmentation_checkpoint: str = ""
        self.__buffer: list = []
        self.__metadata = MetadataCatalog.get("coco_2017_val")
        self.__debug_enable: bool = False
        self.__pytorch_device: str = "cuda"

    def setup(self):
        cnf: Config = Config(self.param)
        self.__input_dimensions = cnf.as_int("input-dimensions")
        self.__reid_dimensions = cnf.as_int("reid-dimensions")
        self.__reid_model_root = cnf.as_string("reid-model-root")
        self.__reid_timm_backbone = cnf.as_string("reid-timm-backbone")
        self.__segmentation_checkpoint = cnf.as_string("segmentation-checkpoint")
        self.__debug_enable = cnf.as_bool("debug-enable")
        self.__pytorch_device = cnf["pytorch-device"]
        self.__model = instantiate(
            get_configuration(
                self.__reid_model_root,
                backbone=self.__reid_timm_backbone,
                dimensions=self.__reid_dimensions,
                device=self.__pytorch_device
            )
        )
        # then load it with the pretrained backbone
        DetectionCheckpointer(self.__model).load(self.__segmentation_checkpoint)
        self.__model.eval()
        self.__model.to(self.__pytorch_device)

    def run(self):
        output: list[Individual] = []
        predictions = predict(
            [image_tensor(
                o.image,
                self.__input_dimensions,
                self.__pytorch_device
            ) for o in self.__current_frames],
            self.__model,
            size=self.__input_dimensions
        )

        for index, prediction in enumerate(predictions):
            prediction = prediction["instances"]
            if self.__debug_enable:
                self.dump_image(self.__current_frames[index].image, prediction)
            if prediction.has("reid_embeddings"):
                _reids = list(prediction.get("reid_embeddings").cpu().numpy().flatten())
                _classes = list(prediction.get("pred_classes").cpu().numpy())
                _boxes = list(prediction.get("pred_boxes").to("cpu"))
                _masks = list(prediction.get("pred_masks").cpu().numpy())
                _scores = list(prediction.get("scores").cpu().numpy())

                # this adds reid support for all the classes, we amalgamate
                # all the instances of a single reid individual. This requires
                # merging bounding boxes and OR all masks for this individual
                #
                # Note this logic is also implemented in the confusion matrix
                class_score: dict = {}
                class_box: dict = {}
                class_mask: dict = {}
                class_count: dict = {}
                class_species: dict = {}
                
                for r, c, s, b, m in zip(_reids, _classes, _scores, _boxes, _masks):
                    r = int(r) # just performing a typecast
                    class_score[r] = float(s + class_score.get(r,0))
                    class_count[r] = int(1 + class_count.get(r,0))
                    class_species[r] = c # just assume all correct
                    # expand all REIDs into a single REID proposal
                    x1, y1, x2, y2 = b.cpu().numpy()
                    xx1, yy1, xx2, yy2 = class_box.get(r,(x1, y1, x2, y2))
                    class_box[r] = (
                        xx1 if xx1 < x1 else x1,
                        yy1 if yy1 < y1 else y1,
                        xx2 if xx2 > x2 else x2,
                        yy2 if yy2 > y2 else y2
                    )
                    class_mask[r] = np.logical_or(class_mask.get(r,m), m)
                
                for key in class_score.keys():
                    if key > -1:
                        output.append(
                            Individual(
                                camera_name=self.__current_frames[index].camera_name,
                                confidence=class_score[r],
                                identity=r,
                                species=class_species[r],
                                x_min=class_box[r][0], 
                                y_min=class_box[r][1], 
                                x_max=class_box[r][2], 
                                y_max=class_box[r][3], 
                                mask=class_mask[r],
                                timestamp=self.__current_frames[index].timestamp
                            )
                        )

        # print(f"detection::run output size = {len(output)}")
        self.__buffer = output

    def teardown(self):
        pass

    def set_inputs(self, values: list):
        """
        Takes frame from multiple cameras 
        """
        self.__current_frames = values

    def get_output(self) -> any:
        """
        Returns a list of individuals (predictions)
        """
        return self.__buffer

    def dump_image(self, image, instance):
        image = np.array(image_tensor(
                image,
                self.__dimensions
            ).cpu()).transpose(1, 2, 0)[ :, : ,[2, 1, 0]]
        image_rgb = image.astype(np.float32) / 255.0
        v = Visualizer(image[:, :, ::-1], metadata=self.__metadata, scale=1.2)
        giraffe_mask = instance.pred_classes == 23
        filtered_instances = Instances((384, 384), **{
            "pred_boxes": instance.pred_boxes[giraffe_mask],
            "scores": instance.scores[giraffe_mask],
            "pred_classes": instance.pred_classes[giraffe_mask]
        })
        out = v.draw_instance_predictions(filtered_instances.to("cpu"))
        output_image = out.get_image()
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        axes[1].imshow(output_image[:, :, ::-1])
        axes[1].set_title("Output Image with Predictions")
        axes[1].axis("off")
        plt.axis("off")
        plt.show()