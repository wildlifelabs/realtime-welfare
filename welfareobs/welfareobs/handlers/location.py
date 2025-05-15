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

from welfareobs.handlers.abstract_handler import AbstractHandler
from PIL import Image
import numpy as np
from welfareobs.models.individual import Individual
from welfareobs.models.intersect import Intersect
from welfareobs.utils.config import Config
from welfareobs.utils.projection_transformer import ProjectionTransformer
from welfareobs.utils.matplotlib_image_wrapper import MatPlotLibImageWrapper


class LocationHandler(AbstractHandler):
    """
    INPUT: array of individual (Individual data class) from a single source image
    OUTPUT: array of intersect (Intersect data class) matching intersections for each individual
    JSON config param is a config JSON filename

    configuration file looks like this:
    {
      "camera-name": "camera-1"
      "camera-projection-filename": "config/camera-1.pkl",
      "y-mask-clipping-threshold": "100",
      "target-width": "384",
      "target-height": "384"
      "debug-enable": "True"
    }

    locations FILTER the input of a detection by the named camera. 
    The name is defined in the main configuration.
    
    """
    def __init__(self, name: str, inputs: list[str], param: str):
        super().__init__(name, inputs, param)
        self.__individual_detections: list[Individual]|None = None
        self.__pt: ProjectionTransformer = ProjectionTransformer()
        self.__clipping_threshold: int = 0
        self.__debug_enable = False
        self.__camera_name_filter = ""

    def setup(self):
        cnf: Config = Config(self.param)
        self.__pt.load(
            cnf.as_string("camera-projection-filename"), 
            target_w=cnf.as_int("target-width"),
            target_h=cnf.as_int("target-height")
        )
        self.__clipping_threshold = cnf.as_int("y-mask-clipping-threshold")
        self.__debug_enable = cnf.as_bool("debug-enable")
        self.__camera_name_filter = cnf.as_string("camera-name")
    
    def run(self):
        pass

    def teardown(self):
        pass

    def set_inputs(self, values: list):
        self.__individual_detections = values[0]

    def get_xy_mask_lower_intersect(self, mask, clipping_threshold):
        """Extracts bottom-most (lowest Y) points of an object mask for each X coordinate."""
        y_indices, x_indices = np.where(mask > 0)
        bottom_points = {}
        max_y = 0
        for x, y in zip(x_indices, y_indices):
            if x not in bottom_points or y > bottom_points[x]:
                bottom_points[x] = y
                if y > max_y:
                    max_y = y
        # setting clipping threshold to 0 allows everything.
        if clipping_threshold == 0:
            clipping_threshold = max_y
        # this drops points that are too far away from the lowest Y point.
        return np.array([(x, y) for x, y in bottom_points.items() if y >= (max_y - clipping_threshold)])

    def valid_camera(self, individual: Individual) -> bool:
        # may need to make this more forgiving?
        return individual.camera_name == self.__camera_name_filter
    
    def get_output(self) -> any:
        output: list[Intersect] = []
        i=0
        for detection in self.__individual_detections:
            if self.valid_camera(detection):
                output.append(
                    Intersect(
                        identity=detection.identity,
                        intersect=self.__pt.get_xz_array(
                            self.get_xy_mask_lower_intersect(
                                detection.mask, 
                                self.__clipping_threshold
                            )
                        ),
                        timestamp=detection.timestamp
                    )
                )
        if self.__debug_enable:
            self.render_output()
        return output

    def render_output(self):
        mw = MatPlotLibImageWrapper(self.__pt.warped_grid_image.copy())  #NB you need to deep copy the numpy array
        i=0
        for detection in self.__individual_detections:
            mw.set_ink(255,255,255)
            # print(np.argwhere(detection.mask==1))
            mw.draw_points(np.argwhere(detection.mask==1)[:, ::-1])
            i += 10
            mw.set_ink(255,i%255,i%255)
            mw.draw_points(self.get_xy_mask_lower_intersect(detection.mask, self.__clipping_threshold))
        mw.render()
        
