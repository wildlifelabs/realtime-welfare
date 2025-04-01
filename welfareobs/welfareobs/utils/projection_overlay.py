# -*- coding: utf-8 -*-
"""
Module Name: projection_overlay.py
Description: This is the simple 2D version of the warp-mapped lookup (before homographic transform)

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
from math import floor

import cv2
import numpy as np


class ProjectionOverlay(object):
    def __init__(self, pixels: int = 4096, calibration_scale: float = 0.1):
        """
        Generate projection overlay for ProjectionTransformer

        This is two single-channel vignettes in perpendicular directions
        that are used as a lookup after the image is warped based on the
        camera's four waypoints that are used for homographic transformation
        from the original camera inputs.

        :param pixels: pixel size of the overlay lookup (default is 4096)
        :param calibration_scale: calibration square as a ratio of the overall map (default to 10%)
        """
        self.__px: int = pixels
        self.__zoom: float = calibration_scale

    def generate_overlay_image(self) -> np.ndarray:
        vignette = np.linspace(0, 255, self.__px, dtype=np.uint8)
        h_r_output = np.array(np.tile(vignette, (self.__px, 1)), dtype=np.uint8).T
        w_g_output = np.array(np.tile(vignette, (self.__px, 1)), dtype=np.uint8)
        z_b_output = np.zeros((self.__px, self.__px), dtype=np.uint8)
        return cv2.merge((h_r_output, w_g_output, z_b_output))

    def get_overlay_corners(self) -> ((int, int), (int, int), (int, int), (int, int)):
        """
        This technique will ensure the grid squares of the calibration image matches the grid of the
        checkerboard calibration. IE: if your real-world grid box is 300mmx300mm, these will be equivalent
        once homographically warped onto the alignment points
        :return: Tuple of (x,y) coordinates, top-left (NW), bottom-left (SW), top-right (NE), bottom-right (SE)
        """
        origin = np.floor(self.__px / 2)
        offset = np.floor(self.__px * self.__zoom)
        return ((origin, origin), # NW
                (origin, origin + offset), #SW
                (origin + offset, origin), #NE
                (origin + offset, origin + offset)) #NW


