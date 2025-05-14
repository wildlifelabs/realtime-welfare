# -*- coding: utf-8 -*-
"""
Module Name: projection_transformer.py
Description: Builds a mapping to transform xy on-camera coordinates to common xz 3D planar coordinates

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
import pickle
from typing import Optional
import numpy as np
import cv2
from sympy import Point2D
from scipy.ndimage import zoom
from welfareobs.utils.image_wrapper import ImageWrapper
from welfareobs.utils.matplotlib_image_wrapper import MatPlotLibImageWrapper
from welfareobs.utils.projection_overlay import ProjectionOverlay


class ProjectionTransformer(object):

    def __init__(self):
        self.warped_grid_image: Optional[np.ndarray] = None

    def load(self, filename, target_w=None, target_h=None):
        """
        Load ProjectionTransformer
        :param filename: name of the saved PT
        :return: None
        """
        p = pickle.load(open(filename, "rb"))
        self.warped_grid_image = p["warped_grid_image"]
        if target_w is not None and target_h is not None:
            self.warped_grid_image = zoom(
                self.warped_grid_image, 
                (
                    target_h / self.warped_grid_image.shape[0], 
                    target_w / self.warped_grid_image.shape[1], 
                    1
                ), 
                order=1
            )
        print(f"Loaded {filename}: dims=({self.warped_grid_image.shape})")

    def save(self, filename):
        """
        Save ProjectionTransformer
        :param filename: name of the PT to save
        :return: None
        """
        p = {
            "warped_grid_image": self.warped_grid_image,
        }
        pickle.dump(p, open(filename, "wb"))

    def render_warp_map(self, src: ImageWrapper):
        """
        Visualisation of the PT over a camera calibration image
        (Requires support of MatPlotLib canvas through RenderTools)
        :param src: CameraCalibrationImage
        :return: None
        """
        w = self.warped_grid_image.shape[1]
        h = self.warped_grid_image.shape[0]
        r = MatPlotLibImageWrapper(src.image(w, h))
        r.set_ink(0, 0, 255)
        if self.warped_grid_image is not None:
            r.add_overlay(self.warped_grid_image)
        r.render()

    def render_xz(self, src: ImageWrapper, xz_pairs, colors, include_warp_map: bool=False):
        """
        Visualisation of the PT over a camera calibration image
        with XZ points for testing homographic calibration is correct
        (Requires support of MatPlotLib canvas through RenderTools)

        Note, this function is not optimised at all and takes ages

        :param src: CameraCalibrationImage
        :param xz_pairs: List of (x, z) tuples
        :param colors: List of (R, G, B) tuples (note you need to have an RGB for each XZ value)
        :param include_warp_map: (default is False) include the overlay of the warp-map on the image
        :return: None
        """
        r = MatPlotLibImageWrapper(src.image(src.width, src.height))
        if (self.warped_grid_image is not None) and include_warp_map:
            r.add_overlay(self.warped_grid_image)
        for xz, color in zip(xz_pairs, colors):
            r.set_ink(color[2], color[1], color[0])
            points = []
            for xx in range(src.width):
                for yy in range(src.height):
                    mx, mz = self.get_xz(xx, yy)
                    if (not np.isnan(mx)) and (not np.isnan(mz)):
                        if (int(mx) == int(xz[0])) and (int(mz) == int(xz[1])):
                            points.append(Point2D(xx,yy))
            r.draw_points(points)
        r.render()

    def __interp(self, src):
        """
        interperet the lut values:
        * we start with an unsigned int (0 - 255) 
        * we want to center this around the calibration point (center of the vignette) so we subtract 128
        * we take limits 128 or -128 as no longer being measurable and anything above or below is NaN.
        """
        output: int = int(src) - 128
        if (output > 128) or (output < -128):
            return np.nan
        return output

    def get_xz_array(self, src) -> np.ndarray:
        return np.array([self.get_xz(*pair) for pair in src])
    
    def get_xz(self, src_x, src_y) -> (float|int, float|int):
        output = (
            self.__interp(self.warped_grid_image[src_y,src_x,1]),
            self.__interp(self.warped_grid_image[src_y,src_x,0])
        )
        # print(f"src=({src_x},{src_y}) out=({output[0]},{output[1]})")
        return output

    def __get_h_matrix(self, source, destination):
        # https://github.com/Socret360/understanding-homography/blob/main/homography.py
        A = []
        b = []
        for i in range(len(source)):
            s_x, s_y = source[i]
            d_x, d_y = destination[i]
            A.append([s_x, s_y, 1, 0, 0, 0, (-d_x) * (s_x), (-d_x) * (s_y)])
            A.append([0, 0, 0, s_x, s_y, 1, (-d_y) * (s_x), (-d_y) * (s_y)])
            b += [d_x, d_y]
        A = np.array(A)
        h = np.linalg.lstsq(A, b, rcond=None)[0]
        h = np.concatenate((h, [1]), axis=-1)
        return np.reshape(h, (3, 3))

    def calibrate(self,
                  camera_image_width: int,
                  camera_image_height: int,
                  north_west: (int, int),
                  south_west: (int, int),
                  north_east: (int, int),
                  south_east: (int, int),
                  overlay_resolution: int = 8192,
                  calibration_scale: float = 0.045) -> any:
        """
        Perform XY to XZ calibration that can be saved
        Tuple of (x,y) coordinates, top-left (NW), bottom-left (SW), top-right (NE), bottom-right (SE)
        :param camera_image_width:
        :param camera_image_height:
        :param north_west: (x,y) tuple representing the pixel on the camera calibration image that aligns with the NW marker in the real world
        :param north_east: (x,y) tuple representing the pixel on the camera calibration image that aligns with the NE marker in the real world
        :param south_west: (x,y) tuple representing the pixel on the camera calibration image that aligns with the SW marker in the real world
        :param south_east: (x,y) tuple representing the pixel on the camera calibration image that aligns with the SE marker in the real world
        :param overlay_resolution: size of the homographic overlay (this affects the accuracy)
        :param calibration_scale: as a ratio of overlay resolution.
        :return: None
        """
        homographic_overlay: ProjectionOverlay = ProjectionOverlay(overlay_resolution, calibration_scale)
        h_corners = np.array(homographic_overlay.get_overlay_corners(), dtype=np.float64)
        # Upscale origin image
        dest_corners = np.array([
            [north_west[0], north_west[1]],
            [north_east[0], north_east[1]],
            [south_west[0], south_west[1]],
            [south_east[0], south_east[1]]
        ], dtype=np.float64)
        h_matrix = self.__get_h_matrix(h_corners, dest_corners)
        self.warped_grid_image = cv2.warpPerspective(homographic_overlay.generate_overlay_image(), h_matrix, (camera_image_width, camera_image_height))

