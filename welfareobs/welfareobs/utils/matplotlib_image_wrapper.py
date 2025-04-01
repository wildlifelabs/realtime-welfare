# -*- coding: utf-8 -*-
"""
Module Name: matplotlib_image_wrapper.py
Description: Quick way to render/manipulate images using MatPlotLib

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
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sympy import Point2D, Line
from typing import Union, List
from welfareobs.utils.type_conv import to_int


class MatPlotLibImageWrapper(object):
    def __init__(self, img):
        self.__img = img
        self.__ink = (0, 0, 0)

    @property
    def image(self):
        return self.__img

    def set_ink(self, r, g, b):
        self.__ink = (r, g, b)

    def draw_points(self, points: Union[np.ndarray, List[Point2D]]):
        if type(points) == np.ndarray:
            for i in range(points.shape[0]):
                cv2.circle(self.__img, (to_int(points[i, 0]), to_int(points[i, 1])), 1, self.__ink, 2)
        else:
            for point in points:
                cv2.circle(self.__img, (to_int(point.x), to_int(point.y)), 1, self.__ink, 2)

    def draw_line(self, line: Line):
        cv2.line(self.__img, (to_int(line.p1.x), to_int(line.p1.y)), (to_int(line.p2.x), to_int(line.p2.y)), self.__ink, 2)

    def add_overlay(self, img):
        self.__img = cv2.addWeighted(self.__img, 1.0, img, 0.9, 1.0)

    def render(self):
        fig: Figure = plt.figure(figsize=(100.0, 25.0))
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(self.__img[..., ::-1])
        plt.show()
