# -*- coding: utf-8 -*-
"""
Module Name: resize transform
Description: perform an aspect-ratio correct transform to be used in TorchVision transform compose 
Based on discussion https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/2

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
import torchvision
import numpy as np


class PaddedSquareTransform:
    """
    Note, this TX needs to be done on a PIL image, it won't work on a raw tensor
    Put into the Compose pipeline accordingly!
    """
    def __init__(self, fill=0, padding_mode='constant'):
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.fill = fill
        self.padding_mode = padding_mode
        
    def __get_padding(self, image):        
        w, h = image.size
        max_wh = np.max([w, h])
        h_padding = (max_wh - w) / 2
        v_padding = (max_wh - h) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5        
        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
        return padding
        
    def __call__(self, image):        
        return torchvision.transforms.Pad(self.__get_padding(image), self.fill, self.padding_mode)(image)
    