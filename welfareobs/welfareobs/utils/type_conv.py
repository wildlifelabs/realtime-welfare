# -*- coding: utf-8 -*-
"""
Module Name: type_conv.py
Description: Handle exotic type conversions, mostly due to using SymPy and Numpy

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
import sympy
from typing import Union, Optional
import numpy as np


def to_nan_float(src: any) -> Optional[float]:
    """
    Convert values to float, but support Numpy nan
    :param src: source value
    :return: Union[float|Numpy.nan]
    """
    if (type(src) == np.nan) or (type(src) is None):
        return np.nan
    return float(src)


def to_int(src: any) -> int:
    try:
        if type(src) == sympy.core.numbers.Infinity:
            return 0
        if type(src) == sympy.core.numbers.NegativeInfinity:
            return 0
        if type(src) == sympy.core.numbers.NaN:
            return 0
        if src is None:
            return 0
        return int(src)
    except Exception as ex:
        raise RuntimeError(f"Could not convert type {str(type(src))}") from ex

def to_int_brute_force(src: any) -> int:
    try:
        return int(src)
    except (ValueError, TypeError):
        return 0
