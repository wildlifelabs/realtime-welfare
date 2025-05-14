# -*- coding: utf-8 -*-
"""
Module Name: individual.py
Description: individual data class

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

from dataclasses import dataclass
import numpy as np
from datetime import datetime


@dataclass
class Individual:
    """
    A data class that represents an individual animal
    """
    confidence: float
    identity: str
    species: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    mask: np.ndarray
    timestamp: datetime
