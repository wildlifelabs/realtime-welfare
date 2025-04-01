from dataclasses import dataclass
import numpy as np


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
