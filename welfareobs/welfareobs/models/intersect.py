from dataclasses import dataclass


@dataclass
class Intersect:
    """
    A data class that represents intersect of an individual animal
    """
    identity: str
    intersect: list[tuple]
    timestamp: float
