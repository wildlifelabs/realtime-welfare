from dataclasses import dataclass


@dataclass
class Frame:
    image: any
    camera_name: str
    timestamp: float
