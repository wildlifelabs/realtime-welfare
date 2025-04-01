from dataclasses import dataclass
from typing import Optional


@dataclass
class Frame:
    image: Optional[any]
    camera_name: str
    timestamp: float
