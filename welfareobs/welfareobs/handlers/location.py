from typing import Optional, List

from welfareobs.handlers.abstract_handler import AbstractHandler
from PIL import Image

from welfareobs.models.individual import Individual
from welfareobs.models.intersect import Intersect
from welfareobs.utils.config import Config
from welfareobs.utils.projection_transformer import ProjectionTransformer


class LocationHandler(AbstractHandler):
    """
    INPUT: array of individual (Individual data class) from a single source image
    OUTPUT: array of intersect (Intersect data class) matching intersections for each individual
    JSON config param is a config JSON filename
    """
    def __init__(self, name: str, inputs: [str], param: str):
        super().__init__(name, inputs, param)
        self.__individual_detections: Optional[List[Individual]] = None
        self.__pt: ProjectionTransformer = ProjectionTransformer()
        self.__clipping_threshold: int = 0

    def setup(self):
        cnf: Config = Config(self.param)
        self.__pt.load(cnf.as_string("camera-projection-filename"))
        self.__clipping_threshold = cnf.as_int("y-mask-clipping-threshold")

    def run(self):
        pass

    def teardown(self):
        pass

    def set_inputs(self, values: [any]):
        self.__individual_detections = values[0]

    def get_output(self) -> any:
        output: [Intersect] = []
        for detection in self.__individual_detections:
            output.append(
                Intersect(
                    identity=detection.identity,
                    intersect=self.__pt.get_xz_mask_lower_intersect(detection.mask, self.__clipping_threshold)
                )
            )
        return output
