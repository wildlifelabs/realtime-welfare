from typing import Optional

from pyarrow import timestamp
from welfareobs.detectron.detectron_configuration import get_configuration
from welfareobs.detectron.detectron_calls import image_tensor, predict
from detectron2.config import instantiate
from detectron2.checkpoint import DetectionCheckpointer
from welfareobs.handlers.abstract_handler import AbstractHandler
from welfareobs.models.frame import Frame
from welfareobs.models.individual import Individual
from welfareobs.utils.config import Config


class DetectionHandler(AbstractHandler):
    """
    INPUT: single image frame in an array wrapper
    OUTPUT: array of individuals List[Individual]
    JSON config param is a config filename with hyperparameters
    """
    def __init__(self, name: str, inputs: [str], param: str):
        super().__init__(name, inputs, param)
        self.__model = None
        self.__current_frame: Optional[Frame] = None
        self.__dimensions: int = 0
        self.__reid_model_root: str = ""
        self.__reid_timm_backbone: str = ""
        self.__segmentation_checkpoint: str = ""
        self.__buffer: list[()] = []

    def setup(self):
        cnf: Config = Config(self.param)
        self.__dimensions = cnf.as_int("dimensions")
        self.__reid_model_root = cnf.as_string("reid-model-root")
        self.__reid_timm_backbone = cnf.as_string("reid-timm-backbone")
        self.__segmentation_checkpoint = cnf.as_string("segmentation-checkpoint")
        self.__model = instantiate(
            get_configuration(
                self.__reid_model_root,
                backbone=self.__reid_timm_backbone,
                dimensions=self.__dimensions
            )
        )
        # then load it with the pretrained backbone
        DetectionCheckpointer(self.__model).load(self.__segmentation_checkpoint)
        self.__model.eval()
        self.__model.to("cuda")

    def run(self):
        output: list[Individual] = []
        prediction = predict(
            [image_tensor(
                self.__current_frame.image,
                self.__dimensions
            )],
            self.__model
        )
        # We assume we are only dealing with one prediction
        # TODO: change the approach to merge the predictions (one detector for 3 images) here
        prediction = prediction[0]["instances"]
        _reids = list(prediction.get("reid_embeddings").cpu().numpy().flatten())
        _classes = list(prediction.get("pred_classes").cpu().numpy())
        _boxes = list(prediction.get("pred_boxes").cpu().numpy())
        _masks = list(prediction.get("pred_masks").cpu().numpy())
        _scores = list(prediction.get("scores").cpu().numpy())
        for i in range(len(prediction)):
            if _classes[i] == 23:
                output.append(
                    Individual(
                        # TODO: map reid and class values to a dict
                        #   This should be in the config
                        confidence=_scores[i],
                        identity=_reids[i],
                        species=_classes[i],
                        x_min=0.0, # TODO fix these
                        y_min=0.0,
                        x_max=0.0,
                        y_max=0.0,
                        mask=_masks[i],
                        timestamp=self.__current_frame.timestamp
                    )
                )
        self.__buffer = output

    def teardown(self):
        pass

    def set_inputs(self, values: [any]):
        self.__current_frame = values[0]

    def get_output(self) -> any:
        """
        Returns a list of individuals (predictions)
        """
        return self.__buffer
