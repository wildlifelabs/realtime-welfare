from datetime import datetime
from typing import Optional
import rtsp
from welfareobs.models.frame import Frame
from welfareobs.handlers.abstract_handler import AbstractHandler
from PIL import Image
import os


class CameraHandler(AbstractHandler):
    """
    RTSP Camera Frame Grabber
    INPUT: nothing
    OUTPUT: Frame object
    JSON config param is camera RTSP URI
    """
    def __init__(self, name: str, inputs: [str], param: str):
        super().__init__(name, inputs, param)
        self.__client = rtsp.Client(rtsp_server_uri=param)
        self.__frame: Optional[any] = None

    def setup(self):
        self.__client.open()

    def teardown(self):
        self.__client.close()

    def run(self):
        self.__frame = self.__client.read()

    def set_inputs(self, values: [any]):
        pass

    def get_output(self) -> any:
        output: Frame = Frame(self.__frame, self.name, datetime.now().timestamp())
        return output


class FauxCameraHandler(AbstractHandler):
    """
    Fake camera handler that just provides a cycle of images for testing
    INPUT: nothing
    OUTPUT: Frame
    JSON config param is top-level path to all images to use for testing
    """
    def __init__(self, name: str, inputs: list[str], param: str):
        super().__init__(name, inputs, param)
        self.__files: list = []
        self.__index = 0

    def __gather(self, directory, suffixes):
        files = []
        for entry in os.listdir(directory):
            path = os.path.join(directory, entry)
            if os.path.isfile(path):
                suffix = os.path.splitext(path)[1].lower()
                if suffix in suffixes:
                    files.append(path)
            elif os.path.isdir(path):
                files.extend(self.__gather(path, suffixes))
        return files

    def setup(self):
        self.__files = self.__gather(
            self.param,
            suffixes=[".jpg", ".png"]
        )
        self.__index = 0
        print(f"found {len(self.__files)} files")

    def teardown(self):
        pass

    def run(self):
        pass

    def set_inputs(self, values: list):
        pass

    def get_output(self) -> any:
        if self.__index >= len(self.__files):
            self.__index = 0
        output: Frame = Frame(
            Image.open(
                self.__files[self.__index]
            ),
            self.name,
            datetime.now().timestamp()
        )
        self.__index += 1
        return output
