# -*- coding: utf-8 -*-
"""
Module Name: 
Description: 

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

from datetime import datetime, timedelta
from typing import Optional
import rtsp
from welfareobs.models.frame import Frame
from welfareobs.handlers.abstract_handler import AbstractHandler
from welfareobs.utils.type_conv import to_int_brute_force
from welfareobs.utils.config import Config
from PIL import Image
import matplotlib.pyplot as plt
import os
import pathlib


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
    JSON config param is filename to the configuration file.

    expected filename format for faux-camera data is:
    c1-savannah-2024_03_26__23_03_54.jpg
    c1 = camera number

    faux camera configuration file looks like this:
        {
            "root": "/project/data/wod_2025/20250220",
            "file-types": [".jpeg",".jpg",".png"],
            "camera-filter": "c1",
            "hour-start-filter": "10",
            "hour-end-filter": "11",
            "timestamp-start": "2025-03-10 10:28:07",
            "timestamp-delta-seconds": "5"
        }
    """
    def __init__(self, name: str, inputs: list[str], param: str):
        super().__init__(name, inputs, param)
        self.__files: list = []
        self.__index = 0
        self.__timestamp = None
        self.__timestamp_delta_seconds = None

    def __gather(self, directory: str, camera_filter: str, hour_start_filter: int, hour_end_filter: int, suffixes: list):
        files = []
        for entry in os.listdir(directory):
            hour = to_int_brute_force(entry[-10:-8])
            if hour_start_filter >= 0 and hour_end_filter >= 0:
                if hour < hour_start_filter or hour > hour_end_filter:
                    continue
            if entry.startswith(camera_filter):
                path = os.path.join(directory, entry)
                if os.path.isfile(path):
                    suffix = os.path.splitext(path)[1].lower()
                    if suffix in suffixes:
                        files.append(path)
                elif os.path.isdir(path):
                    files.extend(self.__gather(path, camera_filter, hour_start_filter, hour_end_filter, suffixes))
        return files

    def setup(self):
        cnf: Config = Config(self.param)
        # use the as_string() in config to allow the element to 
        # not be present in the config without failing.
        print(f"root={cnf["root"]} camera-filter={cnf.as_string("camera-filter")} time-filter={cnf.as_int("hour-start-filter")}->{cnf.as_int("hour-end-filter")} Types: {cnf.as_list("file-types")}")
        self.__files = self.__gather(
            cnf["root"],
            cnf.as_string("camera-filter"),
            cnf.as_int("hour-start-filter"),
            cnf.as_int("hour-end-filter"),
            suffixes=cnf.as_list("file-types")
        )
        self.__files = sorted(self.__files, key=str.lower)
        self.__index = 0
        self.__timestamp = datetime.strptime(cnf.as_string("timestamp-start"), '%Y-%m-%d %H:%M:%S')  
        self.__timestamp_delta_seconds = cnf.as_int("timestamp-delta-seconds")
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
            self.__timestamp
        )
        print(f"Filename: {self.__files[self.__index]} stamp: {self.__timestamp.timestamp()} ({self.__timestamp})")
        self.__index += 1
        self.__timestamp = self.__timestamp + timedelta(seconds=self.__timestamp_delta_seconds)
        # self.dump_output(output)
        return output

    def dump_output(self, output: Frame):
        # img = (image.cpu().permute(1, 2, 0).numpy())[ :, :, [2, 1, 0]]
        # img = np.interp(img, (img.min(), img.max()), (0, 255)).astype(np.uint8)
        plt.imshow(output.image)
        plt.show()
