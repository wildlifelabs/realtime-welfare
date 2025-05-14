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
import csv
from welfareobs.handlers.abstract_handler import AbstractHandler
from welfareobs.models.intersect import Intersect


class SaveIntersectHandler(AbstractHandler):
    """
    INPUT: List[Intersect] list of intersect (one for each individual)
    OUTPUT: Nothing
    JSON config param is CSV output filename
    """
    def __init__(self, name: str, inputs: list[str], param: str):
        super().__init__(name, inputs, param)
        self.__filename = param
        self.__data = None
        self.__sample_run = 1  # sample runs start at 1

    def setup(self):
        pass

    def __render(self, src: list) -> str:
        return " ".join([f"( {int(x)} | {int(y)} )" for x, y in src])

    def run(self):
        with open(self.__filename, 'a') as csvfile:
            writer = csv.writer(csvfile)
            if not csvfile.tell():  # Check if file is empty
                writer.writerow(['sample', 'identity', 'intersect', 'timestamp'])
            if len(self.__data) > 0:
                print(f"Writing sample {self.__sample_run} with {len(self.__data[0])} individuals")
                for item in self.__data[0]:
                    writer.writerow([self.__sample_run, item.identity, self.__render(item.intersect), item.timestamp.strftime('%Y-%m-%d %H:%M:%S')])
            self.__sample_run += 1

    def teardown(self):
        pass

    def set_inputs(self, values: list):
        self.__data = values

    def get_output(self) -> any:
        pass
