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
import pickle
from welfareobs.handlers.abstract_handler import AbstractHandler


class PayloadCreateHandler(AbstractHandler):
    """
    INPUT: anything
    OUTPUT: Nothing (writes pkl file to disk)
    JSON config param is output filename
    """
    def __init__(self, name: str, inputs: list[str], param: str):
        super().__init__(name, inputs, param)
        self.__filename = param
        self.__data = None

    def setup(self):
        pass

    def run(self):
        with open(self.__filename, 'wb') as file:
            pickle.dump(self.__data, file)

    def teardown(self):
        pass

    def set_inputs(self, values: list):
        self.__data = values

    def get_output(self) -> any:
        pass
