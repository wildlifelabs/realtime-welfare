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
from abc import ABC, abstractmethod


class AbstractHandler(ABC):
    def __init__(self, name: str, inputs: list[str], param: str):
        self.__name: str = name
        self.__inputs: list[str] = inputs
        self.__param: str = param

    @property
    def name(self) -> str:
        return self.__name

    @property
    def param(self) -> str:
        return self.__param

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def teardown(self):
        pass

    def required_jobs_for_inputs(self) -> list[str]:
        return self.__inputs

    @abstractmethod
    def set_inputs(self, values: list):
        pass

    @abstractmethod
    def get_output(self) -> any:
        pass

