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

from sklearn.cluster import DBSCAN
import numpy as np

from welfareobs.handlers.abstract_handler import AbstractHandler
from welfareobs.models.intersect import Intersect
from welfareobs.utils.config import Config
from datetime import datetime


class AggregatorHandler(AbstractHandler):
    """
    INPUT: array of arrays of Intersect dataclass
    OUTPUT: single array of intersect dataclass, one for each individual
    JSON config param is configuration filename to configure the DBSCAN

    aggregator configuration file looks like this:
    {
      "dbscan-eps": "2.0",
      "min-samples": "1",
      "individuals": ["Zarafa", "Ebo", "Jimiyu", "Kito"]  
    }    
    """
    def __init__(self, name: str, inputs: list[str], param: str):
        super().__init__(name, inputs, param)
        self.__individuals: {str,list[Intersect]} = {}
        self.__output: list[Intersect] = []
        self.__dbscan_eps = None
        self.__min_samples = None
        self.__names: list = []

    def setup(self):
        cnf: Config = Config(self.param)
        self.__dbscan_eps = cnf.as_float("dbscan-eps")
        self.__min_samples = cnf.as_int("min-samples")
        self.__names = cnf.as_list("individuals")

    def run(self):
        print(f"Aggregator got {len(self.__individuals.keys())} giraffe")
        self.__output = []
        for individual in self.__individuals.keys():
            # use some fancy list comprehension to expand all the intersect arrays in all the intersect classes
            coords = np.array([coord for element in self.__individuals[individual] for coord in element.intersect])
            mask = ~np.isnan(coords).all(axis=1)
            d: DBSCAN = DBSCAN(eps=self.__dbscan_eps, min_samples=self.__min_samples)
            d.fit_predict(coords[mask])
            print(f"Coordinates for {self.__names[individual - 1]}: source={coords.shape} valid={coords[mask].shape} clustered={d.components_.shape}")
            self.__output.append(Intersect(individual, intersect=[tuple(coord) for coord in d.components_], timestamp=self.__individuals[individual][0].timestamp))
            print(self.__individuals[individual][0].timestamp)
    def teardown(self):
        pass

    def set_inputs(self, values: list):
        self.__individuals = {}
        for source in values:
            if type(source) is not list:
                source = [source]
            item: Intersect
            for item in source:
                if type(item) is Intersect:
                    if item.identity not in self.__individuals.keys():
                        self.__individuals[item.identity] = []
                    self.__individuals[item.identity].append(item)

    def get_output(self) -> any:
        return self.__output
