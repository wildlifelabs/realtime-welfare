from typing import List
from sklearn.cluster import DBSCAN
import numpy as np

from welfareobs.handlers.abstract_handler import AbstractHandler
from welfareobs.models.intersect import Intersect
from welfareobs.utils.config import Config


class AggregatorHandler(AbstractHandler):
    """
    INPUT: array of arrays of Intersect dataclass
    OUTPUT: single array of intersect dataclass, one for each individual
    JSON config param is configuration filename to configure the DBSCAN
    """
    def __init__(self, name: str, inputs: [str], param: str):
        super().__init__(name, inputs, param)
        self.__individuals: {str,List[Intersect]} = {}
        self.__output: List[Intersect] = []
        self.__dbscan_eps = None
        self.__min_samples = None

    def setup(self):
        cnf: Config = Config(self.param)
        self.__dbscan_eps = cnf.as_string("dbscan-eps")
        self.__min_samples = cnf.as_int("min-samples")

    def run(self):
        self.__output = []
        for individual in self.__individuals.keys():
            # use some fancy list comprehension to expand all the intersect arrays in all the intersect classes
            coords = np.array([coord for element in self.__individuals[individual] for coord in element.intersect])
            d: DBSCAN = DBSCAN(eps=self.__dbscan_eps, min_samples=self.__min_samples)
            d.fit_predict(coords)
            self.__output.append(Intersect(individual, [tuple(coord) for coord in d.components_]))

    def teardown(self):
        pass

    def set_inputs(self, values: [any]):
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
