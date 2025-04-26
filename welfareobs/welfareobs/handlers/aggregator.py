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
    def __init__(self, name: str, inputs: list[str], param: str):
        super().__init__(name, inputs, param)
        self.__individuals: {str,list[Intersect]} = {}
        self.__output: list[Intersect] = []
        self.__dbscan_eps = None
        self.__min_samples = None

    def setup(self):
        cnf: Config = Config(self.param)
        self.__dbscan_eps = cnf.as_string("dbscan-eps")
        self.__min_samples = cnf.as_int("min-samples")

    def run(self):
        self.__output = []
        sum_timestamp = sum([self.__individuals[o].timestamp for o in self.__individuals.keys()])
        avg_timestamp = sum_timestamp / len(self.__individuals.keys())
        for individual in self.__individuals.keys():
            # use some fancy list comprehension to expand all the intersect arrays in all the intersect classes
            coords = np.array([coord for element in self.__individuals[individual] for coord in element.intersect])
            d: DBSCAN = DBSCAN(eps=self.__dbscan_eps, min_samples=self.__min_samples)
            d.fit_predict(coords)
            self.__output.append(Intersect(individual, intersect=[tuple(coord) for coord in d.components_], timestamp=avg_timestamp))

    def teardown(self):
        pass

    def set_inputs(self, values: list):
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
