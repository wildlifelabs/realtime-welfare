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

    def setup(self):
        pass

    def run(self):
        with open(self.__filename, 'a') as csvfile:
            writer = csv.writer(csvfile)
            if not csvfile.tell():  # Check if file is empty
                writer.writerow(['identity', 'x', 'z', 'timestamp'])
            for item in self.__data:
                writer.writerow([item.identity, item.intersect[0], item.intersect[1], item.timestamp])

    def teardown(self):
        pass

    def set_inputs(self, values: list):
        self.__data = values

    def get_output(self) -> any:
        pass
