import csv

from welfareobs.handlers.abstract_handler import AbstractHandler


class SaveIntersectHandler(AbstractHandler):
    """
    INPUT: List[Intersect] list of intersect (one for each individual)
    OUTPUT: Nothing
    JSON config param is CSV output filename
    """
    def __init__(self, name: str, inputs: [str], param: str):
        super().__init__(name, inputs, param)
        self.__filename = param
        self.__data = None

    def setup(self):
        pass

    def run(self):
        with open(self.__filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # TODO: need to split this data into frame-level observations
            writer.writerow(['identity', 'x', 'z'])
            for intersect in self.__data:
                for row in intersect.intersect:
                    writer.writerow([intersect.identity, row[0], row[1]])

    def teardown(self):
        pass

    def set_inputs(self, values: [any]):
        self.__data = values

    def get_output(self) -> any:
        pass
