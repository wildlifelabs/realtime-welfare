import pickle

from welfareobs.handlers.abstract_handler import AbstractHandler


class PayloadCreateHandler(AbstractHandler):
    """
    INPUT: anything
    OUTPUT: Nothing (writes pkl file to disk)
    JSON config param is output filename
    """
    def __init__(self, name: str, inputs: [str], param: str):
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

    def set_inputs(self, values: [any]):
        self.__data = values

    def get_output(self) -> any:
        pass
