import pickle

from welfareobs.handlers.abstract_handler import AbstractHandler


class PayloadExtractHandler(AbstractHandler):
    """
    INPUT: nothing
    OUTPUT: reconstructs whatever was pkl'ed
    JSON config param is input filename
    """
    def __init__(self, name: str, inputs: [str], param: str):
        super().__init__(name, inputs, param)
        self.__filename = param
        self.__data = None

    def setup(self):
        pass

    def run(self):
        with open(self.__filename, 'rb') as file:
            self.__data = pickle.load(file)

    def teardown(self):
        pass

    def set_inputs(self, values: [any]):
        pass

    def get_output(self) -> any:
        return self.__data
