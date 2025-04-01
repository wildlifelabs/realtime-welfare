from welfareobs.handlers.abstract_handler import AbstractHandler
import time


class StubHandler(AbstractHandler):
    def __init__(self, name: str, inputs: [str], param: str):
        super().__init__(name, inputs, param)
        self.__state = 1
        self.has_run = False
        self.has_setup = False
        self.has_torndown = False

    def setup(self):
        self.has_setup = True

    def run(self):
        # print(f"running {self.name} with params {self.param}")
        time.sleep(0.5)
        self.has_run = True

    def teardown(self):
        self.has_torndown = True

    def set_inputs(self, values: [any]):
        # print(f"Setting values {values}")
        if len(values) > 0:
            self.__state = sum([int(o) for o in values])

    def get_output(self) -> any:
        # print(f"Dumping state {self.__state}")
        return self.__state
