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

