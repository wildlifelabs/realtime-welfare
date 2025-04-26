from collections import deque
import time
import statistics


class PerformanceMonitor(object):
    def __init__(self,
                 label: str = "",
                 history_size: int = 1
                 ):
        self.__label = label
        self.__history = deque(maxlen=history_size)
        self.__overall_execution_time: float = 0.0
        self.__number_of_execution_runs = 0
        self.__this_start_time: float = 0.0

    def track_start(self):
        self.__this_start_time = time.time()

    def track_end(self):
        end_time = time.time()
        self.__history.append(end_time - self.__this_start_time)
        self.__number_of_execution_runs += 1
        self.__overall_execution_time += self.__history[-1]

    def __len__(self):
        return len(self.__history)

    def __getitem__(self, item):
        return self.__history[item]

    def __iter__(self):
        return iter(self.__history)

    @property
    def average(self):
        if len(self.__history) < 1:
            return 0
        return round(sum(self.__history) / len(self.__history),3)

    @property
    def median(self):
        if len(self.__history) < 1:
            return 0
        return round(statistics.median(self.__history),3)

    @property
    def stdev(self):
        if len(self.__history) < 2:
            return 0
        return round(statistics.stdev(self.__history),3)

    @property
    def overall_execution_time(self) -> float:
        return self.__overall_execution_time

    @property
    def number_of_executions(self):
        return self.__number_of_execution_runs

    @property
    def label(self) -> str:
        return self.__label

    def __str__(self):
        return f"{self.__label}: run={self.__number_of_execution_runs} time={self.__history[-1]} avg={self.average} sd={self.stdev}"

