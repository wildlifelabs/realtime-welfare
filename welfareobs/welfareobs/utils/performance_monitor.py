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
from collections import deque
import time
import statistics
import csv


class PerformanceMonitor(object):
    def __init__(self,
                 label: str = "",
                 history_size: int = 1
                 ):
        self.__label = label
        self.__history_size = history_size
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

    def save(self, filename, append=True):
        start_run: int = (self.__number_of_execution_runs - self.__history_size) if self.__history_size < self.__number_of_execution_runs else 0
        end_run: int = self.__number_of_execution_runs
        with open(filename, 'a' if append else 'w') as csvfile:
            writer = csv.writer(csvfile)
            if not csvfile.tell():  # Check if file is empty
                writer.writerow(['run', 'time', 'sum', 'avg', 'med', 'sd'])
            state = []
            for index, item in enumerate(self.__history):
                state.append(item)
                sdev = 0
                if len(state) > 1:
                    sdev = statistics.stdev(state)
                writer.writerow([index + start_run, item, sum(state), sum(state) / len(state), statistics.median(state), sdev])
    